# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Thermodynamic databases for creating chemical species

A database provides thermodynamic data for species and creates
:class:`~atmodeller.containers.ChemicalSpecies` from them, for example:

.. code-block:: python

    db: DataBase = GlennDataBase()
    h2o_g: ChemicalSpecies = db.create_gas("H2O")
    h2o_l: ChemicalSpecies = db.create_condensed("H2O", state="l")

    db_custom: DataBase = GlennDataBase.from_file("my_glenn_coefficients.txt")

    db_default: DataBase = get_default_database()
    c_diamond: ChemicalSpecies = db_default.create_condensed("C", state="diamond")

A condensed species can also be created from its phase components, the gas, liquid or minerals
to which the data refer:

.. code-block:: python

    components_of("SiO2")  # ("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")
    sio2_cd: ChemicalSpecies = db.condensed_from_formula("SiO2")  # all stitched, O2Si_cd
    quartz: ChemicalSpecies = db.condensed_from_components(
        ("quartz-alpha", "quartz-beta"), name="quartz"
    )  # O2Si_quartz
    cristobalite: ChemicalSpecies = db.condensed_from_component("cristobalite-beta")
    h2o_l: ChemicalSpecies = db.condensed_from_component("liquid", formula="H2O")
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, cast

from molmass import Formula

from atmodeller import override
from atmodeller.constants import (
    CONDENSED_STATE,
    GAS_PHASE_COMPONENT,
    GAS_STATE,
    LIQUID_PHASE_COMPONENT,
    LIQUID_STATE,
    SOLID_STATE,
)
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import ChemicalSpeciesData
from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.defaults import default_properties
from atmodeller.thermodata.janaf import (
    NasaGlennThermodynamicProperties,
    glenn_phase_components,
    glenn_properties,
    read_glenn_coefficients,
    read_glenn_phase_components,
    stitch_glenn_properties,
)


class DataBase(ABC):
    """Thermodynamic database for creating chemical species

    Subclasses provide the thermodynamic data, and this base class creates
    :class:`~atmodeller.containers.ChemicalSpecies` from them.

    The database also keeps the thermodynamic data of each phase component, keyed by Hill formula
    and then phase component. Phase components from a data file can be stitched together, whereas
    those added with :meth:`add_species` can only be selected on their own.

    Args:
        phase_components: Thermodynamic properties of each phase component from a data file,
            keyed by Hill formula and then phase component. Defaults to none.
    """

    def __init__(
        self, phase_components: Mapping[str, Mapping[str, ThermodynamicProperties]] | None = None
    ):
        # Copy so that changing this database does not change the data shared by others
        self._phase_components: dict[str, dict[str, ThermodynamicProperties]] = {
            hill_formula: dict(components)
            for hill_formula, components in (phase_components or {}).items()
        }
        self._added_phase_components: dict[str, dict[str, ThermodynamicProperties]] = {}

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        """Creates the default database.

        Returns:
            The default database
        """
        raise NotImplementedError

    @abstractmethod
    def set_thermodynamic_properties(self, name: str, properties: ThermodynamicProperties) -> None:
        """Sets the thermodynamic properties of a species, replacing any existing ones

        Args:
            name: Name of the species, as Hill formula and state, e.g. ``H2O_g``
            properties: Thermodynamic properties
        """
        raise NotImplementedError

    @abstractmethod
    def available_species(self) -> tuple[str, ...]:
        """Available species

        Returns:
            Names of the available species, as Hill formula and state, e.g. ``H2O_g``
        """
        raise NotImplementedError

    @abstractmethod
    def get_thermodynamic_properties(self, name: str) -> ThermodynamicProperties:
        """Gets the thermodynamic properties of a species

        Args:
            name: Name of the species, as Hill formula and state, e.g. ``H2O_g``

        Returns:
            Thermodynamic properties

        Raises:
            KeyError: If the species is not available
        """
        raise NotImplementedError

    def add_species(self, formula: str, state: str, properties: ThermodynamicProperties) -> None:
        """Adds a species to the database, or replaces it if it already exists.

        The species is named from its formula and state in the same way as when it is created, so
        that ``add_species("C", "diamond", ...)`` is found by
        ``create_condensed("C", state="diamond")``.

        Args:
            formula: Formula
            state: State of aggregation, or any other label, e.g. ``diamond``
            properties: Thermodynamic properties
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, state)
        self.set_thermodynamic_properties(species_data.name, properties)

        # Also add it as a phase component, except for the stitched solids (state "s")
        if state == SOLID_STATE:
            return
        component: str = {
            GAS_STATE: GAS_PHASE_COMPONENT,
            LIQUID_STATE: LIQUID_PHASE_COMPONENT,
        }.get(state, state)
        hill_formula: str = species_data.hill_formula
        if component not in (GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT):
            other: list[str] = [
                formula_ for formula_ in self._formulas_of(component) if formula_ != hill_formula
            ]
            if other:
                raise ValueError(f"Phase component {component} is already given for {other[0]}")
        self._phase_components.get(hill_formula, {}).pop(component, None)
        self._added_phase_components.setdefault(hill_formula, {})[component] = properties

    def components_of(self, formula: str) -> tuple[str, ...]:
        """Condensed phase components of a formula from a data file

        Args:
            formula: Formula

        Returns:
            Names of the condensed phase components, in order of temperature, e.g.
            ``("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")`` for ``SiO2``
        """
        components: dict[str, ThermodynamicProperties] = self._phase_components.get(
            Formula(formula).formula, {}
        )

        return tuple(
            sorted(
                (component for component in components if component != GAS_PHASE_COMPONENT),
                key=lambda component: components[component].temperature_range()[0],
            )
        )

    def condensed_from_formula(self, formula: str, **kwargs: Any) -> ChemicalSpecies:
        """Creates a condensed species from all the condensed phase components of a formula

        The phase components from a data file (:meth:`components_of`) are stitched together in
        order of temperature, e.g. quartz to cristobalite to liquid for ``SiO2``, and the species
        is named with :data:`~atmodeller.constants.CONDENSED_STATE`, e.g. ``O2Si_cd``.

        Args:
            formula: Formula
            **kwargs: Keyword arguments for
                :meth:`~atmodeller.containers.ChemicalSpecies.create_condensed`

        Returns:
            A condensed species

        Raises:
            KeyError: If the formula has no condensed phase components
            ValueError: If the phase components cannot be stitched together
        """
        components: tuple[str, ...] = self.components_of(formula)
        if not components:
            raise KeyError(f"{Formula(formula).formula} has no condensed phase components")

        return self.condensed_from_components(
            components, name=CONDENSED_STATE, formula=formula, **kwargs
        )

    def condensed_from_components(
        self, components: Iterable[str], *, name: str, formula: str | None = None, **kwargs: Any
    ) -> ChemicalSpecies:
        """Creates a condensed species by stitching phase components together

        The phase components must be of one formula, and are stitched together in order of
        temperature, without overlaps or gaps. Several phase components must all be from a data
        file and have no volume model, since the pressure dependence would move the transitions
        between them. A single phase component can be from anywhere and keeps any volume model.

        Args:
            components: Names of the phase components, e.g. ``("quartz-alpha", "quartz-beta")``
            name: Suffix of the species name, e.g. ``quartz`` for ``O2Si_quartz``
            formula: Formula. Defaults to ``None``, which takes the formula of the minerals, and
                is only needed when the components are all shared by several formulas, such as
                ``liquid``.
            **kwargs: Keyword arguments for
                :meth:`~atmodeller.containers.ChemicalSpecies.create_condensed`

        Returns:
            A condensed species

        Raises:
            KeyError: If a phase component is not available for the formula
            ValueError: If the phase components cannot be stitched together
        """
        components = tuple(components)
        if not components:
            raise ValueError("At least one phase component is needed")

        # A mineral is a phase component of one formula only, so it gives the formula
        if formula is None:
            minerals: list[str] = [
                component
                for component in components
                if component not in (GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT)
            ]
            if minerals:
                formula = self._get_phase_component(minerals[0], None)[0]

        found: dict[str, tuple[str, ThermodynamicProperties, bool]] = {
            component: self._get_phase_component(component, formula) for component in components
        }
        added: list[str] = [
            component for component, (_, _, from_file) in found.items() if not from_file
        ]
        if len(components) > 1 and added:
            raise ValueError(
                f"Only phase components from a data file can be stitched together, not {added}"
            )
        hill_formula: str = next(iter(found.values()))[0]
        thermo: ThermodynamicProperties = stitch_glenn_properties(
            {
                component: cast(NasaGlennThermodynamicProperties, properties)
                for component, (_, properties, _) in found.items()
            }
        )

        return ChemicalSpecies.create_condensed(hill_formula, state=name, thermo=thermo, **kwargs)

    def condensed_from_component(
        self, component: str, *, formula: str | None = None, **kwargs: Any
    ) -> ChemicalSpecies:
        """Creates a condensed species from one phase component

        The species is named from the phase component, e.g. ``O2Si_cristobalite-beta``, or
        ``H2O_l`` for the liquid.

        Args:
            component: Name of the phase component, e.g. ``cristobalite-beta``
            formula: Formula. Defaults to ``None``, which takes the formula of a mineral, and is
                needed for ``liquid``, which is a phase component of several formulas.
            **kwargs: Keyword arguments for
                :meth:`~atmodeller.containers.ChemicalSpecies.create_condensed`

        Returns:
            A condensed species

        Raises:
            KeyError: If the phase component is not available for the formula
            ValueError: If the phase component is the gas, or the liquid without a formula
        """
        hill_formula, thermo, _ = self._get_phase_component(component, formula)
        state: str = LIQUID_STATE if component == LIQUID_PHASE_COMPONENT else component

        return ChemicalSpecies.create_condensed(hill_formula, state=state, thermo=thermo, **kwargs)

    def _formulas_of(self, component: str) -> list[str]:
        """Hill formulas that have the phase component, added ones first"""
        return [
            hill_formula
            for table in (self._added_phase_components, self._phase_components)
            for hill_formula, components in table.items()
            if component in components
        ]

    def _get_phase_component(
        self, component: str, formula: str | None
    ) -> tuple[str, ThermodynamicProperties, bool]:
        """Gets a condensed phase component

        Args:
            component: Name of the phase component
            formula: Formula, or ``None`` to take the formula of a mineral

        Returns:
            Hill formula, thermodynamic properties, and whether the phase component is from a
            data file
        """
        if component == GAS_PHASE_COMPONENT:
            raise ValueError(
                "The gas is not a condensed phase component. Create it with create_gas."
            )
        formulas: list[str] = self._formulas_of(component)

        if formula is not None:
            hill_formula: str = Formula(formula).formula
            if hill_formula not in formulas:
                raise KeyError(
                    f"{component} is not a phase component of {hill_formula}. Its condensed "
                    f"phase components are {self.components_of(hill_formula)}"
                )
        elif component == LIQUID_PHASE_COMPONENT:
            raise ValueError(
                f"{component} is a phase component of several formulas, so give the formula"
            )
        elif not formulas:
            raise KeyError(f"Phase component {component} is not available")
        else:
            hill_formula = formulas[0]

        added: dict[str, ThermodynamicProperties] = self._added_phase_components.get(
            hill_formula, {}
        )
        if component in added:
            return hill_formula, added[component], False

        return hill_formula, self._phase_components[hill_formula][component], True

    def _get_thermodynamic_properties(self, formula: str, state: str) -> ThermodynamicProperties:
        """Gets the thermodynamic properties of a species with an informative error

        Args:
            formula: Formula
            state: State of aggregation

        Returns:
            Thermodynamic properties
        """
        name: str = ChemicalSpeciesData(formula, state).name

        try:
            return self.get_thermodynamic_properties(name)
        except KeyError:
            raise KeyError(
                f"{name} not available. Available species are {self.available_species()}"
            ) from None

    def create_condensed(
        self, formula: str, *, state: str = SOLID_STATE, **kwargs: Any
    ) -> ChemicalSpecies:
        """Creates a condensed species.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :data:`~atmodeller.constants.SOLID_STATE`.
            **kwargs: Keyword arguments for
                :meth:`~atmodeller.containers.ChemicalSpecies.create_condensed`

        Returns:
            A condensed species
        """
        thermo: ThermodynamicProperties = self._get_thermodynamic_properties(formula, state)

        return ChemicalSpecies.create_condensed(formula, state=state, thermo=thermo, **kwargs)

    def create_gas(
        self, formula: str, *, state: str = GAS_STATE, **kwargs: Any
    ) -> ChemicalSpecies:
        """Creates a gas species.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :data:`~atmodeller.constants.GAS_STATE`.
            **kwargs: Keyword arguments for
                :meth:`~atmodeller.containers.ChemicalSpecies.create_gas`

        Returns:
            A gas species
        """
        thermo: ThermodynamicProperties = self._get_thermodynamic_properties(formula, state)

        return ChemicalSpecies.create_gas(formula, state=state, thermo=thermo, **kwargs)


class GlennDataBase(DataBase):
    """Database of NASA Glenn coefficients :cite:p:`MZG02`

    Args:
        path: Path to a file of NASA Glenn coefficients in the same format as the packaged file.
            Defaults to ``None``, which uses the packaged file.
    """

    def __init__(self, path: str | Path | None = None):
        # The packaged data are already loaded, so reuse them. Copy the dictionary so that adding
        # species does not change the packaged data shared by other databases.
        self._coefficients: dict[str, ThermodynamicProperties] = (
            dict(glenn_properties) if path is None else read_glenn_coefficients(path)
        )
        super().__init__(
            glenn_phase_components if path is None else read_glenn_phase_components(path)
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Creates a database from a file of NASA Glenn coefficients.

        Args:
            path: Path to the file

        Returns:
            A database
        """
        return cls(path)

    @override
    @classmethod
    def default(cls) -> Self:
        """Creates the database of the default data.

        This is the packaged NASA Glenn coefficients with graphite and diamond from
        :data:`~atmodeller.thermodata.defaults.default_properties`, the same data used by
        :meth:`~atmodeller.containers.ChemicalSpecies.create_gas` and
        :meth:`~atmodeller.containers.ChemicalSpecies.create_condensed`.

        Returns:
            The default database
        """
        database = cls()
        for name in ("C_s", "C_graphite"):
            database.set_thermodynamic_properties(name, default_properties[name])
        # Graphite stays a phase component from the data file, with the volume, so it is still
        # selected by condensed_from_formula("C"). Diamond is added and so is only selected alone.
        database._phase_components["C"]["graphite"] = default_properties["C_graphite"]
        database.add_species("C", "diamond", default_properties["C_diamond"])

        return database

    @override
    def set_thermodynamic_properties(self, name: str, properties: ThermodynamicProperties) -> None:
        """Sets the thermodynamic properties of a species, replacing any existing ones

        Args:
            name: Name of the species, as Hill formula and state, e.g. ``H2O_g``
            properties: Thermodynamic properties
        """
        self._coefficients[name] = properties

    @override
    def available_species(self) -> tuple[str, ...]:
        """Available species

        Returns:
            Names of the available species, as Hill formula and state, e.g. ``H2O_g``
        """
        return tuple(self._coefficients)

    @override
    def get_thermodynamic_properties(self, name: str) -> ThermodynamicProperties:
        """Gets the thermodynamic properties of a species

        Args:
            name: Name of the species, as Hill formula and state, e.g. ``H2O_g``

        Returns:
            Thermodynamic properties

        Raises:
            KeyError: If the species is not available
        """
        return self._coefficients[name]


def get_default_database() -> GlennDataBase:
    """Gets the default database

    This is the database of packaged NASA Glenn coefficients with the following changes, which are
    also the defaults of :meth:`~atmodeller.containers.ChemicalSpecies.create_gas` and
    :meth:`~atmodeller.containers.ChemicalSpecies.create_condensed`
    (:data:`~atmodeller.thermodata.defaults.default_properties`):

        - Graphite (``C_s`` and ``C_graphite``) includes the volume
          :data:`~atmodeller.thermodata.gustafson.GRAPHITE_VOLUME_GUSTAFSON`, so that its Gibbs
          energy depends on pressure.
        - Diamond (:data:`~atmodeller.thermodata.gustafson.diamond_gustafson`) is added, created
          with ``create_condensed("C", state="diamond")`` or
          ``condensed_from_component("diamond")``. Its Gibbs energy relative to graphite is from
          the assessment of :cite:t:`Gustafson1986`, valid to 6000 K.

    Further species can be added with :meth:`DataBase.add_species`.

    Returns:
        The default database
    """
    return GlennDataBase.default()


def components_of(formula: str, database: DataBase | None = None) -> tuple[str, ...]:
    """Condensed phase components of a formula from a data file

    Args:
        formula: Formula
        database: Database. Defaults to ``None``, which uses the default database.

    Returns:
        Names of the condensed phase components, in order of temperature, e.g.
        ``("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")`` for ``SiO2``
    """
    return (database or get_default_database()).components_of(formula)
