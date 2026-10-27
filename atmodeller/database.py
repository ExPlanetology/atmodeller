# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Thermodynamic databases of phase components

A database holds the thermodynamic data of each phase component, the gas, liquid or mineral to
which the data refer, keyed by Hill formula and then phase component. Species are created from a
database by :meth:`~atmodeller.containers.ChemicalSpecies.create_gas`,
:meth:`~atmodeller.containers.ChemicalSpecies.from_component` and the methods of
:class:`~atmodeller.phases.PurePhase`, which use the default database
(:func:`get_default_database`) unless another is given, for example:

.. code-block:: python

    db: DataBase = GlennDataBase.from_file("my_glenn_coefficients.txt")
    db.components_of("SiO2")  # ("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")

    h2o_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", database=db)
    h2o_l: ChemicalSpecies = ChemicalSpecies.from_component("liquid", formula="H2O", database=db)
    sio2: PurePhase = PurePhase.from_formula("SiO2", database=db)  # all stitched, O2Si_cd
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Self, cast

from molmass import Formula

from atmodeller import override
from atmodeller.constants import GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT
from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.defaults import default_properties
from atmodeller.thermodata.janaf import (
    NasaGlennThermodynamicProperties,
    glenn_properties,
    read_glenn_coefficients,
    stitch_glenn_properties,
)

_SHARED_COMPONENTS: tuple[str, ...] = (GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT)
"""Phase components that are shared by all formulas, unlike a mineral"""


@cache
def _hill(formula: str) -> str:
    """Hill formula, cached since databases look up formulas many times"""
    return Formula(formula).formula


class DataBase(ABC):
    """Thermodynamic database of phase components

    The thermodynamic data are keyed by Hill formula and then phase component, e.g.
    ``["O2Si"]["quartz-alpha"]`` or ``["H2O"]["gas"]``. The phase components that the database is
    created with are from a data file, and stay so if their data are replaced. Only these can be
    stitched together, whereas phase components added later, such as diamond, can only be selected
    on their own.

    Args:
        phase_components: Thermodynamic properties of each phase component, keyed by Hill formula
            and then phase component. Defaults to none.

    Raises:
        ValueError: If the same mineral is given for more than one formula
    """

    def __init__(
        self, phase_components: Mapping[str, Mapping[str, ThermodynamicProperties]] | None = None
    ):
        self._phase_components: dict[str, dict[str, ThermodynamicProperties]] = {}
        for formula, components in (phase_components or {}).items():
            for component, properties in components.items():
                self.set_phase_component(formula, component, properties)
        self._from_data_file: frozenset[tuple[str, str]] = frozenset(
            (hill_formula, component)
            for hill_formula, components in self._phase_components.items()
            for component in components
        )

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        """Creates the default database.

        Returns:
            The default database
        """
        raise NotImplementedError

    def phase_components(self) -> dict[str, tuple[str, ...]]:
        """Phase components of each formula

        Returns:
            Names of the phase components, keyed by Hill formula, e.g.
            ``{"H2O": ("gas", "ice", "liquid"), ...}``
        """
        return {
            hill_formula: tuple(components)
            for hill_formula, components in self._phase_components.items()
        }

    def get_phase_component(self, formula: str, component: str) -> ThermodynamicProperties:
        """Gets the thermodynamic properties of a phase component

        Args:
            formula: Formula
            component: Name of the phase component, e.g. ``gas``, ``liquid`` or ``quartz-alpha``

        Returns:
            Thermodynamic properties

        Raises:
            KeyError: If the phase component is not available for the formula
        """
        hill_formula: str = _hill(formula)
        if hill_formula not in self._phase_components:
            raise KeyError(f"{hill_formula} is not available")
        components: dict[str, ThermodynamicProperties] = self._phase_components[hill_formula]
        try:
            return components[component]
        except KeyError:
            raise KeyError(
                f"{component} is not a phase component of {hill_formula}. Its phase components "
                f"are {tuple(components)}"
            ) from None

    def set_phase_component(
        self, formula: str, component: str, properties: ThermodynamicProperties
    ) -> None:
        """Sets the thermodynamic properties of a phase component, adding it or replacing it

        A phase component that the database was created with stays a phase component from a data
        file, and can be stitched together with the others (see :meth:`stitch`). A phase component
        that is added can only be selected on its own.

        Args:
            formula: Formula
            component: Name of the phase component, e.g. ``diamond``
            properties: Thermodynamic properties

        Raises:
            ValueError: If the phase component is a mineral of another formula
        """
        hill_formula: str = _hill(formula)
        if component not in _SHARED_COMPONENTS:
            other: list[str] = [
                formula_ for formula_ in self._formulas_of(component) if formula_ != hill_formula
            ]
            if other:
                raise ValueError(f"Phase component {component} is already given for {other[0]}")
        self._phase_components.setdefault(hill_formula, {})[component] = properties

    def formula_of(self, component: str) -> str:
        """Hill formula of a mineral

        Args:
            component: Name of the mineral, e.g. ``quartz-alpha``

        Returns:
            Hill formula

        Raises:
            ValueError: If the phase component is shared by several formulas, such as ``liquid``
            KeyError: If the phase component is not available
        """
        if component in _SHARED_COMPONENTS:
            raise ValueError(
                f"{component} is a phase component of several formulas, so give the formula"
            )
        formulas: list[str] = self._formulas_of(component)
        if not formulas:
            raise KeyError(f"Phase component {component} is not available")

        return formulas[0]

    def components_of(self, formula: str) -> tuple[str, ...]:
        """Condensed phase components of a formula from a data file

        Args:
            formula: Formula

        Returns:
            Names of the condensed phase components, in order of temperature, e.g.
            ``("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")`` for ``SiO2``
        """
        hill_formula: str = _hill(formula)
        components: dict[str, ThermodynamicProperties] = self._phase_components.get(
            hill_formula, {}
        )

        return tuple(
            sorted(
                (
                    component
                    for component in components
                    if component != GAS_PHASE_COMPONENT
                    and (hill_formula, component) in self._from_data_file
                ),
                key=lambda component: components[component].temperature_range()[0],
            )
        )

    def stitch(self, formula: str, components: Iterable[str]) -> ThermodynamicProperties:
        """Stitches condensed phase components of a formula together in order of temperature

        A single phase component is returned as it is, with any volume model. Several phase
        components must all be from a data file, be NASA Glenn polynomials and have no volume
        model, since the pressure dependence would move the transitions between them, and their
        temperature ranges must not overlap or leave a gap.

        Args:
            formula: Formula
            components: Names of the phase components, e.g. ``("quartz-alpha", "quartz-beta")``

        Returns:
            Thermodynamic properties of the phase components stitched together

        Raises:
            KeyError: If a phase component is not available for the formula
            ValueError: If the phase components cannot be stitched together
        """
        components = tuple(components)
        if not components:
            raise ValueError("At least one phase component is needed")
        if GAS_PHASE_COMPONENT in components:
            raise ValueError(
                "The gas is not a condensed phase component. Create it with create_gas."
            )
        found: dict[str, ThermodynamicProperties] = {
            component: self.get_phase_component(formula, component) for component in components
        }
        if len(found) == 1:
            return next(iter(found.values()))

        hill_formula: str = _hill(formula)
        added: list[str] = [
            component
            for component in components
            if (hill_formula, component) not in self._from_data_file
        ]
        if added:
            raise ValueError(
                f"Only phase components from a data file can be stitched together, not {added}"
            )
        not_glenn: list[str] = [
            component
            for component, properties in found.items()
            if not isinstance(properties, NasaGlennThermodynamicProperties)
        ]
        if not_glenn:
            raise ValueError(
                f"Only NASA Glenn polynomials can be stitched together, not {not_glenn}"
            )

        return stitch_glenn_properties(cast(dict[str, NasaGlennThermodynamicProperties], found))

    def _formulas_of(self, component: str) -> list[str]:
        """Hill formulas that have the phase component"""
        return [
            hill_formula
            for hill_formula, components in self._phase_components.items()
            if component in components
        ]


class GlennDataBase(DataBase):
    """Database of NASA Glenn coefficients :cite:p:`MZG02`

    Args:
        path: Path to a file of NASA Glenn coefficients in the same format as the packaged file.
            Defaults to ``None``, which uses the packaged file.
    """

    def __init__(self, path: str | Path | None = None):
        super().__init__(glenn_properties if path is None else read_glenn_coefficients(path))

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
        :data:`~atmodeller.thermodata.defaults.default_properties`.

        Returns:
            The default database
        """
        database = cls()
        # Graphite stays a phase component from the data file, so it is selected by
        # PurePhase.from_formula("C"). Diamond is added, so it is only selected on its own.
        for component in ("graphite", "diamond"):
            database.set_phase_component("C", component, default_properties["C"][component])

        return database


def get_default_database() -> GlennDataBase:
    """Gets the default database

    This is the database of packaged NASA Glenn coefficients with the following changes
    (:data:`~atmodeller.thermodata.defaults.default_properties`):

        - Graphite includes the volume
          :data:`~atmodeller.thermodata.gustafson.GRAPHITE_VOLUME_GUSTAFSON`, so that its Gibbs
          energy depends on pressure.
        - Diamond (:data:`~atmodeller.thermodata.gustafson.diamond_gustafson`) is added, created
          with ``ChemicalSpecies.from_component("diamond")``. Its Gibbs energy relative to graphite
          is from the assessment of :cite:t:`Gustafson1986`, valid to 6000 K.

    Further phase components can be added with :meth:`DataBase.set_phase_component`.

    Returns:
        The default database
    """
    return GlennDataBase.default()


@cache
def _default_database() -> GlennDataBase:
    """The default database shared by the species created without a database, which must not be
    changed. Use :func:`get_default_database` for a copy to change."""
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
    return (database or _default_database()).components_of(formula)
