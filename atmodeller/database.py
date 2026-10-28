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
(:meth:`DataBase.default`) unless another is given, for example:

.. code-block:: python

    db: DataBase = DataBase.from_glenn_file("my_glenn_coefficients.txt")
    db.stitchable_components("SiO2")  # quartz-alpha, quartz-beta, cristobalite-beta, liquid

    h2o_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", database=db)
    h2o_l: ChemicalSpecies = ChemicalSpecies.from_component("liquid", formula="H2O", database=db)
    sio2: PurePhase = PurePhase.from_formula("SiO2", database=db)  # all stitched, O2Si_cd
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Self, cast

from molmass import Formula

from atmodeller.constants import GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT
from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.glenn import (
    NasaGlennThermodynamicProperties,
    glenn_properties,
    read_glenn_coefficients,
)
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson

_SHARED_COMPONENTS: tuple[str, ...] = (GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT)
"""Phase components that are shared by all formulas, unlike a mineral"""


class DataBase:
    """Thermodynamic database of phase components

    The thermodynamic data are keyed by Hill formula and then phase component, e.g.
    ``["O2Si"]["quartz-alpha"]`` or ``["H2O"]["gas"]``. Only the phase components that the
    database is created with, which stay so if their data are replaced, can be stitched together,
    whereas phase components added later, such as diamond, can only be selected on their own.
    The default database is :meth:`default`, and a file of NASA Glenn coefficients is read with
    :meth:`from_glenn_file`. Data from any source can be given, e.g.
    ``DataBase({"H2O": {"gas": ..., "liquid": ...}})``, and
    ``DataBase(glenn_properties)`` (:data:`~atmodeller.thermodata.glenn.glenn_properties`) is the
    packaged NASA Glenn coefficients without the changes of the default database.

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
        self._formula_of_mineral: dict[str, str] = {}
        for formula, components in (phase_components or {}).items():
            for component, properties in components.items():
                self.set_phase_component(formula, component, properties)
        self._created_with: frozenset[tuple[str, str]] = frozenset(
            (hill_formula, component)
            for hill_formula, components in self._phase_components.items()
            for component in components
        )

    @classmethod
    def default(cls) -> Self:
        """Creates the default database

        This is the packaged NASA Glenn coefficients
        (:data:`~atmodeller.thermodata.glenn.glenn_properties`) with the following changes:

            - Graphite includes the volume
              :data:`~atmodeller.thermodata.gustafson.GRAPHITE_VOLUME_GUSTAFSON`, so that its
              Gibbs energy, and hence the equilibrium constant of any reaction involving it,
              depends on pressure.
            - Diamond (:data:`~atmodeller.thermodata.gustafson.diamond_gustafson`) is added,
              created with ``ChemicalSpecies.from_component("diamond")``. Its Gibbs energy is that
              of graphite plus the Gibbs energy of diamond relative to graphite of
              :cite:t:`Gustafson1986`, valid to 6000 K, and it has its own volume.

        The volumes change the Gibbs energies of graphite and diamond, not their activities.
        Gases and other condensates are unchanged. Each call gives a new database, to which
        further phase components can be added with :meth:`set_phase_component`.

        Returns:
            The default database
        """
        database = cls(glenn_properties)
        graphite = database.get_phase_component("C", "graphite")
        # Graphite stays a phase component that the database was created with, so it is selected
        # by PurePhase.from_formula("C"). Diamond is added, so it is only selected on its own.
        database.set_phase_component(
            "C", "graphite", graphite.with_volume(GRAPHITE_VOLUME_GUSTAFSON)
        )
        database.set_phase_component("C", "diamond", diamond_gustafson)

        return database

    @classmethod
    def from_glenn_file(cls, path: str | Path) -> Self:
        """Creates a database from a file of NASA Glenn coefficients :cite:p:`MZG02`

        Only the data of the file are included. For the default database, which also changes
        graphite and adds diamond, use :meth:`default`.

        Args:
            path: Path to a file of NASA Glenn coefficients in the same format as the packaged
                file

        Returns:
            A database
        """
        return cls(read_glenn_coefficients(path))

    def phase_components(
        self, elements: Iterable[str] | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Phase components of each formula

        Args:
            elements: Elements to which the formulas are limited, e.g. ``("C", "H", "O")``.
                Defaults to ``None``, which lists all formulas.

        Returns:
            Names of the phase components, keyed by Hill formula, e.g.
            ``{"H2O": ("gas", "ice", "liquid"), ...}``
        """
        allowed: frozenset[str] | None = None if elements is None else frozenset(elements)

        return {
            hill_formula: tuple(components)
            for hill_formula, components in self._phase_components.items()
            if allowed is None or set(Formula(hill_formula).composition()) <= allowed
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
        hill_formula: str = Formula(formula).formula
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

        A phase component that the database was created with stays so, and can be stitched
        together with the others (see :meth:`stitch`). A phase component that is added can only be
        selected on its own.

        Args:
            formula: Formula
            component: Name of the phase component, e.g. ``diamond``
            properties: Thermodynamic properties

        Raises:
            ValueError: If the phase component is a mineral of another formula
        """
        hill_formula: str = Formula(formula).formula
        if component not in _SHARED_COMPONENTS:
            other: str = self._formula_of_mineral.setdefault(component, hill_formula)
            if other != hill_formula:
                raise ValueError(f"Phase component {component} is already given for {other}")
        self._phase_components.setdefault(hill_formula, {})[component] = properties

    def formula_of(self, *components: str) -> str:
        """Hill formula of phase components, from a mineral among them

        A mineral is a phase component of one formula only, so it gives the formula, whereas the
        gas and liquid are phase components of several formulas.

        Args:
            *components: Names of the phase components, e.g. ``quartz-alpha``, or
                ``quartz-beta`` and ``liquid``

        Returns:
            Hill formula

        Raises:
            ValueError: If no phase component is given, or all are shared by several formulas,
                such as ``liquid``
            KeyError: If the mineral is not available
        """
        if not components:
            raise ValueError("At least one phase component is needed")
        minerals: list[str] = [
            component for component in components if component not in _SHARED_COMPONENTS
        ]
        if not minerals:
            raise ValueError(
                f"{' and '.join(components)} {'is' if len(components) == 1 else 'are'} shared by "
                "several formulas, so give the formula"
            )
        try:
            return self._formula_of_mineral[minerals[0]]
        except KeyError:
            raise KeyError(f"Phase component {minerals[0]} is not available") from None

    def stitchable_components(self, formula: str) -> tuple[str, ...]:
        """Condensed phase components of a formula that can be stitched together

        These are the condensed phase components that the database was created with
        (:meth:`stitch`).

        Args:
            formula: Formula

        Returns:
            Names of the condensed phase components, in order of temperature, e.g.
            ``("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")`` for ``SiO2``
        """
        hill_formula: str = Formula(formula).formula
        components: dict[str, ThermodynamicProperties] = self._phase_components.get(
            hill_formula, {}
        )

        return tuple(
            sorted(
                (
                    component
                    for component in components
                    if component != GAS_PHASE_COMPONENT
                    and (hill_formula, component) in self._created_with
                ),
                key=lambda component: components[component].temperature_range()[0],
            )
        )

    def stitch(self, formula: str, components: Iterable[str]) -> ThermodynamicProperties:
        """Stitches condensed phase components of a formula together in order of temperature

        A single phase component is returned as it is, with any volume model. Several phase
        components must all be ones that the database was created with, be NASA Glenn polynomials
        and have no volume model, since the pressure dependence would move the transitions between
        them, and their temperature ranges must not overlap or leave a gap.

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

        hill_formula: str = Formula(formula).formula
        added: list[str] = [
            component
            for component in components
            if (hill_formula, component) not in self._created_with
        ]
        if added:
            raise ValueError(
                "Only phase components that the database was created with can be stitched "
                f"together, not {added}"
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

        return NasaGlennThermodynamicProperties.stitch(
            cast(dict[str, NasaGlennThermodynamicProperties], found)
        )

