# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Thermodynamic databases of end-members

A database holds the thermodynamic data of each end-member, the gas, liquid or mineral to
which the data refer, keyed by Hill formula and then end-member. Species are created from a
database by :meth:`~atmodeller.containers.ChemicalSpecies.create_gas`,
:meth:`~atmodeller.containers.ChemicalSpecies.from_end_member` and the methods of
:class:`~atmodeller.phases.PurePhase`, which use the default database
(:meth:`DataBase.default`) unless another is given, for example:

.. code-block:: python

    db: DataBase = DataBase.from_glenn_file("my_glenn_coefficients.txt")
    db.stitchable_end_members("SiO2")  # quartz-alpha, quartz-beta, cristobalite-beta, liquid

    h2o_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", database=db)
    h2o_l: ChemicalSpecies = ChemicalSpecies.from_end_member("liquid", formula="H2O", database=db)
    sio2: PurePhase = PurePhase.from_formula("SiO2", database=db)  # all stitched, O2Si_cd
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Self, cast

from molmass import Formula

from atmodeller.constants import GAS_END_MEMBER, LIQUID_END_MEMBER
from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.glenn import (
    NasaGlennThermodynamicProperties,
    glenn_properties,
    read_glenn_coefficients,
)
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson

_SHARED_END_MEMBERS: tuple[str, ...] = (GAS_END_MEMBER, LIQUID_END_MEMBER)
"""End-members that are shared by all formulas, unlike a mineral"""


class DataBase:
    """Thermodynamic database of end-members

    The thermodynamic data are keyed by Hill formula and then end-member, e.g.
    ``["O2Si"]["quartz-alpha"]`` or ``["H2O"]["gas"]``. Only the end-members that the
    database is created with, which stay so if their data are replaced, can be stitched together,
    whereas end-members added later, such as diamond, can only be selected on their own.
    The default database is :meth:`default`, and a file of NASA Glenn coefficients is read with
    :meth:`from_glenn_file`. Data from any source can be given, e.g.
    ``DataBase({"H2O": {"gas": ..., "liquid": ...}})``, and
    ``DataBase(glenn_properties)`` (:data:`~atmodeller.thermodata.glenn.glenn_properties`) is the
    packaged NASA Glenn coefficients without the changes of the default database.

    Args:
        end_members: Thermodynamic properties of each end-member, keyed by Hill formula
            and then end-member. Defaults to none.

    Raises:
        ValueError: If the same mineral is given for more than one formula
    """

    def __init__(
        self, end_members: Mapping[str, Mapping[str, ThermodynamicProperties]] | None = None
    ):
        self._end_members: dict[str, dict[str, ThermodynamicProperties]] = {}
        self._formula_of_mineral: dict[str, str] = {}
        for formula, formula_end_members in (end_members or {}).items():
            for end_member, properties in formula_end_members.items():
                self.set_end_member(formula, end_member, properties)
        self._created_with: frozenset[tuple[str, str]] = frozenset(
            (hill_formula, end_member)
            for hill_formula, formula_end_members in self._end_members.items()
            for end_member in formula_end_members
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
              created with ``ChemicalSpecies.from_end_member("diamond")``. Its Gibbs energy is that
              of graphite plus the Gibbs energy of diamond relative to graphite of
              :cite:t:`Gustafson1986`, valid to 6000 K, and it has its own volume.

        The volumes change the Gibbs energies of graphite and diamond, not their activities.
        Gases and other condensates are unchanged. Each call gives a new database, to which
        further end-members can be added with :meth:`set_end_member`.

        Returns:
            The default database
        """
        database = cls(glenn_properties)
        graphite = database.get_end_member("C", "graphite")
        # Graphite stays an end-member that the database was created with, so it is selected
        # by PurePhase.from_formula("C"). Diamond is added, so it is only selected on its own.
        database.set_end_member("C", "graphite", graphite.with_volume(GRAPHITE_VOLUME_GUSTAFSON))
        database.set_end_member("C", "diamond", diamond_gustafson)

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

    def end_members(self, elements: Iterable[str] | None = None) -> dict[str, tuple[str, ...]]:
        """End-members of each formula

        Args:
            elements: Elements to which the formulas are limited, e.g. ``("C", "H", "O")``.
                Defaults to ``None``, which lists all formulas.

        Returns:
            Names of the end-members, keyed by Hill formula, e.g.
            ``{"H2O": ("gas", "ice", "liquid"), ...}``
        """
        allowed: frozenset[str] | None = None if elements is None else frozenset(elements)

        return {
            hill_formula: tuple(end_members)
            for hill_formula, end_members in self._end_members.items()
            if allowed is None or set(Formula(hill_formula).composition()) <= allowed
        }

    def get_end_member(self, formula: str, end_member: str) -> ThermodynamicProperties:
        """Gets the thermodynamic properties of an end-member

        Args:
            formula: Formula
            end_member: Name of the end-member, e.g. ``gas``, ``liquid`` or ``quartz-alpha``

        Returns:
            Thermodynamic properties

        Raises:
            KeyError: If the end-member is not available for the formula
        """
        hill_formula: str = Formula(formula).formula
        if hill_formula not in self._end_members:
            raise KeyError(f"{hill_formula} is not available")
        end_members: dict[str, ThermodynamicProperties] = self._end_members[hill_formula]
        try:
            return end_members[end_member]
        except KeyError:
            raise KeyError(
                f"{end_member} is not an end-member of {hill_formula}. Its end-members "
                f"are {tuple(end_members)}"
            ) from None

    def set_end_member(
        self, formula: str, end_member: str, properties: ThermodynamicProperties
    ) -> None:
        """Sets the thermodynamic properties of an end-member, adding it or replacing it

        An end-member that the database was created with stays so, and can be stitched
        together with the others (see :meth:`stitch`). An end-member that is added can only be
        selected on its own.

        Args:
            formula: Formula
            end_member: Name of the end-member, e.g. ``diamond``
            properties: Thermodynamic properties

        Raises:
            ValueError: If the end-member is a mineral of another formula
        """
        hill_formula: str = Formula(formula).formula
        if end_member not in _SHARED_END_MEMBERS:
            other: str = self._formula_of_mineral.setdefault(end_member, hill_formula)
            if other != hill_formula:
                raise ValueError(f"End-member {end_member} is already given for {other}")
        self._end_members.setdefault(hill_formula, {})[end_member] = properties

    def formula_of(self, *end_members: str) -> str:
        """Hill formula of end-members, from a mineral among them

        A mineral is an end-member of one chemical formula only, so it gives the formula, whereas
        the gas and liquid are end-members of several formulas.

        Args:
            *end_members: Names of the end-members, e.g. ``quartz-alpha``, or
                ``quartz-beta`` and ``liquid``

        Returns:
            Hill formula

        Raises:
            ValueError: If no end-member is given, or all are shared by several formulas,
                such as ``liquid``
            KeyError: If the mineral is not available
        """
        if not end_members:
            raise ValueError("At least one end-member is needed")
        minerals: list[str] = [
            end_member for end_member in end_members if end_member not in _SHARED_END_MEMBERS
        ]
        if not minerals:
            verb: str = "is" if len(end_members) == 1 else "are"
            raise ValueError(
                f"{' and '.join(end_members)} {verb} shared by several formulas, so give the "
                "formula"
            )
        try:
            return self._formula_of_mineral[minerals[0]]
        except KeyError:
            raise KeyError(f"End-member {minerals[0]} is not available") from None

    def stitchable_end_members(self, formula: str) -> tuple[str, ...]:
        """Condensed end-members of a formula that can be stitched together

        These are the condensed end-members that the database was created with
        (:meth:`stitch`).

        Args:
            formula: Formula

        Returns:
            Names of the condensed end-members, in order of temperature, e.g.
            ``("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")`` for ``SiO2``
        """
        hill_formula: str = Formula(formula).formula
        end_members: dict[str, ThermodynamicProperties] = self._end_members.get(hill_formula, {})

        return tuple(
            sorted(
                (
                    end_member
                    for end_member in end_members
                    if end_member != GAS_END_MEMBER
                    and (hill_formula, end_member) in self._created_with
                ),
                key=lambda end_member: end_members[end_member].temperature_range()[0],
            )
        )

    def stitch(self, formula: str, end_members: Iterable[str]) -> ThermodynamicProperties:
        """Stitches condensed end-members of a formula together in order of temperature

        A single end-member is returned as it is, with any volume model. Several
        end-members must all be ones that the database was created with, be NASA Glenn polynomials
        and have no volume model, since the pressure dependence would move the transitions between
        them, and their temperature ranges must not overlap or leave a gap.

        Args:
            formula: Formula
            end_members: Names of the end-members, e.g. ``("quartz-alpha", "quartz-beta")``

        Returns:
            Thermodynamic properties of the end-members stitched together

        Raises:
            KeyError: If an end-member is not available for the formula
            ValueError: If the end-members cannot be stitched together
        """
        end_members = tuple(end_members)
        if not end_members:
            raise ValueError("At least one end-member is needed")
        if GAS_END_MEMBER in end_members:
            raise ValueError("The gas is not a condensed end-member. Create it with create_gas.")
        found: dict[str, ThermodynamicProperties] = {
            end_member: self.get_end_member(formula, end_member) for end_member in end_members
        }
        if len(found) == 1:
            return next(iter(found.values()))

        hill_formula: str = Formula(formula).formula
        added: list[str] = [
            end_member
            for end_member in end_members
            if (hill_formula, end_member) not in self._created_with
        ]
        if added:
            raise ValueError(
                "Only end-members that the database was created with can be stitched "
                f"together, not {added}"
            )
        not_glenn: list[str] = [
            end_member
            for end_member, properties in found.items()
            if not isinstance(properties, NasaGlennThermodynamicProperties)
        ]
        if not_glenn:
            raise ValueError(
                f"Only NASA Glenn polynomials can be stitched together, not {not_glenn}"
            )

        return NasaGlennThermodynamicProperties.stitch(
            cast(dict[str, NasaGlennThermodynamicProperties], found)
        )
