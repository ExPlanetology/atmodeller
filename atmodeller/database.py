# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from atmodeller import override
from atmodeller.constants import GAS_STATE, SOLID_STATE
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import ChemicalSpeciesData
from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.janaf import glenn_properties, read_glenn_coefficients


class DataBase(ABC):
    """Thermodynamic database for creating chemical species

    Subclasses provide the thermodynamic data, and this base class creates
    :class:`~atmodeller.containers.ChemicalSpecies` from them.
    """

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
        name: str = ChemicalSpeciesData(formula, state).name
        self.set_thermodynamic_properties(name, properties)

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
                :const:`~atmodeller.constants.SOLID_STATE`.
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
                :const:`~atmodeller.constants.GAS_STATE`.
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
        """Creates the database from the packaged NASA Glenn coefficients.

        Returns:
            The default database
        """
        return cls()

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

    This is the database of packaged NASA Glenn coefficients with the following changes:

        - Graphite (``C_s``) includes the volume
          :data:`~atmodeller.thermodata.gustafson.GRAPHITE_VOLUME_GUSTAFSON`, so that its Gibbs
          energy depends on pressure.
        - Diamond (:data:`~atmodeller.thermodata.gustafson.diamond_gustafson`) is added, created
          with ``create_condensed("C", state="diamond")``. Its heat capacity relative to graphite
          is from the assessment of :cite:t:`Gustafson1986`, valid to 6000 K.

    Further species can be added with :meth:`DataBase.add_species`.

    Returns:
        The default database
    """
    database: GlennDataBase = GlennDataBase()

    graphite: ThermodynamicProperties = database.get_thermodynamic_properties("C_s")
    database.add_species("C", "s", graphite.with_volume(GRAPHITE_VOLUME_GUSTAFSON))
    database.add_species("C", "diamond", diamond_gustafson)

    return database
