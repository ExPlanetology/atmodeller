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
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from atmodeller import override
from atmodeller.constants import GAS_STATE, SOLID_STATE
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import ChemicalSpeciesData
from atmodeller.thermodata.core import ThermodynamicProperties
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
        # The packaged data are already loaded, so reuse them
        self._coefficients: dict[str, ThermodynamicProperties] = (
            glenn_properties if path is None else read_glenn_coefficients(path)
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
