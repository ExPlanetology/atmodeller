#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Interfaces for obtaining thermodynamic data"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from atmodeller.core import ChemicalSpecies

logger: logging.Logger = logging.getLogger(__name__)


class RedoxBufferProtocol(Protocol):

    log10_shift: float

    def __init__(self, *args, **kwargs): ...

    def get_log10_value(self, temperature: float, pressure: float, **kwargs) -> float: ...

    def get_value(self, temperature: float, pressure: float, **kwargs) -> float: ...


class ThermodynamicDataForSpeciesProtocol(Protocol):

    @property
    def data_source(self) -> str: ...

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float: ...


class ThermodynamicDataset(ABC):
    """A thermodynamic dataset"""

    _DATA_SOURCE: str
    """Thermodynamic data source"""
    _ENTHALPY_REFERENCE_TEMPERATURE: float
    """Enthalpy reference temperature in K"""
    _STANDARD_STATE_PRESSURE: float
    """Standard state pressure in bar"""

    @abstractmethod
    def get_species_data(
        self, species: ChemicalSpecies, **kwargs
    ) -> ThermodynamicDataForSpeciesProtocol | None:
        """Gets the thermodynamic data for a species.

        Args:
            species: A chemical species
            **kwargs: Arbitrary keyword arguments

        Returns:
            Thermodynamic data for the species or None if not available
        """

    @property
    def data_source(self) -> str:
        """The source of the thermodynamic data."""
        return self._DATA_SOURCE

    @property
    def enthalpy_reference_temperature(self) -> float:
        """Enthalpy reference temperature in K"""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def standard_state_pressure(self) -> float:
        """Standard state pressure in bar"""
        return self._STANDARD_STATE_PRESSURE


class ThermodynamicDataForSpeciesABC(ABC):
    """Thermodynamic data for a species

    Args:
        species: A chemical species
        data_source: Source of the thermodynamic data
        data: Data used for thermodynamic calculations

    Attributes:
        species: A chemical species
        data_source: Source of the thermodynamic data
        data: Data used for thermodynamic calculations
    """

    def __init__(self, species: ChemicalSpecies, data_source: str, data: Any):
        self.species: ChemicalSpecies = species
        self.data_source: str = data_source
        self.data: Any = data

    @abstractmethod
    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        r"""Gets the standard Gibbs free energy of formation.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The standard Gibbs free energy of formation in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
