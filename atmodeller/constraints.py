"""Constraints for the system of equations.

See the LICENSE file for licensing information.

This module defines constraints that can be applied to an interior-atmosphere system.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import UserList
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import ConstantConstraint, ConstraintABC
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class FugacityConstraint(ConstantConstraint):
    """A constant fugacity constraint. See base class."""

    name: str = field(init=False, default="fugacity")


@dataclass(kw_only=True, frozen=True)
class PressureConstraint(ConstantConstraint):
    """A constant pressure constraint. See base class."""

    name: str = field(init=False, default="pressure")


@dataclass(kw_only=True, frozen=True)
class TotalPressureConstraint(ConstantConstraint):
    """Total pressure constraint. See base class.

    'species' is not required so is set to an empty string
    """

    species: str = field(init=False, default="")
    name: str = field(init=False, default="total_pressure")


@dataclass(kw_only=True, frozen=True)
class MassConstraint(ConstantConstraint):
    """A constant mass constraint. See base class."""

    name: str = field(init=False, default="mass")


class SystemConstraints(UserList):
    """A collection of constraints for an interior-atmosphere system.

    A collection of constraints that can be applied to an interior-atmosphere system. It provides
    methods to filter constraints based on their types, such as fugacity, mass conservation,
    pressure, and reaction network constraints.

    Note that solid activity constraints are inherent to the species and are combined with these
    user-prescribed constraints when the interior-atmosphere system is solved.

    Args:
        initlist: Initial list of constraints. Defaults to None.

    Attributes:
        data: A list of constraints for the interior-atmosphere system.
    """

    def __init__(self, initlist=None):
        self.data: list[ConstraintABC]
        super().__init__(initlist)

    def _filter_by_name(self, name: str) -> list[ConstraintABC]:
        """Filters the constraints by a given name.

        Args:
            name: The filter string (e.g., fugacity, pressure, mass).

        Returns:
            A list of filtered constraints.
        """
        filtered: list = []
        for entry in self.data:
            if entry.name == name:
                filtered.append(entry)

        return filtered

    @property
    def fugacity_constraints(self) -> list[ConstraintABC]:
        """Constraints related to fugacity."""
        return self._filter_by_name("fugacity")

    @property
    def mass_constraints(self) -> list[ConstraintABC]:
        """Constraints related to mass conservation."""
        return self._filter_by_name("mass")

    @property
    def pressure_constraints(self) -> list[ConstraintABC]:
        """Constraints related to pressure."""
        return self._filter_by_name("pressure")

    @property
    def total_pressure_constraint(self) -> list[ConstraintABC]:
        """Total pressure constraint."""
        total_pressure: list[ConstraintABC] = self._filter_by_name("total_pressure")
        if len(total_pressure) > 1:
            msg: str = "You can only specify zero or one total pressure constraints"
            logger.error(msg)
            raise ValueError(msg)

        return total_pressure

    @property
    def reaction_network_constraints(self) -> list[ConstraintABC]:
        """Constraints related to the reaction network."""
        filtered_entries: list[ConstraintABC] = self.fugacity_constraints
        filtered_entries.extend(self.pressure_constraints)

        return filtered_entries

    @property
    def number_reaction_network_constraints(self) -> int:
        """Number of constraints related to the reaction network."""
        return len(self.reaction_network_constraints)


@dataclass(kw_only=True, frozen=True)
class RedoxBuffer(ConstraintABC):
    """A mineral redox buffer that constrains a fugacity as a function of temperature

    Args:
        log10_shift: Log10 shift relative to the buffer
    """

    log10_shift: float = 0
    name: str = field(init=False, default="fugacity")

    @abstractmethod
    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """Log10 value at the buffer

        Args:
            temperature: Temperature
            pressure: Pressure. Defaults to 1 bar.
            **kwargs: Arbitrary keyword arguments

        Returns:
            log10 of the fugacity at the buffer
        """

    def get_log10_value(self, **kwargs) -> float:
        """Log10 value including any shift

        Args:
            temperature: Temperature
            pressure: Pressure. Defaults to 1 bar.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """
        log10_value: float = self.get_buffer_log10_value(**kwargs)
        log10_value += self.log10_shift
        return log10_value

    def get_value(self, **kwargs) -> float:
        """See base class"""
        log10_value: float = self.get_log10_value(**kwargs)
        value: float = 10**log10_value
        return value


@dataclass(kw_only=True, frozen=True)
class OxygenFugacityBuffer(RedoxBuffer):
    """A mineral redox buffer that constraints oxygen fugacity as a function of temperature."""

    species: str = field(init=False, default="O2")


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintHirschmann(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from O'Neill and Pownceby (1993) and Hirschmann et al. (2008).

    https://ui.adsabs.harvard.edu/abs/1993CoMP..114..296O/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del kwargs
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintOneill(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from O'Neill and Eggins (2002).

    Gibbs energy of reaction is at 1 bar. See Table 6.
    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del pressure
        del kwargs
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintBallhaus(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from Ballhaus et al. (1991).

    https://ui.adsabs.harvard.edu/abs/1991CoMP..107...27B/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del kwargs
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintFischer(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from Fischer et al. (2011).

    See Table S2 in supplementary materials.
    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del kwargs
        pressure_GPa: float = UnitConversion.bar_to_GPa(pressure)
        a_P: float = 6.44059 + 0.00463099 * pressure_GPa
        b_P: float = (
            -28.1808
            + 0.556272 * pressure_GPa
            - 0.00143757 * pressure_GPa**2
            + 4.0256e-6 * pressure_GPa**3
            - 5.4861e-9 * pressure_GPa**4  # Note typo in Table S2. Must be pressure**4.
        )
        b_P *= 1000 / temperature
        fugacity: float = a_P + b_P

        return fugacity
