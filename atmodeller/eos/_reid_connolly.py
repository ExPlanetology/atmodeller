# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOSs from :cite:t:`RPS77,C16`"""

import logging

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from atmodeller import override
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas, RedlichKwongABC
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration
from atmodeller.thermodata import CriticalData, critical_data_dictionary

logger: logging.Logger = logging.getLogger(__name__)


class RedlichKwong49(RedlichKwongABC):
    """Redlich-Kwong 1949 model

    Repulsive pressure term from van der Waals :cite:p:`RK49,C16{Equation 1}`
    Attractive pressure term from Redlich-Kwong :cite:p:`RK49,C16{Equation 4}`

    Args:
        critical_data: Critical data
    """

    critical_data: CriticalData
    _a: float = eqx.field(converter=float)
    _b: float = eqx.field(converter=float)

    def __init__(self, critical_data: CriticalData):
        r"""Default a (:math:`(\mathrm{m}^3\ \mathrm{mol}^{-1})^2\ \mathrm{K}^{1/2}\ \mathrm{bar}`)
        and b (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`) for SiO calculated from the critical
        pressure and temperature from :cite:p:`C16{Table 2}`"""
        self.critical_data = critical_data
        self._a = 0.000448433
        self._b = 0.00000543922

    @property
    def critical_pressure(self) -> float:
        """Critical pressure (bar)"""
        return self.critical_data.pressure

    @property
    def critical_temperature(self) -> float:
        """Critical temperature (K))"""
        return self.critical_data.temperature

    @classmethod
    def create(cls, hill_formula: str, suffix: str = "") -> "RedlichKwong49":
        """Gets the Redlich-Kwong 1949 (RK49) model for a given species.

        Args:
            hill_formula: Hill formula
            suffix: Suffix. Defaults to an empty string.

        Returns:
            An RK49 model for the species
        """
        critical_data: CriticalData = critical_data_dictionary[f"{hill_formula}{suffix}"]

        return cls(critical_data)

    @override
    def a(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""RK49 `a` parameter :cite:p:`RK49{Equation 4}`.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            RK49 `a` parameter
            (:math:`(\mathrm{m}^3\ \mathrm{mol}^{-1})^2\ \mathrm{K}^{1/2}\ \mathrm{bar}`)
        """
        del temperature
        del pressure

        a: ArrayLike = (
            jnp.power(GAS_CONSTANT_BAR, (2.0))
            * jnp.power(self.critical_temperature, (5.0 / 2))
            / (9 * (jnp.power(2, (1.0 / 3)) - 1))
        ) / self.critical_pressure

        return a

    @override
    def b(self) -> ArrayLike:
        r"""RK49 `b` parameter :cite:p:`RK49{Equation 5}`.

        Returns:
            RK49 `b` parameter (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`).
        """
        b: ArrayLike = (
            (jnp.power(2, (1.0 / 3)) - 1)
            * GAS_CONSTANT_BAR
            * self.critical_temperature
            / (3 * self.critical_pressure)
        )

        return b


experimental_calibration_connolly16: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=1000, temperature_max=10000, pressure_min=1, pressure_max=50e3
)
"""Experimental calibration for :cite:`C16` models"""

experimental_calibration_reid87: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=300, temperature_max=500, pressure_min=1, pressure_max=100
)
"""Experimental calibration for :cite:`RPS77` models"""

OSi_rk49_connolly16: RealGas = RedlichKwong49.create("OSi")
"""OSi Redlich-Kwong :cite:p:`C16`"""
H4Si_rk49_reid87: RealGas = RedlichKwong49.create("H4Si")
"""H4Si Redlich-Kwong :cite:p:`RPS77`"""
CHN_rk49_reid87: RealGas = RedlichKwong49.create("CHN")
"""CHN Redlich-Kwong :cite:p:`RPS77`"""
H3N_rk49_reid87: RealGas = RedlichKwong49.create("H3N")
"""H3N Redlich-Kwong :cite:p:`RPS77`"""

OSi_rk49_connolly16_bounded: RealGas = CombinedRealGas.create(
    [OSi_rk49_connolly16], [experimental_calibration_connolly16]
)
"""OSi Redlich-Kwong bounded :cite:p:`C16`"""


def get_reid_connolly_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of EOS models

    The naming convention is as follows:
        [species]_[eos model]_[citation]

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["OSi_rk49_connolly16"] = OSi_rk49_connolly16_bounded
    eos_models["H4Si_rk49_reid87"] = H4Si_rk49_reid87
    eos_models["CHN_rk49_reid87"] = CHN_rk49_reid87
    eos_models["H3N_rk49_reid87"] = H3N_rk49_reid87

    return eos_models
