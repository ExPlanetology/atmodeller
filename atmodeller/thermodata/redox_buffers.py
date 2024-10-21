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
"""Redox buffers"""

# Use physical symbol conventions so pylint: disable=C0103

import logging
import sys
from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from scipy.optimize import bisect

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import ExperimentalCalibration
from atmodeller.thermodata.interfaces import RedoxBufferProtocol
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class _RedoxBuffer(ABC, RedoxBufferProtocol):
    """A redox buffer

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.
        calibration: Calibration temperature and pressure range. Defaults to empty.

    Attributes:
        log10_shift: Log10 shift relative to the buffer
        calibration: Calibration temperature and pressure range
    """

    def __init__(
        self,
        log10_shift: float = 0,
        *,
        calibration: ExperimentalCalibration = ExperimentalCalibration()
    ):
        self.log10_shift: float = log10_shift
        self.calibration: ExperimentalCalibration = calibration
        logger.info("Setting experimental calibration = %s", calibration)

    @abstractmethod
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        """Log10 value at the buffer

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            log10 of the fugacity at the buffer
        """

    def get_log10_value(
        self, temperature: float, pressure: float, penalty: bool = True, **kwargs
    ) -> float:
        """Log10 value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            penalty: Apply penalty function. Defaults to True.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """

        log10_value: float = self._get_buffer_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        log10_value += self.log10_shift

        if penalty:
            log10_value += self.calibration.get_penalty(temperature, pressure)

        return log10_value

    def get_value(
        self, temperature: float, pressure: float, penalty: bool = True, **kwargs
    ) -> float:
        """Value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            penalty: Apply penalty function. Defaults to True.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Fugacity including any shift
        """
        log10_value: float = self.get_log10_value(
            temperature=temperature, pressure=pressure, penalty=penalty, **kwargs
        )
        value: float = 10**log10_value

        return value


# 27.5 GPa is given in the abstract of :cite:t:`HGD08`
IronWustiteBufferHirschmann08Calibration: ExperimentalCalibration = ExperimentalCalibration(
    pressure_max=UnitConversion.GPa_to_bar(27.5)
)


class IronWustiteBufferHirschmann08(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`"""

    @override
    def __init__(
        self,
        log10_shift: float = 0,
        *,
        calibration: ExperimentalCalibration = IronWustiteBufferHirschmann08Calibration
    ):
        super().__init__(log10_shift, calibration=calibration)

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        del kwargs
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )

        return fugacity


# From :cite:t:`H21`: "It extrapolates smoothly to higher temperature, though not calibrated above
# 3000 K. Extrapolation to lower temperatures (<1000 K) or higher pressures (>100 GPa) is not
# recommended."
IronWustiteBufferHirschmann21Calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=1000, pressure_max=UnitConversion.GPa_to_bar(100)
)


class IronWustiteBufferHirschmann21(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`H21`"""

    @override
    def __init__(
        self,
        log10_shift: float = 0,
        *,
        calibration: ExperimentalCalibration = IronWustiteBufferHirschmann21Calibration
    ):
        super().__init__(log10_shift, calibration=calibration)
        self.a: list[float] = [6.844864, 1.175691e-1, 1.143873e-3, 0, 0]
        self.b: list[float] = [5.791364e-4, -2.891434e-4, -2.737171e-7, 0, 0]
        self.c: list[float] = [-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7]
        self.d: list[float] = [-2.769002e4, 5.285977e2, -2.919275, 0, 0]
        self.e: list[float] = [8.463095, -3.000307e-3, 7.213445e-5, 0, 0]
        self.f: list[float] = [1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0]
        self.g: list[float] = [-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4]
        self.h: list[float] = [-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0]

    def _evaluate_m(self, pressure: float, coefficients: list[float]) -> float:
        """Evaluates an m parameter

        Args:
            pressure: Pressure in GPa
            coefficients: Coefficients

        Return:
            m parameter
        """
        m: float = (
            coefficients[0]
            + coefficients[1] * pressure
            + coefficients[2] * pressure**2
            + coefficients[3] * pressure**3
            + coefficients[4] * pressure ** (1 / 2)
        )

        return m

    def _evaluate_fO2(
        self, temperature: float, pressure: float, coefficients: list[list[float]]
    ) -> float:
        """Evaluates the fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa
            coefficients: Coefficients

        Returns:
            log10fO2
        """
        log10fO2: float = (
            self._evaluate_m(pressure, coefficients[0])
            + self._evaluate_m(pressure, coefficients[1]) * temperature
            + self._evaluate_m(pressure, coefficients[2]) * temperature * np.log(temperature)
            + self._evaluate_m(pressure, coefficients[3]) / temperature
        )

        return log10fO2

    def _fcc_bcc_iron(self, temperature: float, pressure: float) -> float:
        """log10fO2 for fcc and bcc iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for fcc and bcc iron
        """
        log10fO2: float = self._evaluate_fO2(
            temperature, pressure, [self.a, self.b, self.c, self.d]
        )

        return log10fO2

    def _hcp_iron(self, temperature: float, pressure: float) -> float:
        """log10fO2 for hcp iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for hcp iron
        """
        log10fO2: float = self._evaluate_fO2(
            temperature, pressure, [self.e, self.f, self.g, self.h]
        )

        return log10fO2

    def _use_hcp(self, temperature: float, pressure: float) -> bool:
        """Check to use hcp iron formulation for fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa
        """
        x: list[float] = [-18.64, 0.04359, -5.069e-6]
        threshold: float = x[0] + x[1] * temperature + x[2] * temperature**2

        return pressure > threshold

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        del kwargs
        pressure_GPa: float = UnitConversion.bar_to_GPa(pressure)
        if self._use_hcp(temperature, pressure_GPa):
            return self._hcp_iron(temperature, pressure_GPa)
        else:
            return self._fcc_bcc_iron(temperature, pressure_GPa)


class IronWustiteBufferHirschmann(RedoxBufferProtocol):
    """Composite iron-wustite buffer using :cite:t:`OP93,HGD08` and :cite:t:`H21`"""

    @override
    def __init__(
        self,
        log10_shift: float = 0,
    ):
        self.low_temperature_buffer: _RedoxBuffer = IronWustiteBufferHirschmann08(log10_shift)
        self.high_temperature_buffer: _RedoxBuffer = IronWustiteBufferHirschmann21(log10_shift)

    def get_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        """Log10 value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """
        try:
            assert self.high_temperature_buffer.calibration.temperature_min is not None
        except AssertionError as exc:
            raise ValueError(
                "temperature_min is not defined for the high temperature buffer"
            ) from exc

        if temperature < self.high_temperature_buffer.calibration.temperature_min:
            return self.low_temperature_buffer.get_log10_value(temperature, pressure, **kwargs)
        else:
            return self.high_temperature_buffer.get_log10_value(
                temperature=temperature, pressure=pressure, **kwargs
            )

    def get_value(self, temperature: float, pressure: float, **kwargs) -> float:
        """Value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            Fugacity including any shift
        """
        log10_value: float = self.get_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        value: float = 10**log10_value

        return value


class IronWustiteBufferONeill(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`OE02`

    Gibbs energy of reaction is at 1 bar :cite:p:`OE02{Table 6}`.
    """

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        del pressure
        del kwargs
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )

        return fugacity


class IronWustiteBufferBallhaus(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`BBG91`"""

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        del kwargs
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )

        return fugacity


class IronWustiteBufferFischer(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`F11`

    See :cite:t:`F11{Table S2}` in supplementary materials.
    """

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        del kwargs
        pressure_GPa: float = UnitConversion.bar_to_GPa(pressure)
        a_coeff: float = 6.44059 + 0.00463099 * pressure_GPa
        b_coeff: float = (
            -28.1808
            + 0.556272 * pressure_GPa
            - 0.00143757 * pressure_GPa**2
            + 4.0256e-6 * pressure_GPa**3
            - 5.4861e-9 * pressure_GPa**4  # Note typo in Table S2. Must be pressure**4.
        )
        b_coeff *= 1000 / temperature
        fugacity: float = a_coeff + b_coeff

        return fugacity


IronWustiteBuffer: Type[RedoxBufferProtocol] = IronWustiteBufferHirschmann

def solve_for_log10_dIW(
    target_fugacity: float, temperature: float, pressure: float = 1.0, **kwargs
) -> float:
    """Solves for the log10 shift relative to the default Iron-wustite buffer

    The shift is report relative to the standard state defined at temperature and 1 bar pressure.
    If desired, the shift can be reported relative to a standard state defined at an alternative
    pressure.

    Args:
        target_fugacity: Target fugacity in bar
        temperature: Temperature in K
        pressure: Pressure defining the standard state in bar. Defaults to 1 bar.
        **kwargs: Arbitrary keyword arguments

    Returns:
        The required log10_shift to match the target fugacity
    """
    buffer: RedoxBufferProtocol = IronWustiteBuffer()

    def objective_function(log10_shift: float, args):
        """Objective function

        Args:
            log10_shift: Log10 shift
            args: Optional arguments (not used)

        Returns:
            Residual of the objective function
        """
        del args
        buffer.log10_shift = log10_shift
        calculated_log10_fugacity: float = buffer.get_log10_value(
            temperature, pressure, penalty=False, **kwargs
        )

        return calculated_log10_fugacity - np.log10(target_fugacity)

    # solver = optx.Bisection(rtol=1.0e-8, atol=1.0e-8)
    # sol = optx.root_find(
    #     objective_function, solver, jnp.array(-20.0), options=dict(lower=-100, upper=100)
    # )

    sol = bisect(
        objective_function, -100, 100, xtol=1.0e-8, rtol=1.0e-8
    )

    return sol

    # # Success is indicated by no message
    # if optx.RESULTS[sol.result] == "":
    #     value: float = sol.value.item()
    # else:
    #     raise ValueError("Root finding did not converge")

    # return value