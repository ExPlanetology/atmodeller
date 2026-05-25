# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Redox buffers"""

from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from atmodeller import override
from atmodeller.jax_utils import Scalar, as_j64, to_native_floats
from atmodeller.sci_utils import ExperimentalCalibration, unit_conversion


class RedoxBuffer(eqx.Module):
    """Redox buffer

    This must adhere to :class:`atmodeller.interfaces.ActivityConstraintProtocol`.

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
        evaluation_pressure: Pressure (bar) to evaluate the buffer at. Defaults to 1 bar. If
        ``None``, then the total pressure will be used.
    """

    log10_shift: Float[Array, "..."]
    """Log10 shift"""
    evaluation_pressure: Optional[Scalar]
    """Evaluation pressure"""

    def __init__(self, log10_shift: ArrayLike = 0, evaluation_pressure: Optional[Scalar] = 1):
        self.log10_shift = as_j64(log10_shift)
        self.evaluation_pressure = evaluation_pressure

    @abstractmethod
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Converts the pressure units.

        Args:
            pressure: Pressure (bar)

        Returns:
            Pressure in units appropriate for the calculation
        """

    @abstractmethod
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the log10 fugacity at the buffer.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity at the buffer
        """

    def active(self) -> Bool[Array, "..."]:
        """True if the redox buffer is active, otherwise False

        Returns:
            Mask indicating whether the redox buffer is active
        """
        return ~jnp.isnan(self.log10_shift)

    def get_scaled_pressure(self, pressure: ArrayLike) -> ArrayLike:
        """Gets the scaled pressure.

        Args:
            pressure: Pressure (bar)

        Returns:
            Pressure in units appropriate for the calculation
        """
        if self.evaluation_pressure is not None:
            return self.convert_pressure_units(self.evaluation_pressure)
        else:
            return self.convert_pressure_units(pressure)

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the log10 fugacity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity
        """
        return self.log10_fugacity_buffer(temperature, pressure) + self.log10_shift

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Float[Array, "..."]:
        """Gets the log fugacity.

        Note:
            This method is designed to be fully compatible with both :func:`jax.vmap` and explicit
            batched input. It supports broadcasting of ``temperature`` and ``pressure`` to match
            the batch size, as required by output routines (called via :meth:`log_activity`), while
            also working with per-instance vectorization as used by the engine and solver.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log fugacity
        """
        broadcast_shape: tuple[int, ...] = jnp.broadcast_shapes(
            jnp.shape(self.log10_shift), jnp.shape(temperature), jnp.shape(pressure)
        )
        log_fugacity = jnp.log(10) * self.log10_fugacity(temperature, pressure)

        return jnp.broadcast_to(log_fugacity, broadcast_shape)

    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the log activity, which is the same as the log fugacity for a redox buffer.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log fugacity
        """
        return self.log_fugacity(temperature, pressure)


class IronWustiteBufferHirschmann08(RedoxBuffer):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`

    Experimental calibration values are provided in the abstract of :cite:t:`HGD08`.

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
        evaluation_pressure: Pressure (bar) to evaluate the buffer at. Defaults to 1 bar. If
            ``None``, then the total pressure will be used.
    """

    calibration: ExperimentalCalibration
    """Experimental calibration"""

    def __init__(self, log10_shift: ArrayLike = 0, evaluation_pressure: Optional[Scalar] = 1):
        super().__init__(log10_shift, evaluation_pressure)
        self.calibration = ExperimentalCalibration(pressure_max=27.5 * unit_conversion.GPa_to_bar)

    @override
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Units are bar"""
        return pressure

    @override
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the log10 fugacity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity
        """
        scaled_pressure: ArrayLike = self.get_scaled_pressure(pressure)
        log10_fugacity_buffer: Array = (
            -0.8853 * jnp.log(temperature)
            - 28776.8 / temperature
            + 14.057
            + 0.055 * (scaled_pressure - 1) / temperature
        )

        return log10_fugacity_buffer


class IronWustiteBufferHirschmann21(RedoxBuffer):
    """Iron-wustite buffer :cite:p:`H21`

    Regarding the calibration, :cite:t:`H21` states that: 'It extrapolates smoothly to higher
    temperature, though not calibrated above 3000 K. Extrapolation to lower temperatures (<1000 K)
    or higher pressures (>100 GPa) is not recommended.'

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
        evaluation_pressure: Pressure (bar) to evaluate the buffer at. Defaults to 1 bar. If
            ``None``, then the total pressure will be used.
    """

    calibration: ExperimentalCalibration
    """Experimental calibration"""
    a: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """a coefficients"""
    b: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """b coefficients"""
    c: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """c coefficients"""
    d: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """d coefficients"""
    e: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """e coefficients"""
    f: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """f coefficients"""
    g: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """g coefficients"""
    h: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """h coefficients"""
    x: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Coefficients to define the threshold to use the hcp iron formulation"""

    def __init__(self, log10_shift: ArrayLike = 0, evaluation_pressure: Optional[Scalar] = 1):
        super().__init__(log10_shift, evaluation_pressure)
        self.calibration = ExperimentalCalibration(
            temperature_min=1000, pressure_max=100 * unit_conversion.GPa_to_bar
        )
        self.a = (6.844864, 1.175691e-1, 1.143873e-3, 0, 0)
        self.b = (5.791364e-4, -2.891434e-4, -2.737171e-7, 0, 0)
        self.c = (-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7)
        self.d = (-2.769002e4, 5.285977e2, -2.919275, 0, 0)
        self.e = (8.463095, -3.000307e-3, 7.213445e-5, 0, 0)
        self.f = (1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0)
        self.g = (-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4)
        self.h = (-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0)
        self.x = (-18.64, 0.04359, -5.069e-6)

    @override
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Units are GPa"""
        return pressure * unit_conversion.bar_to_GPa

    def _evaluate_m(self, pressure: ArrayLike, coefficients: tuple[float, ...]) -> Array:
        """Evaluates an m parameter

        Args:
            pressure: Pressure in GPa
            coefficients: Coefficients

        Return:
            m parameter
        """
        m: Array = (
            coefficients[4] * jnp.power(pressure, 0.5)
            + coefficients[0]
            + coefficients[1] * pressure
            + coefficients[2] * jnp.power(pressure, 2)
            + coefficients[3] * jnp.power(pressure, 3)
        )

        return m

    def _evaluate_fO2(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        coefficients: tuple[tuple[float, ...], ...],
    ) -> Array:
        """Evaluates the fO2

        Args:
            temperature: Temperature (K)
            pressure: Pressure in GPa
            coefficients: Coefficients

        Returns:
            log10fO2
        """
        log10fO2: Array = (
            self._evaluate_m(pressure, coefficients[0])
            + self._evaluate_m(pressure, coefficients[1]) * temperature
            + self._evaluate_m(pressure, coefficients[2]) * temperature * jnp.log(temperature)
            + self._evaluate_m(pressure, coefficients[3]) / temperature
        )

        return log10fO2

    def _fcc_bcc_iron(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """log10fO2 for fcc and bcc iron

        Args:
            temperature: Temperature (K)
            pressure: Pressure in GPa

        Return:
            log10fO2 for fcc and bcc iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self.a, self.b, self.c, self.d)
        )

        return log10fO2

    def _hcp_iron(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """log10fO2 for hcp iron

        Args:
            temperature: Temperature (K)
            pressure: Pressure in GPa

        Return:
            log10fO2 for hcp iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self.e, self.f, self.g, self.h)
        )

        return log10fO2

    def _use_hcp(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Check to use hcp iron formulation for fO2

        Args:
            temperature: Temperature (K)
            pressure: Pressure in GPa

        Returns:
            True/False whether to use the hcp iron formulation
        """
        threshold: Array = (
            self.x[2] * jnp.power(temperature, 2) + self.x[1] * temperature + self.x[0]
        )

        return jnp.array(pressure) > threshold

    @override
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity
        """
        scaled_pressure: ArrayLike = self.get_scaled_pressure(pressure)

        def hcp_case() -> Array:
            return self._hcp_iron(temperature, scaled_pressure)

        def fcc_bcc_case() -> Array:
            return self._fcc_bcc_iron(temperature, scaled_pressure)

        buffer_value: Array = jnp.where(
            self._use_hcp(temperature, scaled_pressure), hcp_case(), fcc_bcc_case()
        )

        return buffer_value


class IronWustiteBufferHirschmann(RedoxBuffer):
    """Composite iron-wustite buffer using :cite:t:`OP93,HGD08` and :cite:t:`H21`

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.
        evaluation_pressure: Pressure (bar) to evaluate the buffer at. Defaults to 1 bar. If
            ``None``, then the total pressure will be used.
    """

    calibration: ExperimentalCalibration
    """Experimental calibration"""
    low_temperature_buffer: IronWustiteBufferHirschmann08
    """Low temperature buffer"""
    high_temperature_buffer: IronWustiteBufferHirschmann21
    """High temperature buffer"""

    def __init__(self, log10_shift: ArrayLike = 0, evaluation_pressure: Optional[Scalar] = 1):
        super().__init__(log10_shift, evaluation_pressure)
        self.calibration = ExperimentalCalibration(pressure_max=100 * unit_conversion.GPa_to_bar)
        self.low_temperature_buffer = IronWustiteBufferHirschmann08(
            self.log10_shift, self.evaluation_pressure
        )
        self.high_temperature_buffer = IronWustiteBufferHirschmann21(
            self.log10_shift, self.evaluation_pressure
        )

    @override
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Units are bar

        Not used for a composite redox buffer but required by the interface.
        """
        return pressure

    def _use_low_temperature(self, temperature: ArrayLike) -> Array:
        """Check to use the low temperature buffer for fO2

        Args:
            temperature: Temperature (K)

        Returns:
            True/False whether to use the low temperature formulation
        """
        # temperature_min is not None so ignore the typing complaint
        return as_j64(temperature) < self.high_temperature_buffer.calibration.temperature_min  # pyright: ignore

    @override
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the log10 fugacity at the buffer

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity at the buffer
        """

        def low_temperature_case() -> ArrayLike:
            return self.low_temperature_buffer.log10_fugacity_buffer(temperature, pressure)

        def high_temperature_case() -> ArrayLike:
            return self.high_temperature_buffer.log10_fugacity_buffer(temperature, pressure)

        buffer_value: Array = jnp.where(
            self._use_low_temperature(temperature),
            low_temperature_case(),
            high_temperature_case(),
        )

        return buffer_value


IronWustiteBuffer: type[RedoxBuffer] = IronWustiteBufferHirschmann
