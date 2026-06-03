# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Classes that aggregate EOS

Units for temperature and pressure are K and bar, respectively.
"""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.eos.core import IdealGas, RealGas, RealGasBase
from atmodeller.jax_utils import FloatArray, as_j64, to_hashable, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration


class CombinedRealGas(RealGas):
    """Combined real gas EOS with separate volume integrations for each EOS

    This class computes the contribution to the volume integral separately for each EOS based on
    the range covered by its P-T calibration, and then combines them.

    Note:
        This is not presently compatible with batched mole_fractions.

    Args:
        real_gases: Real gases to combine
        calibrations: Experimental calibrations that correspond to ``real_gases``
    """

    real_gases: tuple[RealGas, ...]
    """Real gases to combine"""
    calibrations: tuple[ExperimentalCalibration, ...]
    """Experimental calibrations"""
    upper_pressure_bounds: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Upper pressure bounds"""
    _volume_functions: tuple[Callable, ...]
    _volume_integral_functions: tuple[Callable, ...]

    def __init__(
        self, real_gases: tuple[RealGas, ...], calibrations: tuple[ExperimentalCalibration, ...]
    ):
        self.real_gases = real_gases
        self.calibrations = calibrations
        self.upper_pressure_bounds = self._get_upper_pressure_bounds(calibrations)
        self._volume_functions = tuple(to_hashable(eos.volume) for eos in self.real_gases)
        self._volume_integral_functions = tuple(
            to_hashable(eos.volume_integral) for eos in self.real_gases
        )

    @classmethod
    def create(
        cls,
        real_gases: Sequence[RealGas],
        calibrations: Sequence[ExperimentalCalibration],
        extrapolate: bool = True,
    ) -> RealGas:
        """Create an instance with the given real gases and calibrations

        Reasonable extrapolation behaviour is required to ensure that the function is bounded to
        avoid throwing NaNs or infs which will crash the solver. Physically, it is reasonable to
        extend the lower bound using the ideal gas law and the upper bound assuming a linear
        pressure dependence of the compressibility factor.

        There is no bounding for temperature; hence it is assumed that the extrapolation
        behaviour of temperature is reasonable. This is practically useful because the calibrations
        are often restricted to a lower temperature range than the high temperatures that are
        typically of interest for hot rocks and magma ocean planets.

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to ``real_gases``
            extrapolate: Extrapolate the EOS to have reasonable behaviour below the minimum and
                above the maximum calibration pressure if required. Defaults to ``True``.

        Returns:
            An instance
        """
        real_gases_list: list[RealGas] = list(real_gases)
        calibrations_list: list[ExperimentalCalibration] = list(calibrations)

        if extrapolate:
            if calibrations_list[0].pressure_min is not None:
                cls._append_lower_bound(real_gases_list, calibrations_list)
            if calibrations_list[-1].pressure_max is not None:
                cls._append_upper_bound(real_gases_list, calibrations_list)

        return cls(tuple(real_gases_list), tuple(calibrations_list))

    @staticmethod
    def _append_lower_bound(
        real_gases: list[RealGas], calibrations: list[ExperimentalCalibration]
    ) -> None:
        """Appends the lower bound, which gives ideal gas behaviour

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to ``real_gases``
        """
        assert calibrations[0].pressure_min is not None
        real_gases.insert(0, IdealGas())
        pressure_max: float = calibrations[0].pressure_min
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_max=pressure_max)
        calibrations.insert(0, calibration)

    @staticmethod
    def _append_upper_bound(
        real_gases: list[RealGas], calibrations: list[ExperimentalCalibration]
    ) -> None:
        """Appends the upper bound

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to `real_gases`
        """
        assert calibrations[-1].pressure_max is not None
        pressure_min: float = calibrations[-1].pressure_max
        real_gas: RealGas = UpperBoundRealGas(real_gases[-1], pressure_min)
        real_gases.append(real_gas)
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=pressure_min)
        calibrations.append(calibration)

    @staticmethod
    def _get_upper_pressure_bounds(
        calibrations: tuple[ExperimentalCalibration, ...],
    ) -> tuple[float, ...]:
        """Gets the upper pressure bounds based on each experimental calibration.

        Returns:
            Upper pressure bounds
        """
        upper_pressure_bounds: list[float] = []

        for ii, calibration in enumerate(calibrations):
            try:
                assert calibration.pressure_max is not None
            except AssertionError:
                if ii == len(calibrations) - 1:
                    continue
                else:
                    msg: str = "Maximum pressure cannot be None"
                    raise ValueError(msg)

            pressure_bound = calibration.pressure_max
            upper_pressure_bounds.append(pressure_bound)

        return tuple(upper_pressure_bounds)

    def _get_index(self, pressure: ArrayLike) -> Array:
        """Gets the index of the appropriate EOS model based on ``pressure``.

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """
        index: Array = jnp.searchsorted(
            jnp.array(self.upper_pressure_bounds), pressure, side="right"
        )
        # jax.debug.print("pressure = {pressure}, index = {index}", pressure=pressure, index=index)

        return index

    @override
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        temperature, pressure = jnp.broadcast_arrays(temperature, pressure)
        original_shape: tuple[int, ...] = temperature.shape
        # jax.debug.print("temperature = {out}", out=temperature)
        # jax.debug.print("pressure = {out}", out=pressure)

        # Flatten for vmap
        temperature = temperature.ravel()
        pressure = pressure.ravel()

        indices: Array = self._get_index(pressure)
        # jax.debug.print("indices = {out}", out=indices)

        def apply_volume(index: ArrayLike, temperature, pressure, mole_fractions) -> FloatArray:
            return lax.switch(index, self._volume_functions, temperature, pressure, mole_fractions)

        vmap_volume: Callable = eqx.filter_vmap(apply_volume, in_axes=(0, 0, 0, None))
        volume: FloatArray = vmap_volume(indices, temperature, pressure, mole_fractions)
        # jax.debug.print("volume = {out}", out=volume)

        return jnp.reshape(volume, original_shape)

    @override
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        index: Array = self._get_index(pressure)

        def compute_integral(
            i: Array,
            pressure_high: ArrayLike,
            pressure_low: ArrayLike,
            mole_fractions: ArrayLike | None,
        ) -> FloatArray:
            """Computes the volume integral for the given pressure range.

            Args:
                i: Index of the EOS model
                pressure_high: Upper pressure bound
                pressure_low: Lower pressure bound
                mole_fractions: Mole fractions

            Returns:
                Volume integral
            """
            volume_integral_high: FloatArray = lax.switch(
                i, self._volume_integral_functions, temperature, pressure_high, mole_fractions
            )
            # jax.debug.print(
            #    "compute_integral: volume_integral_high = {out}", out=volume_integral_high
            # )
            volume_integral_low: FloatArray = lax.switch(
                i, self._volume_integral_functions, temperature, pressure_low, mole_fractions
            )
            # jax.debug.print(
            #    "compute_integral: volume_integral_low = {out}", out=volume_integral_low
            # )
            integral: FloatArray = volume_integral_high - volume_integral_low

            return integral

        def scan_fn(carry: FloatArray, i: Array) -> tuple:
            """Scan function for accumulating the volume integral

            Args:
                carry: Accumulated volume integral
                i: Index of the EOS model

            Returns:
                Updated carry and ``None`` (unused)
            """
            # jax.debug.print("scan_fn: i = {out}", out=i)
            pressure_low = lax.dynamic_index_in_dim(
                jnp.array(self.upper_pressure_bounds), i - 1, keepdims=False
            )
            # jax.debug.print("pressure_low = {out}", out=pressure_low)

            # Middle integrals
            mask_middle: Array = i < index
            # jax.debug.print("mask_middle = {out}", out=mask_middle)
            pressure_high = lax.dynamic_index_in_dim(
                jnp.array(self.upper_pressure_bounds), i, keepdims=False
            )
            # jax.debug.print("pressure_high = {out}", out=pressure_high)
            contrib_middle: FloatArray = compute_integral(
                i, pressure_high, pressure_low, mole_fractions
            )
            carry = carry + jnp.where(mask_middle, contrib_middle, 0.0)

            # Final integral
            mask_final: Array = i == index
            # jax.debug.print("mask_final = {out}", out=mask_final)
            contrib_final: FloatArray = compute_integral(i, pressure, pressure_low, mole_fractions)
            carry = carry + jnp.where(mask_final, contrib_final, 0.0)
            # jax.debug.print("carry = {out}", out=carry)

            return carry, None

        def add_first_integral(total_integral: FloatArray) -> FloatArray:
            """Adds the contribution of the first integral to the total integral.

            This is necessary because the first integral is not included in the loop over the
            EOS models.

            Args:
                total_integral: Total integral so far

            Returns:
                Total integral with the first integral contribution added
            """
            # If the index is 0, then the first integral is the only one that is added.
            integral: FloatArray = lax.switch(
                0, self._volume_integral_functions, temperature, pressure, mole_fractions
            )
            # Otherwise, the first integral is added to the total integral.
            pressure_max: FloatArray = lax.dynamic_index_in_dim(
                jnp.array(self.upper_pressure_bounds), 0, keepdims=False
            )
            integral2: FloatArray = lax.switch(
                0, self._volume_integral_functions, temperature, pressure_max, mole_fractions
            )

            # jax.debug.print("add_only_first_integral: integral = {out}", out=integral)
            return jnp.where(index == 0, total_integral + integral, total_integral + integral2)

        # Initialize. Must be 0.0 to ensure float array.
        total_integral: FloatArray = jnp.array(0.0)
        total_integral = add_first_integral(total_integral)

        # Scan over the indices of the EOS models.
        loop_indices: Array = jnp.arange(1, len(self.upper_pressure_bounds) + 1)
        # jax.debug.print("loop_indices = {out}", out=loop_indices)
        total_integral, _ = lax.scan(scan_fn, total_integral, loop_indices)

        return total_integral


class UpperBoundRealGas(RealGas):
    """An upper bound for an EOS

    This is used to extrapolate an EOS assuming that the compressibility factor is a linear
    function of pressure. Importantly, this class is not intended to be used directly, but rather
    as a component of :class:`~atmodeller.eos._aggregators.CombinedRealGas`.

    ``p_eval`` is used to evaluate the compressibility factor and its gradient, and therefore to
    avoid recompilation of JAX functions it is converted to a JAX array of dtype float64 within
    the methods, which is the expected type for the pressure argument of the EOS functions.

    Args:
        real_gas: Real gas to evaluate the compressibility factor at ``p_eval``.
        p_eval: Evaluation pressure in bar. This is usually the maximum calibration pressure
            of ``real_gas``. Defaults to ``1`` bar.
    """

    real_gas: RealGas
    """Real gas to evaluate the compressibility factor at ``p_eval``"""
    p_eval: float = eqx.field(converter=float, default=1.0)
    """Evaluation pressure in bar. Defaults to ``1`` bar."""

    # @eqx.debug.assert_max_traces(max_traces=1)
    def _z0(self, temperature: ArrayLike) -> ArrayLike:
        """Compressibility factor of the previous EOS to blend smoothly with.

        Importantly, we don't want to trigger a recompilation of the compressibility factor, so we
        pass :attr:`p_eval` as an array.

        Args:
            temperature: Temperature in K
        """
        return self.real_gas.compressibility_factor(temperature, as_j64(self.p_eval))

    # @eqx.debug.assert_max_traces(max_traces=1)
    def _dzdp0(self, temperature: ArrayLike) -> ArrayLike:
        """Gradient of the compressibility factor of the previous EOS to blend smoothly with.

        Importantly, we don't want to trigger a recompilation of the compressibility factor, so we
        pass :attr:`p_eval` as an array.

        Args:
            temperature: Temperature in K
        """
        del temperature

        # TODO: In theory it should be possible to use dz/dp, but this is causing the solver to
        # crash for certain parameter ranges due to nans/infs being passed to the linear solver.
        # More investigations are required to understand the root cause. This only affects the
        # behaviour outside of the EOS calibration range.

        return 0.0  # self.real_gas.dzdp(temperature, as_j64(self.p_eval))

    @override
    # @eqx.debug.assert_max_traces(max_traces=1)
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> Array:
        """Log fugacity cannot be computed.

        This method should not be used because the volume integral is only defined above `p_eval`,
        meaning that the log fugacity cannot be calculated.
        """
        del temperature
        del pressure
        del mole_fractions

        raise NotImplementedError("This method should not be used")

    @override
    # @eqx.debug.assert_max_traces(max_traces=1)
    def compressibility_factor(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        del mole_fractions

        compressibility_factor: ArrayLike = self._z0(temperature) + self._dzdp0(temperature) * (
            pressure - self.p_eval
        )

        return compressibility_factor

    @override
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        volume: ArrayLike = (
            self.compressibility_factor(temperature, pressure, mole_fractions)
            * GAS_CONSTANT_BAR
            * temperature
            / pressure
        )

        return volume

    @override
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions

        volume_integral: FloatArray = (
            (
                jnp.log(pressure / self.p_eval)
                * (self._z0(temperature) - self._dzdp0(temperature) * self.p_eval)
                + self._dzdp0(temperature) * (pressure - self.p_eval)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )

        return volume_integral


class CombinedRealGasFugacity(RealGasBase):
    """Combined real gas EOS when only the functional form of the (log) fugacity is known.

    This directly obtains the (log) fugacity at the relevant pressure by selecting the appropriate
    EOS model based on the pressure bounds of each calibration.

    Note:
        This is not presently compatible with batched mole_fractions.

    Args:
        real_gases: Real gases to use
        calibrations: Experimental calibrations that correspond to ``real_gases``
        min_log_fugacity_coefficient: Minimum log fugacity coefficient to avoid numerical issues.
            Defaults to ``-15``.
        max_log_fugacity_coefficient: Maximum log fugacity coefficient to avoid numerical issues.
            Defaults to ``15``.
    """

    real_gases: tuple[RealGasBase, ...]
    """Real gases to use"""
    calibrations: tuple[ExperimentalCalibration, ...]
    """Experimental calibrations"""
    upper_pressure_bounds: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Upper pressure bounds"""
    _log_fugacity_coefficient_functions: tuple[Callable, ...]
    min_log_fugacity_coefficient: float | None
    """Minimum log fugacity coefficient to avoid numerical issues"""
    max_log_fugacity_coefficient: float | None
    """Maximum log fugacity coefficient to avoid numerical issues"""

    def __init__(
        self,
        real_gases: tuple[RealGasBase, ...],
        calibrations: tuple[ExperimentalCalibration, ...],
        min_log_fugacity_coefficient: float | None = None,
        max_log_fugacity_coefficient: float | None = None,
    ):
        self.real_gases = real_gases
        self.calibrations = calibrations
        self.upper_pressure_bounds = self._get_upper_pressure_bounds(calibrations)
        self._log_fugacity_coefficient_functions = tuple(
            to_hashable(eos.log_fugacity_coefficient) for eos in self.real_gases
        )
        self.min_log_fugacity_coefficient = min_log_fugacity_coefficient
        self.max_log_fugacity_coefficient = max_log_fugacity_coefficient

    @staticmethod
    def _get_upper_pressure_bounds(
        calibrations: tuple[ExperimentalCalibration, ...],
    ) -> tuple[float, ...]:
        """Gets the upper pressure bounds based on each experimental calibration.

        Returns:
            Upper pressure bounds
        """
        upper_pressure_bounds: list[float] = []

        for ii, calibration in enumerate(calibrations):
            try:
                assert calibration.pressure_max is not None
            except AssertionError:
                if ii == len(calibrations) - 1:
                    continue
                else:
                    msg: str = "Maximum pressure cannot be None"
                    raise ValueError(msg)

            pressure_bound = calibration.pressure_max
            upper_pressure_bounds.append(pressure_bound)

        return tuple(upper_pressure_bounds)

    def _get_index(self, pressure: ArrayLike) -> Array:
        """Gets the index of the appropriate EOS model based on ``pressure``.

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """
        index: Array = jnp.searchsorted(
            jnp.array(self.upper_pressure_bounds), pressure, side="right"
        )
        # jax.debug.print("pressure = {pressure}, index = {index}", pressure=pressure, index=index)

        return index

    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        temperature, pressure = jnp.broadcast_arrays(temperature, pressure)
        original_shape: tuple[int, ...] = temperature.shape
        # jax.debug.print("temperature = {out}", out=temperature)
        # jax.debug.print("pressure = {out}", out=pressure)

        # Flatten for vmap
        temperature = temperature.ravel()
        pressure = pressure.ravel()

        indices: Array = self._get_index(pressure)
        # jax.debug.print("indices = {out}", out=indices)

        def apply_log_fugacity_coefficient(
            index: ArrayLike, temperature, pressure, mole_fractions
        ) -> FloatArray:
            return lax.switch(
                index,
                self._log_fugacity_coefficient_functions,
                temperature,
                pressure,
                mole_fractions,
            )

        vmap_log_fugacity_coefficient: Callable = eqx.filter_vmap(
            apply_log_fugacity_coefficient, in_axes=(0, 0, 0, None)
        )
        log_fugacity_coefficient: FloatArray = vmap_log_fugacity_coefficient(
            indices, temperature, pressure, mole_fractions
        )

        log_fugacity_coefficient = jnp.clip(
            log_fugacity_coefficient,
            a_min=self.min_log_fugacity_coefficient,
            a_max=self.max_log_fugacity_coefficient,
        )
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        log_fugacity: FloatArray = log_fugacity_coefficient + jnp.log(pressure)

        return jnp.reshape(log_fugacity, original_shape)
