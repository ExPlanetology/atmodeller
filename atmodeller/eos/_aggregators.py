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
"""Classes that aggregate EOS

Units for temperature and pressure are K and bar, respectively.
"""

import logging
import sys
from abc import ABC
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos.core import IdealGas, RealGas
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibration

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


class CombinedRealGasABC(RealGas, ABC):
    """Base class for combining real gas EOS

    Args:
        real_gases: Real gases to combine
        calibrations: Experimental calibrations that correspond to `real_gases`
    """

    def __init__(
        self,
        real_gases: list[RealGasProtocol],
        calibrations: list[ExperimentalCalibration],
    ):
        self._real_gases: list[RealGasProtocol] = real_gases
        self._calibrations: list[ExperimentalCalibration] = calibrations
        self._upper_pressure_bounds: Array = self._get_upper_pressure_bounds()

    @property
    def volume_functions(self) -> list[Callable]:
        """Volume functions"""
        return [eos.volume for eos in self._real_gases]

    @property
    def volume_integral_functions(self) -> list[Callable]:
        """Volume integral functions"""
        return [eos.volume_integral for eos in self._real_gases]

    # Do not jit. Causes problems with initialization.
    def _get_upper_pressure_bounds(self) -> Array:
        """Gets the upper pressure bounds based on each experimental calibration.

        Returns:
            Upper pressure bounds
        """
        upper_pressure_bounds: list[float] = []

        for ii, calibration in enumerate(self._calibrations):
            try:
                assert calibration.pressure_max is not None
            except AssertionError:
                if ii == len(self._calibrations) - 1:
                    continue
                else:
                    msg: str = "Maximum pressure cannot be None"
                    raise ValueError(msg)

            pressure_bound = calibration.pressure_max
            upper_pressure_bounds.append(pressure_bound)

        return jnp.array(upper_pressure_bounds)

    @jit
    def _get_index(self, pressure: ArrayLike) -> int:
        """Gets the index of the appropriate EOS model based on `pressure`.

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """

        def body_fun(carry: int, i: int) -> tuple[int, int]:
            pressure_high: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, i, keepdims=False
            )
            condition: Array = jnp.greater_equal(pressure, pressure_high)
            new_index: int = lax.cond(condition, lambda _: i + 1, lambda _: carry, None)
            return new_index, new_index

        initial_carry: int = 0
        index, _ = lax.scan(body_fun, initial_carry, jnp.arange(len(self._upper_pressure_bounds)))  # type:ignore

        return index

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)
        volume: Array = lax.switch(index, self.volume_functions, temperature, pressure)

        return volume

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "real_gases": self._real_gases,
            "calibrations": self._calibrations,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class UpperBoundRealGas(RealGas):
    """An upper bound for an EOS

    This is used to extrapolate an EOS assuming that the compressibility factor is a linear
    function of pressure.

    Args:
        real_gas: Real gas to evaluate the compressibility factor at `p_eval`.
        p_eval: Evaluation pressure in bar. This is usually the maximum calibration pressure
            of `real_gas`. Defaults to 1 bar.
        dzdp: Gradient of the compressibility factor. Defaults to 0.
    """

    def __init__(self, real_gas: RealGasProtocol, p_eval: float = 1, dzdp: float = 0):
        self._real_gas: RealGasProtocol = real_gas
        self._p_eval: float = p_eval
        self._dzdp: float = dzdp

    @jit
    def _z0(self, temperature: ArrayLike) -> ArrayLike:
        """Compressibility factor of the previous EOS to blend smoothly with.

        Args:
            temperature: Temperature in K
        """
        return self._real_gas.compressibility_factor(temperature, self._p_eval)

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    @override
    @jit
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        compressibility_factor: ArrayLike = self._z0(temperature) + self._dzdp * (
            pressure - self._p_eval
        )

        return compressibility_factor

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        return self.compressibility_factor(temperature, pressure) * self.ideal_volume(
            temperature, pressure
        )

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral in units required for internal Atmodeller operations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        volume_integral: Array = jnp.log(pressure / self._p_eval) * (
            GAS_CONSTANT_BAR * temperature * (self._z0(temperature) - self._dzdp * self._p_eval)
        ) + GAS_CONSTANT_BAR * temperature * self._dzdp * (pressure - self._p_eval)

        return volume_integral

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "real_gas": self._real_gas,
            "p_eval": self._p_eval,
            "dzdp": self._dzdp,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class CombinedRealGasSwitch(CombinedRealGasABC):
    """Combined real gas EOS

    This class selects the EOS to use based on an index. This is only meaningful if subsequent EOS
    contain the contribution of previous EOS, otherwise there will be a mismatch across the joining
    boundary.
    """

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)
        volume_integral: Array = lax.switch(
            index, self.volume_integral_functions, temperature, pressure
        )

        return volume_integral


@register_pytree_node_class
class CombinedRealGas(CombinedRealGasABC):
    """Combined real gas EOS with separate integrations for each EOS

    This class computes the contribution to the volume integral separately for each EOS based on
    the range covered by its P-T calibration, and then combines them.

    This class also automatically applies appropriate extrapolation below the minimum calibration
    pressure and above the maximum calibration pressure. Reasonable extrapolation behaviour is
    required to ensure that the function is bounded to avoid throwing NaNs or infs which will crash
    the solver. Physically, it is reasonable to extend the lower bound using the ideal gas law and
    the upper bound assuming a linear pressure dependence of the compressibility factor.

    There is no bounding for temperature; hence it is assumed that the extrapolation
    behaviour of temperature is reasonable. This is practically useful because the calibrations are
    often restricted to a lower temperature range than the high temperatures that are typically of
    interest for hot rocks and magma ocean planets.

    Args:
        real_gases: Real gases to combine
        calibrations: Experimental calibrations that correspond to `real_gases`
        dzdp: Constant compressibility (pressure) gradient for the upper bound extrapolation (if
            relevant). Defaults to 0.
        extrapolate: Extrapolate the EOS to have reasonable behaviour below the minimum and above
            the maximum calibration pressure if required. This argument is always set to False for
            tree flattening and unflattening operations. Defaults to True.
    """

    @override
    def __init__(
        self,
        real_gases: list[RealGasProtocol],
        calibrations: list[ExperimentalCalibration],
        dzdp: float = 0.0,
        extrapolate: bool = True,
    ):
        self._real_gases: list[RealGasProtocol] = real_gases
        self._calibrations: list[ExperimentalCalibration] = calibrations
        self._dzdp: float = dzdp
        self._extrapolate: bool = extrapolate
        if self.extrapolate_lower:
            self._append_lower_bound()
        if self.extrapolate_upper:
            self._append_upper_bound()
        self._upper_pressure_bounds: Array = self._get_upper_pressure_bounds()

    @property
    def extrapolate_lower(self) -> bool:
        """Include extrapolation below the minimum calibration pressure"""
        return self._calibrations[0].pressure_min is not None and self._extrapolate

    @property
    def extrapolate_upper(self) -> bool:
        """Include extrapolation above the maximum calibration pressure"""
        return self._calibrations[-1].pressure_max is not None and self._extrapolate

    # Do not jit. Causes problems with initialization.
    def _append_lower_bound(self) -> None:
        """Appends the lower bound, which gives ideal gas behaviour"""
        self._real_gases.insert(0, IdealGas())
        pressure_max: float = self._calibrations[0].pressure_min  # type: ignore check done before
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_max=pressure_max)
        self._calibrations.insert(0, calibration)

    # Do not jit. Causes problems with initialization.
    def _append_upper_bound(self) -> None:
        """Appends the upper bound"""
        pressure_min: float = self._calibrations[-1].pressure_max  # type: ignore check done before
        real_gas: RealGasProtocol = UpperBoundRealGas(
            self._real_gases[-1], pressure_min, self._dzdp
        )
        self._real_gases.append(real_gas)
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=pressure_min)
        self._calibrations.append(calibration)

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)

        def compute_integral(ii: int, pressure_high: ArrayLike, pressure_low: ArrayLike) -> Array:
            volume_integral_high: Array = lax.switch(
                ii, self.volume_integral_functions, temperature, pressure_high
            )
            # jax.debug.print("compute_integral: index = {ii}", ii=ii)
            # jax.debug.print(
            #    "compute_integral: volume_integral_high = {out}", out=volume_integral_high
            # )
            volume_integral_low: Array = lax.switch(
                ii, self.volume_integral_functions, temperature, pressure_low
            )
            # jax.debug.print(
            #    "compute_integral: volume_integral_low = {out}", out=volume_integral_low
            # )
            integral_contribution = volume_integral_high - volume_integral_low

            return integral_contribution

        def body_fun(ii: int, carry: Array) -> Array:
            # jax.debug.print("body_fun: index = {out}", out=index)
            pressure_high: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, ii, keepdims=False
            )
            # jax.debug.print("body_fun: pressure_high = {out}", out=pressure_high)
            pressure_low: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, ii - 1, keepdims=False
            )
            # jax.debug.print("body_fun: pressure_low = {out}", out=pressure_low)
            carry = carry + compute_integral(ii, pressure_high, pressure_low)
            # jax.debug.print("body_fun: carry = {out}", out=carry)

            return carry

        # Initialize. Must be 0.0 to ensure float array.
        total_integral: Array = jnp.array(0.0)

        def add_only_first_integral(total_integral: Array) -> Array:
            integral: Array = lax.switch(0, self.volume_integral_functions, temperature, pressure)
            # jax.debug.print("add_only_first_integral: integral = {out}", out=integral)

            return total_integral + integral

        total_integral: Array = lax.cond(
            index == 0, add_only_first_integral, lambda x: x, total_integral
        )

        # FIXME: Use lax scan for reverse differentiation compatibility
        # Loop over and accumulate the vdP integrations over the EOS. This will only do something
        # if index >= 2, so will effectively be ignored for index = 1, as desired
        total_integral: Array = lax.fori_loop(1, index, body_fun, total_integral)
        # jax.debug.print("total_integral after lax.fori_loop = {out}", out=total_integral)

        def add_final_integral(total_integral: Array) -> Array:
            # Account for the first integral, which thus far has not been included for index > 0.
            pressure_high = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, 0, keepdims=False
            )
            # Do not evaluate a difference because log(0) is not defined.
            # TODO: Check. This evaluation for P<1 will be negative for an ideal gas.
            first_integral: Array = lax.switch(
                0, self.volume_integral_functions, temperature, pressure_high
            )
            # jax.debug.print("first_integral = {out}", out=first_integral)

            pressure_low: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, index - 1, keepdims=False
            )
            final_integral: Array = compute_integral(index, pressure, pressure_low)

            return first_integral + final_integral + total_integral

        total_integral = lax.cond(index > 0, add_final_integral, lambda x: x, total_integral)
        # jax.debug.print("total_integral = {out}", out=total_integral)

        return total_integral

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "real_gases": self._real_gases,
            "calibrations": self._calibrations,
            "dzdp": self._dzdp,
            # Must be False to avoid re-appending extrapolation bounds during JAX operations
            "extrapolate": False,
        }
        return (children, aux_data)
