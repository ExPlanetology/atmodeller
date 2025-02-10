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
"""Real gas EOS from :cite:t:`HP91,HP98,HP11`"""

import logging
import sys
from abc import abstractmethod
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from scipy.constants import kilo

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import (
    CORK,
    RealGas,
    RedlichKwongABC,
    RedlichKwongImplicitDenseFluidABC,
    RedlichKwongImplicitGasABC,
    VirialCompensation,
)
from atmodeller.interfaces import RealGasProtocol
from atmodeller.thermodata import CriticalData, select_critical_data
from atmodeller.utilities import ExperimentalCalibration, PyTreeNoData

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


class CorrespondingStatesUnitConverter:
    r"""Unit converter for Holland and Powell CORK corresponding states model

    This converts the coefficient units from Holland and Powell to the required units for
    Atmodeller. This accounts for kilo factors and converting the energy from J to SI volume and
    pressure in bar using:

    .. math::
        1\ \mathrm{J} = 10^{-5}\ \mathrm{m}^3\ \mathrm{bar}
    """

    @staticmethod
    def convert_a_coefficients(a_coefficients: tuple[float, ...]) -> tuple[float, ...]:
        r"""Converts the a coefficients for corresponding states.

        The a coefficients (a0 and a1) have units :cite:p:`HP91{Equation 9}`:

        .. math::
            \left(\frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{K}}\right)^2

        Converting kJ gives rise to a unit conversion factor of :math:`10^{-4}`.

        Args:
            a_coefficients: a coefficients from Holland and Powell

        Returns:
            a coefficients with units of

            .. math::
                \left(\frac{\mathrm{m}^3\ \mathrm{bar}}{\mathrm{mol}\ \mathrm{K}}\right)^2
        """
        factor: float = 1e-4

        return tuple(map(lambda a_coefficient: factor * a_coefficient, a_coefficients))

    @staticmethod
    def convert_b_coefficient(b_coefficient: float) -> float:
        r"""Converts the b coefficient for corresponding states.

        The b coefficient (b0) has units :cite:p:`HP91{Equation 9}`:

        .. math::

            \frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{K}}

        Converting kJ gives rise to a unit conversion factor of :math:`10^{-2}`.

        Args:
            b_coefficient: b coefficient from Holland and Powell

        Returns:
            b coefficient with units of

            .. math::
                \frac{\mathrm{m}^3\ \mathrm{bar}}{\mathrm{mol}\ \mathrm{K}}
        """
        factor: float = 1e-2

        return b_coefficient * factor

    @staticmethod
    def convert_virial_coefficients(virial_coefficients: tuple[float, ...]) -> tuple[float, ...]:
        r"""Converts the virial coefficients for corresponding states

        The virial coefficients, for example associated with coefficients c and d in
        :cite:`HP91{Table 2}`, have units:

        .. math::

            \frac{\mathrm{kJ}}{\mathrm{mol}\ \mathrm{K}}

        Converting kJ gives rise to a unit conversion factor of :math:`10^{-2}`.

        Args:
            virial_coefficient: Virial coefficients from Holland and Powell

        Returns:
            virial coefficients with units of

            .. math::
                \frac{\mathrm{m}^3\ \mathrm{bar}}{\mathrm{mol}\ \mathrm{K}}
        """
        factor: float = 1e-2

        return tuple(
            map(lambda virial_coefficient: factor * virial_coefficient, virial_coefficients)
        )


class FullUnitConverter:
    r"""Unit converter for Holland and Powell full CORK models for H2O and CO2

    This converts the coefficient units from Holland and Powell to the required units for
    Atmodeller. This accounts for kilo factors and converting the energy from J to SI volume and
    pressure in bar using:

    .. math::
        1\ \mathrm{J} = 10^{-5}\ \mathrm{m}^3\ \mathrm{bar}
    """

    @staticmethod
    def convert_a_coefficients(a_coefficients: tuple[float, ...]) -> tuple[float, ...]:
        r"""Converts the a coefficients for the full CORK models

        The a parameter has units :cite:p:`HP91{Table 1}`

        .. math::
            \frac{\mathrm{kJ}^2\ \mathrm{K}^{1/2}}{\mathrm{kbar}\ \mathrm{mol}^2}

        Each a coefficient has a different power of K, but this is dealt with during the
        multiplication with temperature. Here, we just need to deal with kJ and other kilo factors,
        which gives rise to a unit conversion factor of :math:`10^{-7}`.

        Args:
            a_coefficients: a coefficients from Holland and Powell

        Returns:
            a coefficients with units converted such that the a parameter has units of

            .. math::
                \left(\frac{\mathrm{m}^3}{\mathrm{mol}}\right)^2\ \mathrm{bar}\ \mathrm{K}^{1/2}
        """
        factor: float = 1e-7

        return tuple(map(lambda a_coefficient: factor * a_coefficient, a_coefficients))

    @staticmethod
    def convert_b_coefficient(b_coefficient: float) -> float:
        r"""Converts the b coefficient for the full CORK models

        The b parameter has units :cite:p:`HP91{Table 1}`

        .. math::
            \frac{\mathrm{kJ}}{\mathrm{kbar}\ \mathrm{mol}}

        Converting J gives rise to a unit conversion factor of :math:`10^{-5}`.

        Args:
            b_coefficient: b coefficient from Holland and Powell

        Returns:
            b coefficients with units converted such that the b parameter has units of

            .. math::
                \frac{\mathrm{m}^3}{\mathrm{mol}}
        """
        factor: float = 1e-5

        return b_coefficient * factor

    @staticmethod
    def convert_virial_coefficients(
        virial_coefficients: tuple[float, ...], pressure_exponent
    ) -> tuple[float, ...]:
        r"""Converts the virial coefficients for the full CORK models

        The volume correction to the MRK volume, :math:`\Delta V`, based on the parameter that the
        coefficients determine is given by:

        .. math::
            \Delta V = \mathrm{parameter(coefficients)} (P-P_0)^\gamma

        where :math:`\gamma` is `pressure_exponent` and :math:`P_0` is the pressure at which the
        MRK equation begins to overestimate the molar volume significantly.

        Args:
            virial_coefficients: Virial coefficients from Holland and Powell
            pressure_exponent: Pressure exponent :math:`\gamma`, also given by Holland and Powell

        Returns:
            Virial coefficients with the appropriate units for Atmodeller
        """
        factor: float = 1e-5 / kilo**pressure_exponent

        return tuple(
            map(lambda virial_coefficient: factor * virial_coefficient, virial_coefficients)
        )


@register_pytree_node_class
class MRKCorrespondingStatesHP91(RedlichKwongABC):
    """MRK corresponding states :cite:p:`HP91`

    Universal constants from :cite:t:`HP91{Table 2}`

    Args:
        critical_data: Critical data
    """

    def __init__(self, critical_data: CriticalData):
        self._critical_data: CriticalData = critical_data
        self._a_coefficients: tuple[float, ...] = (
            CorrespondingStatesUnitConverter.convert_a_coefficients((5.45963e-5, -8.63920e-6, 0))
        )
        self._b: float = CorrespondingStatesUnitConverter.convert_b_coefficient(9.18301e-4)

    @property
    def critical_pressure(self) -> float:
        """Critical pressure in bar"""
        return self._critical_data.pressure

    @property
    def critical_temperature(self) -> float:
        """Critical temperature in K"""
        return self._critical_data.temperature

    @classmethod
    def get_species(cls, species: str) -> Self:
        """Gets an MRK corresponding states model for a given species.

        Args:
            species: Species

        Returns:
            An MRK corresponding states model for the species
        """
        critical_data: CriticalData = select_critical_data(species)

        return cls(critical_data)

    @override
    @jit
    def a(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""MRK `a` parameter :cite:p:`HP91{Equation 9}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\ \mathrm{mol}^{-1})^2\ \mathrm{K}^{1/2}\ \mathrm{bar}`
        """
        del pressure

        a: ArrayLike = (
            self._a_coefficients[0] * jnp.power(self.critical_temperature, (5.0 / 2))
            + self._a_coefficients[1]
            * jnp.power(self.critical_temperature, (3.0 / 2))
            * temperature
            + self._a_coefficients[2]
            * jnp.power(self.critical_temperature, (1.0 / 2))
            * jnp.square(temperature)
        )
        a = a / self.critical_pressure

        return a

    @override
    @jit
    def b(self) -> ArrayLike:
        r"""MRK `b` parameter computed from :attr:`b0` :cite:p:`HP91{Equation 9}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            MRK `b` parameter in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`.
        """
        b: ArrayLike = self._b * self.critical_temperature / self.critical_pressure

        return b

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children: tuple = ()
        aux_data = (self._critical_data,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(*aux_data)


class MRKImplicitHP91ABCMixin:
    """MRK implicit :cite:p:`HP91`

    Universal constants from :cite:t:`HP91{Table 1}`.

    Args:
        a_coefficients: `a` coefficients
        b: `b` coefficient
        Ta: Temperature at which the `a` parameter is equal for the dense fluid and gas in K
        Tc: Critical temperature in K
    """

    def __init__(self, a_coefficients: tuple[float, ...], b: float, Ta: float, Tc: float):
        self._a_coefficients: tuple[float, ...] = a_coefficients
        self._b: float = b
        self._Ta: float = Ta
        self._Tc: float = Tc

    @abstractmethod
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        """Temperature difference for the calculation of the `a` parameter

        Args:
            temperature: Temperature in K

        Returns:
            Temperature difference in K
        """
        ...

    @jit
    def a(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""MRK `a` parameter :cite:p:`HP91{Equation 6}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            MRK `a` parameter
        """
        del pressure

        delta_temperature: ArrayLike = self.delta_temperature_for_a(temperature)
        a: ArrayLike = (
            self._a_coefficients[0]
            + self._a_coefficients[1] * delta_temperature
            + self._a_coefficients[2] * jnp.square(delta_temperature)
            + self._a_coefficients[3] * jnp.power(delta_temperature, 3)
        )

        return a

    def b(self) -> ArrayLike:
        return self._b

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "a_coefficients": self._a_coefficients,
            "b": self._b,
            "Ta": self._Ta,
            "Tc": self._Tc,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class MRKImplicitGasHP91(MRKImplicitHP91ABCMixin, RedlichKwongImplicitGasABC):
    """MRK for gaseous phase :cite:p:`HP91{Equation 6a}`"""

    @override
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        return self._Ta - temperature


Tc_H2O: float = 695
"""Critical temperature of H2O in K for the MRK/CORK model :cite:p:`HP91`"""
Ta_H2O: float = 673  # K
r"""Temperature at which :math:`a_{\mathrm gas} = a` for H2O by fitting :cite:p:`HP91`"""
b0_H2O: float = FullUnitConverter.convert_b_coefficient(1.465)
"""b parameter value which is the same across all H2O phases :cite:p:`HP91`"""

H2OMrkGasHolland91: MRKImplicitGasHP91 = MRKImplicitGasHP91(
    FullUnitConverter.convert_a_coefficients((1113.4, 5.8487, -2.1370e-2, 6.8133e-5)),
    b0_H2O,
    Ta_H2O,
    Tc_H2O,
)
"""H2O MRK for gas phase :cite:p:`HP91`"""


@register_pytree_node_class
class MRKImplicitLiquidHP91(MRKImplicitHP91ABCMixin, RedlichKwongImplicitDenseFluidABC):
    """MRK for liquid phase :cite:p`HP91{Equation 6}`"""

    @override
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        return self._Ta - temperature


H2OMrkLiquidHolland91: MRKImplicitLiquidHP91 = MRKImplicitLiquidHP91(
    FullUnitConverter.convert_a_coefficients((1113.4, -0.88517, 4.53e-3, -1.3183e-5)),
    b0_H2O,
    Ta_H2O,
    Tc_H2O,
)
"""H2O MRK for liquid phase :cite:p`HP91`"""


@register_pytree_node_class
class MRKImplicitFluidHP91(MRKImplicitHP91ABCMixin, RedlichKwongImplicitDenseFluidABC):
    """MRK for supercritical fluid :cite:p:`HP91{Equation 6}`"""

    @override
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        return temperature - self._Ta

    @override
    @jit
    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Initial guess volume to ensure convergence to the correct root

        See :cite:t:`HP91{Appendix}`. It appears that there is only ever a single root, even if
        Ta < temperature < Tc. Holland and Powell state that a single root exists if
        temperature > Tc, but this appears to be true if temperature > Ta. Nevertheless, the
        initial guess is changed accordingly.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Initial volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """

        def low_temperature_case() -> ArrayLike:
            return self.b() / 2

        def high_temperature_case() -> ArrayLike:
            return GAS_CONSTANT_BAR * temperature / pressure + self.b()

        initial_volume: Array = jnp.where(
            temperature < jnp.array(self._Tc),
            low_temperature_case(),
            high_temperature_case(),
        )

        return initial_volume


H2OMrkFluidHolland91: MRKImplicitFluidHP91 = MRKImplicitFluidHP91(
    FullUnitConverter.convert_a_coefficients(
        (
            1113.4,
            -0.22291,
            -3.8022e-4,
            1.7791e-7,
        )
    ),
    b0_H2O,
    Ta_H2O,
    Tc_H2O,
)
"""H2O MRK for fluid phase :cite:p`HP91`"""

CO2_critical_data: CriticalData = select_critical_data("CO2_g")
"""Alternative values from :cite:t:`HP91` are 304.2 K and 73.8 bar"""
CO2MrkHolland91: MRKImplicitFluidHP91 = MRKImplicitFluidHP91(
    FullUnitConverter.convert_a_coefficients((741.2, -0.10891, -3.4203e-4, 0)),
    FullUnitConverter.convert_b_coefficient(3.057),
    0,
    CO2_critical_data.temperature,
)
"""CO2 MRK :cite:p:`HP91{Above Equation 7}`

Critical behaviour is not considered for CO2 by :cite:t:`HP91`, but for consistency with the 
formulation for H2O, the CO2 critical temperature is set.
"""


@register_pytree_node_class
class H2OMrkGasFluid91(PyTreeNoData, RealGas):
    """A MRK model for H2O for the gas and supercritical fluid

    Args:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        Ta: Temperature at which a_gas = a in the MRK formulation in K
        Tc: Critical temperature in K
    """

    mrk_fluid: MRKImplicitFluidHP91 = H2OMrkFluidHolland91
    """The MRK for the supercritical fluid"""
    mrk_gas: MRKImplicitGasHP91 = H2OMrkGasHolland91
    """The MRK for the subcritical gas"""
    Ta: float = Ta_H2O
    """ Temperature at which a_gas = a in the MRK formulation in K"""
    Tc: float = Tc_H2O
    """Critical temperature in K"""

    @jit
    def _select_condition(self, temperature: ArrayLike) -> Array:
        """Selects the condition

        Args:
            temperature: Temperature in K

        Returns:
            Integer denoting the condition, i.e. the region of phase space
        """
        temperature_array: Array = jnp.asarray(temperature)

        # Supercritical
        cond0: Array = temperature_array >= self.Tc
        # jax.debug.print("cond0 = {cond}", cond=cond0)
        # Below Ta
        cond1: Array = temperature_array <= self.Ta
        # jax.debug.print("cond1 = {cond}", cond=cond1)
        # Below Tc
        cond2: Array = temperature_array < self.Tc
        # Ensure cond2 is exclusive of cond1
        cond2 = jnp.logical_and(cond2, ~cond1)

        # All conditions are mutually exclusive
        condition: Array = jnp.select([cond0, cond1, cond2], [0, 1, 2])
        # jax.debug.print("condition = {condition}", condition=condition)

        return condition

    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        condition: Array = self._select_condition(temperature)

        def volume_integral0() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        def volume_integral1() -> Array:
            return self.mrk_gas.volume_integral(temperature, pressure)

        def volume_integral2() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        volume_integral_funcs: list[Callable] = [
            volume_integral0,
            volume_integral1,
            volume_integral2,
        ]

        volume_integral: Array = lax.switch(condition, volume_integral_funcs)
        # jax.debug.print("volume_integral = {out}", out=volume_integral)

        return volume_integral

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        condition: Array = self._select_condition(temperature)

        def volume0() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        def volume1() -> Array:
            return self.mrk_gas.volume(temperature, pressure)

        def volume2() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        volume_funcs: list[Callable] = [volume0, volume1, volume2]

        volume: Array = lax.switch(condition, volume_funcs)
        # jax.debug.print("volume = {out}", out=volume)

        return volume


@register_pytree_node_class
class H2OMrkHP91(PyTreeNoData, RealGas):
    """A MRK model for H2O that accommodates critical behaviour

    Args:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation in K
        Tc: Critical temperature in K
    """

    mrk_fluid: MRKImplicitFluidHP91 = H2OMrkFluidHolland91
    """The MRK for the supercritical fluid"""
    mrk_gas: MRKImplicitGasHP91 = H2OMrkGasHolland91
    """The MRK for the subcritical gas"""
    mrk_liquid: MRKImplicitLiquidHP91 = H2OMrkLiquidHolland91
    """The MRK for the subcritical liquid"""
    Ta: float = Ta_H2O
    """Temperature at which a_gas = a in the MRK formulation in K"""
    Tc: float = Tc_H2O
    """Critical temperature in K"""

    @jit
    def Psat(self, temperature: ArrayLike) -> Array:
        """Saturation curve

        Compared to :cite:t:`HP91` the pressure is returned in bar, as required by Atmodeller.

        Args:
            temperature: Temperature in K

        Returns:
            Saturation curve pressure in bar
        """
        Psat: Array = (
            -13.627
            + 7.29395e-4 * jnp.square(temperature)
            - 2.34622e-6 * jnp.power(temperature, 3)
            + 4.83607e-12 * jnp.power(temperature, 5)
        )

        return Psat

    @jit
    def _select_condition(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Selects the condition

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Integer denoting the condition, i.e. the region of phase space
        """
        Psat: Array = self.Psat(temperature)
        temperature_array: Array = jnp.asarray(temperature)
        pressure_array: Array = jnp.asarray(pressure)

        # Supercritical (saturation pressure irrelevant)
        cond0: Array = temperature_array >= self.Tc
        # jax.debug.print("cond0 = {cond}", cond=cond0)
        # Below the saturation pressure and below Ta
        cond1: Array = jnp.logical_and(temperature_array <= self.Ta, pressure_array <= Psat)
        # jax.debug.print("cond1 = {cond}", cond=cond1)
        # Below the saturation pressure and below Tc
        cond2: Array = jnp.logical_and(temperature_array < self.Tc, pressure_array <= Psat)
        # Ensure cond2 is exclusive of cond1
        cond2 = jnp.logical_and(cond2, ~cond1)
        # jax.debug.print("cond2 = {cond}", cond=cond2)
        # Above the saturation pressure and below Ta
        cond3: Array = jnp.logical_and(temperature_array <= self.Ta, pressure_array > Psat)
        # jax.debug.print("cond3 = {cond}", cond=cond3)
        # Above the saturation pressure and below Tc
        cond4: Array = jnp.logical_and(temperature_array < self.Tc, pressure_array > Psat)
        # Ensure cond4 is exclusive of cond3
        cond4 = jnp.logical_and(cond4, ~cond3)
        # jax.debug.print("cond4 = {cond}", cond=cond4)

        # All conditions are mutually exclusive
        condition: Array = jnp.select([cond0, cond1, cond2, cond3, cond4], [0, 1, 2, 3, 4])
        # jax.debug.print("condition = {condition}", condition=condition)

        return condition

    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        condition: Array = self._select_condition(temperature, pressure)
        Psat: Array = self.Psat(temperature)

        def volume_integral0() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        def volume_integral1() -> Array:
            return self.mrk_gas.volume_integral(temperature, pressure)

        def volume_integral2() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        def volume_integral3() -> Array:
            value: Array = self.mrk_gas.volume_integral(temperature, Psat)
            value = value - self.mrk_liquid.volume_integral(temperature, Psat)
            value = value + self.mrk_liquid.volume_integral(temperature, pressure)

            return value

        def volume_integral4() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        volume_integral_funcs: list[Callable] = [
            volume_integral0,
            volume_integral1,
            volume_integral2,
            volume_integral3,
            volume_integral4,
        ]

        volume_integral: Array = lax.switch(condition, volume_integral_funcs)
        # jax.debug.print("volume_integral = {out}", out=volume_integral)

        return volume_integral

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        condition: Array = self._select_condition(temperature, pressure)

        def volume0() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        def volume1() -> Array:
            return self.mrk_gas.volume(temperature, pressure)

        def volume2() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        def volume3() -> Array:
            return self.mrk_liquid.volume(temperature, pressure)

        def volume4() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        volume_funcs: list[Callable] = [volume0, volume1, volume2, volume3, volume4]

        volume: Array = lax.switch(condition, volume_funcs)
        # jax.debug.print("volume = {out}", out=volume)

        return volume


H2OMrkHolland91: RealGasProtocol = H2OMrkHP91()
"""H2O MRK that includes critical behaviour (also the liquid phase)"""
CO2_mrk_cs_holland91: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("CO2_g")
"""CO2 MRK corresponding states :cite:p:`HP91`"""
CH4_mrk_cs_holland91: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("CH4_g")
"""CH4 MRK corresponding states :cite:p:`HP91`"""
H2_mrk_cs_holland91: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("H2_g_Holland")
"""H2 MRK corresponding states :cite:p:`HP91`"""
CO_mrk_cs_holland91: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("CO_g")
"""CO MRK corresponding states :cite:p:`HP91`"""
N2_mrk_cs_holland91: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("N2_g")
"""N2 MRK corresponding states :cite:p:`HP91`"""
S2_mrk_cs_holland11: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("S2_g")
"""S2 MRK corresponding states :cite:p:`HP11`"""
H2S_mrk_cs_holland11: RealGasProtocol = MRKCorrespondingStatesHP91.get_species("H2S_g")
"""H2S MRK corresponding states :cite:p:`HP11`"""
H2O_mrk_fluid_holland91: RealGasProtocol = H2OMrkFluidHolland91
"""H2O MRK supercritical fluid :cite:p:`HP91`"""
H2O_mrk_gas_holland91: RealGasProtocol = H2OMrkGasHolland91
"""H2O MRK gas :cite:p:`HP91`"""
H2O_mrk_liquid_holland91: RealGasProtocol = H2OMrkLiquidHolland91
"""H2O MRK liquid :cite:p:`HP91`"""
H2O_mrk_gas_fluid_holland91: RealGasProtocol = H2OMrkGasFluid91()
"""H2O MRK for the gas and supercritical fluid :cite:p:`HP91`"""

coefficients_P: tuple[float, ...] = CorrespondingStatesUnitConverter.convert_virial_coefficients(
    (6.93054e-7, -8.38293e-8)
)
coefficients_sqrtP: tuple[float, ...] = (
    CorrespondingStatesUnitConverter.convert_virial_coefficients((-3.30558e-5, 2.30524e-6))
)
virial_compensation_corresponding_states: VirialCompensation = VirialCompensation(
    coefficients_P, coefficients_sqrtP, (0, 0), 0
)
"""Virial compensation for corresponding states :cite:p:`HP91{Table 2}`

In this case it appears `P0` is always zero, even though for the full CORK equations it determines
whether or not the virial contribution is added. The unit conversions to SI and pressure in bar 
mean that every virial coefficient has been multiplied by 1e-2 compared to the values in 
:cite:t:`HP91{Table 2}`.
"""

experimental_calibration_holland91: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=373,
    temperature_max=1873,
    pressure_min=1,
    pressure_max=50e3,
)
"""Experimental calibration for :cite:`HP91,HP11` models"""
experimental_calibration_holland98: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=373,
    temperature_max=1873,
    pressure_min=1,
    pressure_max=120e3,
)
"""Experimental calibration for :cite:`HP98` models"""

CH4_cork_cs_holland91: RealGasProtocol = CORK(
    CH4_mrk_cs_holland91, virial_compensation_corresponding_states, select_critical_data("CH4_g")
)
"""CH4 CORK corresponding states :cite:p:`HP91`"""
CH4_cork_cs_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [CH4_cork_cs_holland91], [experimental_calibration_holland91]
)
"""CH4 CORK corresponding states bounded :cite:p:`HP91`"""
CO_cork_cs_holland91: RealGasProtocol = CORK(
    CO_mrk_cs_holland91, virial_compensation_corresponding_states, select_critical_data("CO_g")
)
"""CO CORK corresponding states :cite:p:`HP91`"""
CO_cork_cs_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [CO_cork_cs_holland91], [experimental_calibration_holland91]
)
"""CO CORK corresponding states bounded :cite:p:`HP91`"""
CO2_cork_cs_holland91: RealGasProtocol = CORK(
    CO2_mrk_cs_holland91, virial_compensation_corresponding_states, select_critical_data("CO2_g")
)
"""CO2 CORK corresponding states :cite:p:`HP91`"""
CO2_cork_cs_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [CO2_cork_cs_holland91], [experimental_calibration_holland91]
)
"""CO2 CORK corresponding states bounded :cite:p:`HP91`"""
H2_cork_cs_holland91: RealGasProtocol = CORK(
    H2_mrk_cs_holland91,
    virial_compensation_corresponding_states,
    select_critical_data("H2_g_Holland"),
)
"""H2 CORK corresponding states :cite:p:`HP91`"""
H2_cork_cs_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [H2_cork_cs_holland91], [experimental_calibration_holland91]
)
"""H2 CORK corresponding states bounded :cite:p:`HP91`"""
H2S_cork_cs_holland11: RealGasProtocol = CORK(
    H2S_mrk_cs_holland11, virial_compensation_corresponding_states, select_critical_data("H2S_g")
)
"""H2S CORK corresponding states :cite:p:`HP91`"""
H2S_cork_cs_holland11_bounded: RealGasProtocol = CombinedRealGas(
    [H2S_cork_cs_holland11], [experimental_calibration_holland91]
)
"""H2S CORK corresponding states bounded :cite:p:`HP91`"""
N2_cork_cs_holland91: RealGasProtocol = CORK(
    N2_mrk_cs_holland91, virial_compensation_corresponding_states, select_critical_data("N2_g")
)
"""N2 CORK corresponding states :cite:p:`HP91`"""
N2_cork_cs_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [N2_cork_cs_holland91], [experimental_calibration_holland91]
)
"""N2 CORK corresponding states bounded :cite:p:`HP91`"""
S2_cork_cs_holland11: RealGasProtocol = CORK(
    S2_mrk_cs_holland11, virial_compensation_corresponding_states, select_critical_data("S2_g")
)
"""S2 CORK corresponding states :cite:p:`HP91`"""
S2_cork_cs_holland11_bounded: RealGasProtocol = CombinedRealGas(
    [S2_cork_cs_holland11], [experimental_calibration_holland91]
)
"""S2 CORK corresponding states bounded :cite:p:`HP91`"""

dummy_critical_data: CriticalData = CriticalData(1.0, 1.0)
"""Dummy critical data

The full CO2 and H2O CORK models are not a corresponding states model, which can be reproduced by 
ignoring the scaling by the critical temperature and pressure, i.e. setting these quantities to 
unity.
"""
CO2_virial_compensation_holland91: VirialCompensation = VirialCompensation(
    FullUnitConverter.convert_virial_coefficients((1.33790e-2, -1.01740e-5), 1.0),
    FullUnitConverter.convert_virial_coefficients((-2.26924e-1, 7.73793e-5), 0.5),
    (0, 0),
    5000,
)
"""CO2 virial compensation :cite:p:`HP91`"""
CO2_cork_holland91: RealGasProtocol = CORK(
    CO2MrkHolland91, CO2_virial_compensation_holland91, dummy_critical_data
)
"""CO2 cork :cite:p:`HP91`"""
CO2_cork_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [CO2_cork_holland91], [experimental_calibration_holland91]
)
"""CO2 cork bounded :cite:p:`HP91`"""

H2O_virial_compensation_holland91: VirialCompensation = VirialCompensation(
    FullUnitConverter.convert_virial_coefficients((-3.2297554e-3, 2.2215221e-6), 1.0),
    FullUnitConverter.convert_virial_coefficients((-3.025650e-2, -5.343144e-6), 0.5),
    (0, 0),
    2000,
)
"""H2O virial compensation :cite:p:`HP91`"""
H2O_cork_holland91: RealGasProtocol = CORK(
    H2OMrkHolland91, H2O_virial_compensation_holland91, dummy_critical_data
)
"""H2O cork :cite:p:`HP91`"""
H2O_cork_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [H2O_cork_holland91], [experimental_calibration_holland91]
)
"""H2O cork bounded :cite:p:`HP91`"""
H2O_cork_gas_fluid_holland91: RealGasProtocol = CORK(
    H2O_mrk_gas_fluid_holland91, H2O_virial_compensation_holland91, dummy_critical_data
)
"""H2O cork for the gas and supercritical fluid :cite:p:`HP91`"""
H2O_cork_gas_fluid_holland91_bounded: RealGasProtocol = CombinedRealGas(
    [H2O_cork_gas_fluid_holland91], [experimental_calibration_holland91]
)
"""H2O cork for the gas and supercritical fluid bounded :cite:p:`HP91`"""

CO2_virial_compensation_holland98: VirialCompensation = VirialCompensation(
    FullUnitConverter.convert_virial_coefficients((5.40776e-3, -1.59046e-6), 1.0),
    FullUnitConverter.convert_virial_coefficients((-1.78198e-1, 2.45317e-5), 0.5),
    (0, 0),
    5000,
)
"""CO2 virial compensation :cite:p:`HP98`"""
CO2_cork_holland98: RealGasProtocol = CORK(
    CO2MrkHolland91, CO2_virial_compensation_holland98, dummy_critical_data
)
"""CO2 cork :cite:p:`HP98`"""
CO2_cork_holland98_bounded: RealGasProtocol = CombinedRealGas(
    [CO2_cork_holland98], [experimental_calibration_holland98]
)
"""CO2 cork bounded :cite:p:`HP98`"""

H2O_virial_compensation_holland98: VirialCompensation = VirialCompensation(
    FullUnitConverter.convert_virial_coefficients((1.9853e-3, 0), 1.0),
    FullUnitConverter.convert_virial_coefficients((-8.9090e-2, 0), 0.5),
    FullUnitConverter.convert_virial_coefficients((8.0331e-2, 0), 0.25),
    2000,
)
"""H2O virial compensation :cite:p:`HP98`"""
H2O_cork_holland98: RealGasProtocol = CORK(
    H2OMrkHolland91, H2O_virial_compensation_holland98, dummy_critical_data
)
"""H2O cork :cite:p:`HP98`"""
H2O_cork_holland98_bounded: RealGasProtocol = CombinedRealGas(
    [H2O_cork_holland98], [experimental_calibration_holland98]
)
"""H2O cork bounded :cite:p:`HP98`"""
H2O_cork_gas_fluid_holland98: RealGasProtocol = CORK(
    H2O_mrk_gas_fluid_holland91, H2O_virial_compensation_holland98, dummy_critical_data
)
"""H2O cork for the gas and supercritical fluid :cite:p:`HP98`"""
H2O_cork_gas_fluid_holland98_bounded: RealGasProtocol = CombinedRealGas(
    [H2O_cork_gas_fluid_holland98], [experimental_calibration_holland98]
)
"""H2O cork for the gas and supercritical fluid bounded :cite:p:`HP98`"""


def get_holland_eos_models() -> dict[str, RealGasProtocol]:
    """Gets a dictionary of Holland and Powell EOS models that are bounded

    The naming convention is as follows:
        [species]_[eos model]_[citation]

    'cs' refers to corresponding states.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGasProtocol] = {}
    eos_models["CH4_cork_cs_holland91"] = CH4_cork_cs_holland91_bounded
    eos_models["CO_cork_cs_holland91"] = CO_cork_cs_holland91_bounded
    eos_models["CO2_cork_holland91"] = CO2_cork_holland91_bounded
    eos_models["CO2_cork_holland98"] = CO2_cork_holland98_bounded
    eos_models["CO2_cork_cs_holland91"] = CO2_cork_cs_holland91_bounded
    eos_models["H2_cork_cs_holland91"] = H2_cork_cs_holland91_bounded
    eos_models["H2O_cork_holland91"] = H2O_cork_gas_fluid_holland91_bounded
    eos_models["H2O_cork_holland98"] = H2O_cork_gas_fluid_holland98_bounded
    eos_models["H2S_cork_cs_holland11"] = H2S_cork_cs_holland11_bounded
    eos_models["N2_cork_cs_holland91"] = N2_cork_cs_holland91_bounded
    eos_models["S2_cork_cs_holland11"] = S2_cork_cs_holland11_bounded

    return eos_models
