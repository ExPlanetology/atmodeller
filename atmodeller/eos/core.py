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
"""Core classes and functions for real gas equations of state

Units for temperature and pressure are K and bar, respectively.
"""

import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable

import jax.numpy as jnp
import optimistix as optx
from jax import Array, grad, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.interfaces import RealGasProtocol
from atmodeller.thermodata import CriticalData
from atmodeller.utilities import (
    PyTreeNoData,
    safe_exp,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


class RealGas(ABC):
    r"""A real gas equation of state (EOS)

    Fugacity is computed using the standard relation:

    .. math::
        R T \ln f = \int V dP

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`f` is fugacity, :math:`V`
    is volume, and :math:`P` is pressure.
    """

    @abstractmethod
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """

    @abstractmethod
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """

    @abstractmethod
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral in units required for internal Atmodeller operations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """

    @jit
    def volume_integral_J(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral in J

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return 1e5 * self.volume_integral(temperature, pressure)

    @jit
    def dvolume_dpressure(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Derivative of volume with respect to pressure

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Derivative of volume with respect to pressure
        """
        dvolume_dpressure_fn: Callable = grad(self.volume, argnums=1)

        return dvolume_dpressure_fn(temperature, pressure)

    @jit
    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Log activity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log activity, which is dimensionless
        """
        # The standard state is defined at 1 bar (see PRESSURE_REFERENCE), so we do not need to
        # perform a division (by unity) to get activity, which is non-dimensional.

        return self.log_fugacity(temperature, pressure)

    @jit
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Compressibility factor

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Compressibility factor, which is dimensionless
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        volume_ideal: ArrayLike = self.ideal_volume(temperature, pressure)
        compressibility_factor: ArrayLike = volume / volume_ideal

        return compressibility_factor

    @jit
    def fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: Array = safe_exp(self.log_fugacity(temperature, pressure))

        return fugacity

    @jit
    def log_fugacity_coefficient(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log of the fugacity coefficient

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log of the fugacity coefficient, which is dimensionless
        """
        return self.log_fugacity(temperature, pressure) - self.ideal_log_fugacity(
            temperature, pressure
        )

    @jit
    def fugacity_coefficient(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Fugacity coefficient

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            fugacity coefficient, which is dimensionless
        """
        return safe_exp(self.log_fugacity_coefficient(temperature, pressure))

    @jit
    def ideal_log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity of an ideal gas

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity of an ideal gas
        """
        del temperature
        ideal_log_fugacity: ArrayLike = jnp.log(pressure)

        return ideal_log_fugacity

    @jit
    def ideal_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Volume of an ideal gas

        This is required to compute the compressibility factor.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Ideal volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure

        return ideal_volume


@register_pytree_node_class
class IdealGas(PyTreeNoData, RealGas):
    r"""Ideal gas equation of state:

    .. math::

        R T = P V

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`P` is pressure, and
    :math:`V` is volume.
    """

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        return self.ideal_log_fugacity(temperature, pressure)

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        return self.ideal_volume(temperature, pressure)

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        return GAS_CONSTANT_BAR * temperature * self.log_fugacity(temperature, pressure)


class RedlichKwongABC(RealGas):
    r"""Redlich-Kwong EOS:

    .. math::

        P = \frac{RT}{V-b} - \frac{a}{\sqrt{T}V(V+b)}

    where :math:`P` is pressure, :math:`T` is temperature, :math:`V` is the molar volume, :math:`R`
    the gas constant, :math:`a` corrects for the attractive potential of molecules, and :math:`b`
    corrects for the volume.

    This employs an approximation to analytically determine the volume and the volume integral.
    """

    @abstractmethod
    def a(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Gets the `a` parameter

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `a` parameter in
            :math:`(\mathrm{m}^3\ \mathrm{mol}^{-1})^2\ \mathrm{K}^{1/2}\ \mathrm{bar}`
        """

    @abstractmethod
    def b(self) -> ArrayLike:
        r"""Gets the `b` parameter

        Returns:
            `b` parameter in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        a: ArrayLike = self.a(temperature, pressure)

        volume_integral: Array = (
            jnp.log(pressure) * GAS_CONSTANT_BAR * temperature
            + self.b() * pressure
            + a
            / self.b()
            / jnp.sqrt(temperature)
            * (
                jnp.log(GAS_CONSTANT_BAR * temperature + self.b() * pressure)
                - jnp.log(GAS_CONSTANT_BAR * temperature + 2.0 * self.b() * pressure)
            )
        )

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
        r"""Volume-explicit equation :cite:p:`HP91{Equation 7}`

        Without complications of critical phenomena the RK equation can be simplified using the
        approximation:

        .. math::

            V \sim \frac{RT}{P} + b

        where :math:`V` is volume, :math:`R` is the gas constant, :math:`T` is temperature,
        :math:`P` is pressure, and :math:`b` corrects for the volume.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        a: ArrayLike = self.a(temperature, pressure)

        volume: Array = (
            jnp.sqrt(temperature)
            * -1.0
            * a
            * GAS_CONSTANT_BAR
            / (GAS_CONSTANT_BAR * temperature + self.b() * pressure)
            / (GAS_CONSTANT_BAR * temperature + 2.0 * self.b() * pressure)
            + GAS_CONSTANT_BAR * temperature / pressure
            + self.b()
        )

        return volume


class RedlichKwongImplicitABC(RedlichKwongABC):
    r"""Redlich-Kwong EOS in an implicit form

    .. math::

        P = \frac{RT}{V-b} - \frac{a}{\sqrt{T}V(V+b)}

    where :math:`P` is pressure, :math:`T` is temperature, :math:`V` is the molar volume, :math:`R`
    the gas constant, :math:`a` corrects for the attractive potential of molecules, and :math:`b`
    corrects for the volume.
    """

    @abstractmethod
    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Initial guess volume for the solution to ensure convergence to the correct root

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Initial volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        ...

    @jit
    def A_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """`A` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `A` factor, which is dimensionless
        """
        A_factor: ArrayLike = self.a(temperature, pressure) / (
            self.b() * GAS_CONSTANT_BAR * jnp.power(temperature, 1.5)
        )

        return A_factor

    @jit
    def B_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """`B` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `B` factor, which is dimensionless
        """
        B_factor: ArrayLike = self.b() * pressure / (GAS_CONSTANT_BAR * temperature)

        return B_factor

    @override
    @jit
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Compressibility factor

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility factor, which is dimensionless
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        volume_ideal: ArrayLike = self.ideal_volume(temperature, pressure)
        compressibility_factor: ArrayLike = volume / volume_ideal

        return compressibility_factor

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral :cite:p:`HP91{Equation A.2}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        log_fugacity: Array = self.log_fugacity(temperature, pressure)
        volume_integral: Array = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        z: Array = jnp.asarray(self.compressibility_factor(temperature, pressure))
        A: ArrayLike = self.A_factor(temperature, pressure)
        B: ArrayLike = self.B_factor(temperature, pressure)

        log_fugacity_coefficient: Array = -jnp.log(z - B) - A * jnp.log(1 + B / z) + z - 1
        log_fugacity: Array = jnp.log(pressure) + log_fugacity_coefficient

        return log_fugacity

    @jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]
        a: ArrayLike = self.a(temperature, pressure)

        # Coefficients for the polynomial in terms of volume. Unity coefficients are to satisfy
        # type checking.
        rtp: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        coeff2: ArrayLike = -1.0 * rtp
        coeff1: ArrayLike = a / (jnp.sqrt(temperature) * pressure) - 1.0 * self.b() * (
            rtp + self.b()
        )
        coeff0: ArrayLike = -1.0 * a * self.b() / (jnp.sqrt(temperature) * pressure)

        residual: Array = (
            jnp.power(volume, 3) + coeff2 * jnp.square(volume) + coeff1 * volume + coeff0
        )

        return residual

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Solves the RK equation numerically to compute the volume.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        initial_volume: ArrayLike = self.initial_volume(temperature, pressure)
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        # atol reduced since typical volumes are around 1e-5 to 1e-6
        solver = optx.Newton(rtol=1.0e-6, atol=1.0e-12)
        sol = optx.root_find(
            self._objective_function, solver, initial_volume, args=kwargs, throw=False
        )
        volume: ArrayLike = sol.value
        # jax.debug.print("volume = {out}", out=volume)

        return volume


class RedlichKwongImplicitDenseFluidABC(RedlichKwongImplicitABC):
    """MRK for the high density fluid phase :cite:p`HP91{Equation 6}`"""

    @override
    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Initial guess volume to ensure convergence to the correct root

        For the dense fluid phase a suitably low value must be chosen :cite:p:`HP91{Appendix}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Initial volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        del temperature
        del pressure

        initial_volume: ArrayLike = self.b() / 2

        return initial_volume


class RedlichKwongImplicitGasABC(RedlichKwongImplicitABC):
    """MRK for the low density gaseous phase :cite:p:`HP91{Equation 6a}`"""

    @override
    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Initial guess volume to ensure convergence to the correct root

        For the gaseous phase a suitably high value must be chosen :cite:p:`HP91{Appendix}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Initial volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        initial_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure + 10 * self.b()

        return initial_volume


@register_pytree_node_class
class VirialCompensation:
    r"""A virial compensation term for the increasing deviation of the MRK volumes with pressure

    General form of the equation :cite:t:`HP98` and also see :cite:t:`HP91{Equations 4 and 9}`:

    .. math::

        V_\mathrm{virial} = a(P-P0) + b(P-P0)^\frac{1}{2} + c(P-P0)^\frac{1}{4}

    This form also works for the virial compensation term from :cite:t:`HP91`, in which
    case :math:`c=0`.

    Although this looks similar to an EOS, it only calculates an additional perturbation to the
    volume and the volume integral of an MRK EOS, and hence it does not return a meaningful volume
    or volume integral by itself.

    Args:
        a_coefficients: Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`.
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly and may be determined from experimental data.

    Attributes:
        a_coefficients: Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`.
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly and may be determined from experimental data.
    """

    def __init__(
        self,
        a_coefficients: tuple[float, ...],
        b_coefficients: tuple[float, ...],
        c_coefficients: tuple[float, ...],
        P0: float,
    ):
        self.a_coefficients: tuple[float, ...] = a_coefficients
        self.b_coefficients: tuple[float, ...] = b_coefficients
        self.c_coefficients: tuple[float, ...] = c_coefficients
        self.P0: float = P0

    @jit
    def _a(self, temperature: ArrayLike, critical_data: CriticalData) -> Array:
        r"""`a` parameter :cite:p:`HP98`

        This is also the `d` parameter in :cite:t:`HP91`.

        Args:
            temperature: Temperature in K
            critical_data: Critical data

        Returns:
            `a` parameter in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}\ \mathrm{bar}^{-1}`
        """
        a: Array = (
            self.a_coefficients[1] * jnp.asarray(temperature)
            + self.a_coefficients[0] * critical_data.temperature
        )
        a = a / jnp.square(critical_data.pressure)

        return a

    @jit
    def _b(self, temperature: ArrayLike, critical_data: CriticalData) -> Array:
        r"""`b` parameter :cite:p:`HP98`

        This is also the `c` parameter in :cite:t:`HP91`.

        Args:
            temperature: Temperature in K
            critical_data: Critical data

        Returns:
            `b` parameter in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}\ \mathrm{bar}^{-1/2}`
        """
        b: Array = (
            self.b_coefficients[1] * jnp.asarray(temperature)
            + self.b_coefficients[0] * critical_data.temperature
        )
        b = b / jnp.power(critical_data.pressure, (3.0 / 2))

        return b

    @jit
    def _c(self, temperature: ArrayLike, critical_data: CriticalData) -> Array:
        r"""`c` parameter :cite:p:`HP98`

        Args:
            temperature: Temperature in K
            critical_data: Critical data

        Returns:
            `c` parameter in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}\ \mathrm{bar}^{-1/4}`
        """
        c: Array = (
            self.c_coefficients[1] * jnp.asarray(temperature)
            + self.c_coefficients[0] * critical_data.temperature
        )
        c = c / jnp.power(critical_data.pressure, (5.0 / 4))

        return c

    @jit
    def _delta_pressure(self, pressure: ArrayLike) -> Array:
        """Pressure difference

        Args:
            pressure: Pressure in bar

        Returns:
            Pressure difference relative to :attr:`P0` in bar
        """
        pressure_array: Array = jnp.asarray(pressure)
        condition: Array = pressure_array > self.P0

        def pressure_above_P0() -> Array:
            return pressure_array - self.P0

        def pressure_not_above_p0() -> Array:
            return jnp.zeros_like(pressure_array)

        delta_pressure: Array = jnp.where(condition, pressure_above_P0(), pressure_not_above_p0())

        return delta_pressure

    @jit
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, critical_data: CriticalData
    ) -> Array:
        r"""Volume contribution

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            critical_data: Critical data

        Returns:
            Volume contribution in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        delta_pressure: Array = self._delta_pressure(pressure)
        volume: Array = (
            self._a(temperature, critical_data) * delta_pressure
            + self._b(temperature, critical_data) * jnp.sqrt(delta_pressure)
            + self._c(temperature, critical_data) * jnp.power(delta_pressure, 0.25)
        )

        return volume

    @jit
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, critical_data: CriticalData
    ) -> Array:
        r"""Volume integral :math:`V dP` contribution

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            critical_data: Critical data

        Returns:
            Volume integral contribution in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        delta_pressure: Array = self._delta_pressure(pressure)
        volume_integral: Array = (
            self._a(temperature, critical_data) / 2.0 * jnp.square(delta_pressure)
            + 2.0
            / 3.0
            * self._b(temperature, critical_data)
            * jnp.power(delta_pressure, (3.0 / 2.0))
            + 4.0
            / 5.0
            * self._c(temperature, critical_data)
            * jnp.power(delta_pressure, (5.0 / 4.0))
        )

        return volume_integral

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "a_coefficients": self.a_coefficients,
            "b_coefficients": self.b_coefficients,
            "c_coefficients": self.c_coefficients,
            "P0": self.P0,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class CORK(RealGas):
    """A Compensated-Redlich-Kwong (CORK) EOS :cite:p:`HP91`

    Args:
        mrk: MRK model
        virial: Virial compensation term
        critical_data: Critical data
    """

    def __init__(
        self, mrk: RealGasProtocol, virial: VirialCompensation, critical_data: CriticalData
    ):
        self._mrk: RealGasProtocol = mrk
        self._virial: VirialCompensation = virial
        self._critical_data: CriticalData = critical_data

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity :cite:p:`HP91{Equation 8}`

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
        r"""Volume :cite:p:`HP91{Equation 7a}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        volume: Array = self._mrk.volume(temperature, pressure) + self._virial.volume(
            temperature, pressure, self._critical_data
        )

        return volume

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        volume_integral: Array = self._mrk.volume_integral(
            temperature, pressure
        ) + self._virial.volume_integral(temperature, pressure, self._critical_data)

        return volume_integral

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {"mrk": self._mrk, "virial": self._virial, "critical_data": self._critical_data}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)
