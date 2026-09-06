# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions for real gas equations of state

Units for temperature and pressure are K and bar, respectively.
"""

import logging
from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jax import jacfwd
from jaxtyping import ArrayLike

from atmodeller import override
from atmodeller.constants import STANDARD_FUGACITY, STANDARD_PRESSURE
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW, VOLUME_EPSILON
from atmodeller.jax_utils import FloatArray, OptxSolver, as_j64, safe_exp, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT_BAR
from atmodeller.thermodata import CriticalData

logger: logging.Logger = logging.getLogger(__name__)


class RealGasBase(eqx.Module):
    """A real gas equation of state (EOS) without volume calculations

    The methods should support broadcasting of temperature and pressure, but not necessarily of
    mole fractions.
    """

    @abstractmethod
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Log fugacity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log fugacity (bar)
        """

    def fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Fugacity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Fugacity (bar)
        """
        return safe_exp(self.log_fugacity(temperature, pressure, mole_fractions))

    def log_fugacity_coefficient(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Log fugacity coefficient

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log fugacity coefficient (dimensionless)
        """
        return self.log_fugacity(temperature, pressure, mole_fractions) - jnp.log(pressure)

    def fugacity_coefficient(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Fugacity coefficient

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Fugacity coefficient (dimensionless)
        """
        return safe_exp(self.log_fugacity_coefficient(temperature, pressure, mole_fractions))

    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Log activity

        This is the primary access point for calling the EOS within the main engine.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log activity (dimensionless)
        """
        return self.log_fugacity(temperature, pressure, mole_fractions) - jnp.log(
            STANDARD_FUGACITY
        )


class RealGas(RealGasBase):
    r"""A real gas equation of state (EOS) with volume calculations

    Fugacity is computed using the standard relation:

    .. math::
        R T \ln f = \int V dP

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`f` is fugacity, :math:`V`
    is volume, and :math:`P` is pressure.
    """

    @abstractmethod
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        r"""Volume

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """

    @abstractmethod
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral in units required for internal Atmodeller operations.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume integral (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Log fugacity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log fugacity (bar)
        """
        log_fugacity: FloatArray = self.volume_integral(temperature, pressure, mole_fractions) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    def volume_integral_J(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral in J

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume integral (:math:`\mathrm{J}\ \mathrm{mol}^{-1}`)
        """
        return 1e5 * self.volume_integral(temperature, pressure, mole_fractions)

    def dzdp(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Derivative of the compressibility factor with respect to pressure

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Derivative of the compressibility factor with respect to pressure
        """
        temperature = as_j64(temperature)
        pressure = as_j64(pressure)
        # TODO: use Equinox equivalent instead of native JAX?
        dzdp_fn: Callable = jacfwd(self.compressibility_factor, argnums=1)

        return dzdp_fn(temperature, pressure, mole_fractions)

    def dvdp(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Derivative of volume with respect to pressure

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Derivative of volume with respect to pressure
        """
        temperature = as_j64(temperature)
        pressure = as_j64(pressure)
        # TODO: use Equinox equivalent instead of native JAX?
        dvdp_fn: Callable = jacfwd(self.volume, argnums=1)

        return dvdp_fn(temperature, pressure, mole_fractions)

    def compressibility_factor(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        """Compressibility factor

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Compressibility factor (dimensionless)
        """
        volume: ArrayLike = self.volume(temperature, pressure, mole_fractions)
        volume_ideal: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        compressibility_factor: ArrayLike = volume / volume_ideal

        return compressibility_factor


class IdealGas(RealGas):
    r"""Ideal gas equation of state:

    .. math::

        R T = P V

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`P` is pressure, and
    :math:`V` is volume.
    """

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        del mole_fractions

        return GAS_CONSTANT_BAR * temperature / pressure

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions

        return jnp.log(pressure) * GAS_CONSTANT_BAR * temperature


class VirialCompensation(eqx.Module):
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
    """

    a_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`"""
    b_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """As above for the b coefficients"""
    c_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """As above for the c coefficients"""
    P0: float = eqx.field(converter=float)
    """Pressure at which the MRK equation begins to overestimate the molar volume significantly"""

    def _a(self, temperature: ArrayLike, critical_data: CriticalData) -> FloatArray:
        r"""`a` parameter :cite:p:`HP98`

        This is also the `d` parameter in :cite:t:`HP91`.

        Args:
            temperature: Temperature (K)
            critical_data: Critical data

        Returns:
            `a` parameter in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}\ \mathrm{bar}^{-1}`
        """
        a: FloatArray = (
            self.a_coefficients[1] * as_j64(temperature)
            + self.a_coefficients[0] * critical_data.temperature
        )
        a = a / jnp.square(critical_data.pressure)

        return a

    def _b(self, temperature: ArrayLike, critical_data: CriticalData) -> FloatArray:
        r"""`b` parameter :cite:p:`HP98`

        This is also the `c` parameter in :cite:t:`HP91`.

        Args:
            temperature: Temperature (K)
            critical_data: Critical data

        Returns:
            `b` parameter (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}\ \mathrm{bar}^{-1/2}`)
        """
        b: FloatArray = (
            self.b_coefficients[1] * as_j64(temperature)
            + self.b_coefficients[0] * critical_data.temperature
        )
        b = b / jnp.power(critical_data.pressure, (3.0 / 2))

        return b

    def _c(self, temperature: ArrayLike, critical_data: CriticalData) -> FloatArray:
        r"""`c` parameter :cite:p:`HP98`

        Args:
            temperature: Temperature (K)
            critical_data: Critical data

        Returns:
            `c` parameter (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}\ \mathrm{bar}^{-1/4}`)
        """
        c: FloatArray = (
            self.c_coefficients[1] * as_j64(temperature)
            + self.c_coefficients[0] * critical_data.temperature
        )
        c = c / jnp.power(critical_data.pressure, (5.0 / 4))

        return c

    def _delta_pressure(self, pressure: ArrayLike) -> FloatArray:
        """Pressure difference

        Args:
            pressure: Pressure (bar)

        Returns:
            Pressure difference relative to :attr:`P0` (bar)
        """
        pressure_array: FloatArray = as_j64(pressure)
        condition: FloatArray = pressure_array > self.P0

        def pressure_above_P0() -> FloatArray:
            return pressure_array - self.P0

        def pressure_not_above_p0() -> FloatArray:
            return jnp.zeros_like(pressure_array)

        delta_pressure: FloatArray = jnp.where(
            condition, pressure_above_P0(), pressure_not_above_p0()
        )

        return delta_pressure

    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, critical_data: CriticalData
    ) -> FloatArray:
        r"""Volume contribution

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            critical_data: Critical data

        Returns:
            Volume contribution (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        delta_pressure: FloatArray = self._delta_pressure(pressure)
        volume: FloatArray = (
            self._a(temperature, critical_data) * delta_pressure
            + self._b(temperature, critical_data) * jnp.sqrt(delta_pressure)
            + self._c(temperature, critical_data) * jnp.power(delta_pressure, 0.25)
        )

        return volume

    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, critical_data: CriticalData
    ) -> FloatArray:
        r"""Volume integral :math:`V dP` contribution

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            critical_data: Critical data

        Returns:
            Volume integral contribution (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        delta_pressure: FloatArray = self._delta_pressure(pressure)
        volume_integral: FloatArray = (
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


class VanderWaals(RealGas):
    r"""Van der Waals EOS

    Args:
        a: a constant (:math:`\mathrm{m}^6 \mathrm{bar} \mathrm{mol}^{-2}`)
        b: b constant (:math:`\mathrm{m}^3 \mathrm{mol}^{-1}`)
    """

    a: float = eqx.field(converter=float)
    r"""a constant (:math:`\mathrm{m}^6 \mathrm{bar} \mathrm{mol}^{-2}`)"""
    b: float = eqx.field(converter=float)
    r"""b constant (:math:`\mathrm{m}^3 \mathrm{mol}^{-1}`)"""

    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> FloatArray:
        r"""Objective function to solve for the volume

        Args:
            volume: Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

        coeff0: ArrayLike = -self.a * self.b / pressure
        coeff1: ArrayLike = self.a / pressure
        coeff2: ArrayLike = -self.b - GAS_CONSTANT_BAR * temperature / pressure
        coeff3: ArrayLike = 1

        residual: FloatArray = (
            coeff3 * jnp.power(volume, 3)
            + coeff2 * jnp.power(volume, 2)
            + coeff1 * volume
            + coeff0
        )

        return residual

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3 \mathrm{mol}^{-1}`)
        """
        del mole_fractions

        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        # If the ideal volume is around the b constant value then the denominator becomes zero, so
        # shift the volume and add a small epsilon to avoid this.
        safe_volume: ArrayLike = ideal_volume + self.b + VOLUME_EPSILON
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(
            self._objective_function, solver, safe_volume, args=kwargs, throw=THROW
        )
        volume = sol.value
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume integral (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        vol: ArrayLike = self.volume(temperature, pressure, mole_fractions)
        vol0: ArrayLike = self.volume(temperature, STANDARD_PRESSURE, mole_fractions)
        volume_integral: FloatArray = (
            self.b * (vol0 - vol) / ((vol - self.b) * (vol0 - self.b))  # type: ignore
            - jnp.log((vol - self.b) / (vol0 - self.b))
        ) * GAS_CONSTANT_BAR * temperature - 2 * self.a * (1 / vol - 1 / vol0)

        return volume_integral


class CORK(RealGas):
    """A Compensated-Redlich-Kwong (CORK) EOS :cite:p:`HP91`

    Args:
        mrk: MRK model
        virial: Virial compensation term
        critical_data: Critical data
    """

    mrk: RealGas
    """MRK model"""
    virial: VirialCompensation
    """Virial compensation term"""
    critical_data: CriticalData
    """Critical data"""

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        r"""Volume :cite:p:`HP91{Equation 7a}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        volume: ArrayLike = self.mrk.volume(
            temperature, pressure, mole_fractions
        ) + self.virial.volume(temperature, pressure, self.critical_data)

        return volume

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume integral (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        volume_integral: FloatArray = self.mrk.volume_integral(
            temperature, pressure, mole_fractions
        ) + self.virial.volume_integral(temperature, pressure, self.critical_data)

        return volume_integral


def safe_ideal_initial_volume(temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
    r"""Initial guess volume is the ideal gas volume plus a small epsilon

    Args:
        temperature: Temperature (K)
        pressure: Pressure (bar)

    Returns:
        Initial volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
    """
    ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
    safe_volume: FloatArray = as_j64(ideal_volume + VOLUME_EPSILON)
    # jax.debug.print("initial_volume = {out}", out=safe_volume)

    return safe_volume
