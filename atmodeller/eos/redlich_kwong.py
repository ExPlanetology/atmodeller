# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Redlich-Kwong EOS models from :cite:t:`RK49` and associated variants and extensions

Units for temperature and pressure are K and bar, respectively.
"""

import logging
from abc import abstractmethod
from typing import cast

import jax.numpy as jnp
import optimistix as optx
from jaxtyping import ArrayLike

from atmodeller import override
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import FloatArray, OptxSolver, as_j64
from atmodeller.sci_utils import GAS_CONSTANT_BAR
from atmodeller.thermodata import CriticalData, critical_data_dictionary

logger: logging.Logger = logging.getLogger(__name__)


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
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            `a` parameter
            (:math:`(\mathrm{m}^3\ \mathrm{mol}^{-1})^2\ \mathrm{K}^{1/2}\ \mathrm{bar}`)
        """

    @abstractmethod
    def b(self) -> ArrayLike:
        r"""Gets the `b` parameter

        Returns:
            `b` parameter (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """

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
        del mole_fractions

        a: ArrayLike = self.a(temperature, pressure)

        volume_integral: FloatArray = (
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
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume-explicit equation :cite:p:`HP91{Equation 7}`

        Without complications of critical phenomena the RK equation can be simplified using the
        approximation:

        .. math::

            V \sim \frac{RT}{P} + b

        where :math:`V` is volume, :math:`R` is the gas constant, :math:`T` is temperature,
        :math:`P` is pressure, and :math:`b` corrects for the volume.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        del mole_fractions

        a: ArrayLike = self.a(temperature, pressure)

        volume: FloatArray = (
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
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Initial volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        ...

    def A_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """`A` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            `A` factor, which is dimensionless
        """
        A_factor: ArrayLike = self.a(temperature, pressure) / (
            self.b() * GAS_CONSTANT_BAR * jnp.power(temperature, 1.5)
        )

        return A_factor

    def B_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """`B` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            `B` factor, which is dimensionless
        """
        B_factor: ArrayLike = self.b() * pressure / (GAS_CONSTANT_BAR * temperature)

        return B_factor

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral :cite:p:`HP91{Equation A.2}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume integral (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        log_fugacity: FloatArray = self.log_fugacity(temperature, pressure, mole_fractions)
        volume_integral: FloatArray = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Log fugacity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log fugacity (bar)
        """
        z: FloatArray = as_j64(self.compressibility_factor(temperature, pressure, mole_fractions))
        A: ArrayLike = self.A_factor(temperature, pressure)
        B: ArrayLike = self.B_factor(temperature, pressure)

        log_fugacity_coefficient: FloatArray = -jnp.log(z - B) - A * jnp.log(1 + B / z) + z - 1
        log_fugacity: FloatArray = jnp.log(pressure) + log_fugacity_coefficient

        return log_fugacity

    def _objective_function(
        self, volume: ArrayLike, kwargs: dict[str, ArrayLike | None]
    ) -> FloatArray:
        r"""Objective function to solve for the volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = cast(ArrayLike, kwargs["temperature"])
        pressure: ArrayLike = cast(ArrayLike, kwargs["pressure"])

        a: ArrayLike = self.a(temperature, pressure)

        # Coefficients for the polynomial in terms of volume. Unity coefficients are to satisfy
        # type checking.
        rtp: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        coeff2: ArrayLike = -1.0 * rtp
        coeff1: ArrayLike = a / (jnp.sqrt(temperature) * pressure) - 1.0 * self.b() * (
            rtp + self.b()
        )
        coeff0: ArrayLike = -1.0 * a * self.b() / (jnp.sqrt(temperature) * pressure)

        residual: FloatArray = (
            jnp.power(volume, 3) + coeff2 * jnp.square(volume) + coeff1 * volume + coeff0
        )

        return residual

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> ArrayLike:
        r"""Solves the RK equation numerically to compute the volume.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        del mole_fractions

        initial_volume: ArrayLike = self.initial_volume(temperature, pressure)
        kwargs: dict[str, ArrayLike | None] = {"temperature": temperature, "pressure": pressure}

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(
            self._objective_function, solver, initial_volume, args=kwargs, throw=THROW
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
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Initial volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
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
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Initial volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        initial_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure + 10 * self.b()

        return initial_volume


class RedlichKwong49(RedlichKwongABC):
    """Redlich-Kwong 1949 model

    Repulsive pressure term from van der Waals :cite:p:`RK49,C16{Equation 1}`
    Attractive pressure term from Redlich-Kwong :cite:p:`RK49,C16{Equation 4}`

    Args:
        critical_data: Critical data
    """

    critical_data: CriticalData

    def __init__(self, critical_data: CriticalData) -> None:
        self.critical_data = critical_data

    @property
    def critical_pressure(self) -> float:
        """Critical pressure (bar)"""
        return self.critical_data.pressure

    @property
    def critical_temperature(self) -> float:
        """Critical temperature (K)"""
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
