# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Volume models of solids from :cite:t:`HP98` and :cite:t:`HP11`."""

from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from atmodeller.constants import STANDARD_PRESSURE, TEMPERATURE_REFERENCE
from atmodeller.jax_utils import as_j64


class MurnaghanEOS(eqx.Module):
    r"""Volume of a solid with the equation of state of :cite:t:`HP98`.

    The volume at 1 bar follows from the temperature-dependent thermal expansion
    :math:`\alpha_T = a^\circ(1 - 10/\sqrt{T})` and its pressure dependence from the Murnaghan
    equation of state, with a bulk modulus that decreases linearly with temperature.

    The reference temperature is :const:`~atmodeller.constants.TEMPERATURE_REFERENCE` (298.15 K),
    whereas :cite:t:`HP98` write their expressions with 298 K. The difference is negligible.

    Args:
        V0: Volume at the reference temperature and 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        alpha0: Thermal expansion parameter :math:`a^\circ` in :math:`\mathrm{K}^{-1}`
        K0: Bulk modulus at the reference temperature in bar
        dKdP: Pressure derivative of the bulk modulus. Defaults to 4.
    """

    V0: float
    alpha0: float
    K0: float
    dKdP: float = 4.0

    @property
    def dKdT(self) -> float:
        """Temperature derivative of the bulk modulus in bar/K"""
        return -1.0 * self.K0 * 1.5e-4

    def thermal_expansion(self, temperature: ArrayLike) -> Array:
        r"""Gets the thermal expansion.

        Args:
            temperature: Temperature in K

        Returns:
            Thermal expansion in :math:`\mathrm{K}^{-1}`
        """
        return self.alpha0 * (1 - 10 / jnp.sqrt(temperature))

    def bulk_modulus(self, temperature: ArrayLike) -> ArrayLike:
        """Gets the bulk modulus.

        Args:
            temperature: Temperature in K

        Returns:
            Bulk modulus in bar
        """
        return self.K0 + self.dKdT * (temperature - TEMPERATURE_REFERENCE)

    def volume_1bar(self, temperature: ArrayLike) -> Array:
        r"""Gets the volume at 1 bar.

        This integrates the thermal expansion exactly. See :meth:`volume_1bar_linear` for the
        linearised form used by :cite:t:`HP98`.

        Args:
            temperature: Temperature in K

        Returns:
            Volume at 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        """
        volume: Array = self.V0 * jnp.exp(
            self.alpha0 * (temperature - TEMPERATURE_REFERENCE)
            - 2 * 10.0 * self.alpha0 * (jnp.sqrt(temperature) - jnp.sqrt(TEMPERATURE_REFERENCE))
        )

        return volume

    def volume_1bar_linear(self, temperature: ArrayLike) -> Array:
        r"""Gets the volume at 1 bar using the linearised form of :cite:t:`HP98`.

        .. math::

            V_{1,T} = V_{1,T_r}\left[1 + a^\circ(T - T_r) - 20a^\circ(\sqrt{T} - \sqrt{T_r})\right]

        where :math:`T_r` is :const:`~atmodeller.constants.TEMPERATURE_REFERENCE`.
        This is the exact expression used by :cite:t:`HP98` to derive their data set. It is the
        first-order expansion of :meth:`volume_1bar`, which integrates the thermal expansion
        exactly; the two differ by about 0.3% for diamond at 6000 K.

        Args:
            temperature: Temperature in K

        Returns:
            Volume at 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        """
        return cast(
            Array,
            self.V0
            * (
                1
                + self.alpha0 * (temperature - TEMPERATURE_REFERENCE)
                - 2
                * 10.0
                * self.alpha0
                * (jnp.sqrt(temperature) - jnp.sqrt(TEMPERATURE_REFERENCE))
            ),
        )

    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Gets the integral of volume with respect to pressure from 1 bar.

        .. math::

            \int_1^P V\, dP = \frac{V_{1,T}\, K_T}{K' - 1}
                \left[\left(1 + \frac{K'(P - 1)}{K_T}\right)^{1 - 1/K'} - 1\right]

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Integral of volume with respect to pressure in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        bulk_modulus: ArrayLike = self.bulk_modulus(temperature)
        integral: Array = (
            self.volume_1bar(temperature)
            * bulk_modulus
            / (self.dKdP - 1)
            * (
                (1 + self.dKdP * (pressure - STANDARD_PRESSURE) / bulk_modulus)
                ** (1.0 - 1.0 / self.dKdP)
                - 1
            )
        )

        return integral


class TaitEOS(eqx.Module):
    r"""Volume of a solid with the modified Tait equation of state of :cite:t:`HP11`

    The pressure dependence follows the modified Tait equation of state and the temperature
    dependence a thermal pressure from an Einstein model, which together extrapolate to higher
    pressures and temperatures than :class:`MurnaghanEOS` :cite:p:`HP11{Eqs. 3, 11-13}`.

    Args:
        V0: Volume at the reference temperature and 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        alpha0: Thermal expansion at the reference temperature in :math:`\mathrm{K}^{-1}`
        K0: Bulk modulus at the reference temperature in bar
        dKdP: Pressure derivative of the bulk modulus
        einstein_temperature: Einstein temperature in K, which :cite:t:`HP11` estimate as
            :math:`10636 / (S/n + 6.44)` from the entropy :math:`S` and number of atoms :math:`n`
        d2KdP2: Second pressure derivative of the bulk modulus in :math:`\mathrm{bar}^{-1}`.
            Defaults to ``None``, which uses :math:`-K'/K_0` :cite:p:`HP11`.
    """

    V0: float
    alpha0: float
    K0: float
    dKdP: float
    einstein_temperature: float
    d2KdP2: float | None = None

    def _tait_parameters(self) -> tuple[float, float, float]:
        """Gets the parameters a, b and c of the modified Tait equation of state

        Returns:
            Parameters a, b and c :cite:p:`HP11{Eq. 3}`
        """
        d2KdP2: float = -self.dKdP / self.K0 if self.d2KdP2 is None else self.d2KdP2
        a: float = (1 + self.dKdP) / (1 + self.dKdP + self.K0 * d2KdP2)
        b: float = self.dKdP / self.K0 - d2KdP2 / (1 + self.dKdP)
        c: float = (1 + self.dKdP + self.K0 * d2KdP2) / (
            self.dKdP**2 + self.dKdP - self.K0 * d2KdP2
        )

        return a, b, c

    def thermal_pressure(self, temperature: ArrayLike) -> Array:
        """Gets the thermal pressure.

        Args:
            temperature: Temperature in K

        Returns:
            Thermal pressure in bar :cite:p:`HP11`
        """
        u: Array = self.einstein_temperature / as_j64(temperature)
        u0: float = self.einstein_temperature / TEMPERATURE_REFERENCE
        xi0: float = u0**2 * np.exp(u0) / np.expm1(u0) ** 2

        return (
            self.alpha0
            * self.K0
            * self.einstein_temperature
            / xi0
            * (1 / jnp.expm1(u) - 1 / np.expm1(u0))
        )

    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Gets the volume.

        Pressure is measured from 1 bar, so that the volume is :attr:`V0` at the reference
        temperature and 1 bar. :cite:t:`HP11` measure pressure from zero, which differs by a
        relative volume of :math:`1/K_0` (bar).

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{J}\ \mathrm{bar}^{-1}` :cite:p:`HP11{Eq. 12}`
        """
        a, b, c = self._tait_parameters()
        thermal_pressure: Array = self.thermal_pressure(temperature)
        excess_pressure: ArrayLike = pressure - STANDARD_PRESSURE

        return self.V0 * (
            1 - a * (1 - jnp.power(1 + b * (excess_pressure - thermal_pressure), -c))
        )

    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Gets the integral of volume with respect to pressure from 1 bar.

        Pressure is measured from 1 bar, consistent with :meth:`volume`. With
        :math:`P_{ex} = P - 1` bar, the integral is

        .. math::

            \int_1^P V\, dP' = V_0\left[(1 - a)P_{ex} + \frac{a}{b(c - 1)}\left(
                (1 - bP_{th})^{1-c} - \left(1 + b(P_{ex} - P_{th})\right)^{1-c}\right)\right]

        This looks different from :cite:t:`HP11{Eq. 13}`, which factors out :math:`P`, but is the
        same expression multiplied out. This form avoids dividing by :math:`P`, and the term in
        :math:`P_{th}` alone is the lower limit of the integral, which makes it zero at 1 bar.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Integral of volume with respect to pressure in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        a, b, c = self._tait_parameters()
        thermal_pressure: Array = self.thermal_pressure(temperature)
        excess_pressure: ArrayLike = pressure - STANDARD_PRESSURE

        return jnp.asarray(self.V0) * (
            (1 - a) * excess_pressure
            + a
            * (
                jnp.power(1 - b * thermal_pressure, 1 - c)
                - jnp.power(1 + b * (excess_pressure - thermal_pressure), 1 - c)
            )
            / (b * (c - 1))
        )


def einstein_temperature(entropy: float, number_atoms: int) -> float:
    """Gets the Einstein temperature estimated from the entropy :cite:p:`HP11`

    Args:
        entropy: Entropy at 298.15 K and 1 bar in J/K/mol
        number_atoms: Number of atoms in the formula unit

    Returns:
        Einstein temperature in K
    """
    return 10636 / (entropy / number_atoms + 6.44)


DIAMOND_VOLUME_MURNAGHAN: MurnaghanEOS = MurnaghanEOS(0.342, 1.65e-5, 5.8e6)
"""Volume of diamond from :cite:t:`HP98{Table 5}`, with the bulk modulus converted to bar"""
GRAPHITE_VOLUME_MURNAGHAN: MurnaghanEOS = MurnaghanEOS(0.530, 4.84e-5, 3.9e5)
"""Volume of graphite from :cite:t:`HP98{Table 5}`, with the bulk modulus converted to bar"""
DIAMOND_VOLUME_TAIT: TaitEOS = TaitEOS(
    0.342, 0.49e-5, 4.465e6, 1.61, einstein_temperature(2.38, 1)
)
"""Volume of diamond from :cite:t:`HP11{Table 2a}`, with the bulk modulus converted to bar"""
GRAPHITE_VOLUME_TAIT: TaitEOS = TaitEOS(
    0.530, 1.67e-5, 3.12e5, 3.90, einstein_temperature(5.74, 1)
)
"""Volume of graphite from :cite:t:`HP11{Table 2a}`, with the bulk modulus converted to bar"""
