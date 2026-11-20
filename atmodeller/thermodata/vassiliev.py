# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Heat capacity models for diamond, graphite and silicon from :cite:t:`Vassiliev2021`"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import bernoulli, factorial

from atmodeller.jax_utils import FloatArray
from atmodeller.sci_utils import GAS_CONSTANT

_DEBYE_Y_SWITCH: float = 2.0
r"""Value of :math:`y = \Theta/T` at which :func:`_debye_integral` switches from the small-y to the
large-y series. At :math:`y = 2` both series converge to double precision with the number of terms
below."""

# Small-y series. The generating function of the Bernoulli numbers B_n (with B_1 = -1/2) is
#     x / (e^x - 1) = sum_{n>=0} B_n x^n / n!,     |x| < 2*pi
# so x^3 / (e^x - 1) = sum_n B_n x^(n+2) / n!, and integrating term by term from 0 to y gives
#     I(y) = sum_n B_n y^(n+3) / (n! (n+3)).
# Odd B_n vanish for n > 1 and the terms decay like (y / 2 pi)^n, so at y <= 2 terms up to n = 30
# (seventeen non-zero) reach double precision.
# The coefficients are precomputed with scipy rather than jax.scipy.special.bernoulli, which (as of
# JAX 0.11) evaluates B_n via zeta(n) truncated at 49 terms and so is only accurate to ~1e-6 for
# B_4 and ~1e-9 for B_6, limiting I(y) to a relative accuracy of ~1e-7.
_DEBYE_N: np.ndarray = np.arange(31)
_DEBYE_BERNOULLI_COEFFICIENTS: Array = jnp.asarray(
    bernoulli(_DEBYE_N[-1]) / (factorial(_DEBYE_N) * (_DEBYE_N + 3))
)

# Large-y series. Expanding 1 / (e^x - 1) = sum_{k>=1} e^(-kx) and integrating each term by parts
# three times gives
#     int_0^y x^3 e^(-kx) dx = 6/k^4 - e^(-ky) (y^3/k + 3y^2/k^2 + 6y/k^3 + 6/k^4).
# Summing over k with sum_k 6/k^4 = 6 zeta(4) = pi^4/15 yields
#     I(y) = pi^4/15 - sum_{k>=1} e^(-ky) (y^3/k + 3y^2/k^2 + 6y/k^3 + 6/k^4),
# which is the polylogarithm form pi^4/15 + y^3 ln(1-e^-y) - 3y^2 Li_2(e^-y) - 6y Li_3(e^-y)
# - 6 Li_4(e^-y). Terms decay like e^(-ky), so at y >= 2 twenty-four terms reach double precision.
_DEBYE_K: Array = jnp.arange(1, 25, dtype=float)


def _debye_integral(y: ArrayLike) -> Array:
    r"""Evaluates :math:`I(y) = \int_0^y x^3 / (e^x - 1) \, dx` for :math:`y > 0`.

    Uses the Bernoulli series for small y and the exponential (polylogarithm) series for large y,
    so no numerical quadrature is required. Both branches are evaluated with their input clamped to
    their own domain so that the branch discarded by the final ``jnp.where`` stays finite and does
    not corrupt ``jax.grad``.

    Args:
        y: Upper limit of the integral, Theta / T

    Returns:
        The integral, which tends to :math:`\pi^4/15` as y tends to infinity
    """
    y = jnp.asarray(y, dtype=float)

    y_small: Array = jnp.minimum(y, _DEBYE_Y_SWITCH)[..., None]
    small: Array = jnp.sum(_DEBYE_BERNOULLI_COEFFICIENTS * y_small ** (_DEBYE_N + 3), axis=-1)

    y_large: Array = jnp.maximum(y, _DEBYE_Y_SWITCH)[..., None]
    k: Array = _DEBYE_K
    large: Array = jnp.pi**4 / 15 - jnp.sum(
        jnp.exp(-k * y_large)
        * (y_large**3 / k + 3 * y_large**2 / k**2 + 6 * y_large / k**3 + 6 / k**4),
        axis=-1,
    )

    return jnp.where(y < _DEBYE_Y_SWITCH, small, large)


TEMPERATURE_MELTING_CARBON: float = 4600.0
"""Upper bound (K) of the carbon fits, :math:`T_\\mathrm{m}` of :cite:t:`Vassiliev2021{Table 9}`:
the graphite-liquid-vapour triple point (4600 ± 300 K), which the paper uses for both graphite
and diamond. The fits are constrained by data only up to 2000 K at most (Table 1), so above that
the heat capacity follows the extrapolated Maier-Kelley form."""

TEMPERATURE_MELTING_SILICON: float = 1688.0
"""Upper bound (K) of the silicon fits, the melting point :math:`T_\\mathrm{m}` of
:cite:t:`Vassiliev2021{Table 9}` (1688 ± 3 K)"""


class VassilievHeatCapacity(eqx.Module):
    """Heat capacity model for diamond, graphite and silicon from :cite:t:`Vassiliev2021`.

    The fits of :cite:t:`Vassiliev2021{Table 6}` are given for temperatures from 0.1 K to
    :math:`T_\\mathrm{m}`, with :math:`T_\\mathrm{m}` from their Table 9, so that is the temperature
    range of each fit here (:meth:`temperature_range`). Each fit's docstring notes where its data end. The heat
    capacity is per mole of atoms, which for these elements is per mole.

    Args:
        n: Number of experimental (T, Cp) pairs used in the fit (metadata only)
        T0: Temperature (K) in the Cp/Cv factor :cite:p:`Vassiliev2021{Eq. 3}`, controlling the
            transition from :math:`C_p = C_v` at low temperature to the Maier-Kelley form at high
            temperature
        A: Weights of the three Debye functions in Cv :cite:p:`Vassiliev2021{Eq. 1}`, summing to 1
        Theta: Debye temperatures (K) of the three Debye functions
        a: Constant Maier-Kelley coefficient in J/(mol K) :cite:p:`Vassiliev2021{Eq. 4}`
        b: Linear Maier-Kelley coefficient in J/(mol K), multiplying T/1000
        sigma: Standard deviation of the fit in J/(mol K) (metadata only)
        temperature_max: Upper bound (K) of the temperature range of the fit
        temperature_min: Lower bound (K) of the temperature range of the fit. Defaults to 0.1 K,
            as in :cite:t:`Vassiliev2021{Table 6}`.
    """

    n: int
    T0: float
    A: tuple[float, float, float]
    Theta: tuple[float, float, float]
    a: float
    b: float
    sigma: float
    temperature_max: float
    temperature_min: float = 0.1

    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range of the fit.

        Returns:
            Minimum and maximum temperature (K)
        """
        return self.temperature_min, self.temperature_max

    def _cp_over_R(self, temperature: ArrayLike) -> FloatArray:
        """Heat capacity relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`

        Args:
            temperature: Temperature (K)

        Returns:
            Heat capacity relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`
        """
        return self.cp(temperature) / GAS_CONSTANT

    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity at constant pressure.

        Args:
            temperature: Temperature (K)

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        f = (
            self.a
            + self.b * temperature / 1000
            + (3 * GAS_CONSTANT - self.a) / (1 + jnp.square(temperature / self.T0))
        ) / (3 * GAS_CONSTANT)

        return jnp.asarray(f) * self.cv(temperature)

    def cv(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity at constant volume.

        Args:
            temperature: Temperature (K)

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return (
            3
            * GAS_CONSTANT
            * (
                self.A[0] * self._debyes_function(temperature, self.Theta[0])
                + self.A[1] * self._debyes_function(temperature, self.Theta[1])
                + self.A[2] * self._debyes_function(temperature, self.Theta[2])
            )
        )

    def _debyes_function(self, temperature: ArrayLike, theta: float) -> FloatArray:
        r"""Debye's function :cite:p:`Vassiliev2021{Eq. 2}`

        .. math::

            D(T/\Theta) = 12 (T/\Theta)^3 \int_0^{\Theta/T} \frac{x^3}{e^x - 1} dx
                - \frac{3 \Theta/T}{e^{\Theta/T} - 1}

        which tends to 1 at high temperature and to zero as :math:`T^3` at low temperature.

        Args:
            temperature: Temperature (K)
            theta: Characteristic (Debye) temperature (K)

        Returns:
            Debye's function, Cv / 3R for a single Debye term
        """
        y: Array = theta / jnp.asarray(temperature, dtype=float)
        # 3y / (e^y - 1) written as 3y e^-y / (1 - e^-y) so it cannot overflow (and give NaN
        # gradients) at low temperature, where y can reach ~10^4
        return 12 * _debye_integral(y) / y**3 - 3 * y * jnp.exp(-y) / -jnp.expm1(-y)


cp_diamond_1a: VassilievHeatCapacity = VassilievHeatCapacity(
    186,
    812.3,
    (0.454, 0.503, 0.043),
    (1886.0, 1879.6, 1501.6),
    24.59,
    0.287,
    0.06,
    TEMPERATURE_MELTING_CARBON,
)
"""Diamond 1a heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted to the diamond data of :cite:t:`Vassiliev2021{Table 1}` (12.8-2000 K, the
high-temperature part from Cv data), so it is constrained only up to 2000 K."""
cp_diamond_1b: VassilievHeatCapacity = VassilievHeatCapacity(
    32,
    1366.0,
    (0.031, 0.488, 0.482),
    (1833.6, 1968.7, 1824.5),
    24.59,
    0.287,
    0.02,
    TEMPERATURE_MELTING_CARBON,
)
"""Diamond 1b heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted to 32 smoothed low-temperature points (5-160 K) with the Maier-Kelley coefficients from
Cv data at 1200-1600 K, and presented as the diamond Cp from 0 to 2000 K
:cite:p:`Vassiliev2021{Figs. 8, 9}`; constrained only up to 2000 K."""
cp_diamond_1c: VassilievHeatCapacity = VassilievHeatCapacity(
    32,
    602.6,
    (0.730, 0.238, 0.031),
    (1891.0, 1881.1, 1844.9),
    24.943,
    0.0,
    0.02,
    TEMPERATURE_MELTING_CARBON,
)
"""Diamond 1c heat capacity :cite:t:`Vassiliev2021{Table 6}`.

With :math:`a = 3R` and :math:`b = 0`. :cite:t:`Vassiliev2021` call this curve the Cv of
diamond (p. 6, Fig. 9: Cp - Cv = 0.23 J/(mol K) at 2000 K), and state that with these
coefficients function (3) gives the heat capacity of diamond and graphite up to 300 K (p. 3).
Above 300 K it is therefore closer to Cv than to Cp; use 1a or 1b for Cp."""
cp_diamond_1d: VassilievHeatCapacity = VassilievHeatCapacity(
    27,
    242.0,
    (0.884, 0.040, 0.076),
    (1930.7, 2000.8, 1292.7),
    24.943,
    0.0,
    0.05,
    TEMPERATURE_MELTING_CARBON,
)
"""Diamond 1d heat capacity :cite:t:`Vassiliev2021{Table 6}`.

With :math:`a = 3R` and :math:`b = 0`, fitted to the data of :cite:t:`Vassiliev2021`'s
reference [46] (15-1100 K, Table 1), so it is constrained only up to 1100 K. It agrees with 1c
above 600 K (Fig. 10)."""
cp_graphite_2a: VassilievHeatCapacity = VassilievHeatCapacity(
    73,
    288.4,
    (0.770, 0.110, 0.120),
    (1955.9, 424.1, 945.1),
    24.25,
    0.848,
    0.07,
    TEMPERATURE_MELTING_CARBON,
)
"""Graphite 2a heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted to the data of :cite:t:`Vassiliev2021`'s reference [21] (13-301 K) with the Maier-Kelley
coefficients from data at 1600-2000 K (Table 5), and described from 0.1 to 2000 K (Section 5.3);
constrained only up to 2000 K."""
cp_graphite_2b: VassilievHeatCapacity = VassilievHeatCapacity(
    38,
    268.3,
    (0.779, 0.114, 0.107),
    (1937.7, 416.0, 930.3),
    24.25,
    0.848,
    0.09,
    TEMPERATURE_MELTING_CARBON,
)
"""Graphite 2b heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted to the data of :cite:t:`Vassiliev2021`'s reference [14] (5-350 K) with the Maier-Kelley
coefficients from data at 1600-2000 K (Table 5); constrained only up to 2000 K."""
cp_graphite_2c: VassilievHeatCapacity = VassilievHeatCapacity(
    111,
    282.6,
    (0.773, 0.114, 0.114),
    (1949.9, 426.4, 947.9),
    24.25,
    0.848,
    0.08,
    TEMPERATURE_MELTING_CARBON,
)
"""Graphite 2c heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted jointly to :cite:t:`Vassiliev2021`'s references [14] and [21] (5-350 K) with the
Maier-Kelley coefficients from data at 1600-2000 K (Table 5); constrained only up to 2000 K."""
cp_silicon_3a: VassilievHeatCapacity = VassilievHeatCapacity(
    201,
    535.6,
    (0.367, 0.358, 0.275),
    (345.3, 847.6, 866.0),
    23.55,
    3.394,
    0.07,
    TEMPERATURE_MELTING_SILICON,
)
"""Silicon 3a heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted to the silicon data of :cite:t:`Vassiliev2021{Table 2}` (2.2-1600 K). The paper
describes it over 5-2000 K (Section 5.4), but silicon melts at 1688 K, the bound used here."""
cp_silicon_3b: VassilievHeatCapacity = VassilievHeatCapacity(
    24,
    677.0,
    (0.381, 0.337, 0.281),
    (350.4, 852.2, 877.1),
    23.55,
    3.394,
    0.06,
    TEMPERATURE_MELTING_SILICON,
)
"""Silicon 3b heat capacity :cite:t:`Vassiliev2021{Table 6}`.

Fitted to the data of :cite:t:`Vassiliev2021`'s reference [24] only (5-300 K) and shown over
0.1-300 K (Fig. 17), with the Maier-Kelley coefficients of 3a; constrained by its own data only
up to 300 K."""
