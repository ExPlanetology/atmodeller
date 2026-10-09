# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graphite and diamond from the assessment of :cite:t:`Gustafson1986`

The Gibbs energies of graphite and diamond in the SGTE data :cite:p:`Dinsdale1991` are from the
assessment of :cite:t:`Gustafson1986`, which fits them to calorimetric data and to the phase
diagram of carbon. They are valid from 298.15 to 6000 K :cite:p:`Dinsdale1991`, describe the data
between 298 and 5000 K and between 0 and 15 GPa, and extrapolate reasonably to 6000 K and 40 GPa
:cite:p:`Gustafson1986`.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.constants import STANDARD_PRESSURE, TEMPERATURE_REFERENCE
from atmodeller.jax_utils import FloatArray, as_j64, to_native_floats
from atmodeller.thermodata.core import (
    GibbsThermodynamicProperties,
    RelativeThermodynamicProperties,
)
from atmodeller.thermodata.janaf import glenn_properties


class SGTEThermodynamicProperties(GibbsThermodynamicProperties):
    r"""Thermodynamic properties from an SGTE Gibbs energy function :cite:p:`Dinsdale1991`

    The SGTE data give the Gibbs energy at zero pressure relative to the enthalpy of the reference
    phase at 298.15 K (:math:`G - H^{SER}`) as

    .. math::

        G = a + bT + cT\ln T + dT^2 + e/T + f/T^2 + g/T^3

    from which the entropy, enthalpy and heat capacity follow by differentiation
    (:class:`~atmodeller.thermodata.core.GibbsThermodynamicProperties`). Analytically
    :cite:p:`Dinsdale1991`,

    .. math::

        S &= -b - c - c\ln T - 2dT + e/T^2 + 2f/T^3 + 3g/T^4 \\
        H &= a - cT - dT^2 + 2e/T + 3f/T^2 + 4g/T^3 \\
        C_p &= -c - 2dT - 2e/T^2 - 6f/T^3 - 12g/T^4

    The SGTE data add the pressure dependence as a separate term, the integral of the volume from
    zero pressure (:class:`SGTEMurnaghanEOS`). Here the Gibbs energy above is taken as the standard
    state at 1 bar and the volume is integrated from 1 bar, which neglects the integral from zero
    to 1 bar. This is about :math:`V \times 1\ \mathrm{bar}`, 0.53 J/mol for graphite and
    0.34 J/mol for diamond, and is negligible at low pressures :cite:p:`Dinsdale1991`.

    Args:
        coefficients: Coefficients :math:`(a, b, c, d, e, f, g)` of the Gibbs energy in J/mol,
            with :math:`T` in K
        volume_model: Volume model. Defaults to ``None``, which ignores the pressure dependence.
    """

    coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Coefficients (a, b, c, d, e, f, g) of the Gibbs energy"""

    def __check_init__(self) -> None:
        if len(self.coefficients) != 7:
            raise ValueError(
                f"Expected 7 coefficients (a, b, c, d, e, f, g), got {len(self.coefficients)}"
            )

    @override
    def gibbs_energy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets the Gibbs energy at 1 bar.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)
        a, b, c, d, e, f, g = self.coefficients

        return (
            a
            + b * temperature
            + c * temperature * jnp.log(temperature)
            + d * temperature**2
            + e / temperature
            + f / temperature**2
            + g / temperature**3
        )

    @override
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the thermodynamic data are valid.

        Returns:
            Minimum and maximum temperature in K
        """
        return (TEMPERATURE_REFERENCE, 6000.0)


class SGTEMurnaghanEOS(eqx.Module):
    r"""Volume of a solid with the Murnaghan equation of state of the SGTE data
    :cite:p:`Gustafson1986{Eqs. 4-8},Dinsdale1991`

    The volume at zero pressure follows from a thermal expansion linear in temperature,
    :math:`\alpha = \alpha_0 + \alpha_1 T`, integrated from 0 K, and its pressure dependence from
    the Murnaghan equation of state with a compressibility :math:`K` independent of temperature

    .. math::

        V = \frac{A\exp(\alpha_0 T + \alpha_1 T^2/2)}{(1 + nKP)^{1/n}}

    :cite:t:`Gustafson1986` integrate the volume from zero pressure. Here it is integrated from
    1 bar, consistent with the 1-bar standard state of :class:`SGTEThermodynamicProperties`, which
    neglects the integral from zero to 1 bar of about :math:`V \times 1\ \mathrm{bar}`.

    Args:
        A: Volume at 0 K and zero pressure in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        alpha0: Constant term of the thermal expansion in :math:`\mathrm{K}^{-1}`
        alpha1: Linear term of the thermal expansion in :math:`\mathrm{K}^{-2}`
        K: Compressibility in :math:`\mathrm{bar}^{-1}`
        n: Pressure derivative of the bulk modulus
    """

    A: float = eqx.field(converter=float)
    alpha0: float = eqx.field(converter=float)
    alpha1: float = eqx.field(converter=float)
    K: float = eqx.field(converter=float)
    n: float = eqx.field(converter=float)

    def volume_zero_pressure(self, temperature: ArrayLike) -> Array:
        r"""Gets the volume at zero pressure.

        Args:
            temperature: Temperature in K

        Returns:
            Volume at zero pressure in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        """
        temperature = as_j64(temperature)

        return self.A * jnp.exp(self.alpha0 * temperature + self.alpha1 * temperature**2 / 2)

    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Gets the integral of volume with respect to pressure from 1 bar.

        .. math::

            \int_1^P V\, dP = \frac{V_{0,T}}{K(n - 1)}
                \left[(1 + nKP)^{1 - 1/n} - (1 + nK)^{1 - 1/n}\right]

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Integral of volume with respect to pressure in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        exponent: float = 1.0 - 1.0 / self.n

        return (
            self.volume_zero_pressure(temperature)
            / (self.K * (self.n - 1))
            * (
                (1 + self.n * self.K * pressure) ** exponent
                - (1 + self.n * self.K * STANDARD_PRESSURE) ** exponent
            )
        )


GRAPHITE_VOLUME_GUSTAFSON: SGTEMurnaghanEOS = SGTEMurnaghanEOS(
    0.5259, 2.32e-5, 5.7e-9, 3.0e-6, 12.0
)
"""Volume of graphite :cite:p:`Gustafson1986{Table 2},Dinsdale1991`, with :math:`A` and
:math:`K` converted to bar"""
DIAMOND_VOLUME_GUSTAFSON: SGTEMurnaghanEOS = SGTEMurnaghanEOS(0.3412, 2.43e-6, 1.0e-8, 1.7e-7, 5.0)
"""Volume of diamond :cite:p:`Gustafson1986{Table 2},Dinsdale1991`, with :math:`A` and
:math:`K` converted to bar"""

graphite_sgte: SGTEThermodynamicProperties = SGTEThermodynamicProperties(
    (-17368.441, 170.73, -24.3, -4.723e-4, 2562600.0, -2.643e8, 1.2e10),
    volume_model=GRAPHITE_VOLUME_GUSTAFSON,
)
"""Graphite (GHSERCC) from the SGTE data, valid from 298.15 to 6000 K, with the volume
:data:`GRAPHITE_VOLUME_GUSTAFSON` :cite:p:`Gustafson1986,Dinsdale1991{p. 337}`

Graphite is the reference phase of carbon, so its enthalpy is zero at 298.15 K. Its entropy is
5.7423 J/K/mol. The default database instead uses graphite from the NASA Glenn coefficients, with
:data:`diamond_gustafson`."""
diamond_sgte: SGTEThermodynamicProperties = SGTEThermodynamicProperties(
    (-16359.441, 175.61, -24.31, -4.723e-4, 2698000.0, -2.61e8, 1.11e10),
    volume_model=DIAMOND_VOLUME_GUSTAFSON,
)
"""Diamond (GDIACC) from the SGTE data, valid from 298.15 to 6000 K, with the volume
:data:`DIAMOND_VOLUME_GUSTAFSON` :cite:p:`Gustafson1986,Dinsdale1991{p. 337}`

Its enthalpy and entropy at 298.15 K are 1895.79 J/mol and 2.3598 J/K/mol. Use it with
:data:`graphite_sgte`."""

diamond_gustafson: RelativeThermodynamicProperties = RelativeThermodynamicProperties(
    glenn_properties["C_s"], diamond_sgte, graphite_sgte,
    volume_model=DIAMOND_VOLUME_GUSTAFSON,
)
"""Diamond relative to graphite from the assessment of :cite:t:`Gustafson1986`, valid to 6000 K,
with the volume :data:`DIAMOND_VOLUME_GUSTAFSON`.

The Gibbs energy of diamond is that of graphite from the NASA Glenn coefficients (``C_s`` in
:data:`~atmodeller.thermodata.janaf.glenn_properties`) plus the Gibbs energy of diamond relative to
graphite of :cite:t:`Gustafson1986` (:data:`diamond_sgte` minus :data:`graphite_sgte`). This keeps
graphite consistent with the other NASA Glenn species while reproducing the Gibbs energy of diamond
relative to graphite of :cite:t:`Gustafson1986` exactly. At 298.15 K, diamond minus graphite is
1895.79 J/mol in enthalpy and -3.3825 J/K/mol in entropy.

For equilibrium between diamond and graphite at high pressure, use graphite with
:data:`GRAPHITE_VOLUME_GUSTAFSON`, as in :func:`~atmodeller.database.get_default_database`. This
reproduces the graphite-diamond transition of :cite:t:`Gustafson1986`."""
