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

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.constants import STANDARD_PRESSURE, TEMPERATURE_REFERENCE
from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.thermodata.core import (
    Enthalpy,
    Entropy,
    HeatCapacity,
    RelativeHeatCapacity,
    ThermodynamicProperties,
    Volume,
)
from atmodeller.thermodata.vassiliev import graphite


class SGTEHeatCapacity(HeatCapacity):
    r"""Heat capacity from an SGTE Gibbs energy function :cite:p:`Dinsdale1991`

    The Gibbs energy function is

    .. math::

        G = a + bT + cT\ln T + dT^2 + e/T + f/T^2 + g/T^3

    so the heat capacity, :math:`C_p = -T\,\partial^2 G / \partial T^2`, is

    .. math::

        C_p = -c - 2dT - 2e/T^2 - 6f/T^3 - 12g/T^4

    which depends only on the coefficients :math:`c` to :math:`g`.

    Args:
        c: Coefficient of :math:`T\ln T` in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        d: Coefficient of :math:`T^2` in :math:`\mathrm{J}\ \mathrm{K}^{-2} \mathrm{mol}^{-1}`
        e: Coefficient of :math:`1/T` in :math:`\mathrm{J}\ \mathrm{K}\ \mathrm{mol}^{-1}`
        f: Coefficient of :math:`1/T^2` in :math:`\mathrm{J}\ \mathrm{K}^2 \mathrm{mol}^{-1}`
        g: Coefficient of :math:`1/T^3` in :math:`\mathrm{J}\ \mathrm{K}^3 \mathrm{mol}^{-1}`
    """

    c: float
    d: float
    e: float
    f: float
    g: float

    @override
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity at constant pressure.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)

        return (
            -self.c
            - 2 * self.d * temperature
            - 2 * self.e / temperature**2
            - 6 * self.f / temperature**3
            - 12 * self.g / temperature**4
        )

    @override
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the heat capacity model is valid.

        Returns:
            Minimum and maximum temperature in K
        """
        return (TEMPERATURE_REFERENCE, 6000.0)


class SGTEEnthalpy(Enthalpy):
    r"""Enthalpy from an SGTE Gibbs energy function :cite:p:`Dinsdale1991`

    The SGTE data give the Gibbs energy at zero pressure relative to the enthalpy of the reference
    phase at 298.15 K (:math:`G - H^{SER}`) as

    .. math::

        G = a + bT + cT\ln T + dT^2 + e/T + f/T^2 + g/T^3

    so the enthalpy, :math:`H = G - T\,\partial G/\partial T`, is :cite:p:`Dinsdale1991`

    .. math::

        H = a - cT - dT^2 + 2e/T + 3f/T^2 + 4g/T^3

    Its temperature derivative is the heat capacity of :class:`SGTEHeatCapacity`, so :math:`a` is
    the constant of integration. The pressure dependence, which the SGTE data add as a separate
    term, is described by :class:`SGTEMurnaghanEOS`.

    Args:
        heat_capacity_model: SGTE heat capacity model, which holds :math:`c` to :math:`g`
        a: Constant term of the Gibbs energy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
    """

    heat_capacity_model: SGTEHeatCapacity
    """SGTE heat capacity model"""
    a: float
    """Constant term of the Gibbs energy in J/mol"""

    @override
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)
        cp: SGTEHeatCapacity = self.heat_capacity_model

        return (
            self.a
            - cp.c * temperature
            - cp.d * temperature**2
            + 2 * cp.e / temperature
            + 3 * cp.f / temperature**2
            + 4 * cp.g / temperature**3
        )


class SGTEEntropy(Entropy):
    r"""Entropy from an SGTE Gibbs energy function :cite:p:`Dinsdale1991`

    For the Gibbs energy of :class:`SGTEEnthalpy`, the entropy, :math:`S = -\partial G/\partial T`,
    is :cite:p:`Dinsdale1991`

    .. math::

        S = -b - c - c\ln T - 2dT + e/T^2 + 2f/T^3 + 3g/T^4

    Its temperature derivative is :math:`C_p/T` with the heat capacity of
    :class:`SGTEHeatCapacity`, so :math:`b` is the constant of integration.

    Args:
        heat_capacity_model: SGTE heat capacity model, which holds :math:`c` to :math:`g`
        b: Coefficient of :math:`T` in the Gibbs energy in
            :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
    """

    heat_capacity_model: SGTEHeatCapacity
    """SGTE heat capacity model"""
    b: float
    """Coefficient of T in the Gibbs energy in J/K/mol"""

    @override
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)
        cp: SGTEHeatCapacity = self.heat_capacity_model

        return (
            -self.b
            - cp.c
            - cp.c * jnp.log(temperature)
            - 2 * cp.d * temperature
            + cp.e / temperature**2
            + 2 * cp.f / temperature**3
            + 3 * cp.g / temperature**4
        )


def sgte_thermodynamic_properties(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    volume_model: Volume | None = None,
) -> ThermodynamicProperties:
    r"""Creates thermodynamic properties from the coefficients of an SGTE Gibbs energy function

    The Gibbs energy is :math:`G = a + bT + cT\ln T + dT^2 + e/T + f/T^2 + g/T^3`
    (:class:`SGTEEnthalpy`).

    Args:
        a: Constant term in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        b: Coefficient of :math:`T` in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        c: Coefficient of :math:`T\ln T` in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        d: Coefficient of :math:`T^2` in :math:`\mathrm{J}\ \mathrm{K}^{-2} \mathrm{mol}^{-1}`
        e: Coefficient of :math:`1/T` in :math:`\mathrm{J}\ \mathrm{K}\ \mathrm{mol}^{-1}`
        f: Coefficient of :math:`1/T^2` in :math:`\mathrm{J}\ \mathrm{K}^2 \mathrm{mol}^{-1}`
        g: Coefficient of :math:`1/T^3` in :math:`\mathrm{J}\ \mathrm{K}^3 \mathrm{mol}^{-1}`
        volume_model: Volume model. Defaults to ``None``, which ignores the pressure dependence.

    Returns:
        Thermodynamic properties
    """
    heat_capacity_model: SGTEHeatCapacity = SGTEHeatCapacity(c, d, e, f, g)

    return ThermodynamicProperties(
        heat_capacity_model,
        SGTEEnthalpy(heat_capacity_model, a),
        SGTEEntropy(heat_capacity_model, b),
        volume_model,
    )


class SGTEMurnaghanEOS(Volume):
    r"""Volume of a solid with the Murnaghan equation of state of the SGTE data
    :cite:p:`Gustafson1986{Eqs. 4-8},Dinsdale1991`

    The volume at zero pressure follows from a thermal expansion linear in temperature,
    :math:`\alpha = \alpha_0 + \alpha_1 T`, integrated from 0 K, and its pressure dependence from
    the Murnaghan equation of state with a compressibility :math:`K` independent of temperature

    .. math::

        V = \frac{A\exp(\alpha_0 T + \alpha_1 T^2/2)}{(1 + nKP)^{1/n}}

    :cite:t:`Gustafson1986` integrate the volume from zero pressure. Here it is integrated from
    1 bar, consistent with the standard state, which differs by about 0.5 J/mol.

    Args:
        A: Volume at 0 K and zero pressure in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        alpha0: Constant term of the thermal expansion in :math:`\mathrm{K}^{-1}`
        alpha1: Linear term of the thermal expansion in :math:`\mathrm{K}^{-2}`
        K: Compressibility in :math:`\mathrm{bar}^{-1}`
        n: Pressure derivative of the bulk modulus
    """

    A: float
    alpha0: float
    alpha1: float
    K: float
    n: float

    def volume_zero_pressure(self, temperature: ArrayLike) -> Array:
        r"""Gets the volume at zero pressure.

        Args:
            temperature: Temperature in K

        Returns:
            Volume at zero pressure in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        """
        temperature = as_j64(temperature)

        return self.A * jnp.exp(self.alpha0 * temperature + self.alpha1 * temperature**2 / 2)

    @override
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

graphite_sgte: ThermodynamicProperties = sgte_thermodynamic_properties(
    -17368.441, 170.73, -24.3, -4.723e-4, 2562600.0, -2.643e8, 1.2e10, GRAPHITE_VOLUME_GUSTAFSON
)
"""Graphite (GHSERCC) from the SGTE data, valid from 298.15 to 6000 K, with the volume
:data:`GRAPHITE_VOLUME_GUSTAFSON` :cite:p:`Gustafson1986,Dinsdale1991{p. 337}`

Graphite is the reference phase of carbon, so its enthalpy is zero at 298.15 K. Its entropy is
5.7423 J/K/mol. The default database instead uses graphite from the NASA Glenn coefficients, with
:data:`diamond_gustafson`."""
diamond_sgte: ThermodynamicProperties = sgte_thermodynamic_properties(
    -16359.441, 175.61, -24.31, -4.723e-4, 2698000.0, -2.61e8, 1.11e10, DIAMOND_VOLUME_GUSTAFSON
)
"""Diamond (GDIACC) from the SGTE data, valid from 298.15 to 6000 K, with the volume
:data:`DIAMOND_VOLUME_GUSTAFSON` :cite:p:`Gustafson1986,Dinsdale1991{p. 337}`

Its enthalpy and entropy at 298.15 K are 1895.79 J/mol and 2.3598 J/K/mol. Use it with
:data:`graphite_sgte`."""

cp_graphite_gustafson: HeatCapacity = graphite_sgte.heat_capacity_model
"""Graphite heat capacity from :data:`graphite_sgte`"""
cp_diamond_gustafson: HeatCapacity = diamond_sgte.heat_capacity_model
"""Diamond heat capacity from :data:`diamond_sgte`"""

DIAMOND_ENTHALPY_CHANGE: float = float(
    diamond_sgte.enthalpy(TEMPERATURE_REFERENCE) - graphite_sgte.enthalpy(TEMPERATURE_REFERENCE)
)
"""Enthalpy of diamond minus graphite in J/mol at 298.15 K, 1895.79 J/mol, from
:data:`diamond_sgte` and :data:`graphite_sgte`

:cite:t:`Gustafson1986` fitted the Gibbs energies to the enthalpy and entropy of the transition
selected by Wagman et al. (1945), so this is the value of the assessment rather than a measured
value."""
DIAMOND_ENTROPY_CHANGE: float = float(
    diamond_sgte.entropy(TEMPERATURE_REFERENCE) - graphite_sgte.entropy(TEMPERATURE_REFERENCE)
)
"""Entropy of diamond minus graphite in J/K/mol at 298.15 K, -3.3825 J/K/mol, from
:data:`diamond_sgte` and :data:`graphite_sgte`"""

diamond_gustafson: ThermodynamicProperties = ThermodynamicProperties.from_reference_values(
    RelativeHeatCapacity(
        graphite.heat_capacity_model, cp_diamond_gustafson, cp_graphite_gustafson
    ),
    float(graphite.enthalpy(TEMPERATURE_REFERENCE)) + DIAMOND_ENTHALPY_CHANGE,
    float(graphite.entropy(TEMPERATURE_REFERENCE)) + DIAMOND_ENTROPY_CHANGE,
    DIAMOND_VOLUME_GUSTAFSON,
)
"""Diamond relative to graphite from the assessment of :cite:t:`Gustafson1986`, valid to 6000 K,
with the volume :data:`DIAMOND_VOLUME_GUSTAFSON`.

The heat capacity, enthalpy and entropy of diamond are those of graphite from the NASA Glenn
coefficients (:data:`~atmodeller.thermodata.vassiliev.graphite`) plus the differences between
diamond and graphite of :cite:t:`Gustafson1986`. The heat capacity of graphite agrees with that of
:cite:t:`Gustafson1986` to within 0.06 J/K/mol up to 5000 K, so this keeps graphite consistent
with the other NASA Glenn species while reproducing the Gibbs energy of diamond relative to
graphite of :cite:t:`Gustafson1986`.

For equilibrium between diamond and graphite at high pressure, use graphite with
:data:`GRAPHITE_VOLUME_GUSTAFSON`, as in :func:`~atmodeller.database.get_default_database`. This
reproduces the graphite-diamond transition of :cite:t:`Gustafson1986`."""
