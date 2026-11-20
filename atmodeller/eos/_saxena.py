# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Real gas EOSs from :cite:t:`SF87,SF87a,SF88,SS92`

The papers state a volume integration from :math:`P_0` to :math:`P`, where :math:`f(P_0=1)=1`.
Hence for a bounded EOS a minimum pressure of 1 bar is assumed.
"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from atmodeller import override
from atmodeller.interfaces import CatalogueEntry, ModelInfo
from atmodeller.constants import STANDARD_PRESSURE
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import FloatArray, as_j64, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration
from atmodeller.thermodata import CriticalData, critical_data_dictionary


class SaxenaABC(RealGas):
    """A real gas EOS from :cite:t:`SS92`

    This form of the EOS can also be used for the models in :cite:t:`SF87a,SF87,SF88`.

    Args:
        a_coefficients: :math:`a` coefficients
        b_coefficients: :math:`b` coefficients
        c_coefficients: :math:`c` coefficients
        d_coefficients: :math:`d` coefficients
        critical_data: Critical data. Defaults to empty.
    """

    a_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`a` coefficients"""
    b_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`b` coefficients"""
    c_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`c` coefficients"""
    d_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`d` coefficients"""
    critical_data: CriticalData = CriticalData()
    """Critical data"""

    @abstractmethod
    def _get_compressibility_coefficient(
        self, scaled_temperature: ArrayLike, coefficients: tuple[float, ...]
    ) -> FloatArray:
        """General form of the coefficients for the compressibility calculation
        :cite:p:`SS92{Equation 1}`

        Args:
            scaled_temperature: Scaled temperature (dimensionless)
            coefficients: Tuple of the coefficients :math:`a`, :math:`b`, :math:`c`, or :math:`d`.

        Returns
            The relevant coefficient
        """

    @property
    def critical_pressure(self) -> float:
        """Critical Pressure (bar)"""
        return self.critical_data.pressure

    @property
    def critical_temperature(self) -> float:
        """Critical temperature (K)"""
        return self.critical_data.temperature

    def a(self, scaled_temperature: ArrayLike) -> FloatArray:
        """:math:`a` parameter

        Args:
            scaled_temperature: Scaled temperature (dimensionless)

        Returns:
            a parameter
        """
        return self._get_compressibility_coefficient(scaled_temperature, self.a_coefficients)

    def b(self, scaled_temperature: ArrayLike) -> FloatArray:
        """:math:`b` parameter

        Args:
            scaled_temperature: Scaled temperature (dimensionless)

        Returns:
            b parameter
        """
        return self._get_compressibility_coefficient(scaled_temperature, self.b_coefficients)

    def c(self, scaled_temperature: ArrayLike) -> FloatArray:
        """:math:`c` parameter

        Args:
            scaled_temperature: Scaled temperature (dimensionless)

        Returns:
            c parameter
        """
        return self._get_compressibility_coefficient(scaled_temperature, self.c_coefficients)

    def d(self, scaled_temperature: ArrayLike) -> FloatArray:
        """:math:`d` parameter

        Args:
            scaled_temperature: Scaled temperature (dimensionless)

        Returns:
            d parameter
        """
        return self._get_compressibility_coefficient(scaled_temperature, self.d_coefficients)

    def scaled_pressure(self, pressure: ArrayLike) -> ArrayLike:
        """Scaled (reduced) pressure

        Args:
            pressure: Pressure (bar)

        Returns:
            The scaled (reduced) pressure (dimensionless)
        """
        scaled_pressure: ArrayLike = pressure / self.critical_pressure

        return scaled_pressure

    def scaled_temperature(self, temperature: ArrayLike) -> ArrayLike:
        """Scaled (reduced) temperature

        Args:
            temperature: Temperature (K)

        Returns:
            The scaled (reduced) temperature (dimensionless)
        """
        scaled_temperature: ArrayLike = temperature / self.critical_temperature

        return scaled_temperature

    @override
    def compressibility_factor(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Compressibility factor :cite:p:`SS92{Equation 2}`

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            The compressibility factor (dimensionless)
        """
        del mole_fractions

        Tr: ArrayLike = self.scaled_temperature(temperature)
        Pr: ArrayLike = self.scaled_pressure(pressure)
        Z: FloatArray = (
            self.a(Tr)
            + self.b(Tr) * Pr
            + self.c(Tr) * jnp.square(Pr)
            + self.d(Tr) * jnp.power(Pr, 3)
        )

        return Z

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions
        log_fugacity: FloatArray = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume :cite:p:`SS92{Equation 1}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        Z: FloatArray = self.compressibility_factor(temperature, pressure, mole_fractions)
        volume_ideal: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        volume: FloatArray = Z * volume_ideal

        return volume

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral :cite:p:`SS92{Equation 11}`

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

        Tr: ArrayLike = self.scaled_temperature(temperature)
        Pr: ArrayLike = self.scaled_pressure(pressure)
        STANDARD_PRESSURE_SCALED: float = STANDARD_PRESSURE / self.critical_pressure
        volume_integral: FloatArray = (
            (
                self.a(Tr) * jnp.log(Pr / STANDARD_PRESSURE_SCALED)
                + self.b(Tr) * (Pr - STANDARD_PRESSURE_SCALED)
                + (1.0 / 2) * self.c(Tr) * (jnp.square(Pr) - STANDARD_PRESSURE_SCALED**2)
                + (1.0 / 3) * self.d(Tr) * (jnp.power(Pr, 3) - STANDARD_PRESSURE_SCALED**3)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )

        return volume_integral


class SaxenaFiveCoefficients(SaxenaABC):
    """Real gas EOS with five coefficients, which is generally used for low pressures"""

    @override
    def _get_compressibility_coefficient(
        self, scaled_temperature: ArrayLike, coefficients: tuple[float, ...]
    ) -> FloatArray:
        """General form of the coefficients for the compressibility calculation
        :cite:p:`SS92{Equation 3b}`

        Args:
            temperature: Scaled temperature (dimensionless)
            coefficients: Tuple of the coefficients :math:`a`, :math:`b`, :math:`c`, or :math:`d`.

        Returns
            The relevant coefficient
        """
        coefficient: FloatArray = (
            as_j64(coefficients[0])  # To keep type checker happy
            + coefficients[1] / scaled_temperature
            + coefficients[2] / jnp.power(scaled_temperature, 3.0 / 2)
            + coefficients[3] / jnp.power(scaled_temperature, 3)
            + coefficients[4] / jnp.power(scaled_temperature, 4)
        )

        return coefficient


class SaxenaEightCoefficients(SaxenaABC):
    """Real gas EOS with eight coefficients, which is generally used for high pressures"""

    @override
    def _get_compressibility_coefficient(
        self, scaled_temperature: ArrayLike, coefficients: tuple[float, ...]
    ) -> FloatArray:
        """General form of the coefficients for the compressibility calculation
        :cite:p:`SS92{Equation 3a}`

        Args:
            temperature: Scaled temperature (dimensionless)
            coefficients: Tuple of the coefficients :math:`a`, :math:`b`, :math:`c`, or :math:`d`.

        Returns
            The relevant coefficient
        """
        coefficient: FloatArray = (
            as_j64(coefficients[0])  # To keep type checker happy
            + coefficients[1] * scaled_temperature
            + coefficients[2] / scaled_temperature
            + coefficients[3] * jnp.square(scaled_temperature)
            + coefficients[4] / jnp.square(scaled_temperature)
            + coefficients[5] * jnp.power(scaled_temperature, 3)
            + coefficients[6] / jnp.power(scaled_temperature, 3)
            + coefficients[7] * jnp.log(scaled_temperature)
        )

        return coefficient


_H2_low_pressure_SS92: RealGas = SaxenaFiveCoefficients(
    a_coefficients=(1, 0, 0, 0, 0),
    b_coefficients=(0, 0.9827e-1, 0, -0.2709, 0),
    # Saxena and Fei (1987a), Eq. 23, C final coefficient = 0.1472e-1 (not 0.1427e-1)
    c_coefficients=(0, 0, -0.1030e-2, 0, 0.1472e-1),
    d_coefficients=(0, 0, 0, 0, 0),
    critical_data=critical_data_dictionary["H2"],
)
"""H2 low pressure (<1000 bar) :cite:p:`SS92{Table 1b}`

The coefficients are the same as for the corresponding states model in :cite:t:`SS92{Table 1a}`
because they originate from :cite:t:`SF87a{Equation 23}`.

In :cite:t:`SF87a{Equation 23}` the final :math:`c` coefficient is 0.1472e-1, not 0.1427e-1 as
given in :cite:t:`SS92{Table 1b}`. The earlier work is assumed to be correct.
"""
_H2_high_pressure_SS92: RealGas = SaxenaEightCoefficients(
    a_coefficients=(2.2615, 0, -6.8712e1, 0, -1.0573e4, 0, 0, -1.6936e-1),
    b_coefficients=(-2.6707e-4, 0, 2.0173e-1, 0, 4.5759, 0, 0, 3.1452e-5),
    c_coefficients=(-2.3376e-9, 0, 3.4091e-7, 0, -1.4188e-3, 0, 0, 3.0117e-10),
    d_coefficients=(-3.2606e-15, 0, 2.4402e-12, 0, -2.4027e-9, 0, 0, 0),
)
"""H2 high pressure (>1000 bar) :cite:p:`SS92{Table 1b}`

These coefficients are those of :cite:t:`SF88`, rewritten from reduced to absolute temperature (K)
and pressure (bar) using the critical values of :cite:t:`SF88` (:math:`T_c=33.1` K,
:math:`P_c=13` bar), so no critical data are given (they default to 1). Converting the
:cite:t:`SF88` coefficients reproduces this table to four significant figures, provided the final
:math:`c` coefficient of :cite:t:`SF88` is taken as 5.0897e-8; the printed 5.0897e-5 is a misprint.
"""

H2_SS92: RealGas = CombinedRealGas.create(
    [_H2_low_pressure_SS92, _H2_high_pressure_SS92],
    [
        ExperimentalCalibration(pressure_min=1, pressure_max=1000),
        ExperimentalCalibration(pressure_min=1000, pressure_max=1e5),
    ],
)
"""H2 EOS, which combines the low and high pressure EOS :cite:p:`SS92{Table 1b}`

:cite:t:`SF87,SF87a` included H2 in their corresponding states equations, but :cite:t:`SF88`
found that H2, with its softer intermolecular potential, is not well represented by corresponding
states: above 5 kbar the :cite:t:`SF87{Equation 11}` model overestimates the observed
compressibility factor of H2 by 7-10% along the shock-wave data. :cite:t:`SS92` therefore adopt the
H2-specific fit of :cite:t:`SF88` above 1 kbar, which supersedes the corresponding states model for
H2, while the branch below 1 kbar retains the corresponding states form of
:cite:t:`SF87a{Equation 23}`.

The high pressure fit is constrained by static data at a few kbar and by shock-wave data that lie
along a path of rising temperature and pressure, from about 10 kbar at 300 K to about 115 kbar at
3500 K :cite:p:`RRY83,SF87{Table 1}`. Along this path it reproduces the observed compressibility
factor to within 1.5%, but away from it the polynomial is unconstrained. The upper bound of
100 kbar is retained for high temperature use, beyond which the polynomial is not evaluated and
the EOS is extrapolated.

At lower temperatures the fit breaks down well below 100 kbar. The compressibility factor stops
increasing with pressure at about 20, 35, 50 and 70 kbar at 300, 600, 1000 and 2000 K,
respectively, and becomes negative (so the fugacity decreases with pressure) above about 41, 62 and
87 kbar at 300, 500 and 800 K, respectively.

Between 2000 and 3000 K and 1 to 50 kbar, compared with :cite:t:`CD21`, which also reproduces the
shock-wave data, the compressibility factor deviates by up to 6% at 2000 K and up to 25% at 3000 K,
and the fugacity by up to about 8% at 2000 K and 30% at 3000 K. At high temperature the fugacity
error arises mostly at 1 to 5 kbar, where the compressibility factor falls below unity. For this
range the ``H2_chabrier21`` EOS model is preferred.
"""

O2S_SS92: RealGas = SaxenaEightCoefficients(
    a_coefficients=(0.92854, 0.43269e-1, -0.24671, 0, 0.24999, 0, -0.53182, -0.16461e-1),
    b_coefficients=(
        0.84866e-3,
        -0.18379e-2,
        0.66787e-1,
        0,
        -0.29427e-1,
        0,
        0.29003e-1,
        0.54808e-2,
    ),
    c_coefficients=(
        -0.35456e-3,
        0.23316e-4,
        0.94159e-3,
        0,
        -0.81653e-3,
        0,
        0.23154e-3,
        0.55542e-4,
    ),
    d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
    critical_data=critical_data_dictionary["O2S"],
)
"""SO2 EOS :cite:p:`SS92{Table 1c}` from 1 to 10e3 bar"""
O2S_SS92_bounded: RealGas = CombinedRealGas.create(
    [O2S_SS92], [ExperimentalCalibration(pressure_min=1, pressure_max=10000)]
)
"""SO2 EOS :cite:p:`SS92{Table 1c}` from 1 to 10e3 bar"""

_H2S_low_pressure_SS92: RealGas = SaxenaEightCoefficients(
    a_coefficients=(0.14721e1, 0.11177e1, 0.39657e1, 0, -0.10028e2, 0, 0.45484e1, -0.38200e1),
    b_coefficients=(0.16066, 0.10887, 0.29014, 0, -0.99593, 0, -0.18627, -0.45515),
    c_coefficients=(-0.28933, -0.70522e-1, 0.39828, 0, -0.50533e-1, 0, 0.11760, 0.33972),
    d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
    critical_data=critical_data_dictionary["H2S"],
)
"""H2S low pressure (1-500 bar) :cite:p:`SS92{Table 1d}`"""

_H2S_high_pressure_SS92: RealGas = SaxenaEightCoefficients(
    a_coefficients=(0.59941, -0.15570e-2, 0.45250e-1, 0, 0.36687, 0, -0.79248, 0.26058),
    b_coefficients=(
        0.22545e-1,
        0.17473e-2,
        0.48253e-1,
        0,
        -0.19890e-1,
        0,
        0.32794e-1,
        -0.10985e-1,
    ),
    c_coefficients=(
        0.57375e-3,
        -0.20944e-5,
        -0.11894e-2,
        0,
        0.14661e-2,
        0,
        -0.75605e-3,
        -0.27985e-3,
    ),
    d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
    critical_data=critical_data_dictionary["H2S"],
)
"""H2S high pressure (500-10000 bar) :cite:p:`SS92{Table 1d}`"""

H2S_SS92: RealGas = CombinedRealGas.create(
    [_H2S_low_pressure_SS92, _H2S_high_pressure_SS92],
    [
        ExperimentalCalibration(pressure_min=1.0, pressure_max=500),
        ExperimentalCalibration(pressure_min=500, pressure_max=10000),
    ],
)
"""H2S EOS, which combines the low and high pressure EOS :cite:p:`SS92{Table 1d}`"""


def get_corresponding_states_SS92(species: str) -> RealGas:
    """Corresponding states :cite:p:`SS92{Table 1a}`

    Coefficients for the low and medium pressure regimes are from
    :cite:t:`SF87a{Equation 23 and 21}`, respectively, and for the high pressure regime from
    :cite:t:`SF87{Equation 11}`. Two printed values conflict between sources:

    - Low pressure, final :math:`c` coefficient: 0.1472e-1 in :cite:t:`SF87a{Equation 23}`
      (abstract and body) versus 0.1427e-1 in :cite:t:`SS92{Table 1a}`. The original source is
      adopted; the SS92 value is likely a transposition when transcribing from :cite:t:`SF87a`. For
      T >= 400 K the two values differ by up to about 0.03 in Z below 1 kbar, within the stated fit
      error of 0.076.
    - Medium pressure, :math:`c` coefficient of :math:`T_r^{-2}`: -1.4164e-4 in the body of
      :cite:t:`SF87a{Equation 21}` and in :cite:t:`SS92{Table 1a}`, but +1.4164e-4 in the
      abstract of :cite:t:`SF87a`, which is a misprint. The negative value agrees to within a
      few percent with the observed CO2 compressibility factors in :cite:t:`SF87a{Table 1}`
      (the positive value is 13-35% off) and agrees better with independent EOSs (e.g. CORK,
      Zhang and Duan) at 1-5 kbar.

    Although :cite:t:`SF87,SF87a` included H2 in their fits, this function should not be used for
    H2; see :obj:`H2_SS92`.

    Args:
        species: A species

    Returns:
        A corresponding states model for the species
    """
    critical_data: CriticalData = critical_data_dictionary[species]

    # Table 1a, <1000 bar
    low_pressure: RealGas = SaxenaFiveCoefficients(
        a_coefficients=(1, 0, 0, 0, 0),
        b_coefficients=(0, 0.9827e-1, 0, -0.2709, 0),
        # SF87a Eq. 23: final c coefficient 0.1472e-1 (SS92 Table 1a: 0.1427e-1, likely a typo)
        c_coefficients=(0, 0, -0.1030e-2, 0, 0.1472e-1),
        d_coefficients=(0, 0, 0, 0, 0),
        critical_data=critical_data,
    )

    # Table 1a, 1000-5000 bar
    medium_pressure: RealGas = SaxenaEightCoefficients(
        a_coefficients=(1, 0, 0, 0, -5.917e-1, 0, 0, 0),
        b_coefficients=(0, 0, 9.122e-2, 0, 0, 0, 0, 0),
        # SF87a Eq. 21 (body) and SS92 Table 1a: -1.4164e-4 (SF87a abstract misprints the sign)
        c_coefficients=(0, 0, 0, 0, -1.4164e-4, 0, 0, -2.8349e-6),
        d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
        critical_data=critical_data,
    )

    # Table 1a, >5000 bar
    # High precision coefficients taken from Saxena and Fei (1987), GCA, and agree with Table 1(a)
    high_pressure: RealGas = SaxenaEightCoefficients(
        a_coefficients=(2.0614, 0, 0, 0, -2.2351, 0, 0, -3.9411e-1),
        b_coefficients=(0, 0, 5.5125e-2, 0, 3.9344e-2, 0, 0, 0),
        c_coefficients=(0, 0, -1.8935e-6, 0, -1.1092e-5, 0, -2.1892e-5, 0),
        d_coefficients=(0, 0, 5.0527e-11, 0, 0, -6.3033e-21, 0, 0),
        critical_data=critical_data,
    )

    combined_model: RealGas = CombinedRealGas.create(
        [low_pressure, medium_pressure, high_pressure],
        [
            ExperimentalCalibration(pressure_min=1, pressure_max=1000),
            ExperimentalCalibration(pressure_min=1000, pressure_max=5000),
            ExperimentalCalibration(pressure_min=5000),
        ],
    )

    return combined_model


CH4_SS92: RealGas = get_corresponding_states_SS92("CH4")
"""CH4 corresponding states :cite:p:`SS92`"""
CO_SS92: RealGas = get_corresponding_states_SS92("CO")
"""CO corresponding states :cite:p:`SS92`"""
CO2_SS92: RealGas = get_corresponding_states_SS92("CO2")
"""CO2 corresponding states :cite:p:`SS92`"""
COS_SS92: RealGas = get_corresponding_states_SS92("COS")
"""COS corresponding states :cite:p:`SS92`"""
O2_SS92: RealGas = get_corresponding_states_SS92("O2")
"""O2 corresponding states :cite:p:`SS92`"""
S2_SS92: RealGas = get_corresponding_states_SS92("S2")
"""S2 corresponding states :cite:p:`SS92`"""
Ar_SF87: RealGas = get_corresponding_states_SS92("Ar")
"""Ar corresponding states :cite:p:`SF87{Equation 11}`

The low pressure extension given by the corresponding states model of :cite:t:`SS92{Table 1a}` is
also adopted.
"""
N2_SF87: RealGas = get_corresponding_states_SS92("N2")
"""N2 corresponding states :cite:p:`SF87{Equation 11}`

The low pressure extension given by the corresponding states model of :cite:t:`SS92{Table 1a}` is
also adopted.
"""


def get_saxena_eos_catalogue() -> dict[str, CatalogueEntry[RealGas]]:
    """Gets a dictionary of Saxena and colleagues EOS models

    'cs' refers to corresponding states.

    Returns:
        Dictionary of EOS models with their catalogue information
    """
    eos_models: dict[str, CatalogueEntry[RealGas]] = {}
    eos_models["Ar_cs_saxena87"] = CatalogueEntry(
        Ar_SF87, ModelInfo("Ar", sources=("SF87", "SS92"))
    )
    eos_models["CH4_cs_shi92"] = CatalogueEntry(CH4_SS92, ModelInfo("CH4", sources=("SS92",)))
    eos_models["CO_cs_shi92"] = CatalogueEntry(CO_SS92, ModelInfo("CO", sources=("SS92",)))
    eos_models["CO2_cs_shi92"] = CatalogueEntry(CO2_SS92, ModelInfo("CO2", sources=("SS92",)))
    eos_models["COS_cs_shi92"] = CatalogueEntry(COS_SS92, ModelInfo("COS", sources=("SS92",)))
    eos_models["H2_shi92"] = CatalogueEntry(H2_SS92, ModelInfo("H2", sources=("SS92", "SF88")))
    eos_models["H2S_shi92"] = CatalogueEntry(H2S_SS92, ModelInfo("H2S", sources=("SS92",)))
    eos_models["N2_cs_saxena87"] = CatalogueEntry(
        N2_SF87, ModelInfo("N2", sources=("SF87", "SS92"))
    )
    eos_models["O2_cs_shi92"] = CatalogueEntry(O2_SS92, ModelInfo("O2", sources=("SS92",)))
    eos_models["S2_cs_shi92"] = CatalogueEntry(S2_SS92, ModelInfo("S2", sources=("SS92",)))
    eos_models["O2S_shi92"] = CatalogueEntry(O2S_SS92_bounded, ModelInfo("O2S", sources=("SS92",)))

    return eos_models
