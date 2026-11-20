# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Solubility laws for hydrogen species

For every law there should be a test in the test suite.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from molmass import Formula

from atmodeller import override
from atmodeller.eos._chabrier import H2_chabrier21_bounded
from atmodeller.jax_utils import Scalar, as_j64, safe_exp
from atmodeller.sci_utils import unit_conversion
from atmodeller.solubility.core import Solubility, SolubilityPowerLaw, SolubilityPowerLawLog10

H2_andesite_hirschmann12: Solubility = SolubilityPowerLawLog10(1.01058631, 0.60128868)
"""H2 in synthetic andesite :cite:p:`HWA12`

Log-scale linear fit to fH2 vs H2 concentration for andesite in :cite:t:`HWA12{Table 2}`.
Experiments conducted from 0.7-3 GPa at 1400 C.
"""

H2_basalt_hirschmann12: Solubility = SolubilityPowerLawLog10(1.10083602, 0.52413928)
"""H2 in synthetic basalt :cite:p:`HWA12`

Log-scale linear fit to fH2 vs. H2 concentration for basalt in :cite:t:`HWA12{Table 2}`.
Experiments conducted from 0.7-3 GPa, 1400 C.
"""

H2_silicic_melts_gaillard03: Solubility = SolubilityPowerLaw(0.163, 1.252)
"""Fe-H redox exchange in silicate glasses :cite:p:`GSM03`

Power law fit for fH2 vs. H2 (ppm-wt) from :cite:t:`GSM03{Table 4}` data. Experiments at
pressures from 0.02-70 bar, temperatures from 300-1000C.
"""

H2O_ano_dio_newcombe17: Solubility = SolubilityPowerLaw(727, 0.5)
"""H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`

Power law from :cite:t:`NBB17{Figure 5(A)}` for anorthite-diopside glass. Experiments conducted
at 1 atm and 1350 C. Melts equilibrated in 1 atm furnace with H2/CO2 gas mixtures that spanned
fO2 from IW-3 to IW+4.8 and pH2/pH2O from 0.003-24.
"""

H2O_basalt_dixon95: Solubility = SolubilityPowerLaw(965, 0.5)
"""H2O in MORB liquids :cite:p:`DSH95`

Square-root power law refitted by Paolo Sossi to the experimental data of
:cite:t:`DSH95{Figure 4a}`: total dissolved H2O against fH2O for MORB at 1200 C, from
experiments with pure H2O (200-717 bar) and mixed H2O-CO2 vapour (up to 980 bar), spanning fH2O
from 16 to 709 bar. It reproduces the 15 experiments of :cite:t:`DSH95{Table 1}` with an rms
misfit of about 0.07 wt% and no systematic bias.

It is not a fit to the solubility model of :cite:t:`DSH95`, which also includes higher-pressure
data and the speciation of water. Relative to that model (:cite:t:`DSH95{Table 5}`) this law is
about 4% lower over the experimental range and about 12% lower at 1 bar and at 5 kbar. For a fit
to the model, see :data:`H2O_basalt_dixon95_table5`.
"""

H2O_basalt_dixon95_table5: Solubility = SolubilityPowerLaw(1072.64, 0.494861)
r"""H2O in MORB liquids, a power law fitted to the Table 5 model :cite:p:`DSH95`

Power law fitted (least squares in log space) to the model solubility of total H2O against fH2O
for MORB at 1200 C in :cite:t:`DSH95{Table 5}`, from 1 bar to 5 kbar. That model combines
Henrian molecular water with a partial molar volume of 12 :math:`\mathrm{cm^3\,mol^{-1}}` and the
regular-solution speciation of :cite:t:`SS89`, and is calibrated on the experiments of
:cite:t:`DSH95` and higher-pressure data to about 8 kbar. The fit reproduces Table 5 with an rms misfit of 3.5% and a
maximum of 6.4% (at 5 kbar), compared with 8.1% and 13.7% for :data:`H2O_basalt_dixon95`.

The model is for 1200 C only and has no temperature dependence.
"""

H2O_basalt_mitchell17: Solubility = SolubilityPowerLaw(258.946, 0.669)
"""H2O in basaltic melt :cite:p:`MGO17`

Refitted the H2O wt. % vs. fH2O fitted line from :cite:t:`MGO17{Figure 8}` to a power-law.
Experiments conducted at 1200 C and 1000 MPa total pressure. This fit includes data from
their experiments and prior studies on H2O solubility in basaltic melt at 1200 C and P at or
below 600 MPa.
"""

H2O_lunar_glass_newcombe17: Solubility = SolubilityPowerLaw(683, 0.5)
"""H2O in lunar basalt :cite:p:`NBB17`

Power law from :cite:t:`NBB17{Figure 5(A)}` for Lunar glass. Experiments conducted at 1 atm and
1350 C. Melts equilibrated in 1-atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3
to IW+4.8.
"""

H2O_peridotite_sossi23: Solubility = SolubilityPowerLaw(647, 0.5)
"""H2O in peridotite liquids :cite:p:`STB23`

Power law parameters in the abstract for peridotitic glasses. Experiments conducted at 2173 K
and 1 bar and range of fO2 from IW-1.9 to IW+6.0.
"""


class _H2O_moore98(Solubility):
    r"""H2O in natural silicate liquids :cite:p:`MVC98`

    Empirical model of :cite:t:`MVC98{Equation 1}` with the coefficients of
    :cite:t:`MVC98{Table 4}`:

    .. math::

        2 \ln X_{\rm H_2O}^{\rm melt} = \frac{a}{T} + \sum_i b_i X_i \frac{P}{T}
        + c \ln f_{\rm H_2O} + d

    where :math:`X_{\rm H_2O}^{\rm melt}` is the mole fraction of H2O dissolved in the melt,
    :math:`X_i` are the anhydrous mole fractions of the oxide components Al2O3, FeOt (total iron
    as FeO) and Na2O, :math:`T` is temperature in K, and :math:`P` is pressure in bar. The mole
    fractions are on an oxide basis (Al2O3, Na2O, rather than AlO1.5, NaO0.5), which reproduces
    the :math:`X_{\rm H_2O}^{\rm melt}` of :cite:t:`MVC98{Table 2}` without bias.

    The mole fraction is converted to a mass concentration using the mean molar mass of the
    anhydrous melt per mole of oxide components. :cite:t:`MVC98` do not specify their conversion;
    their tabulated mole fractions and mass concentrations imply a slightly smaller molar mass
    (about 61.3 g/mol for the default MORB), so this conversion may give mass concentrations about
    2-3% lower than theirs.

    The model is calibrated on H2O-saturated experiments on natural compositions from basalt to
    rhyolite at 700-1200 C and 1-3000 bar, and reproduces them to 0.5 wt% (two standard
    deviations). :cite:t:`MVC98` caution against extrapolation beyond about 3 kbar, where the
    critical behaviour of silicate-H2O systems also becomes important.

    Args:
        xal2o3: Anhydrous mole fraction of Al2O3. Defaults to 0.0842.
        xfeot: Anhydrous mole fraction of total iron as FeO. Defaults to 0.1082.
        xna2o: Anhydrous mole fraction of Na2O. Defaults to 0.0271.
        molar_mass: Mean molar mass of the anhydrous melt per mole of oxide components in
            g/mol. Defaults to 62.79 g/mol.

    The defaults are for the MORB TT152-21-35 of :cite:t:`MVC98{Table 1}`, which is the MORB of
    :cite:t:`DSH95`.
    """

    xal2o3: float = eqx.field(converter=float)
    """Anhydrous mole fraction of Al2O3"""
    xfeot: float = eqx.field(converter=float)
    """Anhydrous mole fraction of total iron as FeO"""
    xna2o: float = eqx.field(converter=float)
    """Anhydrous mole fraction of Na2O"""
    molar_mass: float = eqx.field(converter=float)
    """Mean molar mass of the anhydrous melt per mole of oxide components in g/mol"""

    def __init__(
        self,
        xal2o3: Scalar = 0.0842,
        xfeot: Scalar = 0.1082,
        xna2o: Scalar = 0.0271,
        molar_mass: Scalar = 62.79,
    ):
        self.xal2o3 = xal2o3
        self.xfeot = xfeot
        self.xna2o = xna2o
        self.molar_mass = molar_mass

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, pressure: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fugacity = as_j64(fugacity)
        temperature = as_j64(temperature)
        pressure = as_j64(pressure)

        # Coefficients from Table 4
        a: float = 2565.0
        b_al2o3: float = -1.997
        b_feot: float = -0.9275
        b_na2o: float = 2.736
        c: float = 1.171
        d: float = -14.21

        compositional_term: float = (
            b_al2o3 * self.xal2o3 + b_feot * self.xfeot + b_na2o * self.xna2o
        )
        two_ln_x: Array = (
            a / temperature
            + compositional_term * pressure / temperature
            + c * jnp.log(fugacity)
            + d
        )
        x_h2o: Array = safe_exp(0.5 * two_ln_x)

        # Mole fraction to mass fraction using the molar mass of H2O and of the anhydrous melt
        molar_mass_h2o: float = Formula("H2O").mass
        mass_fraction: Array = (x_h2o * molar_mass_h2o) / (
            x_h2o * molar_mass_h2o + (1.0 - x_h2o) * self.molar_mass
        )
        ppmw: Array = mass_fraction * unit_conversion.fraction_to_ppm

        return ppmw


H2O_basalt_moore98: Solubility = _H2O_moore98()
"""H2O in MORB liquid :cite:p:`MVC98`

:cite:t:`MVC98{Equation 1}` and coefficients in :cite:t:`MVC98{Table 4}`, with the composition of
the MORB TT152-21-35 of :cite:t:`MVC98{Table 1}`. The model is calibrated at 700-1200 C and
1-3000 bar.
"""


class _H2_chachan18(Solubility):
    """H2 solubility :cite:p:`CS18`

    Args:
        f_calibration: Calibration fugacity
        T_calibration: Calibration temperature
        X_calibration: Mass fraction at calibration conditions
        T0: Arrhenius temperature factor (K), which expresses the repulsive interaction of the
            molecule with magma. Defaults to 4000 K, which is the middle of the range the authors
            explore (from 3000 K to 5000 K).
    """

    f_calibration: float = eqx.field(converter=float)
    """Calibration fugacity"""
    T_calibration: float = eqx.field(converter=float)
    """Calibration temperature"""
    X_calibration: float = eqx.field(converter=float)
    """Mass fraction at calibration conditions"""
    T0: float = eqx.field(converter=float)
    """Arrhenius temperature factor (K)"""
    A: float = eqx.field(converter=float)
    """Constant factor"""

    def __init__(
        self,
        f_calibration: Scalar,
        T_calibration: Scalar,
        X_calibration: Scalar,
        T0: Scalar = 4000,
    ):
        self.f_calibration = f_calibration
        self.T_calibration = T_calibration
        self.X_calibration = X_calibration
        self.T0 = T0
        self.A = np.exp(
            (self.T0 / self.T_calibration) + np.log(self.X_calibration / self.f_calibration)
        )
        # jax.debug.print("A = ", self.A)

    @override
    def concentration(self, fugacity: ArrayLike, *, temperature: ArrayLike, **kwargs) -> Array:
        del kwargs
        mass_fraction: Array = jnp.exp(-self.T0 / temperature) * self.A * fugacity
        ppmw: Array = mass_fraction * unit_conversion.fraction_to_ppm

        return ppmw


H2_chachan18: Solubility = _H2_chachan18(
    f_calibration=1500, T_calibration=1673, X_calibration=0.001
)
"""H2 by combining theory and experiment :cite:p:`CS18`"""

# At 1 GPa in the presence of pure H2, the molecular H2 concentration is 0.19 wt.%, so the
# calibration pressure is converted to H2 fugacity, here with the bounded Chabrier EOS
T_calibration: Scalar = 1673
P_calibration: Scalar = unit_conversion.GPa_to_bar
f_calibration: Array = H2_chabrier21_bounded.fugacity(as_j64(T_calibration), as_j64(P_calibration))
X_calibration: Scalar = 0.0019

H2_kite19: Solubility = _H2_chachan18(
    f_calibration=float(f_calibration), T_calibration=T_calibration, X_calibration=X_calibration
)
"""H2 by combining theory and experiment :cite:p:`KFS19`"""
