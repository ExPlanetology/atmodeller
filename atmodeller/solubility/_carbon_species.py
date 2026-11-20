# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Solubility laws for carbon species

For every law there should be a test in the test suite.
"""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.jax_utils import safe_exp
from atmodeller.sci_utils import unit_conversion
from atmodeller.solubility.core import Solubility


class _CH4_basalt_ardia13(Solubility):
    """CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`

    Experiments conducted at 0.7-3 GPa and 1400-1450 C. :cite:t:`AHW13{Equations 7a, 8}`, values
    for lnK0 and deltaV from the text.
    """

    @override
    def concentration(self, fugacity: ArrayLike, *, pressure: ArrayLike, **kwargs) -> Array:
        del kwargs
        pressure_gpa: ArrayLike = pressure * unit_conversion.bar_to_GPa
        one_bar_in_gpa: ArrayLike = unit_conversion.bar_to_GPa
        k: Array = safe_exp(4.93 - (1.93 * (pressure_gpa - one_bar_in_gpa)))
        ppmw: Array = k * fugacity * unit_conversion.bar_to_GPa

        return ppmw


CH4_basalt_ardia13: Solubility = _CH4_basalt_ardia13()
"""CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`

Experiments conducted at 0.7-3 GPa and 1400-1450 C. :cite:t:`AHW13{Equations 7a, 8}`, values
for lnK0 and deltaV from the text.
"""


class _CO_basalt_armstrong15(Solubility):
    """Volatiles in mafic melts under reduced conditions :cite:p:`AHS15`

    Experiments on Martian and terrestrial basalts at 1.2 GPa and 1400 C with variable fO2 from
    IW-3.65 to IW+1.46. :cite:t:`AHS15{Equation 10}`, log-scale linear fit for CO and includes
    dependence on total pressure. The fitting coefficients also use data from :cite:p:`SHW14`
    (experiments from 1-1.2 GPa).
    """

    @override
    def concentration(self, fugacity: ArrayLike, *, pressure: ArrayLike, **kwargs) -> Array:
        del kwargs
        logco_ppm: Array = -0.738 + (0.876 * jnp.log10(fugacity)) - (5.44e-5 * pressure)
        ppmw: Array = jnp.power(10, logco_ppm)

        return ppmw


CO_basalt_armstrong15: Solubility = _CO_basalt_armstrong15()
"""Volatiles in mafic melts under reduced conditions :cite:p:`AHS15`

Experiments on Martian and terrestrial basalts at 1.2 GPa and 1400 C with variable fO2 from
IW-3.65 to IW+1.46. :cite:t:`AHS15{Equation 10}`, log-scale linear fit for CO and includes
dependence on total pressure. The fitting coefficients also use data from :cite:p:`SHW14`
(experiments from 1-1.2 GPa).
"""


class _CO_basalt_yoshioka19(Solubility):
    """Carbon in silicate melts :cite:p:`YNN19`

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C. Log-scale linear expression for solubility of CO in
    MORB in the abstract.
    """

    @override
    def concentration(self, fugacity: ArrayLike, **kwargs) -> Array:
        del kwargs
        co_wtp: Array = jnp.power(10, (-5.20 + (0.8 * jnp.log10(fugacity))))
        ppmw: Array = co_wtp * unit_conversion.percent_to_ppm

        return ppmw


CO_basalt_yoshioka19: Solubility = _CO_basalt_yoshioka19()
"""Carbon in silicate melts :cite:p:`YNN19`

Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
CO-CO2 fluid phase at 3 GPa and 1500 C. Log-scale linear expression for solubility of CO in
MORB in the abstract.
"""


class _CO_rhyolite_yoshioka19(Solubility):
    """Carbon in silicate melts :cite:p:`YNN19`

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C. A power law in fugacity (exponent 0.52), their
    expression for the solubility of CO in rhyolite in the abstract.
    """

    @override
    def concentration(self, fugacity: ArrayLike, **kwargs) -> Array:
        del kwargs
        co_wtp: Array = jnp.power(10, (-4.08 + (0.52 * jnp.log10(fugacity))))
        ppmw: Array = co_wtp * unit_conversion.percent_to_ppm

        return ppmw


CO_rhyolite_yoshioka19: Solubility = _CO_rhyolite_yoshioka19()
"""Carbon in silicate melts :cite:p:`YNN19`

Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
CO-CO2 fluid phase at 3 GPa and 1500 C. A power law in fugacity (exponent 0.52), their
expression for the solubility of CO in rhyolite in the abstract.
"""


class _CO2_basalt_dixon95(Solubility):
    r"""CO2 in MORB liquids :cite:p:`DSH95`

    :cite:t:`DSH95{Equation 6}` for mole fraction of dissolved carbonate
    (:math:`\mathrm{CO_3^{2-}}`) and then converting to ppmw for CO2 experiments conducted at
    1200 C, 210-980 bars with mixed H2O-CO2 vapor phase (CO2 vapor mole fraction varied from
    0.42-0.97).
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, pressure: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        arg: Array = safe_exp(-23 * (pressure - 1) / (83.15 * temperature)) * fugacity * 3.8e-7
        ppmw: Array = 1.0e4 * (4400 * arg) / (36.6 - 44 * arg)

        return ppmw


CO2_basalt_dixon95: Solubility = _CO2_basalt_dixon95()
r"""CO2 in MORB liquids :cite:p:`DSH95`

:cite:t:`DSH95{Equation 6}` for mole fraction of dissolved carbonate (:math:`\mathrm{CO_3^{2-}}`)
and then converting to ppmw for CO2 experiments conducted at 1200 C, 210-980 bars with mixed
H2O-CO2 vapor phase (CO2 vapor mole fraction varied from 0.42-0.97).
"""
