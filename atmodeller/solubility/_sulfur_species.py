# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Solubility laws for sulfur species

For every law there should be a test in the test suite.
"""

from typing import ClassVar

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.jax_utils import as_j64
from atmodeller.sci_utils import unit_conversion
from atmodeller.solubility.core import Solubility


class _S2_sulfate_andesite_boulliung23(Solubility):
    r"""Sulfur as sulfate :math:`\mathrm{SO_4^{2-}}/\mathrm{S^{6+}}` in andesite
    :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (:math:`C_{\mathrm{S^{6+}}}`) in :cite:t:`BW23corr`. Composition for andesite
    from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
    equilibrated with Air/SO2 mixtures.
    The sulfate capacity is defined with the S content in wt% S, not wt% SO4.
    """

    requires_fO2: ClassVar[bool] = True

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fO2 = jnp.asarray(fO2)
        logcs: Array = -12.948 + (31586.2393 / as_j64(temperature))
        # The sulfate capacity is defined with the S content in wt% S, not wt% SO4 (BW22, p. 160)
        logs_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_andesite_boulliung23: Solubility = _S2_sulfate_andesite_boulliung23()
r"""Sulfur as sulfate :math:`\mathrm{SO_4^{2-}}/\mathrm{S^{6+}}` in andesite
:cite:p:`BW22,BW23corr`

Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
sulfate capacity (:math:`C_{\mathrm{S^{6+}}}`) in :cite:t:`BW23corr`. Composition for andesite from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
equilibrated with Air/SO2 mixtures.
The sulfate capacity is defined with the S content in wt% S, not wt% SO4.
"""


class _S2_sulfide_andesite_boulliung23(Solubility):
    r"""Sulfur as sulfide (:math:`\mathrm{S^{2-}}`) in andesite :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity
    (:math:`C_{\mathrm{S^{2-}}}`). Composition for andesite from :cite:t:`BW22{Table 1}`.
    Experiments conducted at 1 atm, 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2
    conditions were greater than 1 log unit below FMQ.
    """

    requires_fO2: ClassVar[bool] = True

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fO2 = jnp.asarray(fO2)
        logcs: Array = 0.225 - (8921.0927 / as_j64(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_andesite_boulliung23: Solubility = _S2_sulfide_andesite_boulliung23()
r"""Sulfur as sulfide (:math:`\mathrm{S^{2-}}`) in andesite :cite:p:`BW23`

Using expressions in the abstract for S wt.% and sulfide capacity (:math:`C_{\mathrm{S^{2-}}}`).
Composition for andesite from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K
in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
"""


class _S2_andesite_boulliung23(Solubility):
    """S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    requires_fO2: ClassVar[bool] = True

    _sulfide: Solubility
    _sulfate: Solubility

    def __init__(self):
        self._sulfide = S2_sulfide_andesite_boulliung23
        self._sulfate = S2_sulfate_andesite_boulliung23

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        fO2 = jnp.asarray(fO2)
        concentration: ArrayLike = self._sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )
        concentration = concentration + self._sulfate.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )

        return concentration


S2_andesite_boulliung23: Solubility = _S2_andesite_boulliung23()
"""S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""


class _S2_sulfate_basalt_boulliung23(Solubility):
    r"""Sulfur in basalt as sulfate, :math:`\mathrm{SO_4^{2-}}/\mathrm{S^{6+}}`
    :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract and the corrected expression for sulfate capacity
    (:math:`C_{\mathrm{S^{6+}}}`) in :cite:t:`BW23corr`. Composition for Basalt from
    :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure, temperatures from 1473-1773 K
    for silicate melts equilibrated with Air/SO2 mixtures.
    The sulfate capacity is defined with the S content in wt% S, not wt% SO4.
    """

    requires_fO2: ClassVar[bool] = True

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fO2 = jnp.asarray(fO2)
        logcs: Array = -12.948 + (32333.5635 / as_j64(temperature))
        # The sulfate capacity is defined with the S content in wt% S, not wt% SO4 (BW22, p. 160)
        logs_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_basalt_boulliung23: Solubility = _S2_sulfate_basalt_boulliung23()
r"""Sulfur in basalt as sulfate, :math:`\mathrm{SO_4^{2-}}/\mathrm{S^{6+}}` :cite:p:`BW22,BW23corr`

Using the first equation in the abstract and the corrected expression for sulfate capacity
(:math:`C_{\mathrm{S^{6+}}}`) in :cite:t:`BW23corr`. Composition for Basalt from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure, temperatures from 1473-1773 K for
silicate melts equilibrated with Air/SO2 mixtures.
The sulfate capacity is defined with the S content in wt% S, not wt% SO4.
"""


class _S2_sulfide_basalt_boulliung23(Solubility):
    r"""Sulfur in basalt as sulfide (:math:`\mathrm{S^{2-}}`) :cite:p:`BW23`

    Using expressions in the abstract for S wt% and sulfide capacity (:math:`C_{\mathrm{S^{2-}}}`).
    Composition for basalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure
    and temperatures from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were
    greater than 1 log unit below FMQ.
    """

    requires_fO2: ClassVar[bool] = True

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fO2 = jnp.asarray(fO2)
        logcs: Array = 0.225 - (8045.7465 / as_j64(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_basalt_boulliung23: Solubility = _S2_sulfide_basalt_boulliung23()
r"""Sulfur in basalt as sulfide (:math:`\mathrm{S^{2-}}`) :cite:p:`BW23`

Using expressions in the abstract for S wt% and sulfide capacity (:math:`C_{\mathrm{S^{2-}}}`).
Composition for basalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure and
temperatures from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater
than 1 log unit below FMQ.
"""


class _S2_basalt_boulliung23(Solubility):
    """Sulfur in basalt due to sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    requires_fO2: ClassVar[bool] = True

    _sulfide: Solubility
    _sulfate: Solubility

    def __init__(self):
        self._sulfide = S2_sulfide_basalt_boulliung23
        self._sulfate = S2_sulfate_basalt_boulliung23

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        fO2 = jnp.asarray(fO2)
        concentration: ArrayLike = self._sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )
        concentration = concentration + self._sulfate.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )

        return concentration


S2_basalt_boulliung23: Solubility = _S2_basalt_boulliung23()
"""Sulfur in basalt due to sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""


class _S2_sulfate_trachybasalt_boulliung23(Solubility):
    r"""Sulfur as sulfate :math:`\mathrm{SO_4^{2-}}/\mathrm{S^{6+}}` in trachybasalt
    :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (:math:`C_{\mathrm{S^{6+}}}`) in :cite:t:`BW23corr`. Composition for
    trachybasalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for
    silicate melts equilibrated with Air/SO2 mixtures.
    The sulfate capacity is defined with the S content in wt% S, not wt% SO4.
    """

    requires_fO2: ClassVar[bool] = True

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fO2 = jnp.asarray(fO2)
        logcs: Array = -12.948 + (32446.366 / as_j64(temperature))
        # The sulfate capacity is defined with the S content in wt% S, not wt% SO4 (BW22, p. 160)
        logs_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_trachybasalt_boulliung23: Solubility = _S2_sulfate_trachybasalt_boulliung23()
r"""Sulfur as sulfate :math:`\mathrm{SO_4^{2-}}/\mathrm{S^{6+}}` in trachybasalt
:cite:p:`BW22,BW23corr`

Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for sulfate
capacity (:math:`C_{\mathrm{S^{6+}}}`) in :cite:t:`BW23corr`. Composition for trachybasalt from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
equilibrated with Air/SO2 mixtures.
The sulfate capacity is defined with the S content in wt% S, not wt% SO4.
"""


class _S2_sulfide_trachybasalt_boulliung23(Solubility):
    r"""Sulfur as sulfide (:math:`\mathrm{S^{2-}}`) in trachybasalt :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity
    (:math:`C_{\mathrm{S^{2-}}}`). Composition for trachybasalt from :cite:t:`BW22{Table 1}`.
    Experiments conducted at 1 atm, 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2
    conditions were greater than 1 log unit below FMQ.
    """

    requires_fO2: ClassVar[bool] = True

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        fO2 = jnp.asarray(fO2)
        logcs: Array = 0.225 - (7842.5 / as_j64(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_trachybasalt_boulliung23: Solubility = _S2_sulfide_trachybasalt_boulliung23()
r"""Sulfur as sulfide (:math:`\mathrm{S^{2-}}`) in trachybasalt :cite:p:`BW23`

Using expressions in the abstract for S wt.% and sulfide capacity (:math:`C_{\mathrm{S^{2-}}}`).
Composition for trachybasalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm,
1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below
FMQ.
"""


class _S2_trachybasalt_boulliung23(Solubility):
    """Sulfur in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    requires_fO2: ClassVar[bool] = True

    _sulfide: Solubility
    _sulfate: Solubility

    def __init__(self):
        self._sulfide = S2_sulfide_trachybasalt_boulliung23
        self._sulfate = S2_sulfate_trachybasalt_boulliung23

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        fO2 = jnp.asarray(fO2)
        concentration: ArrayLike = self._sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )
        concentration = concentration + self._sulfate.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )

        return concentration


S2_trachybasalt_boulliung23: Solubility = _S2_trachybasalt_boulliung23()
"""Sulfur in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""
