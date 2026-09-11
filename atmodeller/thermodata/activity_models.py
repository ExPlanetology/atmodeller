# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Custom activity models for specific species in specific phases."""

from dataclasses import KW_ONLY

import jax.numpy as jnp
from jaxtyping import ArrayLike

from atmodeller.jax_utils import FloatArray
from atmodeller.thermodata.core import ActivityCoefficient

T0_BADRO15: float = 1873.0
"""Reference temperature (K) at which :cite:t:`Badro2015`'s Table S1 interaction-parameter
regressions are tabulated"""
LN_GAMMA_SI0_T0_BADRO15: float = -6.65
r""":math:`\ln \gamma_{\rm Si}^0` at :data:`T0_BADRO15`, i.e. Si's activity coefficient at
infinite dilution in Fe, from :cite:t:`Badro2015` Table S1"""
EPSILON_SI_SI_T0_BADRO15: float = 12.41
r"""Si self-interaction parameter :math:`\varepsilon_{\rm Si}^{\rm Si}` at :data:`T0_BADRO15`,
from :cite:t:`Badro2015` Table S1"""


class Si_iron_badro15(ActivityCoefficient):
    r"""Activity coefficient of Si in Fe :cite:p:`Badro2015`.

    Reduces the general multicomponent interaction-parameter formalism of :cite:t:`Badro2015`'s
    Eqns. 3-4 to a Si-in-Fe system under two assumptions:

    1. Fe (the solvent) mixes ideally, i.e. :math:`\gamma_{\rm Fe} = 1` at any composition, so
       the :math:`\ln \gamma_{\rm Fe}` term in Eqn. 4 vanishes by default. This is a weaker
       assumption than requiring Si to be dilute: it holds at any :math:`X_{\rm Si}`, at the cost
       of neglecting how Fe's own non-ideality (Eqn. 3) actually varies with composition. The
       ``gamma`` argument below is exactly this :math:`\gamma_{\rm Fe}` term (not an arbitrary
       scale on Si's own coefficient): it multiplies the result, and setting it away from its
       default of 1 relaxes the ideal-Fe assumption to a caller-supplied value.
    2. Cross-interaction terms with any other solute (e.g. O) are neglected; only Si's
       self-interaction parameter :math:`\varepsilon_{\rm Si}^{\rm Si}` is retained. This is
       exact if Si is the only solute in the metal phase, and approximate otherwise.

    This gives:

    .. math::
        \ln \gamma_{\rm Si}(T, X_{\rm Si}) = \ln \gamma_{\rm Si}^0(T)
            - \varepsilon_{\rm Si}^{\rm Si}(T) \ln(1 - X_{\rm Si})

    where :math:`\ln \gamma_{\rm Si}^0(T)` and :math:`\varepsilon_{\rm Si}^{\rm Si}(T)` are
    extrapolated from their values at :data:`T0_BADRO15` (:data:`LN_GAMMA_SI0_T0_BADRO15`,
    :data:`EPSILON_SI_SI_T0_BADRO15`) via :cite:t:`Badro2015`'s stated :math:`T_0/T` scaling.

    Args:
        gamma: Fe's own activity coefficient :math:`\gamma_{\rm Fe}` (see assumption 1 above).
            Effectively a multiplicative factor on the returned activity coefficient, but not an
            arbitrary one: it is Eqn. 4's :math:`\gamma_{\rm Fe}` term. Defaults to 1 (ideal Fe).
        Si_index: Index of silicon in the mole fraction array for the metal species
    """

    _: KW_ONLY
    Si_index: int
    """Index of silicon in the mole fraction array for the metal species"""

    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: FloatArray | None = None
    ) -> FloatArray:
        """Log of the activity coefficient (dimensionless).

        This is the primary access point for calling the EOS within the main engine so must adhere
        to the expected interface for activity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions. Defaults to ``None`` if unused.

        Returns:
            Log activity coefficient
        """
        del pressure

        if mole_fractions is None:
            raise ValueError("Mole fractions must be provided for this activity model.")

        ln_gamma_Si0: ArrayLike = LN_GAMMA_SI0_T0_BADRO15 * T0_BADRO15 / temperature
        epsilon: ArrayLike = EPSILON_SI_SI_T0_BADRO15 * T0_BADRO15 / temperature
        x_Si: FloatArray = mole_fractions[..., self.Si_index]

        # Fabian's original test
        # return jnp.log(self.gamma * 0.3 * x_si)

        return jnp.log(self.gamma) + ln_gamma_Si0 - epsilon * jnp.log(1 - x_Si)
