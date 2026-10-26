# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Physical and numerical constants

This module defines reference thermodynamic conditions and numerical limits.
"""

# Thermodynamic standard state
TEMPERATURE_REFERENCE: float = 298.15
"""Enthalpy reference temperature (K) (:math:`T_r` in the JANAF tables) :cite:p:`MZG02,Cha98`"""
STANDARD_PRESSURE: float = 1.0
"""Standard state pressure (bar)"""
STANDARD_FUGACITY: float = STANDARD_PRESSURE
"""Standard fugacity for gases (bar)"""
GAS_STATE: str = "g"
"""Suffix to identify gases as per JANAF convention for the state of aggregation"""
SOLID_STATE: str = "s"
"""Suffix to identify solids as per JANAF convention for the state of aggregation"""
DISSOLVED_STATE: str = "d"
"""Suffix to identify dissolved species for output purposes"""

# Numerical floor for dissolved-species solubility in dissolution reactions
DISSOLUTION_PPMW_FLOOR: float = 1.0e-20
"""Minimum dissolved concentration (ppmw) used before taking logs in dissolution reactions.

Numerical floor for dissolved-species solubility in dissolution reactions. This prevents ``log(0)``
while keeping the floor large enough to avoid excessive solver stiffness in underflow-prone
regimes.
"""

# Penalty for a species outside the temperature range of its thermodynamic data
NO_DATA_GIBBS_OVER_RT: float = 100.0
"""Standard Gibbs energy over RT added to a species outside the temperature range of its data
when extrapolation is off.

This reduces the equilibrium activity of the species by a factor of :math:`\\exp(-100)`, so that
it is not stable. The value is a numerical margin, not a thermodynamic estimate. Liquid water is
the most demanding case: with its Gibbs energy held at the 600 K value, even :math:`10^5` bar of
H2O is supersaturated by only :math:`\\ln(10^5/124) \\approx 7`, where 124 bar is the saturation
pressure at 600 K. In the TRAPPIST-1e example, values of 10, 100 and 1000 give the same
equilibrium, and only the trace amount of the suppressed condensate changes (roughly as the
inverse of the value).
"""

# Phase indices. These are used to index into the phase system. TODO: Make Atmodeller completely
# agnostic to phase ordering.
GAS_PHASE_INDEX: int = 0
"""Gas phase index"""
SILICATE_MELT_PHASE_INDEX: int = 1
"""Silicate melt phase index"""
SILICATE_SOLID_PHASE_INDEX: int = 2
"""Silicate solid phase index"""
METAL_PHASE_INDEX: int = 3
"""Metal phase index"""

# Default lower and upper bounds on the hypercube which contains the root, used by
# atmodeller.containers.SolverParameters. These are somewhat empirically calibrated to bound the
# expected values for typical models, and can be adjusted for edge cases through SolverParameters.
LOG_NUMBER_MOLES_LOWER: float = -400.0
"""Lower log number of moles for a species

Minor species can be far below -200 at equilibrium, e.g. CH4 at about -220 in an O2-rich
atmosphere over liquid water at about 370 K.
"""
LOG_NUMBER_MOLES_UPPER: float = 80.0
"""Upper log number of moles for a species"""
LOG_STABILITY_LOWER: float = -700.0
r"""Lower stability for a species

Derived to ensure that the exponential function :math:`\exp(x)` does not underflow to zero
"""
LOG_STABILITY_UPPER: float = 35.0
"""Upper stability for a species

Empirically determined.
"""
# Default tau settings for species stability, used by atmodeller.containers.SolverParameters (tau,
# tau_max and tau_num), where they can be adjusted.
TAU_MAX: float = 1.0e-3
"""Maximum tau scaling factor for species stability, at the start of the tau sweep"""
TAU: float = 1.0e-25
"""Desired (i.e. final/minimum) tau scaling factor for species stability :cite:p:`LKK16`.

Tau effectively controls the minimum non-zero number of moles of unstable species. Formally, it
defines the number of moles of an unstable pure condensate with an activity of ``1/e``, which
corresponds to a log stability of zero.
"""
TAU_NUM: int = 2
"""Number of tau values solved between :data:`TAU_MAX` and :data:`TAU` (inclusive).

Used by the tau sweep. Empirically, once a solution has been found for :data:`TAU_MAX`, the solver
can typically proceed directly to :data:`TAU` and converge within a few steps on the first
attempt.
"""
