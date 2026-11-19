# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Element-potential initializer for the solver's initial guess

Provides :class:`ElementPotentialInitializer`, the default initializer of
:class:`~atmodeller.classes.EquilibriumModel`, which solves an ideal-gas equilibrium in the element
potentials. See its docstring for the model and method.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Integer

from atmodeller.constants import GAS_PHASE_INDEX
from atmodeller.engine import compute_implied_log_stability
from atmodeller.initializer.core import (
    CONDENSATE_ALLOCATION_FRACTION,
    LOG_TRACE_VALUE,
    max_moles_by_limiting_element,
)
from atmodeller.jax_utils import safe_exp
from atmodeller.parameters import Parameters
from atmodeller.phases import BasePhase

SWEEP_MIN_SLOPE: float = 0.1
"""Floor on the slope of each element's 1-D mass balance in the sweep

The slope is the mean of d ln n / d lambda_e over the carriers of the element, weighted by their
shares. It is at least one for gas species, but a dissolved species can contribute much less (the
exponent of its solubility law, or zero at the mole-fraction cap), so when the melt holds most of
an element the slope can approach zero. The floor bounds the step by ``|f| / SWEEP_MIN_SLOPE``; it
acts only when the true slope is smaller, and then shortens the step without changing its
direction."""

SWEEP_MAX_STEP: float = 10.0
"""Largest change in lambda_e per iteration of each element's 1-D Newton in the sweep

With gas carriers only the mass balance is convex with slope at least one, so a Newton step is at
most ``|f|`` and this rarely acts. With dissolved carriers it couples to the pressure and fO2
through the solubility laws and can overshoot from far away; the clip bounds that."""


def _gibbs_over_RT_all_species(
    parameters: Parameters, temperature: Array, pressure: Array
) -> Float[Array, " n_species"]:
    """Standard Gibbs energy over RT for every species, zero for species not in the reactions

    Args:
        parameters: Parameters of a single system
        temperature: Temperature (K)
        pressure: Pressure (bar)

    Returns:
        Standard Gibbs energy over RT in the full species space
    """
    species = parameters.reaction_system.species
    reaction_names: list[str] = list(species.reaction_species.species_names)
    gibbs_reaction: Float[Array, " n_reaction_species"] = (
        parameters.reaction_system.reaction.standard_gibbs_over_RT(temperature, pressure)
    )

    return jnp.stack(
        [
            gibbs_reaction[reaction_names.index(name)] if name in reaction_names else 0.0
            for name in species.species_names
        ]
    )


class ElementPotentialInitializer(eqx.Module):
    r"""Initializer that solves an ideal-gas equilibrium in the element potentials

    The cold guess is the solution of a simplified equilibrium problem: ideal gas, pure
    condensates and standard Gibbs energies held fixed, solved for the element potentials. This
    is the default initializer of :class:`~atmodeller.classes.EquilibriumModel`.

    Notation: :math:`\lambda_e` is the potential of element :math:`e` in units of :math:`RT`,
    :math:`a_{es}` the formula matrix (moles of element :math:`e` in species :math:`s`),
    :math:`G_s/RT` the standard Gibbs energy of species :math:`s`, :math:`b_e` the element
    budget (moles), :math:`n_s` and :math:`p_s` the moles and partial pressure of species
    :math:`s`, :math:`N_\mathrm{gas}` the total gas moles, and
    :math:`d_c = \sum_e a_{ec} \lambda_e - G_c/RT` the driving force of condensate :math:`c`
    (positive if supersaturated).

    .. rubric:: Model

    - Gas: ideal, :math:`\ln p_s = \sum_e a_{es} \lambda_e - G_s/RT` and
      :math:`n_s = (p_s / \sum_\mathrm{gas} p)\, N_\mathrm{gas}`.
    - Condensates: pure phases; a present condensate has unit activity, :math:`d_c = 0`.
    - Dissolved species (if the system has dissolution reactions): their log moles are
      unknowns, and the dissolution rows of the solver's residual (the solubility laws) are part
      of the residual. In the element sweep they follow the potentials (see below).
    - :math:`G_s/RT` is evaluated at the pressure of the start. The gas values are for the 1-bar
      standard state and do not change; the condensate values are re-evaluated at the pressure
      reached after each switching pass (see below), and are held fixed within each solve.

    The unknowns are :math:`z = [\lambda, \ln N_\mathrm{gas}, \text{one slot per species}]`,
    where the slots hold :math:`\ln n` of present condensates and dissolved species (the others
    are unused). The residual has a fixed shape, with inapplicable rows zero:

    .. math::

        \begin{array}{ll}
        \ln \left( \sum_s a_{es} n_s / b_e \right) & \text{mass balance, constrained elements} \\
        \ln p_s - \ln f_s & \text{imposed gas fugacities} \\
        d_c & \text{present condensates} \\
        \ln n_c - \texttt{log\_trace\_value} & \text{absent condensate slots} \\
        \ln \sum_\mathrm{gas} p - \ln P(n) & \text{pressure closure} \\
        \text{dissolution residual} & \text{dissolved species}
        \end{array}

    The mass balance sums gas, present condensates and dissolved species. The pressure closure
    and imposed fugacities are needed: the pressure of a planet depends on its atmosphere, and
    an element that is not mass constrained (e.g. O under an fO\ :sub:`2` buffer) has its
    potential fixed only by an imposed fugacity.

    .. rubric:: Method

    1. Start from each gas and dissolved species' share of its limiting element,
       :math:`\min_e b_e / \sum_{s' \ni e} a_{es'}` over the mass-constrained elements of the
       species (the sum over every gas and dissolved species containing :math:`e`), so that the
       start satisfies every element budget. :math:`\lambda` and the normalisation are fitted by
       least squares to the gas shares and to the imposed fugacities (:math:`\sum_e a_{es}
       \lambda_e - G_s/RT = \ln f_s`), so an element that is not mass constrained starts from
       its target. Gas species with neither are left out of the fit. The dissolved slots start at
       their shares, and the condensate slots at trace. No knowledge of the answer is used. An
       even split of the total over the gas species (as CEA) is not used: for a skewed inventory
       (e.g. a primordial H/He atmosphere with trace C, N, O and S) it puts the dissolved species
       of the minor elements above their whole budgets.
    2. Phase selection in :math:`2 n_\mathrm{condensates} + 1` passes, then a final solve (a
       system without condensates has nothing to switch, and is solved once). Each pass solves,
       re-evaluates the condensate :math:`G_c/RT` at the pressure reached, switches on the most
       supersaturated absent condensate (if :math:`d_c > 10^{-6}`; it starts at
       :data:`~atmodeller.initializer.core.CONDENSATE_ALLOCATION_FRACTION` of the budget of its
       limiting element, as the heuristic initializer allocates), and switches off present
       condensates whose log moles fall below ``log_trace_value`` + 1. Switching one at a time
       means that a present polymorph fixes its element's potential and the other becomes
       undersaturated. The start pressure (from the shares) can differ from the converged
       pressure by an order of magnitude, enough to put a condensate near saturation on the wrong
       side of its boundary through the pressure dependence of its Gibbs energy, hence the
       re-evaluation.
    3. Every solve is an element sweep followed by damped Newton:

       - *Sweep* (FastChem): up to ``sweeps`` Gauss-Seidel passes over the elements in order
         of decreasing abundance, each solving its own element's mass balance for
         :math:`\lambda_e` (see below). Each 1-D Newton stops when its step is below
         ``sweep_step_tolerance`` (at most ``sweep_iterations``), and the passes stop when no
         :math:`\lambda_e` changes by more than ``sweep_tolerance``. The sweep only sets the
         scale of every species, and Newton then converges the coupled problem, so the sweep
         tolerances are loose: tightening them (to ``1e-10`` and ``1e-8``) gave the same guesses
         and solves on the test suite and on a stress family of magma-ocean planets, and made the
         guess about 1.5 times slower on the TRAPPIST-1e example with solubility.
       - *Newton*: least-squares direction from the damped normal equations (see below),
         scaled so that no :math:`\lambda_e` and no carried species' log moles change by more
         than ``max_log_step``; stops when the residual norm is below ``residual_tolerance``, or
         after ``newton_iterations``.

    4. Absent condensates are given the solver's stability representation: the solver's
       residual for an absent condensate is :math:`-\nu (d_c + \zeta_c)`, so
       :math:`\zeta_c = -d_c`, and its log moles follow from the implied-stability relation
       (absent condensates with :math:`d_c \geq 0` stay at trace).
    5. If the result is not finite, the start (the shares) is returned instead.

    .. rubric:: Why the sweep

    In the element-potential form a gas species is not an unknown. It enters Newton only
    through the mass-balance rows, with a sensitivity proportional to its share of the element
    budget. If it is driven far below the major carriers of its elements, that sensitivity
    underflows and Newton cannot bring it back. This is the *water trap*: with liquid water
    present and hydrogen slightly in excess of H\ :sub:`2`\ O stoichiometry, the excess
    hydrogen must be H\ :sub:`2` gas, but a start with H\ :sub:`2` at
    :math:`\ln n \approx -18` next to H\ :sub:`2`\ O\ :sub:`l` at 52.7 leaves the hydrogen
    balance only a few percent off, with a derivative of about :math:`e^{-70}` in the
    H\ :sub:`2` direction.

    FastChem (:cite:p:`Stock2018,Stock2022,KSP24`) solves for the same unknowns (its log
    free-atom densities differ from :math:`\lambda_e` by a constant) but by Gauss-Seidel over
    the elements: each element's own mass balance, solved on its own scale, so a minor species
    is found however small it starts. Gauss-Seidel alone converges slowly when elements are
    strongly coupled and knows nothing of condensates, fugacities or the pressure closure, so
    here it only sets the scale of every species, and Newton then solves the coupled problem.

    The number of sweeps is capped by ``sweeps``. Too few can leave a minor species far from its
    scale, as in the water trap but for O\ :sub:`2` when H\ :sub:`2`\ O dominates both the
    hydrogen and the oxygen budgets, since each sweep then moves it only a little. Newton cannot
    recover within ``newton_iterations``, and the guess can end with the wrong condensates. In the
    TRAPPIST-1e example at 280 K (1000 conditions), a cap of 3 left 12 guesses with the wrong
    condensates, whereas with 10 every guess was at the equilibrium, at a cost of about 10% in
    the run time of the guess for systems that are already easy. More sweeps without more Newton
    iterations were worse, because Newton then hit its cap in more switching passes.

    *The 1-D problem for element e.* Hold the other potentials, the moles in present condensates
    (:math:`C_e`) and the gas normalisation
    :math:`\sigma = \ln N_\mathrm{gas} - \ln \sum_\mathrm{gas} p` fixed, so that a gas species
    has :math:`\ln n_s = a_{es} x + \beta_s` with :math:`x = \lambda_e`. The carriers are the gas
    and the dissolved species, with the log mass balance

    .. math::

        f(x) = \ln \sum_{s \in \mathrm{gas} \cup \mathrm{dissolved},\, a_{es} > 0}
               a_{es}\, n_s(x) - \ln(b_e - C_e).

    A dissolved species follows the potentials like a gas species: its moles are set where its
    dissolution row is zero, at the gas fugacities :math:`\ln p_s = \sum_e a_{es} \lambda_e -
    G_s/RT` (and their sum as the pressure, as in the model above; the gas equations of state are
    not evaluated in the sweep), by two steps (the row has a slope of about one in
    :math:`\ln n_d`), with its mole fraction capped at one and its log moles within the solver's
    bounds (``log_number_moles_lower`` and ``log_number_moles_upper`` of the solver
    parameters). Without the cap,
    when the dissolved species are not part of the melt's mass, a solubility law can ask for a
    mole fraction above one, and the moles run away. Condensates are the dual case: a present
    condensate has unit activity, so its equilibrium fixes the potentials and not its amount, and
    it is held fixed in the sweep. With gas carriers only, :math:`f'(x) = \sum_s w_s a_{es} \geq
    1` and :math:`f''(x) \geq 0` (mean and variance of :math:`a_{es}` under the weights
    :math:`w_s = a_{es} n_s / \sum a n`): :math:`f` is convex and increasing, with one root that
    Newton reaches from any start. The slope is taken from a forward-mode derivative, floored at
    :data:`~atmodeller.initializer.element_potential.SWEEP_MIN_SLOPE`, and each step is clipped to
    :data:`~atmodeller.initializer.element_potential.SWEEP_MAX_STEP`. Freezing :math:`\sigma` is
    FastChem's fixed total density: it keeps each 1-D problem separate. After each sweep the
    dissolved moles are set at the swept potentials, and :math:`\ln N_\mathrm{gas}` is reset so
    that the normalised gas moles equal the swept ones. An element is skipped if it is not mass
    constrained, if :math:`C_e \geq b_e`, or if no gas species contains it. The sweep ignores
    every row other than the mass balances and the dissolution rows.

    .. rubric:: Newton direction

    The fixed-shape system has zero rows (inactive constraints, such as an element without a mass
    balance or a gas without an imposed fugacity, which may differ between batch elements) and
    zero columns (the slots of gas species, whose moles follow from :math:`\lambda`). The
    Jacobian :math:`J` is therefore taken only over the *live* unknowns: :math:`\lambda`,
    :math:`\ln N_\mathrm{gas}`, and the slots of condensates and dissolved species. These are
    fixed by the species and found once, outside JIT. The direction solves the damped normal
    equations

    .. math::

        (J^\mathsf{T} J + \mu^2 I)\, \Delta z = -J^\mathsf{T} r

    by Cholesky factorisation (``damping`` is :math:`\mu`). Zero rows drop out of
    :math:`J^\mathsf{T} J`, so every constraint row is kept and constraints can differ between
    batch elements. The damping keeps the matrix positive definite in degenerate steps, such as an
    element that is neither mass constrained nor fixed by a fugacity, where it gives a zero step,
    as the minimum-norm least-squares solution would. Compared with a singular value decomposition
    of the full Jacobian, this needs fewer Jacobian columns and a much cheaper factorisation.

    .. rubric:: JAX

    The guess is computed per system and vmapped over the batch by the solver. The switching
    passes and the loop over elements in a sweep have fixed counts (``lax.fori_loop``). The
    sweep passes, the 1-D Newton iterations, and each Newton solve are ``lax.while_loop``, which
    under vmap run until every system in the batch has converged (or hit the cap). The element
    order comes from ``argsort`` of the (possibly traced) abundances. Gradients are stopped
    through the guess, since the equilibrium solve is differentiated implicitly (and
    ``while_loop`` is not reverse-differentiable).

    Args:
        newton_iterations: Maximum number of damped Newton iterations per solve. Defaults to
            ``100``.
        residual_tolerance: Residual norm at which a Newton solve stops. Defaults to ``1e-10``.
        max_log_step: Maximum change in the log moles of any carried species (or in any
            :math:`\lambda_e`) per Newton iteration. Defaults to ``4.0``.
        sweeps: Maximum number of Gauss-Seidel element sweeps at the start of every Newton
            solve. Defaults to ``10``.
        sweep_iterations: Maximum number of 1-D Newton iterations per element in a sweep.
            Defaults to ``30``.
        sweep_step_tolerance: Step size at which a 1-D Newton iteration in a sweep stops.
            Defaults to ``1e-4``.
        sweep_tolerance: Largest change in any :math:`\lambda_e` over a sweep below which the
            sweeps stop. Defaults to ``1e-2``.
        damping: Damping :math:`\mu` of the normal equations for the Newton direction. Defaults
            to ``1e-6``.
        log_trace_value: Log number of moles of absent species (absent condensates, and species
            outside the reaction network). Defaults to
            :data:`~atmodeller.initializer.core.LOG_TRACE_VALUE`.
    """

    newton_iterations: int = 100
    """Maximum number of damped Newton iterations per solve"""
    residual_tolerance: float = 1e-10
    """Residual norm at which a Newton solve stops"""
    max_log_step: float = 4.0
    r"""Maximum change in the log moles of any carried species (or in any :math:`\lambda_e`) per
    Newton iteration"""
    sweeps: int = 10
    """Maximum number of Gauss-Seidel element sweeps at the start of every Newton solve"""
    sweep_iterations: int = 30
    """Maximum number of 1-D Newton iterations per element in a sweep (each step is at most the
    log residual)"""
    sweep_step_tolerance: float = 1e-4
    """Step size at which a 1-D Newton iteration in a sweep stops"""
    sweep_tolerance: float = 1e-2
    r"""Largest change in any :math:`\lambda_e` over a sweep below which the sweeps stop"""
    damping: float = 1e-6
    r"""Damping :math:`\mu` of the normal equations for the Newton direction, which keeps
    :math:`J^\mathsf{T} J + \mu^2 I` positive definite in degenerate steps"""
    log_trace_value: float = LOG_TRACE_VALUE
    """Log number of moles of absent species"""

    def start(self, parameters: Parameters) -> Float[Array, " twice_species"]:
        """Cold initial guess from the element-potential equilibrium.

        Args:
            parameters: Parameters of a single system

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        # The guess needs no gradients (the solve is differentiated implicitly), and reverse-mode
        # differentiation cannot pass through the while loops
        parameters = jax.tree_util.tree_map(
            lambda x: lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, parameters
        )

        # -----------------------------------------------------------------------------------
        # STEP 1. Setup: species classes, formula matrix, element budgets, constraints.
        #
        # Gas species and condensates are in the reaction network; dissolved species are not
        # (they have no standard Gibbs energy) and are handled through the dissolution rows.
        # Unconstrained elements have a NaN budget.
        # -----------------------------------------------------------------------------------
        reaction_system = parameters.reaction_system
        species = reaction_system.species
        reaction_names: list[str] = list(species.reaction_species.species_names)
        n_species: int = species.number_species
        formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
            reaction_system.formula_matrix, dtype=float
        )
        n_elements: int = formula_matrix.shape[0]
        in_reaction: np.ndarray = np.array(
            [name in reaction_names for name in species.species_names]
        )
        gas: Bool[Array, " n_species"] = (
            jnp.asarray(reaction_system.phase_system.gas_species_mask) & in_reaction
        )
        condensate: Bool[Array, " n_species"] = (
            jnp.asarray(reaction_system.phase_system.condensates_species_mask) & in_reaction
        )
        has_dissolution: bool = reaction_system.dissolution.number_reactions > 0
        dissolved: Bool[Array, " n_species"] = jnp.asarray(~in_reaction & has_dissolution)
        temperature: Array = jnp.asarray(parameters.state.temperature)
        abundance: Float[Array, " n_elements"] = jnp.reshape(
            parameters.mass_constraints.abundance(), (n_elements,)
        )
        constrained: Bool[Array, " n_elements"] = ~jnp.isnan(abundance) & (abundance > 0)
        log_b: Float[Array, " n_elements"] = jnp.log(jnp.where(constrained, abundance, 1.0))
        imposed: Bool[Array, " n_species"] = gas & parameters.activity_constraints.active()

        # Live unknowns of z = [lambda, ln N_gas, slots] for the Newton direction: lambda,
        # ln N_gas, and the slots of condensates and dissolved species (gas-species slots are
        # never used, so their Jacobian columns are always zero). Built with NumPy from the gas
        # mask (from the static phase structure) and in_reaction, so the indices are static
        # under JIT; constraint-dependent masks are traced and are not used here.
        gas_static: np.ndarray = np.asarray(reaction_system.phase_system.gas_species_mask)
        slot_used: np.ndarray = (~gas_static & in_reaction) | (~in_reaction & has_dissolution)
        live: np.ndarray = np.concatenate(
            [np.arange(n_elements + 1), n_elements + 1 + np.flatnonzero(slot_used)]
        )

        # -----------------------------------------------------------------------------------
        # STEP 2. Proportional start and standard Gibbs energies at the start pressure. Each gas
        # and dissolved species starts at its share of its limiting mass-constrained element, so
        # that the start satisfies every element budget; species without a mass-constrained
        # element have no share (budgeted is False). The condensate G/RT is updated after each
        # switching pass (STEP 7), so G/RT is an argument of the functions below.
        # -----------------------------------------------------------------------------------
        share: Float[Array, " n_species"] = max_moles_by_limiting_element(
            formula_matrix, jnp.where(constrained, abundance, jnp.nan), gas | dissolved
        )
        budgeted: Bool[Array, " n_species"] = (gas | dissolved) & (share > 0)
        log_share: Float[Array, " n_species"] = jnp.log(jnp.where(budgeted, share, 1.0))
        start_log_n: Float[Array, " n_species"] = jnp.where(
            budgeted, log_share, self.log_trace_value
        )
        start_pressure: Array = parameters.state.get_pressure(start_log_n)
        gibbs: Float[Array, " n_species"] = _gibbs_over_RT_all_species(
            parameters, temperature, start_pressure
        )

        # -----------------------------------------------------------------------------------
        # STEP 3. Species moles from the unknowns z = [lambda, ln N_gas, slots].
        #
        # Gas from the element potentials (normalised to N_gas); present condensates and
        # dissolved species from their slots; absent condensates at trace.
        # -----------------------------------------------------------------------------------
        def log_moles(z: Array, present: Array, gibbs: Array) -> tuple[Array, Array, Array]:
            lam, log_n_gas, slots = z[:n_elements], z[n_elements], z[n_elements + 1 :]
            # Not clipped, since a clip would zero the derivatives of the log-space rows (fugacity,
            # pressure closure, dissolution); logsumexp is stable, and moles are formed with
            # safe_exp
            log_p: Array = formula_matrix.T @ lam - gibbs
            log_p_sum: Array = logsumexp(jnp.where(gas, log_p, -jnp.inf))
            log_n: Array = jnp.where(
                gas,
                log_p - log_p_sum + log_n_gas,
                jnp.where((condensate & present) | dissolved, slots, self.log_trace_value),
            )
            return log_n, log_p, log_p_sum

        # -----------------------------------------------------------------------------------
        # STEP 4. Residual (fixed shape; inapplicable rows zero). See the class docstring.
        #
        # The mass balance is a log ratio, so every element weighs the same whatever its
        # abundance. Fugacity targets may depend on pressure.
        # -----------------------------------------------------------------------------------
        def residual(z: Array, present: Array, gibbs: Array) -> Array:
            lam, slots = z[:n_elements], z[n_elements + 1 :]
            log_n, log_p, log_p_sum = log_moles(z, present, gibbs)
            pressure: Array = parameters.state.get_pressure(log_n)
            carried: Array = gas | (condensate & present) | dissolved
            element_moles: Array = formula_matrix @ jnp.where(carried, safe_exp(log_n), 0.0)
            mass: Array = jnp.where(
                constrained, jnp.log(jnp.where(constrained, element_moles, 1.0)) - log_b, 0.0
            )
            target: Array = parameters.activity_constraints.log_activity(temperature, pressure)
            fugacity: Array = jnp.where(imposed, log_p - jnp.where(imposed, target, 0.0), 0.0)
            condensed: Array = jnp.where(
                condensate & present,
                formula_matrix.T @ lam - gibbs,
                jnp.where(condensate, slots - self.log_trace_value, 0.0),
            )
            closure: Array = jnp.atleast_1d(log_p_sum - jnp.log(pressure))
            rows: list[Array] = [mass, fugacity, condensed, closure]
            if has_dissolution:
                # The solver's dissolution rows
                rows.append(
                    reaction_system.get_dissolution_residual(
                        log_n,
                        reaction_system.get_log_activity(log_n, temperature, pressure),
                        temperature,
                        pressure,
                    )
                )
            return jnp.concatenate(rows)

        # -----------------------------------------------------------------------------------
        # STEP 5. Element sweep (FastChem): each element's own mass balance, solved for
        # lambda_e.
        #
        # See the class docstring for the 1-D problem. The carriers are the gas and the dissolved
        # species, whose moles both follow the potentials (the dissolved species through their
        # solubility laws). Held fixed: the other potentials (updated in Gauss-Seidel order, most
        # abundant element first), the moles in present condensates and the gas normalisation
        # sigma.
        # -----------------------------------------------------------------------------------
        order: Array = jnp.argsort(-jnp.where(constrained, abundance, -jnp.inf))
        # ln a_es; the placeholder 1.0 where a_es = 0 is masked out by the carrier mask
        log_a: Array = jnp.log(jnp.where(formula_matrix > 0, formula_matrix, 1.0))
        # The dissolution row of each dissolved species, and whether it has one (no rows, nothing
        # dissolved, without dissolution). Traced under vmap (a leaf of the parameters), so
        # jax.numpy throughout
        dissolution_row_of: Integer[Array, " n_species"] = jnp.zeros(n_species, dtype=int)
        has_dissolution_row: Bool[Array, " n_species"] = jnp.zeros(n_species, dtype=bool)
        if has_dissolution:
            dissolution_matrix: Float[Array, "n_dissolution n_species"] = jnp.asarray(
                reaction_system.dissolution.dissolution_matrix
            )
            dissolution_row_of = jnp.argmax(dissolution_matrix > 0, axis=0)
            has_dissolution_row = dissolved & jnp.any(dissolution_matrix > 0, axis=0)
        # Bounds on the log number of moles of a dissolved species, as for the solver
        log_n_lower: float = parameters.solver_parameters.log_number_moles_lower
        log_n_upper: float = parameters.solver_parameters.log_number_moles_upper

        phases: tuple[BasePhase, ...] = reaction_system.phase_system.phases

        def condensed_log_activity(
            log_n: Float[Array, " n_species"],
            log_p_gas: Float[Array, " n_species"],
            pressure: Array,
        ) -> Float[Array, " n_species"]:
            """Log activities with the gas at its ideal partial pressures, ``log_p_gas``, and the
            other phases from their own activity models. The gas equation of state is not
            evaluated."""
            parts: list[Array] = []
            for phase_index, phase in enumerate(phases):
                phase_slice: slice = reaction_system.phase_system.phase_slice(phase_index)
                if phase_index == GAS_PHASE_INDEX:
                    parts.append(log_p_gas[phase_slice])
                else:
                    parts.append(phase.get_log_activity(log_n[phase_slice], temperature, pressure))
            return jnp.concatenate(parts)

        def equilibrate_dissolved(
            log_n: Float[Array, " n_species"], log_p_gas: Float[Array, " n_species"]
        ) -> Float[Array, " n_species"]:
            """Sets ln n of every dissolved species where its dissolution row is zero, by two
            Newton-like steps (slope about one in ln n_d), with the mole fraction capped at one.

            The gas activities are the ideal partial pressures from the potentials, ``log_p_gas``
            (a_s.lambda - G_s/RT), and the pressure is their sum, as in the element-potential
            model, so that the dissolved moles follow the potentials.
            """
            if not has_dissolution:
                return log_n
            pressure: Array = jnp.exp(logsumexp(jnp.where(gas, log_p_gas, -jnp.inf)))
            for _ in range(2):
                log_activity: Array = condensed_log_activity(log_n, log_p_gas, pressure)
                step: Array = reaction_system.get_dissolution_residual(
                    log_n, log_activity, temperature, pressure
                )[dissolution_row_of]
                # The mole fraction is capped at one, i.e. ln n_d at most ln n_d - ln a_d: when
                # the dissolved species are not part of the melt's mass, a solubility law can ask
                # for a mole fraction above one, and the moles would run away
                new_log_n: Array = jnp.minimum(log_n - step, log_n - log_activity)
                log_n = jnp.where(
                    has_dissolution_row & jnp.isfinite(step),
                    jnp.clip(new_log_n, log_n_lower, log_n_upper),
                    log_n,
                )
            return log_n

        def sweep(z: Array, present: Array, gibbs: Array) -> Array:
            log_n_now, _, log_p_sum = log_moles(z, present, gibbs)
            sigma: Array = z[n_elements] - log_p_sum
            in_condensates: Array = formula_matrix @ jnp.where(
                condensate & present, safe_exp(log_n_now), 0.0
            )

            def log_n_at(
                lam: Float[Array, " n_elements"], slots: Float[Array, " n_species"]
            ) -> Float[Array, " n_species"]:
                """Species moles at the potentials ``lam``: the gas from lambda and sigma, the
                dissolved species by their solubility laws, present condensates from ``slots``"""
                log_p_gas: Array = formula_matrix.T @ lam - gibbs
                log_n: Array = jnp.where(
                    gas,
                    log_p_gas + sigma,
                    jnp.where((condensate & present) | dissolved, slots, self.log_trace_value),
                )
                return equilibrate_dissolved(log_n, log_p_gas)

            def element(k: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                lam, slots = carry
                e: Array = order[k]
                a_e: Array = formula_matrix[e]
                carrier: Array = (gas | dissolved) & (a_e > 0)
                available: Array = abundance[e] - in_condensates[e]
                solvable: Array = constrained[e] & (available > 0) & jnp.any(gas & (a_e > 0))
                log_target: Array = jnp.log(jnp.where(solvable, available, 1.0))

                def mass_balance(x: Array) -> Array:
                    log_n: Array = log_n_at(lam.at[e].set(x), slots)
                    return logsumexp(jnp.where(carrier, log_a[e] + log_n, -jnp.inf)) - log_target

                # 1-D Newton until the step is below sweep_step_tolerance. The slope (from a jvp)
                # is at least one for gas carriers; SWEEP_MIN_SLOPE and SWEEP_MAX_STEP are guards
                def newton(carry_: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                    i, x, _ = carry_
                    value, slope = jax.jvp(mass_balance, (x,), (jnp.ones_like(x),))
                    dx: Array = jnp.clip(
                        value / jnp.maximum(slope, SWEEP_MIN_SLOPE),
                        -SWEEP_MAX_STEP,
                        SWEEP_MAX_STEP,
                    )
                    dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
                    return i + 1, x - dx, dx

                def unfinished(carry_: tuple[Array, Array, Array]) -> Array:
                    i, _, dx = carry_
                    return (i < self.sweep_iterations) & (jnp.abs(dx) > self.sweep_step_tolerance)

                x: Array = lax.while_loop(
                    unfinished, newton, (jnp.array(0), lam[e], jnp.array(jnp.inf))
                )[1]
                return lam.at[e].set(jnp.where(solvable & jnp.isfinite(x), x, lam[e])), slots

            # Sweeps until no lambda_e changes by more than sweep_tolerance (at most sweeps), so a
            # state that is already converged costs a single sweep
            def sweeping(carry: tuple[Array, Array, Array, Array]) -> Array:
                i, _, _, change = carry
                return (i < self.sweeps) & (change > self.sweep_tolerance)

            def one_sweep(
                carry: tuple[Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array]:
                i, lam, slots, _ = carry
                lam_new, slots = lax.fori_loop(0, n_elements, element, (lam, slots))
                # The dissolved slots follow the potentials
                slots = jnp.where(dissolved, log_n_at(lam_new, slots), slots)
                return i + 1, lam_new, slots, jnp.max(jnp.abs(lam_new - lam))

            _, lam, slots, _ = lax.while_loop(
                sweeping,
                one_sweep,
                (jnp.array(0), z[:n_elements], z[n_elements + 1 :], jnp.array(jnp.inf)),
            )
            slots = jnp.where(dissolved, log_n_at(lam, slots), slots)
            # Reset ln N_gas so that the normalised gas moles of STEP 3 equal the swept ones
            log_p_sum_new: Array = logsumexp(
                jnp.where(gas, formula_matrix.T @ lam - gibbs, -jnp.inf)
            )
            return (
                z.at[:n_elements]
                .set(lam)
                .at[n_elements]
                .set(log_p_sum_new + sigma)
                .at[n_elements + 1 :]
                .set(slots)
            )

        # -----------------------------------------------------------------------------------
        # STEP 6. Solve: sweep, then damped Newton to residual_tolerance (at most
        # newton_iterations).
        #
        # Step control: no lambda and no carried species' log moles (via a jvp through STEP 3)
        # change by more than max_log_step. Without it, Newton on exponentials overshoots minor
        # species into flat regions where it stalls.
        # -----------------------------------------------------------------------------------
        def solve(z: Array, present: Array, gibbs: Array) -> Array:
            carried: Array = gas | (condensate & present) | dissolved

            def newton(carry):
                k, z, r = carry
                # Jacobian over the live unknowns only (the other entries of z are held fixed)
                jacobian = lx.materialise(
                    lx.JacobianLinearOperator(
                        lambda z_live, _: residual(z.at[live].set(z_live), present, gibbs), z[live]
                    )
                )
                # Damped normal equations (J^T J + mu^2 I) dz = -J^T r, solved by Cholesky.
                # Inactive constraint rows are zero and drop out of J^T J; the damping keeps the
                # operator positive definite if a direction is undetermined in this step.
                normal = jacobian.T @ jacobian + self.damping**2 * lx.IdentityLinearOperator(
                    jax.eval_shape(lambda: z[live])
                )
                normal = lx.TaggedLinearOperator(normal, lx.positive_semidefinite_tag)
                dz_live: Array = lx.linear_solve(
                    normal, -jacobian.T.mv(r), lx.Cholesky(), throw=False
                ).value
                # Unused slots get no step
                dz: Array = jnp.zeros_like(z).at[live].set(dz_live)
                dz = jnp.where(jnp.isfinite(dz), dz, 0.0)
                _, dlog_n = jax.jvp(lambda z_: log_moles(z_, present, gibbs)[0], (z,), (dz,))
                biggest: Array = jnp.maximum(
                    jnp.max(jnp.where(carried, jnp.abs(dlog_n), 0.0)),
                    jnp.max(jnp.abs(dz[:n_elements])),
                )
                scale: Array = jnp.minimum(1.0, self.max_log_step / jnp.maximum(biggest, 1e-12))
                z = z + scale * dz
                return k + 1, z, residual(z, present, gibbs)

            def unfinished(carry):
                k, _, r = carry
                return (k < self.newton_iterations) & (
                    jnp.linalg.norm(r) > self.residual_tolerance
                )

            z = sweep(z, present, gibbs)
            return lax.while_loop(
                unfinished, newton, (jnp.array(0), z, residual(z, present, gibbs))
            )[1]

        # -----------------------------------------------------------------------------------
        # STEP 7. Phase selection: switch condensates on and off one at a time.
        #
        # Starting unknowns: lambda and c fitted by least squares to two kinds of row: the gas
        # species with a share (ln n_s = a_s.lambda - G_s/RT + c), and the gas species with an
        # imposed fugacity (a_s.lambda - G_s/RT = ln f_s, with no c), which start the potential of
        # an element without a mass constraint from its target. Gas species with neither carry no
        # information and are left out. ln N_gas is the total of the gas shares, the dissolved
        # slots start at their shares (the sweep then sets them by their solubility laws), and
        # the condensate slots at trace.
        # 2 * n_condensates + 1 passes let each condensate be switched on and off once; a final
        # solve follows. After each pass's solve, the condensate G/RT is re-evaluated at the
        # pressure reached (gas G/RT is for the 1-bar standard state), before the on/off decision.
        # The final solve uses the G/RT of the last switching pass, which is not re-evaluated at
        # the pressure of the final solve.
        # -----------------------------------------------------------------------------------
        share_row: Array = gas & budgeted & ~imposed
        target_fugacity: Array = parameters.activity_constraints.log_activity(
            temperature, start_pressure
        )
        design: Array = jnp.concatenate(
            [
                jnp.where((share_row | imposed)[:, None], formula_matrix.T, 0.0),
                jnp.where(share_row, 1.0, 0.0)[:, None],
            ],
            -1,
        )
        rhs: Array = jnp.where(
            share_row,
            log_share + gibbs,
            jnp.where(imposed, jnp.where(imposed, target_fugacity, 0.0) + gibbs, 0.0),
        )
        fit: Array = jnp.linalg.lstsq(design, rhs)[0]
        log_n_gas_start: Array = logsumexp(jnp.where(gas & budgeted, log_share, -jnp.inf))
        log_n_gas_start = jnp.where(jnp.isfinite(log_n_gas_start), log_n_gas_start, 0.0)
        z0: Array = jnp.concatenate(
            [
                fit[:n_elements],
                jnp.atleast_1d(log_n_gas_start),
                jnp.where(dissolved & budgeted, log_share, self.log_trace_value),
            ]
        )

        def switching_pass(k, carry):
            z, present, gibbs = carry
            z = solve(z, present, gibbs)
            # The last iteration is the final solve: no Gibbs update and no switching. Running it
            # inside the loop traces solve only once, which keeps the compiled program smaller.
            switching: Array = k < 2 * n_condensates + 1
            pressure: Array = parameters.state.get_pressure(log_moles(z, present, gibbs)[0])
            gibbs = jnp.where(
                condensate & switching,
                _gibbs_over_RT_all_species(parameters, temperature, pressure),
                gibbs,
            )
            slots: Array = z[n_elements + 1 :]
            driving_force: Array = formula_matrix.T @ z[:n_elements] - gibbs
            candidates: Array = condensate & ~present & (driving_force > 1e-6)
            best: Array = jnp.argmax(jnp.where(candidates, driving_force, -jnp.inf))
            on: Array = (jnp.arange(n_species) == best) & jnp.any(candidates) & switching
            off: Array = condensate & present & (slots < self.log_trace_value + 1.0) & switching
            # A newly present condensate starts at a fraction (half) of the budget of its limiting
            # element, as the heuristic initializer allocates
            n_limit: Array = max_moles_by_limiting_element(
                formula_matrix, jnp.where(constrained, abundance, jnp.nan), on
            )
            log_start: Array = jnp.log(
                CONDENSATE_ALLOCATION_FRACTION
                * jnp.where(n_limit > 0, n_limit, jnp.exp(self.log_trace_value))
            )
            slots = jnp.where(on, log_start, jnp.where(off, self.log_trace_value, slots))
            return z.at[n_elements + 1 :].set(slots), (present | on) & ~off, gibbs

        n_condensates: int = len(reaction_names) - len(reaction_system.phase_system.gas.species)
        present: Array = jnp.zeros(n_species, dtype=bool)
        if n_condensates == 0:
            # Nothing to switch, so a single solve (a switching pass would only re-evaluate the
            # condensate G/RT, of which there are none, before an identical final solve)
            z: Array = solve(z0, present, gibbs)
        else:
            # 2 * n_condensates + 1 switching passes, then the final solve as the last iteration
            z, present, gibbs = lax.fori_loop(
                0, 2 * n_condensates + 2, switching_pass, (z0, present, gibbs)
            )

        # -----------------------------------------------------------------------------------
        # STEP 8. Output: absent condensates in the solver's stability representation
        # (zeta_c = -d_c), then [log moles, log stability], or the start (the shares) if not
        # finite. Species that are neither reaction nor dissolved species are set to trace.
        # -----------------------------------------------------------------------------------
        log_n: Array = log_moles(z, present, gibbs)[0]
        driving_force: Array = formula_matrix.T @ z[:n_elements] - gibbs
        absent: Array = condensate & ~present & (driving_force < 0)
        implied_constant: Array = compute_implied_log_stability(parameters, jnp.zeros(n_species))
        log_n = jnp.where(
            absent, implied_constant - jnp.log(jnp.where(absent, -driving_force, 1.0)), log_n
        )
        log_n = jnp.where(in_reaction | dissolved, log_n, self.log_trace_value)
        result: Array = jnp.concatenate([log_n, compute_implied_log_stability(parameters, log_n)])
        start: Array = jnp.concatenate(
            [start_log_n, compute_implied_log_stability(parameters, start_log_n)]
        )

        return jnp.where(jnp.all(jnp.isfinite(result)), result, start)
