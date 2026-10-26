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
from jaxtyping import Array, Bool, Float

from atmodeller.engine import compute_implied_log_stability
from atmodeller.initializer.core import LOG_TRACE_VALUE, max_moles_by_limiting_element
from atmodeller.jax_utils import safe_exp
from atmodeller.parameters import Parameters


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
      of the residual.
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

    1. Start from an even split of the total constrained element moles over the gas species (as
       CEA), with :math:`\lambda` fitted to it by least squares. No knowledge of the answer is
       used.
    2. Phase selection in :math:`2 n_\mathrm{condensates} + 1` passes, then a final solve. Each
       pass solves, re-evaluates the condensate :math:`G_c/RT` at the pressure reached, switches
       on the most supersaturated absent condensate (if :math:`d_c > 10^{-6}`; it starts at half
       the budget of its limiting element), and switches off present condensates whose log moles
       fall below ``log_trace_value`` + 1. Switching one at a
       time means that a present polymorph fixes its element's potential and the other becomes
       undersaturated. The start pressure (from the even split) can differ from the converged
       pressure by an order of magnitude, enough to put a condensate near saturation on the wrong
       side of its boundary through the pressure dependence of its Gibbs energy, hence the
       re-evaluation.
    3. Every solve is an element sweep followed by damped Newton:

       - *Sweep* (FastChem): up to ``sweeps`` Gauss-Seidel passes over the elements in order
         of decreasing abundance, each solving its own element's mass balance exactly for
         :math:`\lambda_e` (see below). Each 1-D Newton stops when its step is below
         ``sweep_step_tolerance`` (at most ``sweep_iterations``), and the passes stop when no
         :math:`\lambda_e` changes by more than ``sweep_tolerance``.
       - *Newton*: least-squares direction from the damped normal equations (see below),
         scaled so that no :math:`\lambda_e` and no carried species' log moles change by more
         than ``max_log_step``; stops when the residual norm is below ``residual_tolerance``, or
         after ``newton_iterations``.

    4. Absent condensates are given the solver's stability representation: the solver's
       residual for an absent condensate is :math:`-\nu (d_c + \zeta_c)`, so
       :math:`\zeta_c = -d_c`, and its log moles follow from the implied-stability relation
       (absent condensates with :math:`d_c \geq 0` stay at trace).
    5. If the result is not finite, the even-split start is returned instead.

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

    *The 1-D problem for element e.* Hold the other potentials, the moles held outside the gas
    (:math:`C_e`, in present condensates and dissolved species) and the gas normalisation
    :math:`\sigma = \ln N_\mathrm{gas} - \ln \sum_\mathrm{gas} p` fixed, so that
    :math:`\ln n_s = a_{es} x + \beta_s` with :math:`x = \lambda_e`. The log mass balance

    .. math::

        f(x) = \ln \sum_{s \in \mathrm{gas},\, a_{es} > 0}
               \exp\left(\ln a_{es} + a_{es} x + \beta_s\right) - \ln(b_e - C_e)

    has :math:`f'(x) = \sum_s w_s a_{es} \geq 1` and :math:`f''(x) \geq 0` (mean and variance
    of :math:`a_{es}` under the weights :math:`w_s = a_{es} n_s / \sum a n`). It is convex and
    increasing with slope at least one, so it has one root and Newton reaches it from any
    start, with every step at most :math:`|f|`. Freezing :math:`\sigma` is FastChem's fixed
    total density: it keeps each 1-D problem separate and solvable when :math:`b_e > C_e`.
    After the sweep :math:`\ln N_\mathrm{gas}` is reset so that the normalised moles equal the
    swept ones. An element is skipped if it is not mass constrained, if :math:`C_e \geq b_e`, or
    if no gas species contains it. The sweep ignores every row other than the mass balances.

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
            Defaults to ``1e-10``.
        sweep_tolerance: Largest change in any :math:`\lambda_e` over a sweep below which the
            sweeps stop. Defaults to ``1e-8``.
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
    sweep_step_tolerance: float = 1e-10
    """Step size at which a 1-D Newton iteration in a sweep stops"""
    sweep_tolerance: float = 1e-8
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
        # STEP 2. Even-split start and standard Gibbs energies at the start pressure. The
        # condensate values are updated after each switching pass (STEP 7), so G/RT is an
        # argument of the functions below.
        # -----------------------------------------------------------------------------------
        total: Array = jnp.sum(jnp.where(constrained, abundance, 0.0))
        total = jnp.where(total > 0, total, 1.0)
        log_n_each: Array = jnp.log(total / jnp.maximum(jnp.sum(gas), 1))
        start_log_n: Float[Array, " n_species"] = jnp.where(gas, log_n_each, self.log_trace_value)
        gibbs: Float[Array, " n_species"] = _gibbs_over_RT_all_species(
            parameters, temperature, parameters.state.get_pressure(start_log_n)
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
        # See the class docstring for the 1-D problem. Held fixed: the other potentials (updated
        # in Gauss-Seidel order, most abundant element first), the moles outside the gas (C_e)
        # and the gas normalisation sigma.
        # -----------------------------------------------------------------------------------
        order: Array = jnp.argsort(-jnp.where(constrained, abundance, -jnp.inf))
        # ln a_es; the placeholder 1.0 where a_es = 0 is masked out by the carrier mask
        log_a: Array = jnp.log(jnp.where(formula_matrix > 0, formula_matrix, 1.0))

        def sweep(z: Array, present: Array, gibbs: Array) -> Array:
            log_n, _, log_p_sum = log_moles(z, present, gibbs)
            sigma: Array = z[n_elements] - log_p_sum
            outside_gas: Array = formula_matrix @ jnp.where(
                (condensate & present) | dissolved, safe_exp(log_n), 0.0
            )

            def element(k, lam):
                e: Array = order[k]
                a_e: Array = formula_matrix[e]
                carrier: Array = gas & (a_e > 0)
                gas_share: Array = abundance[e] - outside_gas[e]
                solvable: Array = constrained[e] & (gas_share > 0) & jnp.any(carrier)
                log_target: Array = jnp.log(jnp.where(solvable, gas_share, 1.0))
                # beta_s: ln n_s without its a_es lambda_e term
                beta: Array = formula_matrix.T @ lam - gibbs + sigma - a_e * lam[e]

                def newton_step(x):
                    terms: Array = jnp.where(carrier, log_a[e] + a_e * x + beta, -jnp.inf)
                    log_g: Array = logsumexp(terms)
                    # f' = mean stoichiometry of the carriers, >= 1 (floor for non-integer
                    # formulae)
                    slope: Array = jnp.maximum(jnp.sum(jnp.exp(terms - log_g) * a_e), 1.0)
                    # Clipped only as a guard against extreme values
                    return jnp.clip((log_g - log_target) / slope, -50.0, 50.0)

                # 1-D Newton until the step is below sweep_step_tolerance. The problem is convex
                # with slope >= 1, so it converges in a few iterations; an element that is
                # already solved stops after one.
                def unfinished(carry):
                    i, _, dx = carry
                    return (i < self.sweep_iterations) & (jnp.abs(dx) > self.sweep_step_tolerance)

                def newton(carry):
                    i, x, _ = carry
                    dx: Array = newton_step(x)
                    return i + 1, x - dx, dx

                x: Array = lax.while_loop(unfinished, newton, (0, lam[e], jnp.array(jnp.inf)))[1]
                return lam.at[e].set(jnp.where(solvable & jnp.isfinite(x), x, lam[e]))

            # Sweeps until no lambda_e changes by more than sweep_tolerance (at most sweeps), so a
            # state that is already converged costs a single sweep
            def sweeping(carry):
                i, _, change = carry
                return (i < self.sweeps) & (change > self.sweep_tolerance)

            def one_sweep(carry):
                i, lam, _ = carry
                lam_new: Array = lax.fori_loop(0, n_elements, element, lam)
                return i + 1, lam_new, jnp.max(jnp.abs(lam_new - lam))

            lam: Array = lax.while_loop(
                sweeping, one_sweep, (0, z[:n_elements], jnp.array(jnp.inf))
            )[1]
            # Reset ln N_gas so that the normalised gas moles of STEP 3 equal the swept ones
            log_p_sum_new: Array = logsumexp(
                jnp.where(gas, formula_matrix.T @ lam - gibbs, -jnp.inf)
            )
            return z.at[:n_elements].set(lam).at[n_elements].set(log_p_sum_new + sigma)

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
        # Starting unknowns: lambda fitted to the even split by least squares
        # (ln n_s = a_s.lambda - G_s/RT + c over the gas species), ln N_gas = ln(total),
        # dissolved slots at the even split, condensate slots at trace.
        # 2 * n_condensates + 1 passes let each condensate be switched on and off once; a final
        # solve follows. After each pass's solve, the condensate G/RT is re-evaluated at the
        # pressure reached (gas G/RT is for the 1-bar standard state), before the on/off decision.
        # The final solve uses the G/RT of the last switching pass, which is not re-evaluated at
        # the pressure of the final solve.
        # -----------------------------------------------------------------------------------
        design: Array = jnp.where(
            gas[:, None], jnp.concatenate([formula_matrix.T, jnp.ones((n_species, 1))], -1), 0.0
        )
        fit: Array = jnp.linalg.lstsq(design, jnp.where(gas, log_n_each + gibbs, 0.0))[0]
        z0: Array = jnp.concatenate(
            [
                fit[:n_elements],
                jnp.atleast_1d(jnp.log(total)),
                jnp.where(dissolved, log_n_each, self.log_trace_value),
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
            # A newly present condensate starts at half the budget of its limiting element
            n_limit: Array = max_moles_by_limiting_element(
                formula_matrix, jnp.where(constrained, abundance, jnp.nan), on
            )
            log_start: Array = jnp.log(
                0.5 * jnp.where(n_limit > 0, n_limit, jnp.exp(self.log_trace_value))
            )
            slots = jnp.where(on, log_start, jnp.where(off, self.log_trace_value, slots))
            return z.at[n_elements + 1 :].set(slots), (present | on) & ~off, gibbs

        n_condensates: int = len(reaction_names) - len(reaction_system.phase_system.gas.species)
        # 2 * n_condensates + 1 switching passes, then the final solve as the last iteration
        z, present, gibbs = lax.fori_loop(
            0,
            2 * n_condensates + 2,
            switching_pass,
            (z0, jnp.zeros(n_species, dtype=bool), gibbs),
        )

        # -----------------------------------------------------------------------------------
        # STEP 8. Output: absent condensates in the solver's stability representation
        # (zeta_c = -d_c), then [log moles, log stability], or the even-split start if not
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
