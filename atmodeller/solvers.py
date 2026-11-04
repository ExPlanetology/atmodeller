# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Atmodeller's solvers for chemical equilibrium

These assemble the generic building blocks of :mod:`atmodeller.solver_utils` for Atmodeller's
model:

- **Single solve:** :func:`solve_single` supplies Atmodeller's objective function and solver
  configuration to :func:`~atmodeller.solver_utils.checked_root_find`, which solves a single
  system and checks it. This is the only place the objective function is used.
- **Batched solve:** :func:`make_batch_solver` vmaps :func:`solve_single` over the batch.
- **Retries:** :func:`make_batch_retry_solver_from_parameters` retries the models that failed,
  from initial guesses perturbed by :func:`perturb_log_number_moles`, with the generic
  :func:`~atmodeller.solver_utils.make_batch_retry_solver`.
- **Tau sweep:** :func:`make_tau_sweep_solver` solves systems with active stability species
  along a log-spaced schedule of tau.
- **Initial guess:** rows without an initial guess are completed with the initializer's cold
  guess (:func:`~atmodeller.initializer.batch.make_batch_initializer`).

Main entry points, all compatible with JIT-compiled workflows:

- :func:`make_solver_with_jit`: JIT-compiled solver (default: single path)
- :func:`make_solver_with_jit_single_path`: JIT-compiled solver (single path only)
- :func:`make_solver_with_jit_dual_path`: JIT-compiled solver (both branches, most flexible)
- :func:`make_solver_with_jit_batch_only`: JIT-compiled solver (batch solver only, fastest
  compilation)
- :func:`make_solver`: non-JIT solver (can be wrapped with JIT externally)

Quick guide:

- ``dual_path``: Most flexible, highest compilation cost
- ``single_path``: Faster compilation while retaining retry and stability support
- ``batch_only``: Fastest compilation; can still solve active-stability systems, but skips retry
    and tau-sweep robustness logic

The solvers return a checked :class:`~atmodeller.solver_utils.CheckedSolution`, or an
:class:`~atmodeller.output.Output` for the entry points, with the result of each model.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

from atmodeller.engine import (
    compute_implied_log_stability,
    get_residual_by_constraint,
    is_valid_model,
    objective_function,
)
from atmodeller.initializer import InitializerProtocol, make_batch_initializer
from atmodeller.jax_utils import FloatArray, vmap_axes_spec
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.solver_utils import (
    RESULTS,
    SOLUTION_BATCH_AXIS,
    CheckedSolution,
    checked_root_find,
    implicit_polish,
    make_batch_retry_solver,
    perturb_around_converged,
)


def solve_single(initial_guess: FloatArray, parameters: Parameters) -> CheckedSolution:
    """Solves a single (unbatched) system, and checks it.

    Supplies Atmodeller's objective function and solver configuration, read from
    ``parameters.solver_parameters``, to :func:`~atmodeller.solver_utils.checked_root_find`. The
    system is solved if the solver converged and the residual of the objective function is within
    ``solver_parameters.atol``. Intended to be wrapped with :func:`equinox.filter_vmap` by
    :func:`make_batch_solver` rather than called directly.

    Args:
        initial_guess: Initial guess for the solution vector
        parameters: Parameters, which provide the configuration of the solver and the bounds on
            the solution

    Returns:
        Checked :class:`~atmodeller.solver_utils.CheckedSolution` of this single attempt
    """
    solver_parameters = parameters.solver_parameters
    number_species: int = parameters.species.number_species

    checked_sol: CheckedSolution = checked_root_find(
        objective_function,
        initial_guess,
        parameters,
        solver_parameters,
        lower=solver_parameters.get_lower_bound(number_species),
        upper=solver_parameters.get_upper_bound(number_species),
    )

    # A model that is not well posed can never be solved, which its result reports
    result: RESULTS = RESULTS.where(
        is_valid_model(parameters), checked_sol.result, RESULTS.invalid_model
    )

    return eqx.tree_at(lambda sol: sol.result, checked_sol, result)


def make_batch_solver(parameters: Parameters) -> Callable:
    """Gets a vmapped batch solver for independent systems.

    Wraps :func:`solve_single` with :func:`equinox.filter_vmap` so that each batch element is
    solved independently, producing per-element convergence statistics. The vmapping axes are
    fixed from ``parameters`` at construction time.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable that returns a checked :class:`~atmodeller.solver_utils.CheckedSolution` of a
        single attempt
    """
    solver_function_vmapped: Callable = eqx.filter_vmap(
        solve_single, in_axes=(SOLUTION_BATCH_AXIS, vmap_axes_spec(parameters))
    )

    def batch_single_pass_solver(solution: Array, parameters: Parameters) -> CheckedSolution:
        """Runs the vmapped single-pass batch solve.

        Args:
            solution: Batched initial guess with shape ``(batch, solution)``
            parameters: Parameters passed through to each vmapped :func:`solve_single` call

        Returns:
            Checked :class:`~atmodeller.solver_utils.CheckedSolution` of a single attempt
        """
        # jax.debug.print("Running batch single-pass solver with vmapped single solve")
        # jax.debug.print("solution = {out}", out=solution)
        # jax.debug.print("tau = {out}", out=parameters.solver_parameters.tau)
        return solver_function_vmapped(solution, parameters)

    return batch_single_pass_solver


def make_batch_residual(parameters: Parameters) -> Callable:
    """Gets a vmapped residual of every constraint, for the output.

    Wraps :func:`~atmodeller.engine.get_residual_by_constraint` with :func:`equinox.filter_vmap`.
    The vmapping axes are fixed from ``parameters`` at construction time, as for
    :func:`make_batch_solver`, since they cannot be derived from traced parameters.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable ``(solution, parameters) -> residual`` with the residual of every constraint for
        each model, NaN for inactive constraints
    """
    return eqx.filter_vmap(
        get_residual_by_constraint, in_axes=(SOLUTION_BATCH_AXIS, vmap_axes_spec(parameters))
    )


def perturb_log_number_moles(
    solution: Float[Array, "... twice_species"],
    attempt: Integer[Array, "..."],
    key: PRNGKeyArray,
    parameters: Parameters,
) -> Float[Array, "... twice_species"]:
    """Perturbs the solution of the models that failed, for Atmodeller's solution layout.

    The solution is split into ``log_number_moles`` and ``log_stability``. Only the former is
    perturbed around its already-converged rows (via
    :func:`~atmodeller.solver_utils.perturb_around_converged`), and the latter is re-derived via
    :func:`~atmodeller.engine.compute_implied_log_stability`. This is the ``perturb_fn`` of the
    generic :func:`~atmodeller.solver_utils.make_batch_retry_solver`.

    Args:
        solution: Batched solution, ``log_number_moles`` followed by ``log_stability``
        attempt: Attempt on which each model was solved, or ``0`` if it was not solved
        key: JAX PRNG key
        parameters: Parameters

    Returns:
        Candidate perturbed solution for every model
    """
    log_number_moles, _ = jnp.split(solution, 2, axis=-1)

    perturbed_log_number_moles: Float[Array, "... n_species"] = perturb_around_converged(
        log_number_moles,
        attempt,
        key,
        parameters.solver_parameters.log_number_moles_lower,
        parameters.solver_parameters.log_number_moles_upper,
    )
    new_log_stability: Float[Array, "... n_species"] = compute_implied_log_stability(
        parameters, perturbed_log_number_moles
    )

    return jnp.concatenate([perturbed_log_number_moles, new_log_stability], axis=-1)


def make_batch_retry_solver_from_parameters(parameters: Parameters) -> Callable:
    """Gets a batch retry solver, constructing the vmapped batch solver internally.

    Builds the batch solver (:func:`make_batch_solver`), whose vmapping axes are fixed from
    ``parameters``, and passes it with :func:`perturb_log_number_moles` to the generic
    :func:`~atmodeller.solver_utils.make_batch_retry_solver`. The retry-loop mechanics (perturb
    failed rows, re-solve, keep newly-successful rows, repeat) live in the generic factory, which
    is shared with downstream packages that have a different solution layout.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable that returns a :class:`~atmodeller.solver_utils.CheckedSolution` object
    """
    batch_solver: Callable = make_batch_solver(parameters)

    return make_batch_retry_solver(batch_solver, perturb_log_number_moles)


def make_tau_sweep_solver(batch_retry_solver: Callable, batch_solver: Callable) -> Callable:
    """Gets a solver function that performs a tau sweep for active stability systems.

    Closes over the provided ``batch_retry_solver`` and ``batch_solver``. The returned callable
    first attempts a solve at the final tau, ``solver_parameters.tau``; if all batch elements
    converge it returns immediately, otherwise it runs a full log-spaced sweep of
    ``solver_parameters.tau_num`` values from ``solver_parameters.tau_max`` down to the final tau
    for every batch element.

    The solves in the sweep only locate the root. The derivatives of the solution are attached
    once at the end by a final solve at the final tau (see
    :func:`~atmodeller.solver_utils.implicit_polish`), so they are those of the root at the final
    tau and do not depend on the sweep.

    Args:
        batch_retry_solver: Pre-built batch retry solver, e.g. from
            :func:`make_batch_retry_solver_from_parameters`
        batch_solver: Pre-built batch solver, e.g. from :func:`make_batch_solver`, used for the
            final solve that attaches the derivatives

    Returns:
        Callable that returns a :class:`~atmodeller.solver_utils.CheckedSolution` object
    """
    get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731

    def tau_sweep_solver(
        initial_guess: Float[Array, "... solution"], parameters: Parameters, key: PRNGKeyArray
    ) -> CheckedSolution:
        """Attempts a solve at the final tau and, if any element fails, runs a full tau sweep.

        First tries to solve all batch elements at the final tau (``solver_parameters.tau``) with
        multistart retry. If every element converges, the result is returned immediately.
        Otherwise a log-spaced schedule of ``solver_parameters.tau_num`` values from
        ``solver_parameters.tau_max`` down to the final tau is swept via :func:`jax.lax.scan`,
        applying the solver at each step for all batch elements. Because ``tau`` must remain a
        scalar throughout the scan (to keep the vmapping axes consistent), converged and failed
        elements both run the full sweep; converged ones simply re-solve quickly from their
        existing solution, but keep the result of their solve at the final tau. The other elements
        report the final step of the sweep, which solves at the final tau.

        Args:
            initial_guess: Batched array of initial guesses
            parameters: :class:`~atmodeller.parameters.Parameters` whose ``tau`` leaf will be
                replaced at each scan step.
            key: JAX PRNG key for reproducible random perturbations

        Returns:
            :class:`~atmodeller.solver_utils.CheckedSolution` object
        """

        def solve_tau_step(carry: tuple, tau: Float[Array, " ..."]) -> tuple[tuple, None]:
            """Performs a single batched solver step for one scalar tau value.

            Intended for use inside :func:`jax.lax.scan`. Receives a scalar ``tau`` from
            ``varying_schedule``, injects it into a copy of ``parameters``, and runs the batch
            retry solver on all elements from the solution carried forward from the previous step.

            Args:
                carry: Tuple of ``(key, checked_sol)`` where ``checked_sol`` is the
                    :class:`~atmodeller.solver_utils.CheckedSolution` of the previous step
                tau: Scalar tau value for this scan step

            Returns:
                Updated ``(key, checked_sol)`` carry tuple and no stacked output
            """
            key, checked_sol = carry
            key, subkey = jax.random.split(key)

            # Get new parameters with tau value
            new_parameters: Parameters = eqx.tree_at(get_leaf, parameters, tau)
            # jax.debug.print("tau = {out}", out=new_parameters.solver_parameters.tau)

            new_sol: CheckedSolution = batch_retry_solver(
                checked_sol.value,
                new_parameters,
                subkey,
                parameters.solver_parameters.max_starts,
                implicit_gradient=False,
            )

            return (key, new_sol), None

        # Log-spaced from tau_max down to the final tau. tau_num is a Python int, so the length of
        # the scan is fixed at compile time. The last value is set to the final tau exactly, which
        # logspace reproduces only to within rounding.
        solver_parameters = parameters.solver_parameters
        varying_schedule: Float[Array, " tau"] = jnp.logspace(
            jnp.log10(solver_parameters.tau_max),
            jnp.log10(solver_parameters.tau),
            num=solver_parameters.tau_num,
        )
        varying_schedule = varying_schedule.at[-1].set(solver_parameters.tau)

        # Initial solve at the final tau
        key, subkey = jax.random.split(key)

        first_sol: CheckedSolution = batch_retry_solver(
            initial_guess,
            parameters,
            subkey,
            parameters.solver_parameters.max_starts,
            implicit_gradient=False,
        )
        first_success: Bool[Array, "..."] = first_sol.success
        # A model that is not well posed keeps its first result, since a sweep cannot solve it
        first_final: Bool[Array, "..."] = jnp.logical_or(
            first_success, first_sol.result == RESULTS.invalid_model
        )

        def run_scan(key: PRNGKeyArray) -> CheckedSolution:
            """Run the full tau sweep scan across all batch elements.

            Called when at least one element failed the initial solve at the final tau. All batch
            elements (including those that already converged) run every step of the
            ``varying_schedule`` scan, but elements that converged at the final tau on the first
            attempt keep the result of that solve, so their result does not depend on whether
            other batch elements failed. The remaining elements report the result of the final
            step, since only the final step solves at the final tau; the earlier steps at larger
            tau only guide the solution towards the root at the final tau.
            """
            (_, sweep_sol), _ = jax.lax.scan(solve_tau_step, (key, first_sol), varying_schedule)

            def keep_first(first: Array, sweep: Array) -> Array:
                """Selects ``first`` for elements that converged on the first attempt, or that are
                not well posed"""
                mask: Bool[Array, "..."] = jnp.expand_dims(
                    first_final, tuple(range(first_final.ndim, first.ndim))
                )
                return jnp.where(mask, first, sweep)

            return jax.tree.map(keep_first, first_sol, sweep_sol)

        def run_single_step(_: PRNGKeyArray) -> CheckedSolution:
            """All batch elements converged at the final tau on the first attempt: return
            immediately."""
            return first_sol

        # If all entries converged at the final tau on the first attempt, skip the sweep entirely
        all_converged: Bool[Array, ""] = jnp.all(first_final)
        checked_sol: CheckedSolution = lax.cond(all_converged, run_single_step, run_scan, key)

        # The sweep finishes at the final tau, so attach the derivatives of the root there
        polished_solution: Float[Array, "... solution"] = implicit_polish(
            batch_solver, checked_sol.value, checked_sol.success, parameters
        )
        checked_sol = eqx.tree_at(lambda m: m.solution.value, checked_sol, polished_solution)

        return checked_sol

    return tau_sweep_solver


def make_solver(
    parameters: Parameters, initializer: InitializerProtocol | None = None
) -> Callable:
    """General assembly function that constructs and returns the dual-path solver.

    Builds a :func:`make_batch_retry_solver_from_parameters` and a tau sweep solver from
    ``parameters`` at construction time, sharing the same ``batch_retry_solver`` instance between
    both paths. The returned callable dispatches at runtime via :func:`jax.lax.cond` based on
    whether any species have active stability — routing to the tau sweep solver when stability
    species are present, or to the generic multistart solver otherwise.

    Note:
        ``active_stability`` is currently not a traced JAX array; its size must be fixed at
        compile time because it determines the shape of the residual vector. The
        :func:`jax.lax.cond` branch therefore compiles *both* paths even though only one will
        execute at runtime. This retains generality for future capabilities (e.g. dynamically
        switching solver strategy based on active species) at the expense of additional — currently
        unnecessary — compilation time.

    Args:
        parameters: Parameters used to derive the vmapping axes and build the sub-solvers at
            construction time
        initializer: Initializer for the cold initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.

    Returns:
        Callable that returns a :class:`~atmodeller.output.Output` object
    """
    batch_initializer: Callable = make_batch_initializer(parameters, initializer)
    batch_residual: Callable = make_batch_residual(parameters)
    batch_retry_solver: Callable = make_batch_retry_solver_from_parameters(parameters)
    tau_sweep_solver: Callable = make_tau_sweep_solver(
        batch_retry_solver, make_batch_solver(parameters)
    )

    # For debugging to determine if this function is jittable in isolation
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def dispatch_solver(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "n_batch twice_species"],
    ) -> Output:
        """Entry point that dispatches to the appropriate solver branch.

        Checks whether any species have active stability and routes accordingly via
        :func:`jax.lax.cond`: the tau sweep solver is used when stability species are
        present, otherwise the generic multistart retry solver is used. Both branches are
        compiled at trace time.

        Args:
            parameters: Parameters; array leaves are traced, non-array leaves are static
            key: JAX PRNG key
            base_solution_array: Initial guess with shape ``(batch_size, 2 * n_species)``

        Returns:
            :class:`~atmodeller.output.Output` object
        """
        base_solution_array = batch_initializer(parameters, base_solution_array)

        # Define the condition to check if active stability is enabled
        condition: Bool[Array, ""] = jnp.any(parameters.reaction_system.species.active_stability)
        # jax.debug.print("condition (active stability) = {out}", out=condition)

        def solve_with_stability(key):
            """Routes to the tau sweep solver for systems with active stability species."""
            _, subkey = jax.random.split(key)
            return tau_sweep_solver(base_solution_array, parameters, subkey)

        def solve_without_stability(key):
            """Routes to the generic multistart retry solver for systems without stability."""
            _, subkey = jax.random.split(key)
            return batch_retry_solver(
                base_solution_array,
                parameters,
                subkey,
                parameters.solver_parameters.max_starts,
            )

        checked_sol = lax.cond(
            condition, solve_with_stability, solve_without_stability, operand=key
        )
        output: Output = Output(
            parameters,
            checked_sol,
            base_solution_array,
            batch_residual(checked_sol.value, parameters),
        )

        return output

    return dispatch_solver


# For testing and debugging
# @eqx.debug.assert_max_traces(max_traces=1)
def make_solver_with_jit_dual_path(
    parameters: Parameters, initializer: InitializerProtocol | None = None
) -> Callable:
    """Gets a JIT-compiled solver with both runtime branches compiled.

    A convenience wrapper around :func:`make_solver` that applies :func:`equinox.filter_jit` to
    the returned solver function. This version compiles both the tau sweep and batch retry paths
    at trace time, dispatching at runtime via :func:`jax.lax.cond`.

    Use this when the model structure may change dynamically between solver construction and
    runtime, or when runtime flexibility is more important than compilation speed.

    Args:
        parameters: Parameters used to derive the vmapping axes and build the sub-solvers at
            construction time
        initializer: Initializer for the cold initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.

    Returns:
        Callable that returns a :class:`~atmodeller.output.Output` object
    """
    dual_path_solver: Callable = make_solver(parameters, initializer)

    return eqx.filter_jit(dual_path_solver)


# Useful for debugging to ensure that the solver is not being re-compiled unnecessarily during
# development iterations
# @eqx.debug.assert_max_traces(max_traces=5)
def make_solver_with_jit_single_path(
    parameters: Parameters, initializer: InitializerProtocol | None = None
) -> Callable:
    """Gets a JIT-compiled solver with optimised compilation by eliminating unused branches.

    This function inspects ``active_stability`` at construction time (Python-level) to determine
    which solver path to use, then compiles only that path. This results in faster JIT
    compilation, but requires that the active stability structure remain fixed for the lifetime
    of the solver. If the active stability structure changes, a new solver must be constructed.

    Args:
        parameters: Parameters used to derive the vmapping axes and build the sub-solvers at
            construction time. The ``active_stability`` structure is inspected to choose the
            solver path.
        initializer: Initializer for the cold initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.

    Returns:
        JIT-compiled callable that returns a :class:`~atmodeller.output.Output` object
    """
    batch_initializer: Callable = make_batch_initializer(parameters, initializer)
    batch_residual: Callable = make_batch_residual(parameters)
    batch_retry_solver: Callable = make_batch_retry_solver_from_parameters(parameters)
    tau_sweep_solver: Callable = make_tau_sweep_solver(
        batch_retry_solver, make_batch_solver(parameters)
    )

    # Determine the solver path at construction time (Python-level evaluation).
    has_active_stability: bool = bool(
        jnp.any(parameters.reaction_system.species.active_stability).item()
    )

    @eqx.filter_jit
    def solver_with_stability(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "n_batch twice_species"],
    ) -> Output:
        """Tau sweep solver path for systems with active stability species."""
        _, subkey = jax.random.split(key)
        base_solution_array = batch_initializer(parameters, base_solution_array)
        checked_sol = tau_sweep_solver(base_solution_array, parameters, subkey)
        return Output(
            parameters,
            checked_sol,
            base_solution_array,
            batch_residual(checked_sol.value, parameters),
        )

    @eqx.filter_jit
    def solver_without_stability(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "n_batch twice_species"],
    ) -> Output:
        """Generic batch retry solver path for systems without active stability."""
        _, subkey = jax.random.split(key)
        base_solution_array = batch_initializer(parameters, base_solution_array)
        checked_sol = batch_retry_solver(
            base_solution_array,
            parameters,
            subkey,
            parameters.solver_parameters.max_starts,
        )
        return Output(
            parameters,
            checked_sol,
            base_solution_array,
            batch_residual(checked_sol.value, parameters),
        )

    # Return the appropriate pre-compiled solver based on the construction-time check.
    return solver_with_stability if has_active_stability else solver_without_stability


def make_solver_with_jit_batch_only(
    parameters: Parameters, initializer: InitializerProtocol | None = None
) -> Callable:
    """Gets a JIT-compiled solver with minimal compilation overhead.

    This is the fastest compilation option: it uses only :func:`make_batch_solver` without
    retry logic or stability sweep. It can still be used for systems with active stability, but
    it bypasses the tau-sweep path that is designed to improve robustness for those systems.

    This is suitable for:
    - Systems where direct batch solves are usually sufficient
    - Cases where robust convergence retry is not needed
    - Rapid iteration during development

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time
        initializer: Initializer for the cold initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.

    Returns:
        JIT-compiled callable that returns a :class:`~atmodeller.output.Output` object
    """
    batch_initializer: Callable = make_batch_initializer(parameters, initializer)
    batch_solver: Callable = make_batch_solver(parameters)
    batch_residual: Callable = make_batch_residual(parameters)

    @eqx.filter_jit
    def batch_only_solver(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "n_batch twice_species"],
    ) -> Output:
        """Basic batch solver path without retry or tau-sweep enhancements.

        Each model is solved on its single attempt if it passes the check in
        :func:`solve_single`, so the reported successful attempt and steps are zero otherwise.
        """
        # Note: key is unused in basic batch solver, but included for interface compatibility
        del key
        base_solution_array = batch_initializer(parameters, base_solution_array)
        checked_sol: CheckedSolution = batch_solver(base_solution_array, parameters)

        return Output(
            parameters,
            checked_sol,
            base_solution_array,
            batch_residual(checked_sol.value, parameters),
        )

    return batch_only_solver


# Select the default JIT solver factory here for development and benchmarking.
make_solver_with_jit: Callable = make_solver_with_jit_single_path
# make_solver_with_jit: Callable = make_solver_with_jit_dual_path
# make_solver_with_jit: Callable = make_solver_with_jit_batch_only
