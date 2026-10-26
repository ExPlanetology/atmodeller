# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""JAX-compatible non-linear solvers for chemical equilibrium.

Key features:

- **Single and batched solvers:**
    - Solve individual or batched systems with automatic initial guess generation
    - Batch solvers use :func:`equinox.filter_vmap` for efficient parallelism
- **Robust convergence:**
    - Retry logic for failed solves with random perturbations and multiple attempts
    - Objective-based convergence validation, independent of solver's internal status
- **Tau sweep for stability:**
    - Automatic tau sweep for systems with active stability species, using a log-spaced schedule
    - Efficiently finds solutions across a range of tau values when needed
- **JIT compilation:**
    - All main solver entry points are JIT-compatible and can be used in compiled workflows
- **Flexible construction:**
    - Main entry points allow construction of solvers with or without JIT, and with custom
      retry/tau sweep logic

Main entry points:

- :func:`make_solver_with_jit`: Returns a JIT-compiled solver (default: single-path)
- :func:`make_solver_with_jit_single_path`: Returns a JIT-compiled solver (single path only)
- :func:`make_solver_with_jit_dual_path`: Returns a JIT-compiled solver (both branches for maximum flexibility)
- :func:`make_solver_with_jit_batch_only`: Returns a JIT-compiled solver (batch solver only, fastest compilation)
- :func:`make_solver`: Returns a non-JIT solver (can be wrapped with JIT externally)
- :func:`make_batch_retry_solver_from_parameters`: Builds a batch retry solver from parameters
- :func:`make_tau_sweep_solver`: Returns a tau sweep solver for active stability systems

Quick guide:

- ``dual_path``: Most flexible, highest compilation cost
- ``single_path``: Faster compilation while retaining retry and stability support
- ``batch_only``: Fastest compilation; can still solve active-stability systems, but skips retry
    and tau-sweep robustness logic

Most solvers return results as :class:`~atmodeller.jax_utils.MultiAttemptSolution` or
:class:`~atmodeller.output.Output` objects, with detailed convergence and step statistics.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import lax
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

from atmodeller.engine import compute_implied_log_stability, objective_function
from atmodeller.initializer import ElementPotentialInitializer, InitializerProtocol
from atmodeller.jax_utils import (
    FloatArray,
    MultiAttemptSolution,
    implicit_polish,
    max_norm,
    perturb_around_converged,
    vmap_axes_spec,
)
from atmodeller.jax_utils import make_batch_retry_solver as generic_make_batch_retry_solver
from atmodeller.output import Output
from atmodeller.parameters import Parameters

LOG_NUMBER_MOLES_VMAP_AXES: int = 0
"""Axis index for the solution array in the vmapped batch solver"""


def solve_single(initial_guess: FloatArray, parameters: Parameters) -> optx.Solution:
    """Solves a single (unbatched) system using :func:`optimistix.root_find`.

    Intended to be wrapped with :func:`equinox.filter_vmap` by :func:`make_batch_solver`
    rather than called directly. All solver configuration is read from
    ``parameters.solver_parameters``.

    Args:
        initial_guess: Initial guess for the solution vector
        parameters: Parameters providing the solver instance, step limit, and options

    Returns:
        :class:`optimistix.Solution` object
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        parameters.solver_parameters.get_solver_instance(),
        initial_guess,
        args=parameters,
        throw=parameters.solver_parameters.throw,
        max_steps=parameters.solver_parameters.max_steps,
        options=parameters.solver_parameters.get_options(parameters.species.number_species),
    )
    # jax.debug.print("solution = {out}", out=sol.value)

    return sol


def make_initial_guess_function(
    parameters: Parameters, initializer: InitializerProtocol | None = None
) -> Callable:
    """Gets a function that completes a batched initial guess with the initializer's cold guess.

    Rows of the initial guess are used as given, and rows that contain ``NaN`` are replaced by the
    cold guess from ``initializer``. The initializer acts on a single system, so it is wrapped
    with :func:`equinox.filter_vmap` with the vmapping axes fixed from ``parameters`` at
    construction time, as for :func:`make_batch_solver`. The decision to evaluate it is made for
    the whole batch, outside the vmap, so that the cold guess is skipped when no row contains
    ``NaN``.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time
        initializer: Initializer for the cold initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.

    Returns:
        Callable that returns the initial guess with shape ``(batch, solution)``
    """
    initializer_: InitializerProtocol = (
        ElementPotentialInitializer() if initializer is None else initializer
    )

    def cold_start(
        parameters: Parameters, base_solution: Float[Array, " solution"]
    ) -> Float[Array, " solution"]:
        """Cold guess for a single system; the solution sets the batch size, as for solving"""
        del base_solution

        return initializer_.start(parameters)

    cold_start_vmapped: Callable = eqx.filter_vmap(
        cold_start, in_axes=(vmap_axes_spec(parameters), LOG_NUMBER_MOLES_VMAP_AXES)
    )

    def initial_guess_function(
        parameters: Parameters, base_solution_array: Float[Array, "batch solution"]
    ) -> Float[Array, "batch solution"]:
        """Replaces rows of ``base_solution_array`` that contain ``NaN`` with the cold guess.

        Args:
            parameters: Parameters
            base_solution_array: Initial guess with shape ``(batch, solution)``. Rows that
                contain ``NaN`` have no initial guess.

        Returns:
            Initial guess with shape ``(batch, solution)``
        """
        # Mask: True for rows where any entry is NaN along the last axis
        nan_mask: Bool[Array, "batch 1"] = jnp.any(
            jnp.isnan(base_solution_array), axis=-1, keepdims=True
        )

        def fill_from_cold_start(
            base_solution_array: Float[Array, "batch solution"],
        ) -> Float[Array, "batch solution"]:
            """Selects the cold guess for rows with any NaN, otherwise keeps the given row."""
            return jnp.where(
                nan_mask,
                cold_start_vmapped(parameters, base_solution_array),
                base_solution_array,
            )

        # The predicate is a scalar for the whole batch, so the cold guess is skipped when every
        # row has an initial guess
        result: Float[Array, "batch solution"] = lax.cond(
            jnp.any(nan_mask), fill_from_cold_start, lambda x: x, base_solution_array
        )

        result = eqx.error_if(
            result,
            jnp.any(jnp.isnan(result)),
            "NaN encountered in initial guess, which will break the solver.",
        )

        return result

    return initial_guess_function


def make_batch_solver(parameters: Parameters) -> Callable:
    """Gets a vmapped batch solver for independent systems.

    Wraps :func:`solve_single` with :func:`equinox.filter_vmap` so that each batch element is
    solved independently, producing per-element convergence statistics. The vmapping axes are
    fixed from ``parameters`` at construction time.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable that returns a :class:`~atmodeller.jax_utils.MultiAttemptSolution` with
        ``attempts=1``
    """
    solver_function_vmapped: Callable = eqx.filter_vmap(
        solve_single, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )

    def batch_single_pass_solver(solution: Array, parameters: Parameters) -> MultiAttemptSolution:
        """Runs the vmapped single-pass batch solve.

        ``_attempts`` is set to ``1`` unconditionally; objective-based per-element convergence
        checking is delegated to the retry wrapper in ``batch_retry_solver``.

        Args:
            solution: Batched initial guess with shape ``(batch, solution)``
            parameters: Parameters passed through to each vmapped :func:`solve_single` call

        Returns:
            :class:`~atmodeller.jax_utils.MultiAttemptSolution` with ``attempts=1`` for all batch elements
        """
        # jax.debug.print("Running batch single-pass solver with vmapped single solve")
        # jax.debug.print("solution = {out}", out=solution)
        # jax.debug.print("tau = {out}", out=parameters.solver_parameters.tau)
        sol: optx.Solution = solver_function_vmapped(solution, parameters)

        return MultiAttemptSolution(sol, _attempts=1)

    return batch_single_pass_solver


def make_batch_retry_solver(solver_function: Callable, objective_fn: Callable) -> Callable:
    """Makes a batch retry solver.

    A thin, atmodeller-specific wrapper around the generic
    :func:`~atmodeller.jax_utils.make_batch_retry_solver`: supplies a ``perturb_fn`` that splits
    the solution into ``log_number_moles``/``log_stability``, perturbs only the former around its
    already-converged rows (via :func:`~atmodeller.jax_utils.perturb_around_converged`), and
    re-derives the latter via :func:`~atmodeller.engine.compute_implied_log_stability`. The
    retry-loop mechanics themselves (perturb failed rows, re-solve, check objective-based
    convergence, keep newly-successful rows, repeat) live in the generic factory, shared with
    downstream packages (e.g. mcmodeller) that have a different solution-vector layout.

    ``solver_function`` and ``objective_fn`` must be pure JAX-callable functions compatible
    with :func:`equinox.filter_jit`. They must not close over non-JAX state or produce Python side
    effects.

    Args:
        solver_function: Callable that performs a single solve. Must accept arguments of an initial
            guess and a pytree of parameters.
        objective_fn: Callable for the objective function

    Returns:
        Callable that returns a :class:`~atmodeller.jax_utils.MultiAttemptSolution` object
    """

    def perturb_fn(
        solution: Float[Array, "... twice_species"],
        attempt: Integer[Array, "..."],
        key: PRNGKeyArray,
        parameters: Parameters,
    ) -> Float[Array, "... twice_species"]:
        """Perturbs ``log_number_moles`` around its converged rows, then re-derives stability."""
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

    return generic_make_batch_retry_solver(solver_function, objective_fn, perturb_fn)


def make_batch_retry_solver_from_parameters(parameters: Parameters) -> Callable:
    """Gets a batch retry solver, constructing the vmapped batch solver and objective internally.

    A convenience wrapper around :func:`make_batch_retry_solver` that accepts
    ``parameters`` directly, deriving the ``vmap`` axes and building both the batch solver and
    vmapped objective function at construction time. Use this in preference to calling
    :func:`make_batch_retry_solver` directly when the vmap axes are not already available.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable that returns a :class:`~atmodeller.jax_utils.MultiAttemptSolution` object
    """
    batch_solver: Callable = make_batch_solver(parameters)
    objective_function_vmapped: Callable = eqx.filter_vmap(
        objective_function, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )
    batch_retry_solver: Callable = make_batch_retry_solver(
        batch_solver, objective_function_vmapped
    )

    return batch_retry_solver


def make_tau_sweep_solver(batch_retry_solver: Callable, batch_solver: Callable) -> Callable:
    """Gets a solver function that performs a tau sweep for active stability systems.

    Closes over the provided ``batch_retry_solver`` and ``batch_solver``. The returned callable
    first attempts a solve at the final tau, ``solver_parameters.tau``; if all batch elements
    converge it returns immediately, otherwise it runs a full log-spaced sweep of
    ``solver_parameters.tau_num`` values from ``solver_parameters.tau_max`` down to the final tau
    for every batch element.

    The solves in the sweep only locate the root. The derivatives of the solution are attached
    once at the end by a final solve at the final tau (see
    :func:`~atmodeller.jax_utils.implicit_polish`), so they are those of the root at the final tau
    and do not depend on the sweep.

    Args:
        batch_retry_solver: Pre-built batch retry solver, e.g. from
            :func:`make_batch_retry_solver_from_parameters`
        batch_solver: Pre-built batch solver, e.g. from :func:`make_batch_solver`, used for the
            final solve that attaches the derivatives

    Returns:
        Callable that returns a :class:`~atmodeller.jax_utils.MultiAttemptSolution` object
    """
    get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731

    def tau_sweep_solver(
        initial_guess: Float[Array, "... solution"], parameters: Parameters, key: PRNGKeyArray
    ) -> MultiAttemptSolution:
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
            :class:`~atmodeller.jax_utils.MultiAttemptSolution` object
        """

        def solve_tau_step(carry: tuple, tau: Float[Array, " ..."]) -> tuple[tuple, None]:
            """Performs a single batched solver step for one scalar tau value.

            Intended for use inside :func:`jax.lax.scan`. Receives a scalar ``tau`` from
            ``varying_schedule``, injects it into a copy of ``parameters``, and runs the batch
            retry solver on all elements from the solution carried forward from the previous step.

            Args:
                carry: Tuple of ``(key, multi_sol)`` where ``multi_sol`` is the
                    :class:`~atmodeller.jax_utils.MultiAttemptSolution` of the previous step
                tau: Scalar tau value for this scan step

            Returns:
                Updated ``(key, multi_sol)`` carry tuple and no stacked output
            """
            key, multi_sol = carry
            key, subkey = jax.random.split(key)

            # Get new parameters with tau value
            new_parameters: Parameters = eqx.tree_at(get_leaf, parameters, tau)
            # jax.debug.print("tau = {out}", out=new_parameters.solver_parameters.tau)

            new_sol: MultiAttemptSolution = batch_retry_solver(
                multi_sol.value,
                new_parameters,
                subkey,
                parameters.solver_parameters.max_starts - 1,
                parameters.solver_parameters.atol,
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

        first_sol: MultiAttemptSolution = batch_retry_solver(
            initial_guess,
            parameters,
            subkey,
            parameters.solver_parameters.max_starts - 1,
            parameters.solver_parameters.atol,
            implicit_gradient=False,
        )
        first_success: Bool[Array, "..."] = first_sol.success

        def run_scan(key: PRNGKeyArray) -> MultiAttemptSolution:
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
                """Selects ``first`` for elements that converged on the first attempt"""
                mask: Bool[Array, "..."] = jnp.expand_dims(
                    first_success, tuple(range(first_success.ndim, first.ndim))
                )
                return jnp.where(mask, first, sweep)

            return jax.tree.map(keep_first, first_sol, sweep_sol)

        def run_single_step(_: PRNGKeyArray) -> MultiAttemptSolution:
            """All batch elements converged at the final tau on the first attempt: return
            immediately."""
            return first_sol

        # If all entries converged at the final tau on the first attempt, skip the sweep entirely
        all_converged: Bool[Array, ""] = jnp.all(first_success)
        multi_sol: MultiAttemptSolution = lax.cond(all_converged, run_single_step, run_scan, key)

        # The sweep finishes at the final tau, so attach the derivatives of the root there
        polished_solution: Float[Array, "... solution"] = implicit_polish(
            batch_solver, multi_sol.value, multi_sol.success, parameters
        )
        multi_sol = eqx.tree_at(lambda m: m.solution.value, multi_sol, polished_solution)

        return multi_sol

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
    initial_guess_function: Callable = make_initial_guess_function(parameters, initializer)
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
        base_solution_array = initial_guess_function(parameters, base_solution_array)

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
                parameters.solver_parameters.max_starts - 1,
                parameters.solver_parameters.atol,
            )

        multi_sol = lax.cond(condition, solve_with_stability, solve_without_stability, operand=key)
        output: Output = Output(parameters, multi_sol, base_solution_array)

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
    initial_guess_function: Callable = make_initial_guess_function(parameters, initializer)
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
        base_solution_array = initial_guess_function(parameters, base_solution_array)
        multi_sol = tau_sweep_solver(base_solution_array, parameters, subkey)
        return Output(parameters, multi_sol, base_solution_array)

    @eqx.filter_jit
    def solver_without_stability(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "n_batch twice_species"],
    ) -> Output:
        """Generic batch retry solver path for systems without active stability."""
        _, subkey = jax.random.split(key)
        base_solution_array = initial_guess_function(parameters, base_solution_array)
        multi_sol = batch_retry_solver(
            base_solution_array,
            parameters,
            subkey,
            parameters.solver_parameters.max_starts - 1,
            parameters.solver_parameters.atol,
        )
        return Output(parameters, multi_sol, base_solution_array)

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
    initial_guess_function: Callable = make_initial_guess_function(parameters, initializer)
    batch_solver: Callable = make_batch_solver(parameters)
    objective_function_vmapped: Callable = eqx.filter_vmap(
        objective_function, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )

    @eqx.filter_jit
    def batch_only_solver(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "n_batch twice_species"],
    ) -> Output:
        """Basic batch solver path without retry or tau-sweep enhancements.

        A model is solved on its single attempt if it passes the same objective-based check and
        solver status as in the batch retry solver, so the reported attempts and steps are zero
        otherwise.
        """
        # Note: key is unused in basic batch solver, but included for interface compatibility
        del key
        base_solution_array = initial_guess_function(parameters, base_solution_array)
        multi_sol: MultiAttemptSolution = batch_solver(base_solution_array, parameters)

        converged: Bool[Array, "..."] = (
            max_norm(objective_function_vmapped, multi_sol.value, parameters)
            < parameters.solver_parameters.atol
        )
        success: Bool[Array, "..."] = jnp.logical_and(converged, multi_sol.solver_success)
        solution: optx.Solution = eqx.tree_at(
            lambda sol: sol.stats["num_steps"],
            multi_sol.solution,
            jnp.where(success, multi_sol.num_steps, jnp.zeros_like(multi_sol.num_steps)),
        )
        multi_sol = MultiAttemptSolution(solution, success.astype(int))

        return Output(parameters, multi_sol, base_solution_array)

    return batch_only_solver


# Select the default JIT solver factory here for development and benchmarking.
make_solver_with_jit: Callable = make_solver_with_jit_single_path
# make_solver_with_jit: Callable = make_solver_with_jit_dual_path
# make_solver_with_jit: Callable = make_solver_with_jit_batch_only
