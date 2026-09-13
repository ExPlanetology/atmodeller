# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

Most solvers return results as :class:`atmodeller.containers.MultiAttemptSolution` or
:class:`~atmodeller.output.Output` objects, with detailed convergence and step statistics.
"""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray
from optimistix import Solution

from atmodeller.constants import (
    LOG_NUMBER_MOLES_LOWER,
    LOG_NUMBER_MOLES_UPPER,
    TAU,
    TAU_MAX,
    TAU_NUM,
)
from atmodeller.engine import compute_implied_log_stability, objective_function
from atmodeller.initial_solution import generate_initial_guess
from atmodeller.jax_utils import (
    FloatArray,
    MultiAttemptSolution,
    perturb_around_converged,
    vmap_axes_spec,
)
from atmodeller.jax_utils import make_batch_retry_solver as generic_make_batch_retry_solver
from atmodeller.output import Output
from atmodeller.parameters import Parameters

LOG_NUMBER_MOLES_VMAP_AXES: int = 0
"""Axis index for the solution array in the vmapped batch solver"""
POSTCHECK_TOLERANCE: float = 1.0e-6
"""Default tolerance for the objective-based convergence validation performed after each solve
attempt"""


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


def make_batch_solver(parameters: Parameters) -> Callable:
    """Gets a vmapped batch solver for independent systems.

    Wraps :func:`solve_single` with :func:`equinox.filter_vmap` so that each batch element is
    solved independently, producing per-element convergence statistics. The vmapping axes are
    fixed from ``parameters`` at construction time.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable that returns a :class:`atmodeller.containers.MultiAttemptSolution` with
        ``attempts=1``
    """
    solver_function_vmapped: Callable = eqx.filter_vmap(
        solve_single, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )

    def batch_single_pass_solver(
        solution: Array, parameters: Parameters, *args
    ) -> MultiAttemptSolution:
        """Runs the vmapped single-pass batch solve.

        ``_attempts`` is set to ``1`` unconditionally; objective-based per-element convergence
        checking is delegated to the retry wrapper in ``batch_retry_solver``.

        Args:
            solution: Batched initial guess with shape ``(batch, solution)``
            parameters: Parameters passed through to each vmapped :func:`solve_single` call
            *args: Unused; present for interface consistency with :func:`make_batch_retry_solver`

        Returns:
            :class:`atmodeller.containers.MultiAttemptSolution` with ``attempts=1`` for all batch elements
        """
        del args
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
        Callable that returns a :class:`atmodeller.containers.MultiAttemptSolution` object
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
            LOG_NUMBER_MOLES_LOWER,
            LOG_NUMBER_MOLES_UPPER,
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
        Callable that returns a :class:`atmodeller.containers.MultiAttemptSolution` object
    """
    batch_solver: Callable = make_batch_solver(parameters)
    objective_function_vmapped: Callable = eqx.filter_vmap(
        objective_function, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )
    batch_retry_solver: Callable = make_batch_retry_solver(
        batch_solver, objective_function_vmapped
    )

    return batch_retry_solver


def make_tau_sweep_solver(batch_retry_solver: Callable) -> Callable:
    """Gets a solver function that performs a tau sweep for active stability systems.

    Closes over the provided ``batch_retry_solver``. The returned callable first attempts a solve
    at ``TAU``; if all batch elements converge it returns immediately, otherwise it runs a
    full log-spaced sweep from ``TAU_MAX`` down to ``TAU`` for every batch element.

    Args:
        batch_retry_solver: Pre-built batch retry solver, e.g. from
            :func:`make_batch_retry_solver_from_parameters`

    Returns:
        Callable that returns a :class:`atmodeller.containers.MultiAttemptSolution` object
    """
    get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731
    varying_schedule: Float[Array, " tau"] = jnp.logspace(
        jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
    )

    def tau_sweep_solver(
        initial_guess: Float[Array, "... solution"], parameters: Parameters, key: PRNGKeyArray
    ) -> MultiAttemptSolution:
        """Attempts a solve at ``TAU`` and, if any element fails, runs a full tau sweep.

        First tries to solve all batch elements at ``TAU`` with multistart retry. If every element
        converges, the result is returned immediately. Otherwise a log-spaced schedule from
        ``TAU_MAX`` down to ``TAU`` is swept via :func:`jax.lax.scan`, applying the solver at each
        step for all batch elements. Because ``tau`` must remain a scalar throughout the scan
        (to keep the vmapping axes consistent), converged and failed elements both run the full
        sweep; converged ones simply re-solve quickly from their existing solution.

        Args:
            initial_guess: Batched array of initial guesses
            parameters: :class:`~atmodeller.parameters.Parameters` whose ``tau`` leaf will be
                replaced at each scan step.
            key: JAX PRNG key for reproducible random perturbations

        Returns:
            :class:`atmodeller.containers.MultiAttemptSolution` object
        """

        def solve_tau_step(carry: tuple, tau: Float[Array, " ..."]) -> tuple[tuple, tuple]:
            """Performs a single batched solver step for one scalar tau value.

            Intended for use inside :func:`jax.lax.scan`. Receives a scalar ``tau`` from
            ``varying_schedule``, injects it into a copy of ``parameters``, and runs the batch
            retry solver on all elements. The solution carried forward is the best result from this
            step; the outputs stacked by ``scan`` capture the full history across tau steps.

            Args:
                carry: Tuple of ``(key, solution)`` where ``solution`` has shape
                    ``(batch, solution)``
                tau: Scalar tau value for this scan step

            Returns:
                Updated ``(key, solution)`` carry tuple and an output tuple of
                ``(solution, result_value, num_steps, attempts)``
            """
            (key, solution) = carry
            key, subkey = jax.random.split(key)

            # Get new parameters with tau value
            new_parameters: Parameters = eqx.tree_at(get_leaf, parameters, tau)
            # jax.debug.print("tau = {out}", out=new_parameters.solver_parameters.tau)

            new_sol: MultiAttemptSolution = batch_retry_solver(
                solution,
                new_parameters,
                subkey,
                parameters.solver_parameters.max_starts - 1,
                parameters.solver_parameters.atol,
            )

            new_solution: Float[Array, "... solution"] = new_sol.value
            new_result: optx.RESULTS = new_sol.result
            new_steps: Integer[Array, "..."] = new_sol.stats["num_steps"]
            success_attempt: Integer[Array, "..."] = new_sol.attempts

            new_carry: tuple[PRNGKeyArray, Float[Array, "... solution"]] = (key, new_solution)

            # Output current solution for this tau step
            out: tuple[Array, ...] = (new_solution, new_result._value, new_steps, success_attempt)  # pyright: ignore

            return new_carry, out

        # Initial solve at TAU
        key, subkey = jax.random.split(key)
        initial_parameters: Parameters = eqx.tree_at(get_leaf, parameters, jnp.array(TAU))

        first_sol: MultiAttemptSolution = batch_retry_solver(
            initial_guess,
            initial_parameters,
            subkey,
            parameters.solver_parameters.max_starts - 1,
            parameters.solver_parameters.atol,
        )
        first_solution: Float[Array, "... solution"] = first_sol.value
        # jax.debug.print("first_solution = {out}", out=first_solution)
        # jax.debug.print("solver success = {out}", out=first_sol.result._value)
        first_converged: Bool[Array, "..."] = first_sol.attempts > 0
        # jax.debug.print("first_converged = {out}", out=first_converged)
        first_steps: Integer[Array, "..."] = first_sol.stats["num_steps"]
        # jax.debug.print("first_steps = {out}", out=first_steps)

        def run_scan(args_in: tuple) -> MultiAttemptSolution:
            """Run the full tau sweep scan across all batch elements.

            Called when at least one element failed the initial solve at ``TAU``. All batch
            elements (including those that already converged) run every step of the
            ``varying_schedule`` scan. The final solution, result, and the maximum step count
            and attempt index across all tau steps are returned.
            """
            key, guess, first_converged, first_steps = args_in
            initial_carry_: tuple[PRNGKeyArray, Float[Array, "... solution"]] = (key, guess)
            _, results_ = jax.lax.scan(solve_tau_step, initial_carry_, varying_schedule)
            solution_, result_value_, steps_, attempts_ = results_
            final_result_: optx.RESULTS = cast(
                optx.RESULTS,
                EnumerationItem(result_value_[-1], optx.RESULTS),  # pyright: ignore
            )
            # For steps, report the maximum, either from the initial solve at TAU (if the case
            # converged) or from the tau sweep, to capture the worst-case performance across all
            # tau steps.
            report_steps_ = jnp.where(first_converged, first_steps, jnp.max(steps_, axis=0))

            sol_: Solution = Solution(
                solution_[-1], final_result_, None, {"num_steps": report_steps_}, None
            )
            return MultiAttemptSolution(sol_, jnp.max(attempts_, axis=0))

        def run_single_step(_: tuple) -> MultiAttemptSolution:
            """All batch elements converged at ``TAU`` on the first attempt: return immediately."""
            # jax.debug.print("All converged at TAU on the first attempt. Skipping tau sweep.")
            return first_sol

        # If all entries converged at TAU on the first attempt, skip the sweep entirely
        all_converged: Bool[Array, ""] = jnp.all(first_converged)
        multi_sol: MultiAttemptSolution = lax.cond(
            all_converged,
            run_single_step,
            run_scan,
            operand=(key, first_solution, first_converged, first_steps),
        )

        return multi_sol

    return tau_sweep_solver


def make_solver(parameters: Parameters) -> Callable:
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

    Returns:
        Callable that returns a :class:`~atmodeller.output.Output` object
    """
    batch_retry_solver: Callable = make_batch_retry_solver_from_parameters(parameters)
    tau_sweep_solver: Callable = make_tau_sweep_solver(batch_retry_solver)

    # For debugging to determine if this function is jittable in isolation
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def dispatch_solver(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "#n_batch twice_species"],
    ) -> Output:
        """JIT-compiled entry point that dispatches to the appropriate solver branch.

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
        base_solution_array = generate_initial_guess(parameters, base_solution_array)

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
        output: Output = Output(parameters, multi_sol)

        return output

    return dispatch_solver


# For testing and debugging
# @eqx.debug.assert_max_traces(max_traces=1)
def make_solver_with_jit_dual_path(parameters: Parameters) -> Callable:
    """Gets a JIT-compiled solver with both runtime branches compiled.

    A convenience wrapper around :func:`make_solver` that applies :func:`equinox.filter_jit` to
    the returned solver function. This version compiles both the tau sweep and batch retry paths
    at trace time, dispatching at runtime via :func:`jax.lax.cond`.

    Use this when the model structure may change dynamically between solver construction and
    runtime, or when runtime flexibility is more important than compilation speed.

    Args:
        parameters: Parameters used to derive the vmapping axes and build the sub-solvers at
            construction time

    Returns:
        Callable that returns a :class:`~atmodeller.output.Output` object
    """
    dual_path_solver: Callable = make_solver(parameters)

    return eqx.filter_jit(dual_path_solver)


# Useful for debugging to ensure that the solver is not being re-compiled unnecessarily during
# development iterations
# @eqx.debug.assert_max_traces(max_traces=5)
def make_solver_with_jit_single_path(parameters: Parameters) -> Callable:
    """Gets a JIT-compiled solver with optimized compilation by eliminating unused branches.

    This function inspects ``active_stability`` at construction time (Python-level) to determine
    which solver path to use, then compiles only that path. This results in faster JIT
    compilation, but requires that the active stability structure remain fixed for the lifetime
    of the solver. If the active stability structure changes, a new solver must be constructed.

    Args:
        parameters: Parameters used to derive the vmapping axes and build the sub-solvers at
            construction time. The ``active_stability`` structure is inspected to choose the
            solver path.

    Returns:
        JIT-compiled callable that returns a :class:`~atmodeller.output.Output` object
    """
    batch_retry_solver: Callable = make_batch_retry_solver_from_parameters(parameters)
    tau_sweep_solver: Callable = make_tau_sweep_solver(batch_retry_solver)

    # Determine the solver path at construction time (Python-level evaluation).
    has_active_stability: bool = bool(
        jnp.any(parameters.reaction_system.species.active_stability).item()
    )

    @eqx.filter_jit
    def solver_with_stability(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "#n_batch twice_species"],
    ) -> Output:
        """Tau sweep solver path for systems with active stability species."""
        _, subkey = jax.random.split(key)
        base_solution_array = generate_initial_guess(parameters, base_solution_array)
        multi_sol = tau_sweep_solver(base_solution_array, parameters, subkey)
        return Output(parameters, multi_sol)

    @eqx.filter_jit
    def solver_without_stability(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "#n_batch twice_species"],
    ) -> Output:
        """Generic batch retry solver path for systems without active stability."""
        _, subkey = jax.random.split(key)
        base_solution_array = generate_initial_guess(parameters, base_solution_array)
        multi_sol = batch_retry_solver(
            base_solution_array,
            parameters,
            subkey,
            parameters.solver_parameters.max_starts - 1,
            parameters.solver_parameters.atol,
        )
        return Output(parameters, multi_sol)

    # Return the appropriate pre-compiled solver based on the construction-time check.
    return solver_with_stability if has_active_stability else solver_without_stability


def make_solver_with_jit_batch_only(parameters: Parameters) -> Callable:
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

    Returns:
        JIT-compiled callable that returns a :class:`~atmodeller.output.Output` object
    """
    batch_solver: Callable = make_batch_solver(parameters)

    @eqx.filter_jit
    def batch_only_solver(
        parameters: Parameters,
        key: PRNGKeyArray,
        base_solution_array: Float[Array, "#n_batch twice_species"],
    ) -> Output:
        """Basic batch solver path without retry or tau-sweep enhancements."""
        # Note: key is unused in basic batch solver, but included for interface compatibility
        del key
        base_solution_array = generate_initial_guess(parameters, base_solution_array)
        multi_sol = batch_solver(base_solution_array, parameters)
        return Output(parameters, multi_sol)

    return batch_only_solver


# Select the default JIT solver factory here for development and benchmarking.
make_solver_with_jit: Callable = make_solver_with_jit_single_path
# make_solver_with_jit: Callable = make_solver_with_jit_dual_path
# make_solver_with_jit: Callable = make_solver_with_jit_batch_only
