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
from typing import Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax, random
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
from atmodeller.jax_utils import FloatArray, MultiAttemptSolution, max_norm, vmap_axes_spec
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

    # For debugging to determine if this function is jittable in isolation
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def batch_retry_solver(
        initial_guess: Float[Array, "... solution"],
        parameters: Parameters,
        key: PRNGKeyArray,
        max_retries: int,
        tolerance: float = POSTCHECK_TOLERANCE,
    ) -> MultiAttemptSolution:
        """Batched solver with retry and perturbation for failed cases

        Runs a batched solver function on a set of initial guesses. If some entries fail to
        converge, the function perturbs only the failed solutions and retries, up to
        ``max_retries``. Successfully converged solutions are kept fixed throughout.

        This approach is useful when solving large batches of nonlinear systems where certain
        initial guesses may fail. Perturbations help the solver escape poor local minima or flat
        regions of the objective function.

        Note:
            - ``solution.result``: solver's internal convergence classification
            - ``attempts``: first iteration satisfying objective-based check
            - ``attempts == 0``: never converged within the initial attempt plus
              ``max_retries`` retries

        Args:
            initial_guess: Batched array of initial guesses for the solver
            parameters: Model parameters passed to the solver
            key: JAX PRNG key for reproducible random perturbations
            max_retries: Maximum number of solver retries per batch entry
            tolerance: Tolerance for the objective-based convergence validation performed after
                each solve attempt. Defaults to :obj:`POSTCHECK_TOLERANCE`.

        Returns:
            :class:`atmodeller.containers.MultiAttemptSolution` object
        """

        def body_fn(state: tuple[Array, Array, Array, Array, Array, Array]) -> tuple:
            """Performs one retry iteration for failed solutions.

            This function executes a single iteration of the solver retry loop. It perturbs only
            the solutions that previously failed, reruns the solver, and updates the batch state
            accordingly. Successfully converged entries remain unchanged.

            Args:
                tuple:
                    i: Current attempt index
                    key: Random key for perturbation generation
                    solution: Current batch of solution estimates
                    result_value: Current result value of the solver for each entry
                    steps: Number of solver steps recorded for each entry
                    attempt: Attempt index when each entry first succeeded or 0 if it did not
                         converge at all.

            Returns:
                Updated state tuple with the same structure as in the input
            """
            i, key, solution, result_value, steps, attempt = state
            # jax.debug.print("Iteration: {out}", out=i)

            failed_mask: Bool[Array, "..."] = attempt == 0  # Not yet converged per objective check
            # jax.debug.print("failed_mask = {out}", out=failed_mask)

            # Split solution into log_number_moles and log_stability
            log_number_moles, log_stability = jnp.split(solution, 2, axis=-1)

            # Perturbation for log number of moles
            key, subkey = random.split(key)
            perturb_shape: tuple[int, ...] = log_number_moles.shape
            raw_perturb = random.uniform(subkey, shape=perturb_shape, minval=-1.0, maxval=1.0)
            # jax.debug.print("raw_perturb = {out}", out=raw_perturb)

            # Compute a central tendency for perturbation:
            # - In a batch, leverage the successful solves to find a central value for each species
            #   (column), and perturb failed cases around this value. This assumes batch entries
            #   are similar enough for the median to be a meaningful reference, which is often true
            #   in practice.
            # - For a single system, simply use the median of the log_number_moles as the central
            #   value for perturbation, and similarly compute the data range across all species.

            if log_number_moles.ndim == 2:
                # Batched: shape (batch, n_species)
                axis: Literal[0, None] = 0
            else:
                # Single system: shape (n_species,)
                axis = None

            central_value: FloatArray = jnp.median(log_number_moles, axis=axis, keepdims=True)
            data_max: FloatArray = jnp.max(log_number_moles, axis=axis, keepdims=True)
            data_min: FloatArray = jnp.min(log_number_moles, axis=axis, keepdims=True)
            data_range: FloatArray = data_max - data_min
            # jax.debug.print("central_value = {out}", out=central_value)
            # jax.debug.print("data_range = {out}", out=data_range)
            perturbed: Float[Array, "... n_species"] = data_range / 2 * raw_perturb + central_value
            perturbed = jnp.minimum(perturbed, LOG_NUMBER_MOLES_UPPER)
            perturbed = jnp.maximum(perturbed, LOG_NUMBER_MOLES_LOWER)
            # jax.debug.print("perturbed = {out}", out=perturbed)

            # Perturb only the failed cases, keep successful cases unchanged
            new_log_number_moles: Float[Array, "... n_species"] = cast(
                Float[Array, "... n_species"],
                jnp.where(failed_mask[..., None], perturbed, log_number_moles),
            )
            # jax.debug.print("new_log_number_moles = {out}", out=new_log_number_moles)

            # Re-compute log stability for the new perturbed guess for failed cases, keep unchanged
            # for successful cases
            new_log_stability: Float[Array, "... n_species"] = compute_implied_log_stability(
                parameters, new_log_number_moles
            )
            new_log_stability = jnp.where(failed_mask[..., None], new_log_stability, log_stability)
            # jax.debug.print("new_log_stability = {out}", out=new_log_stability)

            # Recombine
            new_initial_solution: Float[Array, "... twice_species"] = jnp.concatenate(
                [new_log_number_moles, new_log_stability], axis=-1
            )
            # jax.debug.print("new_initial_solution = {out}", out=new_initial_solution)

            new_sol: MultiAttemptSolution = solver_function(new_initial_solution, parameters)
            new_solution: Float[Array, "... solution"] = new_sol.value
            # jax.debug.print("new_solution = {out}", out=new_solution)

            new_result_value: Integer[Array, "..."] = new_sol.result._value  # pyright: ignore
            # jax.debug.print("new_result_value = {out}", out=new_result_value)

            # Perform a per-system convergence check.
            new_converged: Bool[Array, "..."] = (
                max_norm(objective_fn, new_solution, parameters) < tolerance
            )
            # jax.debug.print("new_converged = {out}", out=new_converged)

            # Also get the status of the solver
            new_solver_success: Bool[Array, "..."] = new_sol.solver_success
            # jax.debug.print("new_solver_success = {out}", out=new_solver_success)

            # Failback solution to initial guess for failed models
            new_success: Bool[Array, "..."] = jnp.logical_and(new_converged, new_solver_success)
            # jax.debug.print("new_success = {out}", out=new_success)

            new_num_steps: Integer[Array, "..."] = new_sol.stats["num_steps"]
            # jax.debug.print("new_num_steps = {out}", out=new_num_steps)

            # Determine which entries to update: previously failed, now succeeded
            update_mask: Bool[Array, "..."] = jnp.logical_and(failed_mask, new_success)
            # jax.debug.print("update_mask = {out}", out=update_mask)
            updated_solution: Float[Array, "... solution"] = cast(
                Array, jnp.where(update_mask[..., None], new_solution, solution)
            )
            updated_result_value: Integer[Array, "..."] = jnp.where(
                update_mask, new_result_value, result_value
            )
            # jax.debug.print("updated_result_value = {out}", out=updated_result_value)
            updated_num_steps: Integer[Array, "..."] = cast(
                Array, jnp.where(update_mask, new_num_steps, steps)
            )
            # jax.debug.print("updated_num_steps = {out}", out=updated_num_steps)
            updated_attempt: Array = jnp.where(update_mask, i, attempt)  # pyright: ignore
            # jax.debug.print("updated_attempt = {out}", out=updated_attempt)

            return (
                i + 1,
                key,
                updated_solution,
                updated_result_value,
                updated_num_steps,
                updated_attempt,
            )

        def cond_fn(
            state: tuple[Array, Array, Array, Array, Array, Array],
        ) -> Bool[Array, "..."]:
            """Determines whether additional solver retries are needed.

            This condition function controls the ``lax.while_loop``. The retry loop continues as
            long as at least one batch entry has not converged and the maximum number of attempts
            has not been reached.

            Args:
                tuple:
                    i: Current attempt index
                    _: Unused (PRNG key)
                    _: Unused (current batch solution)
                    _: Unused (result value)
                    _: Unused (number of steps)
                    attempts: Unused (success attempt index)

            Returns:
                ``True`` if any entry has failed and the number of attempts is less than
                    ``max_retries``; otherwise ``False``.
            """
            i, _, _, _, _, attempt = state

            # For debugging to force the loop to run to the maximum allowable value
            # return jnp.logical_and(i < max_retries, True)

            # Convergence is determined by `check_convergence`, which enforces the objective
            # tolerance on each batch entry individually. We track the first successful attempt
            # index in `attempts`. An entry is considered converged if attempts > 0, ensuring
            # consistency with the convergence mask used elsewhere in the code.
            # i starts at 2 (second overall attempt), so to allow max_retries retries we need
            # the body to run at i in {2, ..., max_retries+1}, hence the condition
            # i < max_retries+2.
            continue_loop: Bool[Array, "..."] = jnp.logical_and(
                jnp.any(attempt == 0), i < max_retries + 2
            )

            return continue_loop

        # Try first solution
        # jax.debug.print("Iteration: 1")
        first_sol: MultiAttemptSolution = solver_function(initial_guess, parameters)
        first_solution: Float[Array, "... solution"] = first_sol.value
        # jax.debug.print("first_solution = {out}", out=first_solution)

        # Check the solver result
        # jax.debug.print("first_sol.result = {out}", out=first_sol.result)

        # Perform a per-system check
        first_converged: Bool[Array, "..."] = (
            max_norm(objective_fn, first_solution, parameters) < tolerance
        )
        # jax.debug.print("first_converged = {out}", out=first_converged)

        first_solver_success: Bool[Array, "..."] = first_sol.solver_success
        # jax.debug.print("first_solver_success = {out}", out=first_solver_success)

        first_result_value: Integer[Array, "..."] = jnp.broadcast_to(
            first_sol.result._value, first_converged.shape
        )
        # jax.debug.print("first_result_value = {out}", out=first_result_value)

        first_num_steps: Integer[Array, "..."] = jnp.broadcast_to(
            first_sol.stats["num_steps"], first_converged.shape
        )
        # jax.debug.print("first_num_steps = {out}", out=first_num_steps)

        # Failback solution to initial guess for failed models
        first_success: Bool[Array, "..."] = jnp.logical_and(first_converged, first_solver_success)
        # jax.debug.print("first_success = {out}", out=first_success)

        solution: Float[Array, "... solution"] = cast(
            Array, jnp.where(first_success[..., None], first_solution, initial_guess)
        )
        # jax.debug.print("solution = {out}", out=solution)
        # jax.debug.print("Completed iteration: 1")

        initial_state: tuple = (
            jnp.array(2),  # Second overall attempt
            key,
            solution,
            first_result_value,
            first_num_steps,
            first_success.astype(int),  # 1 for solved, otherwise 0
        )

        _, _, final_solution, final_result_value, final_num_steps, final_attempt = lax.while_loop(
            cond_fn, body_fn, initial_state
        )
        # jax.debug.print("After lax.while_loop")

        # jax.debug.print("final_solution = {out}", out=final_solution)
        # jax.debug.print("final_result_value = {out}", out=final_result_value)
        # jax.debug.print("final_num_steps = {out}", out=final_num_steps)
        # jax.debug.print("final_attempt = {out}", out=final_attempt)

        # Bundle the final outputs into a single optimistix Solution object
        final_result: optx.RESULTS = cast(
            optx.RESULTS,
            EnumerationItem(final_result_value, optx.RESULTS),  # pyright: ignore
        )

        # This solution instance does not return all the information from the solves, but it
        # encapsulates the most important (final) quantities. Zero out steps for failed entries so
        # that reported steps are not misleadingly non-zero for models that never converged.
        final_num_steps_out: Integer[Array, "..."] = cast(
            Array, jnp.where(final_attempt > 0, final_num_steps, jnp.zeros_like(final_num_steps))
        )
        sol: optx.Solution = optx.Solution(
            final_solution, final_result, None, {"num_steps": final_num_steps_out}, None
        )
        multi_sol: MultiAttemptSolution = MultiAttemptSolution(sol, final_attempt)

        return multi_sol

    return batch_retry_solver


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
