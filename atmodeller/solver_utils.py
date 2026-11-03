# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Generic helpers for solving batches of systems of equations with Optimistix

This module provides solver building blocks that do not depend on Atmodeller's model, so they can
be reused by other packages:

- :class:`RootFindParameters`: the configuration of an Optimistix root finder
- :class:`RESULTS` and :class:`CheckedSolution`: the checked outcome of a solve for each
  model, alongside the Optimistix solution
- :func:`checked_root_find`: solves a single system with any objective function and checks the
  residual, which is the only place the objective function is needed
- :func:`make_batch_retry_solver`: retries the models of a batch that failed, from perturbed
  initial guesses, e.g. with :func:`perturb_around_converged`
- :func:`implicit_polish`: attaches the derivatives of a root found by a procedure that cannot be
  reverse-differentiated

It depends only on :mod:`atmodeller.jax_utils` within the atmodeller package.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, PyTree
from lineax import AbstractLinearSolver

from atmodeller.jax_utils import OptxSolver, stop_gradient_arrays

logger: logging.Logger = logging.getLogger(__name__)

SOLUTION_BATCH_AXIS: int = 0
"""Batch axis of the solution array, which is vmapped by the batched solvers and initializers"""


class RootFindParameters(eqx.Module):
    """Parameters for Optimistix root finding

    Args:
        solver: Solver. Defaults to :class:`optimistix.Newton`.
        atol: Absolute tolerance. Defaults to ``1.0e-6``.
        rtol: Relative tolerance. Defaults to ``1.0e-6``.
        linear_solver: Linear solver. Defaults to ``AutoLinearSolver(well_posed=None)``.
        norm: Norm. Defaults to :func:`optimistix.max_norm`.
        throw: How to report any failures. Defaults to ``False``.
        max_steps: The maximum number of steps the solver can take. Defaults to ``256``.
        jac: Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian.
            Can be either ``fwd`` or ``bwd``. Defaults to ``fwd``.
        best_so_far: Whether to wrap the solver in :class:`optimistix.BestSoFarRootFinder`, which
            returns the iterate with the smallest residual rather than the last one, e.g. for a
            solve that converges and then diverges. For a solve that fails, this may report a point
            closer to a root than :func:`checked_root_find`, which compares only the starting
            point and the final iterate. However, the wrapper scores each iterate before its step,
            so the final iterate is never scored, and a solve that converges returns the iterate
            before it, whose residual may exceed the tolerance of the check. It is therefore off by
            default. Defaults to ``False``.
        max_starts: Maximum number of attempts for each model, the first attempt and its retries
            (see :func:`make_batch_retry_solver`). Defaults to ``10``.
    """

    solver: type[OptxSolver] = optx.Newton
    """Solver"""
    atol: float = 1.0e-6
    """Absolute tolerance"""
    rtol: float = 1.0e-6
    """Relative tolerance"""
    linear_solver: AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    """Linear solver (see https://docs.kidger.site/lineax/api/solvers/)"""
    norm: Callable = optx.max_norm
    """Norm"""
    throw: bool = False
    """How to report any failures"""
    max_steps: int = 256
    """Maximum number of steps the solver can take"""
    jac: Literal["fwd", "bwd"] = "fwd"
    """Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian"""
    best_so_far: bool = False
    """Whether to wrap the solver in :class:`optimistix.BestSoFarRootFinder`"""
    max_starts: int = 10
    """Maximum number of attempts for each model"""

    def get_solver_instance(self) -> OptxSolver:
        """Instantiates the solver, wrapped in :class:`optimistix.BestSoFarRootFinder` if
        ``best_so_far``"""
        solver: OptxSolver = self.solver(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            linear_solver=self.linear_solver,  # type: ignore because there is a parameter
            # For debugging LM solver. Not valid for all solvers (e.g. Newton)
            # verbose=frozenset({"step_size", "y", "loss", "accepted"}),
        )
        if self.best_so_far:
            return optx.BestSoFarRootFinder(solver)  # pyright: ignore[reportArgumentType]

        return solver


class RESULTS(eqx.Enumeration):
    """Result of a solve for each model

    A model is solved only if the solver converged and the residual of the objective function is
    within the tolerance. For a model that was not solved, the result describes its best attempt,
    the one with the smallest residual. The result of the solver itself is :class:`optimistix.RESULTS`
    (:attr:`CheckedSolution.optimistix_result`).
    """

    successful = ""
    solver_failed = (
        "The solver did not converge on the best attempt, and every attempt failed. The "
        "Optimistix result gives the reason."
    )
    objective_check_failed = (
        "The solver converged on the best attempt, but the residual of the objective function "
        "exceeds the tolerance, and every attempt failed."
    )


class CheckedSolution(eqx.Module):  # pragma: no cover
    """Checked solution of a solve with one or more attempts for each model

    A model is solved if the solver converged and the residual of the objective function is within
    the tolerance, on one of its attempts. This is always the final, checked outcome of a solve.

    Args:
        solution: Optimistix solution of the attempt kept for each model: the attempt on which the
            model was solved or, for a model that was not solved, its best attempt, the one with
            the smallest residual. The solutions of a batch may come from different attempts. The
            value of a model that was not solved is the point of its best attempt closest to a
            root, which is either the final iterate or the starting point (see
            :func:`checked_root_find`), and ``aux`` and ``state`` are not kept.
        result: Result for each model
        stats: Statistics for each model: ``successful_attempt``, the attempt on which the model
            was solved (``0`` if it was not solved), ``num_steps``, the number of solver steps on
            that attempt (``0`` if it was not solved), and ``residual``, the infinity norm of the
            objective function at the value
    """

    solution: optx.Solution
    result: RESULTS
    stats: dict[str, PyTree[ArrayLike]]

    @property
    def aux(self) -> Any:
        """Auxiliary payload returned by the underlying solver"""
        return self.solution.aux

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape (all dimensions except the trailing solution dimension)"""
        return self.solution.value.shape[:-1]

    @property
    def num_steps(self) -> Integer[Array, " batch"]:
        """Number of solver steps on the attempt on which each model was solved, or ``0``"""
        return jnp.broadcast_to(self.stats["num_steps"], self.batch_shape)

    @property
    def optimistix_result(self) -> optx.RESULTS:
        """Optimistix result of the attempt kept for each model"""
        return self.solution.result

    @property
    def residual(self) -> Float[Array, " batch"]:
        """Infinity norm of the objective function at the value of each model"""
        return jnp.broadcast_to(self.stats["residual"], self.batch_shape)

    @property
    def state(self) -> Any:
        """Internal solver state from the underlying Optimistix solution"""
        return self.solution.state

    @property
    def successful_attempt(self) -> Integer[Array, " batch"]:
        """Attempt on which each model was solved, or ``0`` if it was not solved"""
        return jnp.broadcast_to(self.stats["successful_attempt"], self.batch_shape)

    @property
    def solver_success(self) -> Bool[Array, " batch"]:
        """Whether the solver converged on the attempt kept for each model"""
        return jnp.broadcast_to(
            self.optimistix_result == optx.RESULTS.successful, self.batch_shape
        )

    @property
    def success(self) -> Bool[Array, " batch"]:
        """Whether each model is solved"""
        return jnp.broadcast_to(self.result == RESULTS.successful, self.batch_shape)

    @property
    def value(self) -> Float[Array, "batch solution"]:
        return self.solution.value

    def asdict(self) -> dict[str, ArrayLike]:
        """Converts pertinent solution statistics to a dictionary"""
        return {
            "solver_success": self.solver_success,
            "steps": self.num_steps,
            "successful_attempt": self.successful_attempt,
            "success": self.success,
            "result": jnp.broadcast_to(self.result._value, self.batch_shape),  # pyright: ignore
            "residual": self.residual,
        }

    def stats_to_logger(self, logger_: logging.Logger = logger) -> None:
        """Logs solver statistics.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            logger_: Logger to log the statistics to. Defaults to the module logger.
        """
        total_models: int = int(self.success.size)
        num_successful_models: int = jnp.count_nonzero(self.success).item()
        num_failed_models: int = jnp.count_nonzero(~self.success).item()

        logger_.info(
            "Solve complete: %d (%0.2f%%) successful model(s)",
            num_successful_models,
            num_successful_models * 100 / total_models,
        )
        if num_failed_models > 0:
            logger_.warning(
                "%d (%0.2f%%) model(s) still failed",
                num_failed_models,
                num_failed_models * 100 / total_models,
            )

        # Count unique values and their frequencies over the successful models
        successful_attempts = self.successful_attempt[self.success]
        unique_vals, counts = jnp.unique(successful_attempts, return_counts=True)
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            if val == 1:
                msg = "Solve attempt %d: %d (%0.2f%%) model(s) solved"
            else:
                msg = "Solve attempt %d: %d (%0.2f%%) additional model(s) solved"
            logger_.info(msg, val, count, count * 100 / total_models)

        # Maximum over the successful models, or 0 if none succeeded, as failed models report 0
        max_steps: int = int(jnp.max(jnp.where(self.success, self.num_steps, 0)).item())
        logger_.info("Solver steps (max) = %d", max_steps)


def checked_root_find(
    fn: Callable,
    y0: Float[Array, " solution"],
    args: PyTree,
    root_find_parameters: RootFindParameters,
    *,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
) -> CheckedSolution:  # pragma: no cover
    """Solves a single system with :func:`optimistix.root_find`, and checks it.

    The solver and its configuration (``throw``, ``max_steps`` and ``jac``) come from
    ``root_find_parameters``. The system is solved if the solver converged and the residual of
    ``fn`` at the solution is within ``root_find_parameters.atol`` in the infinity norm. This is
    the only place where the objective function is needed: the batched and retry solvers built on
    it receive checked solutions.

    A system that is not solved reports the point closer to a root, by the residual: the final
    iterate or ``y0``. The final iterate of a failed solve may have diverged, so the value of the
    returned Optimistix solution is then ``y0``. A residual that is NaN counts as infinite.

    Args:
        fn: Objective function ``fn(y, args)`` returning the residual, as for
            :func:`optimistix.root_find`
        y0: Initial guess
        args: Arguments passed to ``fn``, e.g. the parameters of the model
        root_find_parameters: Configuration of the root finder
        lower: Lower bound on the solution, for a solver that supports it (e.g.
            :class:`optimistix.Newton`). Defaults to ``None``, i.e. no bound.
        upper: Upper bound on the solution, as for ``lower``. Defaults to ``None``.

    Returns:
        Checked :class:`CheckedSolution` of this single attempt
    """

    def residual_norm(y: Float[Array, " solution"]) -> Float[Array, ""]:
        """Infinity norm of the residual at ``y``, with NaN as infinity"""
        norm: Array = jnp.linalg.norm(fn(y, args), ord=jnp.inf)
        return jnp.where(jnp.isnan(norm), jnp.inf, norm)

    options: dict[str, Any] = {"jac": root_find_parameters.jac}
    if lower is not None:
        options["lower"] = lower
    if upper is not None:
        options["upper"] = upper

    sol: optx.Solution = optx.root_find(
        fn,
        root_find_parameters.get_solver_instance(),
        y0,
        args=args,
        options=options,
        max_steps=root_find_parameters.max_steps,
        throw=root_find_parameters.throw,
    )

    solver_success: Bool[Array, ""] = sol.result == optx.RESULTS.successful
    residual: Float[Array, ""] = residual_norm(sol.value)
    converged: Bool[Array, ""] = residual < root_find_parameters.atol
    success: Bool[Array, ""] = jnp.logical_and(solver_success, converged)
    result: RESULTS = RESULTS.where(
        success,
        RESULTS.successful,
        RESULTS.where(solver_success, RESULTS.objective_check_failed, RESULTS.solver_failed),
    )

    # A failed solve reports its starting point if that is closer to a root than its final iterate
    initial_residual: Float[Array, ""] = residual_norm(y0)
    use_start: Bool[Array, ""] = jnp.logical_and(~success, initial_residual <= residual)
    sol = eqx.tree_at(lambda s: s.value, sol, jnp.where(use_start, y0, sol.value))

    stats: dict[str, Array] = {
        "successful_attempt": success.astype(int),
        "num_steps": jnp.where(success, sol.stats["num_steps"], 0),
        "residual": jnp.where(use_start, initial_residual, residual),
    }

    return CheckedSolution(sol, result, stats)


def perturb_around_converged(
    array: Float[Array, "batch dim"],
    attempt: Integer[Array, " batch"],
    key: PRNGKeyArray,
    lower: ArrayLike,
    upper: ArrayLike,
) -> Float[Array, "batch dim"]:  # pragma: no cover
    """Perturbs each column of ``array`` around the central tendency of its converged rows.

    Intended for use inside a :func:`make_batch_retry_solver` ``perturb_fn``: computes a per-column
    median and range from the rows that have already converged (``attempt > 0``), then draws a
    uniform perturbation scaled by that range around the median. Each retry draws a fresh key, so
    repeated calls already sample different candidate points without needing to widen the window --
    an earlier version scaled the window by the retry count, but that made the window's endpoint
    depend on ``max_starts`` (an unrelated, caller-side setting) and, for a large retry budget,
    collapsed into clipping almost every draw onto ``lower``/``upper`` instead of usefully covering
    a wide range. Falls back to the whole array's own statistics on the rare chance that no row has
    converged yet.

    Note:
        Returns candidate perturbed values for *every* row, including already-converged ones.
        Callers are expected to select which rows to actually keep; :func:`make_batch_retry_solver`
        only substitutes rows where ``attempt == 0``, so this function does not need to preserve
        converged rows itself.

    Args:
        array: Batched array to perturb, shape ``(batch, dim)``
        attempt: Attempt index at which each row first converged (``0`` if not yet converged)
        key: JAX PRNG key
        lower: Lower bound to clip the perturbed values to
        upper: Upper bound to clip the perturbed values to

    Returns:
        Candidate perturbed array, same shape as ``array``
    """
    converged_mask: Bool[Array, " batch"] = attempt > 0
    has_converged: Bool[Array, ""] = jnp.any(converged_mask)

    reference: Float[Array, "batch dim"] = jnp.where(converged_mask[:, None], array, jnp.nan)
    central_value: Float[Array, "1 dim"] = jnp.where(
        has_converged,
        jnp.nanmedian(reference, axis=0, keepdims=True),
        jnp.median(array, axis=0, keepdims=True),
    )
    data_max: Float[Array, "1 dim"] = jnp.where(
        has_converged,
        jnp.nanmax(reference, axis=0, keepdims=True),
        jnp.max(array, axis=0, keepdims=True),
    )
    data_min: Float[Array, "1 dim"] = jnp.where(
        has_converged,
        jnp.nanmin(reference, axis=0, keepdims=True),
        jnp.min(array, axis=0, keepdims=True),
    )
    data_range: Float[Array, "1 dim"] = data_max - data_min

    raw_perturb: Float[Array, "batch dim"] = jax.random.uniform(
        key, shape=array.shape, minval=-1.0, maxval=1.0
    )
    perturbed: Float[Array, "batch dim"] = data_range / 2 * raw_perturb + central_value

    return cast(Array, jnp.clip(perturbed, lower, upper))


def implicit_polish(
    solver_function: Callable,
    solution: Float[Array, "... solution"],
    success: Bool[Array, "..."],
    parameters: PyTree,
) -> Float[Array, "... solution"]:
    """Re-solves from a converged solution to attach derivatives with respect to ``parameters``.

    The solve starts at the root, so it converges in about one step and leaves the solution
    unchanged to within the solver tolerance. Because :func:`optimistix.root_find` differentiates
    with the implicit function theorem, the returned solution has the derivatives of the root with
    respect to ``parameters``, in both forward and reverse mode, regardless of how the root was
    found. This allows the root to be located by procedures that cannot be reverse-differentiated,
    such as a :func:`jax.lax.while_loop`, provided they run on inputs with stopped gradients.

    Args:
        solver_function: Callable that performs a single (batched) solve. Must accept an initial
            guess and a pytree of parameters, and return a :class:`CheckedSolution`.
        solution: Converged solution, which must not carry gradients
        success: Boolean mask of the batch entries that converged. Other entries are returned
            unchanged and have zero derivatives.
        parameters: Parameters to differentiate with respect to

    Returns:
        Solution with derivatives with respect to ``parameters``
    """
    polished: CheckedSolution = solver_function(solution, parameters)

    return cast(Array, jnp.where(success[..., None], polished.value, solution))


def make_batch_retry_solver(
    solver_function: Callable,
    perturb_fn: Callable[
        [Float[Array, "... solution"], Integer[Array, "..."], PRNGKeyArray, PyTree],
        Float[Array, "... solution"],
    ],
) -> Callable:  # pragma: no cover
    """Makes a generic batch retry solver, with the perturbation strategy supplied by the caller.

    This factors out the retry-loop mechanics (perturb failed rows, re-solve, keep newly-successful
    rows and the best attempt of the others, repeat) that are identical across every batch-retry
    solver built on top of :class:`CheckedSolution` -- both within this package and in
    downstream packages -- which otherwise differ only in what the solution vector represents and
    how it should be perturbed. See :func:`perturb_around_converged` for a ready-made perturbation
    strategy that can be composed into ``perturb_fn``.

    Each attempt is checked by ``solver_function``, for example a batch of
    :func:`~atmodeller.solver_utils.checked_root_find`, so no objective function is needed here.
    A model that is not solved keeps its best attempt, the one with the smallest ``residual``.

    ``solver_function`` and ``perturb_fn`` must be pure JAX-callable functions compatible with
    :func:`equinox.filter_jit`. They must not close over non-JAX state or produce Python side
    effects.

    Args:
        solver_function: Callable that performs a single (batched) solve. Must accept an initial
            guess and a pytree of parameters, and return a checked :class:`CheckedSolution` with
            ``residual`` in its ``stats``.
        perturb_fn: Callable ``(solution, attempt, key, parameters) -> perturbed`` returning a
            candidate perturbed solution for *every* row (converged or not); this
            factory only substitutes the rows where ``attempt == 0``, so ``perturb_fn`` does not
            need to preserve converged rows itself

    Returns:
        Callable that returns a :class:`CheckedSolution` object
    """

    def batch_retry_solver(
        initial_guess: Float[Array, "... solution"],
        parameters: PyTree,
        key: PRNGKeyArray,
        max_starts: int,
        implicit_gradient: bool = True,
    ) -> CheckedSolution:
        """Batched solver with retry and perturbation for failed cases.

        Runs a batched solver function on a set of initial guesses. If some entries fail to
        converge, the function perturbs only the failed solutions (via ``perturb_fn``) and
        retries, up to ``max_starts`` attempts in total. Successfully converged solutions are kept fixed throughout.

        The retries only locate the root, so they run with stopped gradients. With
        ``implicit_gradient``, a final solve from the root (see :func:`implicit_polish`) attaches
        the derivatives of the root with respect to ``parameters``, which makes the solver
        differentiable in both forward and reverse mode. The retry history does not contribute to
        the derivatives.

        Note:
            - ``result``: :class:`RESULTS` of each model, which for a model that was not solved
              describes its best attempt, the one with the smallest residual
            - ``successful_attempt``: attempt on which each model was solved, or ``0`` if it was
              not solved within ``max_starts`` attempts

        Args:
            initial_guess: Batched array of initial guesses for the solver
            parameters: Model parameters passed to the solver and to ``perturb_fn``
            key: JAX PRNG key for reproducible random perturbations
            max_starts: Maximum number of attempts for each model, the first attempt and its
                retries, e.g. :attr:`RootFindParameters.max_starts`
            implicit_gradient: Whether to attach derivatives with respect to ``parameters`` with a
                final solve. Set to ``False`` when the caller attaches them itself, for example
                after a sequence of solves. Defaults to ``True``.

        Returns:
            :class:`CheckedSolution` object
        """
        # The retries only locate the root, so stop gradients to avoid differentiating through the
        # while loop, which does not support reverse mode
        initial_guess = lax.stop_gradient(initial_guess)
        search_parameters: PyTree = stop_gradient_arrays(parameters)

        def attempt_solve(guess: Float[Array, "... solution"]) -> tuple[Array, ...]:
            """Solves from ``guess``.

            Returns:
                Value, Optimistix result value, :class:`RESULTS` value, success, number of solver
                steps and residual for each model
            """
            sol: CheckedSolution = solver_function(guess, search_parameters)
            value: Float[Array, "... solution"] = sol.value
            batch_shape: tuple[int, ...] = value.shape[:-1]

            result_value: Integer[Array, "..."] = jnp.broadcast_to(
                sol.optimistix_result._value,  # pyright: ignore
                batch_shape,
            )
            status: Integer[Array, "..."] = jnp.broadcast_to(
                sol.result._value,  # pyright: ignore
                batch_shape,
            )
            num_steps: Integer[Array, "..."] = jnp.broadcast_to(
                sol.solution.stats["num_steps"], batch_shape
            )

            return value, result_value, status, sol.success, num_steps, sol.residual

        def body_fn(state: tuple[Array, ...]) -> tuple[Array, ...]:
            """Performs one retry iteration for failed solutions."""
            i, key, solution, result_value, status, steps, attempt, residual = state

            failed_mask: Bool[Array, "..."] = attempt == 0

            key, subkey = jax.random.split(key)
            perturbed: Float[Array, "... solution"] = perturb_fn(
                solution, attempt, subkey, search_parameters
            )
            new_initial_solution: Float[Array, "... solution"] = cast(
                Array, jnp.where(failed_mask[..., None], perturbed, solution)
            )

            (
                new_solution,
                new_result_value,
                new_status,
                new_success,
                new_num_steps,
                new_residual,
            ) = attempt_solve(new_initial_solution)

            # Entries that previously failed take every quantity from this attempt if it solved
            # them or, failing that, if its residual is smaller, so a model that is not solved
            # keeps its best attempt
            solved_mask: Bool[Array, "..."] = jnp.logical_and(failed_mask, new_success)
            update_mask: Bool[Array, "..."] = jnp.logical_and(
                failed_mask, jnp.logical_or(new_success, new_residual < residual)
            )
            updated_solution: Float[Array, "... solution"] = cast(
                Array, jnp.where(update_mask[..., None], new_solution, solution)
            )
            updated_result_value: Integer[Array, "..."] = jnp.where(
                update_mask, new_result_value, result_value
            )
            updated_status: Integer[Array, "..."] = jnp.where(update_mask, new_status, status)
            updated_num_steps: Integer[Array, "..."] = cast(
                Array, jnp.where(solved_mask, new_num_steps, steps)
            )
            updated_attempt: Array = jnp.where(solved_mask, i, attempt)  # pyright: ignore
            updated_residual: Float[Array, "..."] = jnp.where(update_mask, new_residual, residual)

            return (
                i + 1,
                key,
                updated_solution,
                updated_result_value,
                updated_status,
                updated_num_steps,
                updated_attempt,
                updated_residual,
            )

        def cond_fn(state: tuple[Array, ...]) -> Bool[Array, "..."]:
            """Determines whether additional solver retries are needed.

            ``i`` is the attempt about to be made, starting at 2 (the second overall attempt), so the
            body runs while ``i`` is in ``{2, ..., max_starts}``.
            """
            i, _, _, _, _, _, attempt, _ = state

            continue_loop: Bool[Array, "..."] = jnp.logical_and(
                jnp.any(attempt == 0), i <= max_starts
            )

            return continue_loop

        # Try first solution. A failed model reports its initial guess if the final iterate is no
        # closer to a root (see checked_root_find).
        (
            first_solution,
            first_result_value,
            first_status,
            first_success,
            first_num_steps,
            first_residual,
        ) = attempt_solve(initial_guess)

        initial_state: tuple = (
            jnp.array(2),  # Second overall attempt
            key,
            first_solution,
            first_result_value,
            first_status,
            first_num_steps,
            first_success.astype(int),  # 1 for solved, otherwise 0
            first_residual,
        )

        (
            _,
            _,
            final_solution,
            final_result_value,
            final_status,
            final_num_steps,
            final_attempt,
            final_residual,
        ) = lax.while_loop(cond_fn, body_fn, initial_state)

        if implicit_gradient:
            final_solution = implicit_polish(
                solver_function, final_solution, final_attempt > 0, parameters
            )

        # The Optimistix solution of the attempt kept for each model. It does not keep the
        # auxiliary output or the internal state of the solves.
        final_result: optx.RESULTS = cast(
            optx.RESULTS,
            EnumerationItem(final_result_value, optx.RESULTS),  # pyright: ignore
        )
        sol: optx.Solution = optx.Solution(
            final_solution, final_result, None, {"num_steps": final_num_steps}, None
        )

        # Zero out steps for failed entries so that reported steps are not misleadingly non-zero
        # for models that never converged
        stats: dict[str, Array] = {
            "successful_attempt": final_attempt,
            "num_steps": cast(
                Array,
                jnp.where(final_attempt > 0, final_num_steps, jnp.zeros_like(final_num_steps)),
            ),
            "residual": final_residual,
        }
        result: RESULTS = cast(
            RESULTS,
            EnumerationItem(final_status, RESULTS),  # pyright: ignore
        )

        return CheckedSolution(sol, result, stats)

    return batch_retry_solver
