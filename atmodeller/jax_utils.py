# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone helper functions and utilities for JAX-based scientific modeling

This module provides type aliases, numerically stable mathematical operations, masking utilities,
batch handling, and linear algebra helpers for use with JAX, NumPy, and related libraries. It is
designed to be standalone and does not depend on other modules within the atmodeller package.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pandas as pd
from equinox._enum import EnumerationItem
from jax import lax
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, PyTree
from lineax import AbstractLinearSolver

MAX_FLOAT64 = np.finfo(np.float64).max
"""Largest finite value representable by float64 (approximately 1.8e308)"""
MIN_FLOAT64 = np.finfo(np.float64).min
"""Most negative finite value representable by float64 (approximately -1.8e308)"""
TINY_FLOAT64 = np.finfo(np.float64).tiny
"""Smallest positive normal value representable by float64 (approximately 2.2e-308)"""

# Type aliases
FloatArray: TypeAlias = Float[Array, "..."]
"""Type alias for a JAX float array of any shape"""
NpArray: TypeAlias = np.ndarray
"""Type alias for a NumPy array"""
NpBool: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
"""Type alias for a :obj:`numpy.bool_` array"""
NpFloat: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
"""Type alias for a :obj:`numpy.float64` array"""
NpInt: TypeAlias = np.ndarray[Any, np.dtype[np.intp]]
"""Type alias for a :obj:`numpy.intp` array"""
Scalar: TypeAlias = int | float
"""Scalar"""
OptxSolver: TypeAlias = (
    optx.AbstractRootFinder | optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser
)
"""Optimistix solver"""

logger: logging.Logger = logging.getLogger(__name__)


def as_j64(x: ArrayLike | tuple) -> FloatArray:  # pragma: no cover
    """Converts an array-like or tuple to a JAX array of dtype float64.

    Args:
        x: An array-like object or tuple to convert

    Returns:
        A JAX array of dtype float64 with the same shape as the input
    """
    return jnp.asarray(x, dtype=jnp.float64)


def power_law(
    values: ArrayLike, constant: ArrayLike, exponent: ArrayLike
) -> Array:  # pragma: no cover
    """Computes a power law.

    Args:
        values: Array of input values
        constant: Power law constant
        exponent: Power law exponent

    Returns:
        Array of the same shape as ``values`` containing the result of the power law
    """
    return jnp.power(values, exponent) * constant


def safe_exp(x: ArrayLike) -> Array:  # pragma: no cover
    """Computes a numerically stable elementwise exponential with explicit handling of -inf.

    This function extends :func:`jax.numpy.exp` with safeguards for common numerical issues:

    - Clips inputs to prevent overflow of ``exp(x)`` for large positive values.
    - Clips inputs to prevent underflow of ``exp(x)`` for large negative values.
    - Treats ``-inf`` inputs explicitly and returns 0 for those entries
      (i.e., preserves the identity ``exp(-inf) = 0``).
    - Avoids nans by replacing invalid values before applying ``exp``.

    Note:
        This function is intended for computations performed in log-space or masked representations
        where ``-inf`` denotes absent or zero-valued contributions. It preserves standard autodiff
        behavior by default (no gradient suppression).

    Args:
        x: Input array-like values

    Returns:
        Array of the same shape as ``x`` containing ``exp(x)`` computed in a numerically stable way
        with special handling of ``-inf``.
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.float64)

    is_neg_inf: Bool[Array, "..."] = jnp.isneginf(x)

    # Replace -inf with something safe before clipping
    x_safe: Array = jnp.where(is_neg_inf, 0.0, x)

    # Define lower and upper bounds for clipping from the active dtype.
    finfo = jnp.finfo(x.dtype)
    min_clip = jnp.log(finfo.tiny)
    max_clip = jnp.log(finfo.max)

    # Clip to prevent both underflow and overflow (except for -inf)
    x_clipped: Array = jnp.clip(x_safe, min_clip, max_clip)

    y: Array = jnp.exp(x_clipped)

    # Restore semantics: exp(-inf) = 0
    y = jnp.where(is_neg_inf, 0.0, y)

    # Kill gradients through masked entries
    # y = jnp.where(is_neg_inf, jax.lax.stop_gradient(y), y)

    return y


def safe_log10(x: ArrayLike) -> Array:  # pragma: no cover
    """Computes a numerically stable elementwise base-10 logarithm.

    Floors the input at the smallest positive representable value for its dtype before taking the
    logarithm, so ``log10(0)`` (``-inf``) cannot occur.

    Args:
        x: Input array-like values

    Returns:
        Array of the same shape as ``x`` containing ``log10(x)``, with non-positive entries floored
        at ``log10(finfo(dtype).tiny)``.
    """
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.float64)

    tiny = jnp.finfo(x.dtype).tiny

    return jnp.log10(jnp.maximum(x, tiny))


def safe_pow10(x: ArrayLike) -> Float[Array, "..."]:
    """Numerically safe ``10**x``"""
    return safe_exp(jnp.asarray(x) * jnp.log(10.0))


def masked_logsumexp(
    log_x: Float[Array, "... n"], axis: int = -1, keepdims: bool = True
) -> FloatArray:  # pragma: no cover
    """Computes a numerically stable log-sum-exp with explicit masking of -inf values.

    This function extends the standard :func:`jax.scipy.special.logsumexp` with support for masked
    inputs, where ``-inf`` values are treated as absent (i.e., zero contribution in linear space).

    - Replaces ``-inf`` values with a large negative finite number to ensure numerical stability
      during computation while preserving masking semantics.
    - Allows ``+inf`` values to propagate through :func:`jax.scipy.special.logsumexp`.
    - Computes the log-sum-exp in a stable manner using JAX's implementation.
    - Preserves the semantics that if all values along the reduction axis are masked,
      the result is ``-inf``.
    - Avoids nans and remains compatible with automatic differentiation.

    Note:
        This function is intended for use in log-domain computations (e.g., probabilities,
        thermodynamic quantities) where ``-inf`` encodes zero or absent contributions.

    Args:
        log_x: Input array of log-values, where ``-inf`` indicates masked entries
        axis: Axis or axes over which to compute the log-sum-exp. Defaults to ``-1`` (last axis).
        keepdims: Whether to retain reduced dimensions with length 1. Defaults to ``True``.

    Returns:
        An array containing the log-sum-exp over the specified axis, with masked inputs properly
        handled and ``-inf`` returned if all entries are masked.
    """
    dtype = log_x.dtype
    neg_large = jnp.finfo(dtype).min

    is_neg_inf: Bool[Array, "..."] = jnp.isneginf(log_x)

    # Replace -inf with large negative number (safe for autodiff)
    safe_log_x: FloatArray = jnp.where(is_neg_inf, neg_large, log_x)

    out: FloatArray = logsumexp(safe_log_x, axis=axis, keepdims=keepdims)

    # If everything was masked -> return -inf (strict logic)
    all_invalid: Bool[Array, "..."] = jnp.all(is_neg_inf, axis=axis, keepdims=keepdims)
    out: FloatArray = jnp.where(all_invalid, -jnp.inf, out)

    # Kill gradients if nothing exists
    # out: FloatArray = jnp.where(all_invalid, jax.lax.stop_gradient(out), out)

    return out


def to_hashable(x: Callable) -> Callable:  # pragma: no cover
    """Wraps a callable to make it hashable for JAX transformations.

    This wrapper is useful when passing bound methods of Equinox PyTrees (with JAX arrays as
    attributes) to transformations like :func:`jax.jit`, :func:`jax.vmap`, or :func:`jax.lax.scan`.
    It wraps the callable in a lambda to forward all arguments while avoiding JAX trying to trace
    the method itself. See discussion: https://github.com/patrick-kidger/equinox/issues/1011

    Args:
        x: A callable to wrap

    Returns:
        A hashable lambda forwarding all arguments to the original callable.
    """
    return lambda *args, **kwargs: x(*args, **kwargs)


def get_batch_size(x: PyTree) -> int:  # pragma: no cover
    """Determines the maximum batch size (i.e., length along axis ``0``) amongst all JAX arrays.

    This inspects every leaf in the pytree and checks whether it is a JAX array. Scalars contribute
    a size of ``1``, while arrays contribute the length of their leading dimension (``shape[0]``).
    The result is the largest such size found.

    Args:
        x: Pytree of nested containers

    Returns:
        The maximum leading dimension size across all JAX arrays
    """
    max_size: int = 1
    for leaf in jax.tree_util.tree_leaves(x):
        # logger.debug("leaf = %s", leaf)
        if isinstance(leaf, jax.Array):
            # logger.debug("Found JAX array")
            max_size = max(max_size, leaf.shape[0] if leaf.ndim else 1)
            # logger.debug("max_size: %s", max_size)

    return max_size


def to_native_floats(value: Any) -> Any:
    """Recursively converts any structure to nested tuples of native floats.

    This is useful for converting entries that should be static (non-array) to store in a pytree
    structure, such as when using JAX or Equinox, where static (non-traceable) values must be
    Python floats or tuples thereof.

    Args:
        value: A scalar, list/tuple/array of floats, or nested thereof

    Returns:
        A float or nested tuple of floats
    """
    # Scalars (covers Python, NumPy, JAX scalars)
    if jnp.isscalar(value):
        return float(value)

    # Pandas DataFrame: convert to list of rows (as tuples)
    if isinstance(value, pd.DataFrame):
        iterable: Iterable = value.itertuples(index=False, name=None)
        return tuple(to_native_floats(row) for row in iterable)

    # Array-like (NumPy, JAX)
    if hasattr(value, "ndim"):
        return tuple(to_native_floats(sub) for sub in value.tolist())

    # Generic iterables (lists, tuples, etc.)
    try:
        iterable = list(value)
    except Exception:  # pragma: no cover
        raise TypeError(f"Cannot convert to float or iterate over type {type(value)}")

    return tuple(to_native_floats(item) for item in iterable)


def get_batch_axis(x: Any) -> Literal[0, None]:
    """Determines the batch axis for a JAX array.

    This function checks if the input is a JAX array and has at least one dimension. If so, it
    returns ``0``, indicating that the array should be batched along the leading dimension for use
    with :func:`jax.vmap`. Otherwise, it returns ``None``, indicating that the input should not be
    treated as batched.

    Note:
        This function only considers JAX arrays for batching. While :func:`equinox.is_array`
        regards both JAX and NumPy arrays as arrays for tracing, NumPy arrays are treated here as
        static constants and are never batched. This allows fixed matrices to remain inside pytrees
        without being inadvertently vectorised.

    Args:
        x: Object to check for batching

    Returns:
        ``0`` if batched along axis ``0``, otherwise ``None``
    """
    if isinstance(x, jax.Array):
        if x.ndim >= 1:
            return 0
    return None  # explicit fallback


def vmap_axes_spec(x: PyTree) -> PyTree[Literal[0, None]]:
    """Recursively generate ``in_axes`` for :func:`jax.vmap` over a pytree.

    Only JAX arrays are considered for batching. NumPy arrays and other objects are treated as
    static constants (not batched).

    Args:
        x: A pytree potentially containing JAX arrays, NumPy arrays, or scalars

    Returns:
        A pytree with the same structure as ``x``. Each leaf is ``0`` if batched, or ``None``
        if not.
    """
    return tree_map(get_batch_axis, x)


def partial_rref(matrix: NpArray) -> NpArray:
    """Computes a partial reduced row echelon form (RREF) to determine linear components.

    This function performs the computation using NumPy in-place operations and is therefore not
    compatible with JAX transformations. The returned matrix represents the linear components of
    the input, extracted from the augmented RREF procedure.

    Args:
        matrix: A 2-D NumPy array of shape (nrows, ncols).

    Returns:
        A matrix containing the linear components.
    """
    nrows, ncols = matrix.shape

    augmented_matrix: NpArray = np.hstack((matrix, np.eye(nrows)))
    # logger.debug("augmented_matrix = \n%s", augmented_matrix)
    # Permutation matrix
    # P: NpArray = np.eye(nrows)

    # Forward elimination with partial pivoting
    for i in range(min(nrows, ncols)):
        # Pivot selection with check
        nonzero: NpInt = np.flatnonzero(augmented_matrix[i:, i])
        # logger.debug("nonzero = %s", nonzero)
        if nonzero.size == 0:
            # logger.debug("i: %d. No pivot in this column.", i)
            continue  # no pivot in this column
        # Absolute row index of first non-zero index
        pivot_row: np.int_ = nonzero[0] + i
        # Swap if pivot row is not already in place
        if pivot_row != i:
            augmented_matrix[[i, pivot_row], :] = augmented_matrix[[pivot_row, i], :]
            # P[[i, nonzero_row], :] = P[[nonzero_row, i], :]

        # Perform row operations to eliminate values below the pivot.
        pivot_value: np.float64 = augmented_matrix[i, i]
        if i + 1 < nrows:
            factors = augmented_matrix[i + 1 :, i : i + 1] / pivot_value  # shape (nrows-i-1, 1)
            augmented_matrix[i + 1 :] -= factors * augmented_matrix[i]

    # logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

    # Backward substitution
    for i in range(min(nrows, ncols) - 1, -1, -1):
        pivot_value = augmented_matrix[i, i]
        if pivot_value == 0:
            # logger.debug("i: %d. Pivot is zero, skipping backward elimination.", i)
            continue  # skip columns with no pivot
        # Normalize the pivot row.
        augmented_matrix[i] /= augmented_matrix[i, i]

        # Eliminate entries above the pivot
        if i > 0:
            factors = augmented_matrix[:i, i : i + 1] / pivot_value  # shape (i, 1)
            augmented_matrix[:i] -= factors * augmented_matrix[i]

    # logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

    # reduced_matrix: NpArray = augmented_matrix[:, :ncols]
    component_matrix: NpArray = augmented_matrix[min(ncols, nrows) :, ncols:]
    # logger.debug("reduced_matrix = \n%s", reduced_matrix)
    # logger.debug("component_matrix = \n%s", component_matrix)
    # logger.debug("permutation_matrix = \n%s", P)

    return component_matrix


def max_norm(
    objective_function: Callable, solution: Float[Array, "... solution"], parameters: PyTree
) -> FloatArray:  # pragma: no cover
    """Computes the L-infinity norm of batched objective residuals.

    Evaluates the objective function for each model in the batch and returns the maximum absolute
    residual across all components of each system. This is a vectorised variant of
    :func:`optimistix.max_norm`, producing one scalar L-infinity norm per system in the batch.

    See: https://docs.kidger.site/optimistix/api/norms/

    Args:
        objective_function: A callable taking ``solution`` and ``parameters`` that returns the
            objective residuals for each model in the batch
        solution: Batched array of candidate solutions
        parameters: Parameters passed to the objective function

    Returns:
        L-infinity norm
    """
    return jnp.linalg.norm(objective_function(solution, parameters), ord=jnp.inf, axis=-1)


def stack_broadcast(arrays: Sequence[Array], axis: int = -1) -> Array:
    """Broadcasts a sequence of arrays to a common shape and stacks them.

    Args:
        arrays: sequence of arrays with identical trailing semantics but possibly different leading
            batch shapes.
        axis: axis along which to stack after broadcasting. Defaults to ``-1`` (last axis).

    Returns:
        Stacked array with broadcasted leading dimensions.
    """
    shapes: list[tuple[int, ...]] = [arr.shape for arr in arrays]
    target_shape: tuple[int, ...] = jnp.broadcast_shapes(*shapes)
    arrays_: list[Array] = [jnp.broadcast_to(arr, target_shape) for arr in arrays]

    return jnp.stack(arrays_, axis=axis)


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

    def get_solver_instance(self) -> OptxSolver:
        """Instantiates the solver"""
        return self.solver(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            linear_solver=self.linear_solver,  # type: ignore because there is a parameter
            # For debugging LM solver. Not valid for all solvers (e.g. Newton)
            # verbose=frozenset({"step_size", "y", "loss", "accepted"}),
        )


class MultiAttemptSolution(eqx.Module):  # pragma: no cover
    """A solution wrapper for handling multiple solver attempts per problem

    This class standardises solver outputs from multi-attempt strategies. Some attributes
    (e.g. ``converged``, ``solver_success``, ``num_steps``) are broadcast to the batch dimension
    to ensure consistent shapes across all outputs, whether the underlying solver returns scalar
    or per-attempt values.

    Args:
        solution: Optimistix solution
        _attempts: Number of attempts required for each batch element to converge (``0`` indicates
            no successful attempt). Defaults to ``0``.
    """

    solution: optx.Solution
    _attempts: ArrayLike = 0

    @property
    def attempts(self) -> Integer[Array, " batch"]:
        return jnp.broadcast_to(self._attempts, self.batch_shape)

    @property
    def aux(self) -> Any:
        """Auxiliary payload returned by the underlying solver"""
        return self.solution.aux

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape (all dimensions except the trailing solution dimension)"""
        return self.solution.value.shape[:-1]

    @property
    def converged(self) -> Bool[Array, " batch"]:
        """Boolean mask indicating objective-based convergence"""
        return jnp.broadcast_to(self.attempts > 0, self.batch_shape)

    @property
    def num_steps(self) -> Integer[Array, " batch"]:
        """Number of steps"""
        return jnp.broadcast_to(self.stats["num_steps"], self.batch_shape)

    @property
    def result(self) -> optx.RESULTS:
        """Raw Optimistix solver result code"""
        return self.solution.result

    @property
    def value(self) -> Float[Array, "batch solution"]:
        return self.solution.value

    @property
    def solver_success(self) -> Bool[Array, " batch"]:
        """Whether the underlying solver claims success"""
        return jnp.broadcast_to(self.solution.result == optx.RESULTS.successful, self.batch_shape)

    @property
    def state(self) -> Any:
        """Internal solver state from the underlying Optimistix solution"""
        return self.solution.state

    @property
    def stats(self) -> dict[str, PyTree[ArrayLike]]:
        """Solver statistics from the underlying Optimistix solution"""
        return self.solution.stats

    @property
    def success(self) -> Bool[Array, " batch"]:
        """Whether the solution is successful based on both convergence and solver success"""
        return jnp.logical_and(self.converged, self.solver_success)

    def asdict(self) -> dict[str, ArrayLike]:
        """Converts pertinent solution statistics to a dictionary"""
        return {
            "solver_success": self.solver_success,
            "steps": self.num_steps,
            "attempts": self.attempts,
            "converged": self.converged,
            "success": self.success,
        }

    def stats_to_logger(self, logger_: logging.Logger = logger) -> None:
        """Logs solver statistics.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            logger_: Logger to log the statistics to. Defaults to :obj:`logger`.
        """
        total_models: int = int(self.converged.size)
        num_successful_models: int = jnp.count_nonzero(self.converged).item()
        num_failed_models: int = jnp.count_nonzero(~self.converged).item()

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

        # Count unique values and their frequencies, ignoring failed models (attempts == 0)
        successful_attempts = self.attempts[self.attempts > 0]
        unique_vals, counts = jnp.unique(successful_attempts, return_counts=True)
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            if val == 1:
                msg = "Solve attempt %d: %d (%0.2f%%) model(s) solved"
            else:
                msg = "Solve attempt %d: %d (%0.2f%%) additional model(s) solved"
            logger_.info(msg, val, count, count * 100 / total_models)

        # Steps of 0 indicate no solution; replace with nan and report the max over solved models
        steps_float: Array = cast(
            Array, jnp.where(self.num_steps == 0, jnp.nan, self.num_steps.astype(float))
        )
        max_steps: Array = jnp.nanmax(steps_float)
        logger_.info("Solver steps (max) = %s", int(max_steps.item()))


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
    depend on ``max_retries`` (an unrelated, caller-side setting) and, for a large retry budget,
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


def make_batch_retry_solver(
    solver_function: Callable,
    objective_fn: Callable,
    perturb_fn: Callable[
        [Float[Array, "... solution"], Integer[Array, "..."], PRNGKeyArray, PyTree],
        Float[Array, "... solution"],
    ],
) -> Callable:  # pragma: no cover
    """Makes a generic batch retry solver, with the perturbation strategy supplied by the caller.

    This factors out the retry-loop mechanics (perturb failed rows, re-solve, check objective-based
    convergence, keep newly-successful rows, repeat) that are identical across every batch-retry
    solver built on top of :class:`MultiAttemptSolution` -- both within this package and in
    downstream packages -- which otherwise differ only in what the solution vector represents and
    how it should be perturbed. See :func:`perturb_around_converged` for a ready-made perturbation
    strategy that can be composed into ``perturb_fn``.

    ``solver_function``, ``objective_fn``, and ``perturb_fn`` must be pure JAX-callable functions
    compatible with :func:`equinox.filter_jit`. They must not close over non-JAX state or produce
    Python side effects.

    Args:
        solver_function: Callable that performs a single (batched) solve. Must accept an initial
            guess and a pytree of parameters, and return a :class:`MultiAttemptSolution`.
        objective_fn: Callable for the (batched) objective function
        perturb_fn: Callable ``(solution, attempt, key, parameters) -> perturbed`` returning a
            candidate perturbed solution for *every* row (converged or not); this
            factory only substitutes the rows where ``attempt == 0``, so ``perturb_fn`` does not
            need to preserve converged rows itself

    Returns:
        Callable that returns a :class:`MultiAttemptSolution` object
    """

    def batch_retry_solver(
        initial_guess: Float[Array, "... solution"],
        parameters: PyTree,
        key: PRNGKeyArray,
        max_retries: int,
        tolerance: float = 1.0e-6,
    ) -> MultiAttemptSolution:
        """Batched solver with retry and perturbation for failed cases.

        Runs a batched solver function on a set of initial guesses. If some entries fail to
        converge, the function perturbs only the failed solutions (via ``perturb_fn``) and
        retries, up to ``max_retries``. Successfully converged solutions are kept fixed throughout.

        Note:
            - ``solution.result``: solver's internal convergence classification
            - ``attempts``: first iteration satisfying objective-based check
            - ``attempts == 0``: never converged within the initial attempt plus ``max_retries``
              retries

        Args:
            initial_guess: Batched array of initial guesses for the solver
            parameters: Model parameters passed to the solver and to ``perturb_fn``
            key: JAX PRNG key for reproducible random perturbations
            max_retries: Maximum number of solver retries per batch entry
            tolerance: Tolerance for the objective-based convergence validation performed after
                each solve attempt. Defaults to ``1.0e-6``.

        Returns:
            :class:`MultiAttemptSolution` object
        """

        def body_fn(state: tuple[Array, Array, Array, Array, Array, Array]) -> tuple:
            """Performs one retry iteration for failed solutions."""
            i, key, solution, result_value, steps, attempt = state

            failed_mask: Bool[Array, "..."] = attempt == 0

            key, subkey = jax.random.split(key)
            perturbed: Float[Array, "... solution"] = perturb_fn(
                solution, attempt, subkey, parameters
            )
            new_initial_solution: Float[Array, "... solution"] = cast(
                Array, jnp.where(failed_mask[..., None], perturbed, solution)
            )

            new_sol: MultiAttemptSolution = solver_function(new_initial_solution, parameters)
            new_solution: Float[Array, "... solution"] = new_sol.value
            new_result_value: Integer[Array, "..."] = new_sol.result._value  # pyright: ignore

            new_converged: Bool[Array, "..."] = (
                max_norm(objective_fn, new_solution, parameters) < tolerance
            )
            new_solver_success: Bool[Array, "..."] = new_sol.solver_success
            new_success: Bool[Array, "..."] = jnp.logical_and(new_converged, new_solver_success)
            new_num_steps: Integer[Array, "..."] = new_sol.num_steps

            # Determine which entries to update: previously failed, now succeeded
            update_mask: Bool[Array, "..."] = jnp.logical_and(failed_mask, new_success)
            updated_solution: Float[Array, "... solution"] = cast(
                Array, jnp.where(update_mask[..., None], new_solution, solution)
            )
            updated_result_value: Integer[Array, "..."] = jnp.where(
                update_mask, new_result_value, result_value
            )
            updated_num_steps: Integer[Array, "..."] = cast(
                Array, jnp.where(update_mask, new_num_steps, steps)
            )
            updated_attempt: Array = jnp.where(update_mask, i, attempt)  # pyright: ignore

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

            ``i`` starts at 2 (the second overall attempt), so to allow ``max_retries`` retries the
            body must run while ``i`` is in ``{2, ..., max_retries + 1}``, hence the ``+ 2`` below.
            """
            i, _, _, _, _, attempt = state

            continue_loop: Bool[Array, "..."] = jnp.logical_and(
                jnp.any(attempt == 0), i < max_retries + 2
            )

            return continue_loop

        # Try first solution
        first_sol: MultiAttemptSolution = solver_function(initial_guess, parameters)
        first_solution: Float[Array, "... solution"] = first_sol.value

        # Perform a per-system check
        first_converged: Bool[Array, "..."] = (
            max_norm(objective_fn, first_solution, parameters) < tolerance
        )
        first_solver_success: Bool[Array, "..."] = first_sol.solver_success
        first_result_value: Integer[Array, "..."] = jnp.broadcast_to(
            first_sol.result._value,  # pyright: ignore
            first_converged.shape,
        )
        first_num_steps: Integer[Array, "..."] = jnp.broadcast_to(
            first_sol.num_steps, first_converged.shape
        )

        # Failback solution to initial guess for failed models
        first_success: Bool[Array, "..."] = jnp.logical_and(first_converged, first_solver_success)
        solution: Float[Array, "... solution"] = cast(
            Array, jnp.where(first_success[..., None], first_solution, initial_guess)
        )

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
