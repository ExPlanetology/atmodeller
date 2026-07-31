# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone helper functions and utilities for JAX-based scientific modeling

This module provides type aliases, numerically stable mathematical operations, masking utilities,
batch handling, and linear algebra helpers for use with JAX, NumPy, and related libraries. It is
designed to be standalone and does not depend on other modules within the atmodeller package.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree

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
    except Exception:
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
