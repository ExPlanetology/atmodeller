# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Batched initializer, which completes a batch of initial guesses with a cold guess"""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float

from atmodeller.initializer.core import InitializerProtocol
from atmodeller.initializer.element_potential import ElementPotentialInitializer
from atmodeller.jax_utils import vmap_axes_spec
from atmodeller.parameters import Parameters
from atmodeller.solver_utils import SOLUTION_BATCH_AXIS


def make_batch_initializer(
    parameters: Parameters, initializer: InitializerProtocol | None = None
) -> Callable:
    """Gets a function that completes a batched initial guess with the initializer's cold guess.

    Rows of the initial guess are used as given, and rows that contain ``NaN`` are replaced by the
    cold guess from ``initializer``. The initializer acts on a single system, so it is wrapped
    with :func:`equinox.filter_vmap` with the vmapping axes fixed from ``parameters`` at
    construction time, as for :func:`~atmodeller.solvers.make_batch_solver`. The decision to
    evaluate it is made for the whole batch, outside the vmap, so that the cold guess is skipped
    when no row contains ``NaN``.

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
        cold_start, in_axes=(vmap_axes_spec(parameters), SOLUTION_BATCH_AXIS)
    )

    def batch_initializer(
        parameters: Parameters, initial_guess: Float[Array, "batch solution"]
    ) -> Float[Array, "batch solution"]:
        """Replaces rows of ``initial_guess`` that contain ``NaN`` with the cold guess.

        Args:
            parameters: Parameters
            initial_guess: Initial guess with shape ``(batch, solution)``. Rows that
                contain ``NaN`` have no initial guess.

        Returns:
            Initial guess with shape ``(batch, solution)``
        """
        # Mask: True for rows where any entry is NaN along the last axis
        nan_mask: Bool[Array, "batch 1"] = jnp.any(
            jnp.isnan(initial_guess), axis=-1, keepdims=True
        )

        def fill_from_cold_start(
            initial_guess: Float[Array, "batch solution"],
        ) -> Float[Array, "batch solution"]:
            """Selects the cold guess for rows with any NaN, otherwise keeps the given row."""
            return jnp.where(
                nan_mask,
                cold_start_vmapped(parameters, initial_guess),
                initial_guess,
            )

        # The predicate is a scalar for the whole batch, so the cold guess is skipped when every
        # row has an initial guess
        result: Float[Array, "batch solution"] = lax.cond(
            jnp.any(nan_mask), fill_from_cold_start, lambda x: x, initial_guess
        )

        result = eqx.error_if(
            result,
            jnp.any(jnp.isnan(result)),
            "NaN encountered in initial guess, which will break the solver.",
        )

        return result

    return batch_initializer
