# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Interface and shared infrastructure for initializers of the solver's initial guess

An initializer provides the cold initial guess, the concatenated array of [log_number_moles,
log_stability], from which the solver starts when no previous solution is available. It acts on a
single system and is vmapped over the batch by the solver, in the same way as
:func:`~atmodeller.solvers.solve_single`.

The solver decides which batch elements need a cold guess: rows of the initial guess passed to the
solver are used as given, and rows that contain ``NaN`` are replaced by the initializer's guess.
If no row contains ``NaN``, the initializer is not evaluated.

Concrete implementations are in :mod:`atmodeller.initializer.heuristic` and
:mod:`atmodeller.initializer.element_potential`. This module also holds what they share: the trace
value for species with a negligible element budget, and the moles of species limited by the
element budget, the fraction of its limiting element's budget that a condensate predicted (or
switched on) as stable starts with.
"""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from atmodeller.parameters import Parameters

LOG_TRACE_VALUE: float = -20.0
"""Small trace value (in log space) to assign to species that have a negligible element budget"""

CONDENSATE_ALLOCATION_FRACTION: float = 0.5
"""Fraction of the budget of its limiting element that a condensate predicted (or switched on) as
stable starts with

This value, although arbitrary to some extent, prevents condensates from being allocated an
excessively large portion of the element budget, which can drive gas species to unrealistically
low initial abundances.
"""


def max_moles_by_limiting_element(
    formula_matrix: Float[Array, "n_elements n_species"],
    element_abundance: Float[Array, "... n_elements"],
    mask: Bool[Array, "... n_species"],
) -> Float[Array, "... n_species"]:
    r"""Maximum moles by limiting element for species in ``mask``

    For each species in ``mask``, the mole count is estimated by asking: given the available
    element budget, how many moles of this species could be formed if that element were shared
    equally among all masked species that contain it? The tightest such constraint across all
    elements that appear in the species formula determines the estimate. For species :math:`s` in
    the mask :math:`M`, with formula matrix :math:`a_{es}` and element abundance :math:`b_e`,

    .. math::

        \hat{n}_s = \min_{e:\ a_{es} > 0,\ b_e\ \text{not NaN}}
            \frac{b_e}{\sum_{s' \in M} a_{es'}},

    which is the moles at which the masked species that contain element :math:`e`, all with the
    same moles, together use all of :math:`e`. Returns ``NaN`` for species not in ``mask`` or not
    constrained by any available element.

    Args:
        formula_matrix: Matrix of elemental formulas for each species
        element_abundance: Element abundance. ``NaN`` for unconstrained elements.
        mask: Boolean mask selecting the species to allocate budget to.

    Returns:
        Per-species mole estimates; ``NaN`` where unconstrained.
    """
    constrained_element: Bool[Array, "... n_elements"] = ~jnp.isnan(element_abundance)

    # Broadcast all masks to (..., n_elements, n_species)
    fm_mask: Bool[Array, "n_elements n_species"] = formula_matrix > 0
    mask_b: Bool[Array, "... 1 n_species"] = mask[..., None, :]
    constrained_element_b: Bool[Array, "... n_elements 1"] = constrained_element[..., :, None]

    stoich_total: Float[Array, "... n_elements"] = jnp.sum(formula_matrix * mask_b, axis=-1)
    safe_stoich: Float[Array, "... n_elements"] = jnp.where(stoich_total > 0, stoich_total, 1.0)
    share: Float[Array, "... n_elements"] = element_abundance / safe_stoich

    is_constrained: Bool[Array, "... n_elements n_species"] = (
        fm_mask & mask_b & constrained_element_b
    )
    # jax.debug.print("is_constrained = {out}", out=is_constrained)
    implied: Float[Array, "... n_elements n_species"] = jnp.where(
        is_constrained, share[..., :, None], jnp.nan
    )
    # jax.debug.print("implied = {out}", out=implied)

    max_moles_by_limiting_element: Float[Array, "... n_species"] = jnp.nanmin(implied, axis=-2)
    # jax.debug.print("max_moles_by_limiting_element = {out}", out=max_moles_by_limiting_element)

    return max_moles_by_limiting_element


@runtime_checkable
class InitializerProtocol(Protocol):
    """Protocol for an initializer that provides the cold initial guess for the solver

    Implementations should be stateless :class:`equinox.Module` instances whose settings are plain
    Python numbers, which are constant under JIT, so reusing an initializer never triggers
    retracing.
    """

    def start(self, parameters: Parameters) -> Float[Array, " twice_species"]:
        """Cold initial guess for a single system.

        Args:
            parameters: Parameters of a single system

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        ...
