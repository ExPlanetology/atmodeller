# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Initial solution estimation

All functions in this module are designed to be compatible with both :func:`jax.vmap`, as used by
the engine and solver routines, and with explicit batched input arrays, as used by output routines.
This means that each function should correctly handle both single-instance and batched input,
broadcasting and returning outputs with shapes consistent with the input batch dimensions. This
ensures seamless integration with both vectorized and batch-processing workflows throughout the
codebase.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Integer

from atmodeller.engine import compute_implied_log_stability
from atmodeller.jax_utils import FloatArray
from atmodeller.parameters import Parameters

LOG_TRACE_VALUE: float = -20.0
"""Small trace value (in log space) to assign to species that have a negligible element budget"""


def max_moles_by_limiting_element(
    formula_matrix: Float[Array, "n_elements n_species"],
    element_abundance: Float[Array, "... n_elements"],
    mask: Bool[Array, "... n_species"],
) -> Float[Array, "... n_species"]:
    """Maximum moles by limiting element for species in ``mask``

    For each species in ``mask``, the mole count is estimated by asking: given the available
    element budget, how many moles of this species could be formed if that element were shared
    equally among all masked species that contain it? The tightest such constraint across all
    elements that appear in the species formula determines the estimate. Returns ``NaN`` for
    species not in ``mask`` or not constrained by any available element.

    Args:
        formula_matrix: Matrix of elemental formulas for each species
        element_abundance: Element abundance. ``NaN`` for unconstrained elements.
        mask: Boolean mask selecting the species to allocate budget to.

    Returns:
        Per-species mole estimates; ``NaN`` where unconstrained.
    """
    constrained_element: Bool[Array, "... n_elements"] = ~jnp.isnan(element_abundance)

    # Broadcast all masks to (..., n_elements, n_species)
    fm_mask: Bool[Array, "1 n_elements n_species"] = (formula_matrix > 0)[None, ...]
    mask_b: Bool[Array, "... 1 n_species"] = mask[..., None, :]
    constrained_element_b: Bool[Array, "... n_elements 1"] = constrained_element[..., :, None]

    stoich_total: Float[Array, "... n_elements"] = jnp.sum(
        formula_matrix[None, ...] * mask_b, axis=-1
    )
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

    # An extra dimension was introduced to ensure correct broadcasting for both 1-D and 2-D
    # (batched) cases, but we can now squeeze it back out for the single-case scenario to be
    # consistent with the shape of the elemental abundance input.
    return jnp.squeeze(max_moles_by_limiting_element)


def allocate_element_budget(
    parameters: Parameters,
    element_abundance: Float[Array, "... n_elements"],
    condensate_stable_mask: Bool[Array, "... n_species"],
) -> tuple[Float[Array, "... n_species"], Float[Array, "... n_species"]]:
    """Allocates the element budget between predicted-stable condensates and other species.

    This function first assigns as much of each element as possible to the species predicted to be
    stable condensates, according to the limiting-reagent principle. It then computes the remaining
    element budget and allocates it to the non-condensate species, again using the limiting-reagent
    logic. Finally, fallback logic is applied to assign small nonzero values or the geometric mean.

    Args:
        parameters: Parameters
        element_abundance: Available abundance of each element
        condensate_stable_mask: Boolean mask indicating which species are predicted-stable
            condensates

    Returns:
        Tuple of two arrays:
            - n_condensate: Moles allocated to each condensate species
            - n_other: Moles allocated to each non-condensate species
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )

    # Allocate element budget to predicted-stable condensates
    n_condensate: Float[Array, "... n_species"] = max_moles_by_limiting_element(
        formula_matrix, element_abundance, condensate_stable_mask
    )
    # jax.debug.print("n_condensate = {out}", out=n_condensate)

    # Assign fallback values to unconstrained (NaN) entries after stable-condensate allocation.
    # Species with NaN allocation are considered unstable or absent for element accounting and are
    # assigned a small trace value (exp(LOG_TRACE_VALUE)). This ensures they do not consume element
    # budget and prevents NaN propagation. Any values assigned to non-condensate species here are
    # ignored in subsequent steps by masking.
    n_condensate_fallback: Float[Array, "... n_species"] = jnp.where(
        jnp.isnan(n_condensate), jnp.exp(LOG_TRACE_VALUE), n_condensate
    )
    # jax.debug.print("n_condensate_fallback = {out}", out=n_condensate_fallback)

    # Total moles of each element consumed by the allocated condensate budget. Note here that the
    # mask ensures that only predicted-stable condensates contribute to the element consumption.
    element_used: Float[Array, "... n_elements"] = jnp.einsum(
        "es,...s->...e", formula_matrix, condensate_stable_mask * n_condensate_fallback
    )
    # jax.debug.print("element_used = {out}", out=element_used)

    # Remaining element budget after stable-condensate allocation, which must be positive or zero.
    remaining_b: Float[Array, "... n_elements"] = jnp.maximum(
        element_abundance - element_used, 0.0
    )
    # jax.debug.print("remaining_b = {out}", out=remaining_b)

    # If present, exclude O2 from the remaining budget, because otherwise the initial estimate for
    # O2 tends to be unreasonably high since oxygen is equally shared among all O-bearing species.
    O2_index: Float[Array, ""] = parameters.reaction_system.phase_system.gas.O2_index
    non_condensate_no_O2_mask: Bool[Array, "... n_species"] = lax.cond(
        jnp.isnan(O2_index),
        lambda: ~condensate_mask,
        lambda: (~condensate_mask).at[O2_index.astype(int)].set(False),  # Additionally turn off O2
    )

    # Allocate remaining budget to non-condensate species excluding O2
    n_other: Float[Array, "... n_species"] = max_moles_by_limiting_element(
        formula_matrix, remaining_b, non_condensate_no_O2_mask
    )
    # NaNs are either O2 (if present) or condensates, both of which should be zeroed for
    # calculating the budget
    n_other = jnp.where(jnp.isnan(n_other), 0.0, n_other)
    # jax.debug.print("n_other = {out}", out=n_other)

    # Recalculate element_used after n_other allocation
    element_used_no_O2: Float[Array, "... n_elements"] = jnp.einsum(
        "es,...s->...e", formula_matrix, n_other
    )
    # jax.debug.print("element_abundance = {out}", out=element_abundance)
    # jax.debug.print("element_used_no_O2 = {out}", out=element_used_no_O2)

    # Update element budget after allocation excluding O2, which must be positive or zero.
    remaining_b = jnp.maximum(element_abundance - element_used - element_used_no_O2, 0.0)
    # jax.debug.print("remaining_b (after n_other) = {out}", out=remaining_b)

    def allocate_to_O2(
        n_other: Float[Array, "... n_species"],
        remaining_b: Float[Array, "... n_elements"],
        formula_matrix: Float[Array, "n_elements n_species"],
        O2_index_int: Integer[Array, ""],
    ) -> Float[Array, "... n_species"]:
        """Allocates remaining O budget to O2 if O2 is present in the system."""
        # NOTE: A sentinel value is required since all branches of lax.cond are traced, but
        # O_index will only ever be meaningful if O2 is present in the system.
        O_index: int = (
            parameters.element_names.index("O") if "O" in parameters.element_names else -1
        )
        # Compute how much O is left
        O_remaining: Float[Array, "..."] = remaining_b[..., O_index]
        # Get stoichiometry of O in O2 (should be 2)
        O2_stoich: Float[Array, ""] = formula_matrix[O_index, O2_index_int]
        n_O2: Float[Array, "..."] = O_remaining / O2_stoich
        # Set the O2 entry in n_other
        n_other = n_other.at[..., O2_index_int].set(n_O2)

        return n_other

    n_other = lax.cond(
        jnp.isnan(O2_index),
        lambda n_other: n_other,
        lambda n_other: allocate_to_O2(n_other, remaining_b, formula_matrix, O2_index.astype(int)),
        n_other,
    )
    # jax.debug.print("n_other after O2 allocation = {out}", out=n_other)

    # Apply fallback logic to non-condensate species.
    # If a species is unconstrained by any available element or constrained to zero moles, fallback
    # values are assigned. For unconstrained species, the fallback value is the geometric mean of
    # the finite estimates, which keeps the magnitude of missing species comparable to known ones.
    # For exhausted elements, the fallback value is a small trace value.
    unconstrained: Bool[Array, "... n_species"] = jnp.isnan(n_other) & ~condensate_mask
    # jax.debug.print("unconstrained = {out}", out=unconstrained)
    exhausted: Bool[Array, "... n_species"] = n_other == 0 & ~condensate_mask
    # jax.debug.print("exhausted = {out}", out=exhausted)
    valid: Bool[Array, "... n_species"] = ~unconstrained & ~exhausted & ~condensate_mask
    # jax.debug.print("valid = {out}", out=valid)

    log_geometric_mean: FloatArray = jnp.sum(
        jnp.where(valid, jnp.log(n_other), 0.0), axis=-1, keepdims=True
    ) / jnp.sum(valid, axis=-1, keepdims=True).astype(float)
    # jax.debug.print("log_geometric_mean = {out}", out=log_geometric_mean)

    n_other_fallback = jnp.where(unconstrained, jnp.exp(log_geometric_mean), n_other)
    n_other_fallback = jnp.where(exhausted, jnp.exp(LOG_TRACE_VALUE), n_other_fallback)

    # jax.debug.print("n_condensate after fallback = {out}", out=n_condensate_fallback)
    # jax.debug.print("n_other after fallback = {out}", out=n_other_fallback)

    return n_condensate_fallback, n_other_fallback


def get_log_activity_estimate(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Estimates log activity for gas species based on ideal-gas assumptions.

    For gas species, the log activity is estimated as log(x_i * P) = log(n_i/n_total) + log(P).
    For non-gas species (melt, solid, pure-phase condensates), the log activity is set to 0,
    i.e. unit activity. This ignores dilution (mole fraction < 1) and activity coefficients
    for melt/solid solution species, but is intentional: the pre-screen is a cheap heuristic
    and calling the EOS/mixing models here would be circular and expensive. Activity
    coefficients are in any case unavailable without a complete solution.

    Args:
        parameters: Parameters containing the reaction system and state information
        species_abundance: Estimated abundance of each species

    Returns:
        Estimated log activity for each species
    """
    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )

    # Must be broadcast to a column array
    pressure: Float[Array, "... 1"] = parameters.state.get_pressure(jnp.log(species_abundance))[
        ..., None
    ]
    n_gas_total: Float[Array, "... 1"] = jnp.nansum(
        jnp.where(gas_mask, species_abundance, 0.0), axis=-1, keepdims=True
    )
    safe_n_gas_total: Float[Array, "... 1"] = jnp.where(n_gas_total > 0, n_gas_total, 1.0)

    # TODO: Here, activity of dissolved species is also computed as unity. To improve.
    log_activity: Float[Array, "... n_species"] = jnp.where(
        gas_mask,
        jnp.log(jnp.where(gas_mask, species_abundance, 1.0))
        - jnp.log(safe_n_gas_total)
        + jnp.log(jnp.where(pressure > 0, pressure, 1.0)),
        0.0,
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # activity: Float[Array, "... n_species"] = jnp.exp(log_activity)
    # jax.debug.print("activity = {out}", out=activity)

    return log_activity


def get_stability_signal(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_reactions n_species"]:
    """Computes the stability signal for each reaction and condensate species.

    The stability signal is defined as sm[r,c] * (log_Kp[r] - log_Q[r]), where sm is the
    stability matrix, Kp is the equilibrium constant, and Q is the reaction quotient. A positive
    signal means the reaction is driven toward condensate formation given the current assumed
    activities.

    Args:
        parameters: Parameters containing the reaction system information
        species_abundance: Estimated abundance of each species, used to estimate activities

    Returns:
        Stability signal for each reaction and condensate species
    """
    reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)
    stability_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_stability_matrix_full
    )
    # jax.debug.print("stability_matrix = {out}", out=stability_matrix)

    temperature: FloatArray = parameters.state.temperature
    log_Kp: Float[Array, "... n_reactions"] = parameters.reaction_system.reaction.get_log_Kp(
        temperature
    )
    # jax.debug.print("log_Kp = {out}", out=log_Kp)

    log_activity: Float[Array, "... n_species"] = get_log_activity_estimate(
        parameters, species_abundance
    )

    # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
    log_Q: Float[Array, "... n_reactions"] = jnp.einsum(
        "rs,...s->...r", reaction_matrix, log_activity
    )
    stability_signal: Float[Array, "... n_reactions n_species"] = (
        stability_matrix * jnp.expand_dims(log_Kp - log_Q, axis=-1)
    )
    # jax.debug.print("stability_signal = {out}", out=stability_signal)

    return stability_signal


def get_soft_activation(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Computes soft activation weights for condensate species using a sigmoid function.

    Args:
        parameters: Parameters containing the reaction system information
        species_abundance: Estimated abundance of each species, used to estimate activities

    Returns:
        Activation weights in [0, 1] for each condensate (non-condensates get 0)
    """
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )
    stability_signal: Float[Array, "... n_reactions n_species"] = get_stability_signal(
        parameters, species_abundance
    )

    # Mask out non-condensate species in the stability signal before max, so only real condensates
    # can have nonzero activation
    stability_signal = jnp.where(condensate_mask, stability_signal, -jnp.inf)

    # Take the maximum stability signal for each species across all reactions
    max_affinity: Float[Array, "... n_species"] = jnp.nanmax(stability_signal, axis=-2)
    # jax.debug.print("max_affinity = {out}", out=max_affinity)

    # Sigmoid activation: 1 / (1 + exp(-affinity)). This smoothly maps the stability signal to an
    # activation weight between 0 and 1. Note that the RT scaling is already baked into the
    # stability signal via log_Kp, so no additional temperature scaling is needed here.
    activation: Float[Array, "... n_species"] = jax.nn.sigmoid(max_affinity)

    # Only condensates get nonzero activation
    activation = jnp.where(condensate_mask, activation, 0.0)

    return activation


def generate_auto_initial_guess(parameters: Parameters) -> Float[Array, "... twice_species"]:
    """Generates an automatic initial guess for the solution vector.

    The algorithm:
      - Iteratively predicts stable condensates by allocating element budgets and evaluating
        stability signals until convergence.
      - Allocates element budgets to predicted-stable condensates first, then distributes the
        remainder to other species.
      - Handles fugacity constraints for gas species if present.
      - Initializes log stability for predicted-stable condensates and uses a default value for
        others.

    Args:
        parameters: Parameters for a single batch element

    Returns:
        Concatenated array of [log_number_moles, log_stability]
    """
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )

    temperature: FloatArray = parameters.state.temperature

    element_abundance: Float[Array, "... n_elements"] = parameters.mass_constraints.abundance()
    # jax.debug.print("element_abundance = {out}", out=element_abundance)

    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )
    active_activity_constraints: Bool[Array, "... n_species"] = (
        parameters.activity_constraints.active()
    )
    # jax.debug.print("active_activity_constraints = {out}", out=active_activity_constraints)

    gas_no_imposed_fugacity: Bool[Array, "... n_species"] = gas_mask & ~active_activity_constraints
    # jax.debug.print("gas_no_imposed_fugacity = {out}", out=gas_no_imposed_fugacity)

    def _one_stability_pass(
        condensate_stable_known: Bool[Array, "... n_species"],
    ) -> Bool[Array, "... n_species"]:
        """One stability-prediction pass

        Given a set of already-known-stable condensates, allocates element budget to them first,
        distributes the remainder to gas species, computes ideal activities, evaluates the reaction
        K vs Q signal, and returns the monotone union of the new predictions with the input mask.

        Args:
            condensate_stable_known: Boolean mask of condensates predicted stable so far

        Returns:
            Updated mask - superset of ``condensate_stable_known``.
        """
        _, n_other = allocate_element_budget(
            parameters, element_abundance, condensate_stable_known
        )
        # jax.debug.print("n_other in stability pass = {out}", out=n_other)
        stability_signal: Float[Array, "... n_reactions n_species"] = get_stability_signal(
            parameters, n_other
        )
        # jax.debug.print("stability_signal in stability pass = {out}", out=stability_signal)

        # TODO: For debugging and testing
        # activation_weights: Float[Array, "... n_species"] = get_soft_activation(
        #    parameters, n_other
        # )
        # jax.debug.print("activation_weights in stability pass = {out}", out=activation_weights)

        new_predictions: Bool[Array, "... n_species"] = (
            jnp.any(stability_signal > 0, axis=-2) & condensate_mask
        )
        # jax.debug.print("new_predictions in stability pass = {out}", out=new_predictions)
        # jax.debug.print("new_predictions = {out}", out=new_predictions)

        # Monotone union: condensates are never retracted once predicted stable. This ensures the
        # set grows by at least one entry per iteration, guaranteeing fixed-point convergence
        # without a cap. The solver makes the definitive stable/absent decision.
        return condensate_stable_known | new_predictions

    # Iterate until the predicted-stable set stops growing
    # Initialize with no condensates known stable; the first body call is the gas-only pre-screen.
    # The monotone union in _one_stability_pass guarantees termination.
    init_stable: Bool[Array, " n_species"] = jnp.zeros_like(condensate_mask)
    # jax.debug.print("init_stable = {out}", out=init_stable)
    first_stable: Bool[Array, "... n_species"] = _one_stability_pass(init_stable)
    # jax.debug.print("first_stable = {out}", out=first_stable)

    init_stable_broadcasted: Bool[Array, "... n_species"] = jnp.broadcast_to(
        init_stable, first_stable.shape
    )
    # jax.debug.print("init_stable_broadcasted = {out}", out=init_stable_broadcasted)

    def _cond_fn(carry: tuple) -> Bool[Array, "..."]:
        prev, curr, i = carry
        return jnp.any(prev != curr)

    def _body_fn(carry: tuple) -> tuple:
        _, curr, i = carry
        return curr, _one_stability_pass(curr), i + 1

    _, condensate_stable_predicted, _ = lax.while_loop(
        _cond_fn, _body_fn, (init_stable_broadcasted, first_stable, jnp.array(0))
    )

    n_condensate, n_other = allocate_element_budget(
        parameters, element_abundance, condensate_stable_predicted
    )

    # Combine: predicted-stable condensates use their budget estimate; all others use the
    # non-condensate remainder. Fallback for species not covered by any constrained element:
    # geometric mean of finite estimates keeps missing species at a comparable magnitude.
    n_estimate: Float[Array, "... n_species"] = jnp.where(condensate_mask, n_condensate, n_other)
    # jax.debug.print("n_estimate after merge = {out}", out=n_estimate)

    log_number_moles: Float[Array, "... n_species"] = jnp.log(n_estimate)
    # jax.debug.print("log_number_moles = {out}", out=log_number_moles)

    # Fugacity-constrained gas species
    log_n_gas_known_total: Float[Array, "..."] = logsumexp(
        jnp.where(gas_no_imposed_fugacity, log_number_moles, -jnp.inf), axis=-1, keepdims=True
    )
    # jax.debug.print("log_n_gas_known_total = {out}", out=log_n_gas_known_total)

    pressure: Float[Array, "..."] = parameters.state.get_pressure(log_number_moles)
    # jax.debug.print("pressure = {out}", out=pressure)

    # Pressure must be 1-D in this function
    log_fug: Float[Array, "... n_species"] = parameters.activity_constraints.log_activity(
        temperature, pressure
    )
    # jax.debug.print("log_fug = {out}", out=log_fug)

    # Pressure must be a column vector
    log_n_fug: Float[Array, "... n_species"] = (
        log_fug + log_n_gas_known_total - jnp.log(pressure)[..., None]
    )
    # jax.debug.print("log_n_fug = {out}", out=log_n_fug)

    log_number_moles = jnp.where(
        gas_mask & active_activity_constraints, log_n_fug, log_number_moles
    )
    # jax.debug.print("log_number_moles after fugacity constraints = {out}", out=log_number_moles)

    log_stability = compute_implied_log_stability(parameters, log_number_moles)
    # jax.debug.print("log_stability = {out}", out=log_stability

    result: Float[Array, "... twice_species"] = jnp.concatenate(
        (log_number_moles, log_stability), axis=-1
    )
    # jax.debug.print("Initial guess (log_number_moles, log_stability) = {out}", out=result)

    result = eqx.error_if(
        result,
        jnp.any(jnp.isnan(result)),
        "Initial guess contains NaNs, which will break the solver.",
    )

    return result


def generate_initial_guess(
    parameters: Parameters, initial_guess: Float[Array, "... twice_species"]
) -> Float[Array, "... twice_species"]:
    """Generates an initial guess for the solution vector.

    This function checks the provided ``initial_guess`` for NaN values. If any NaNs are present for
    a given batch element, an automatic initial guess is generated for that element using
    :func:`generate_auto_initial_guess`.

    Args:
        parameters: Parameters

    Returns:
        Concatenated array of [log_number_moles, log_stability]
    """
    auto_initial_guess: Float[Array, "... twice_species"] = generate_auto_initial_guess(parameters)

    # Mask: True for rows where any entry is NaN along the last axis
    nan_mask: Bool[Array, " ... 1"] = jnp.any(jnp.isnan(initial_guess), axis=-1, keepdims=True)
    # Select auto_initial_guess for rows with any NaN, otherwise keep initial_guess
    result: Float[Array, "... twice_species"] = jnp.where(
        nan_mask, auto_initial_guess, initial_guess
    )

    result = eqx.error_if(
        result,
        jnp.any(jnp.isnan(result)),
        "NaN encountered in initial guess, which will break the solver.",
    )

    return result
