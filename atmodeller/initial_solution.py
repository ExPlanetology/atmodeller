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
from jaxtyping import Array, Bool, Float

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


def initial_moles(
    parameters: Parameters,
    element_abundance: Float[Array, "... n_elements"],
    species_score: Float[Array, "... n_species"],
) -> Float[Array, "... n_species"]:
    """Allocates the element budget between predicted-stable condensates and other species.

    This function first assigns as much of each element as possible to the species predicted to be
    stable condensates, according to the limiting-reagent principle. It then computes the remaining
    element budget and allocates it to the non-condensate species, again using the limiting-reagent
    logic. Finally, fallback logic is applied to assign small nonzero values or the geometric mean.

    Args:
        parameters: Parameters
        element_abundance: Available abundance of each element
        species_score: Score indicating the stability of each species

    Returns:
        Tuple of two arrays:
            - n_condensate: Moles allocated to each condensate species
            - n_other: Moles allocated to each non-condensate species
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )

    # Convert score to soft weights
    weights = species_score
    weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-30)
    jax.debug.print("weights = {out}", out=weights)

    # weighted limiting reagent allocation
    return max_moles_by_limiting_element(formula_matrix, element_abundance, weights > 0)


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


def compute_reaction_affinity(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_reactions"]:
    """Computes a masked thermodynamic driving force (affinity) for each reaction.

    This function evaluates the reaction-level driving force

        Δ_r = log_Kp[r] - log_Q[r],

    and projects it onto a subset of species defined by a stability mask. The mask identifies
    species that are considered "eligible" to participate in stability evaluation (e.g.
    condensates, or more generally any subset of species of interest).

    The returned affinity is

        A_r = sum_i (R[r, i] * m[i]) * Δ_r,

    where:
        - R[r, i] is the reaction stoichiometric matrix
        - m[i] is the species-level stability mask (broadcast across reactions)

    Interpretation:
        - Δ_r encodes the thermodynamic driving force of the reaction.
        - The mask m[i] restricts this driving force to contributions from selected species.
        - The sum over species aggregates the participation of all allowed species,
          yielding a single smooth, reaction-level score.

    Notes:
        - The mask is not limited to condensates; it can include any subset of species
          (e.g. condensates, gas species, or hybrid selections).
        - This is a heuristic projection used for initialization / activation purposes,
          not a strict thermodynamic quantity.
        - The formulation is fully differentiable and avoids discontinuities that arise
          from max-based selection.

    Args:
        parameters: Parameters containing the reaction system information.
        species_abundance: Estimated abundance of each species, used to compute activities.

    Returns:
        A reaction-level affinity array with shape (..., n_reactions), where larger
        values indicate stronger driving toward formation of the masked species.
    """
    reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)

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
    delta_r: Float[Array, "... n_reactions"] = log_Kp - log_Q

    return delta_r


def compute_species_score(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Computes soft activation weights for condensate species using a sigmoid function.

    Args:
        parameters: Parameters containing the reaction system information
        species_abundance: Estimated abundance of each species, used to estimate activities

    Returns:
        Activation weights in [0, 1] for each condensate (non-condensates get 0)
    """
    reaction_matrix = jnp.asarray(parameters.reaction_system.reaction.reaction_matrix_full)

    A_r: Float[Array, "... n_reactions"] = compute_reaction_affinity(parameters, species_abundance)

    # Map reaction -> species via participation
    participation = jnp.abs(reaction_matrix)
    jax.debug.print("participation = {out}", out=participation)

    species_score = jnp.sum(participation * jnp.exp(A_r[..., :, None]), axis=-2)
    jax.debug.print("species_score = {out}", out=species_score)

    return species_score


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
    temperature: FloatArray = parameters.state.temperature

    element_abundance: Float[Array, "... n_elements"] = parameters.mass_constraints.abundance()
    # jax.debug.print("element_abundance = {out}", out=element_abundance)

    n_seed = 1e-12 * jnp.ones(parameters.species.number_species)

    species_score = compute_species_score(parameters, n_seed)

    n = initial_moles(parameters, element_abundance, species_score)

    log_n = jnp.log(n + 1e-30)

    return log_n

    # gas_mask: Bool[Array, " n_species"] = jnp.asarray(
    #     parameters.reaction_system.phase_system.gas_species_mask
    # )
    # condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
    #     parameters.reaction_system.phase_system.condensates_species_mask
    # )
    # active_activity_constraints: Bool[Array, "... n_species"] = (
    #     parameters.activity_constraints.active()
    # )
    # # jax.debug.print("active_activity_constraints = {out}", out=active_activity_constraints)

    # gas_no_imposed_fugacity: Bool[Array, "... n_species"] = gas_mask & ~active_activity_constraints
    # # jax.debug.print("gas_no_imposed_fugacity = {out}", out=gas_no_imposed_fugacity)

    # n_condensate, n_other = allocate_element_budget(
    #     parameters, element_abundance, condensate_stable_predicted
    # )

    # # Combine: predicted-stable condensates use their budget estimate; all others use the
    # # non-condensate remainder. Fallback for species not covered by any constrained element:
    # # geometric mean of finite estimates keeps missing species at a comparable magnitude.
    # n_estimate: Float[Array, "... n_species"] = jnp.where(condensate_mask, n_condensate, n_other)
    # # jax.debug.print("n_estimate after merge = {out}", out=n_estimate)

    # log_number_moles: Float[Array, "... n_species"] = jnp.log(n_estimate)
    # # jax.debug.print("log_number_moles = {out}", out=log_number_moles)

    # # Fugacity-constrained gas species
    # log_n_gas_known_total: Float[Array, "..."] = logsumexp(
    #     jnp.where(gas_no_imposed_fugacity, log_number_moles, -jnp.inf), axis=-1, keepdims=True
    # )
    # # jax.debug.print("log_n_gas_known_total = {out}", out=log_n_gas_known_total)

    # pressure: Float[Array, "..."] = parameters.state.get_pressure(log_number_moles)
    # # jax.debug.print("pressure = {out}", out=pressure)

    # # Pressure must be 1-D in this function
    # log_fug: Float[Array, "... n_species"] = parameters.activity_constraints.log_activity(
    #     temperature, pressure
    # )
    # # jax.debug.print("log_fug = {out}", out=log_fug)

    # # Pressure must be a column vector
    # log_n_fug: Float[Array, "... n_species"] = (
    #     log_fug + log_n_gas_known_total - jnp.log(pressure)[..., None]
    # )
    # # jax.debug.print("log_n_fug = {out}", out=log_n_fug)

    # log_number_moles = jnp.where(
    #     gas_mask & active_activity_constraints, log_n_fug, log_number_moles
    # )
    # # jax.debug.print("log_number_moles after fugacity constraints = {out}", out=log_number_moles)

    # log_stability = compute_implied_log_stability(parameters, log_number_moles)
    # # jax.debug.print("log_stability = {out}", out=log_stability

    # result: Float[Array, "... twice_species"] = jnp.concatenate(
    #     (log_number_moles, log_stability), axis=-1
    # )
    # # jax.debug.print("Initial guess (log_number_moles, log_stability) = {out}", out=result)

    # result = eqx.error_if(
    #     result,
    #     jnp.any(jnp.isnan(result)),
    #     "Initial guess contains NaNs, which will break the solver.",
    # )

    # return result


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
