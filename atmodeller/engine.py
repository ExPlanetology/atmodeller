# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JAX-based model functions for atmospheric and chemical equilibrium calculations.

This module defines the core single-instance model functions used by the equilibrium solver
(e.g., activity/mass/stability residual construction and active-constraint masking). Functions
operate on one input instance at a time, with no implicit batching.

These functions form the building blocks for solving the coupled system of equations governing the
model (e.g., mass balance, activity constraints, phase stability), and are intended to be:

    1. Pure: No side effects, deterministic outputs for given inputs.
    2. JAX-compatible: Written with ``jax.numpy`` and compatible with transformations such as
       ``jit``, ``grad``, and ``vmap``.
    3. Shape-consistent: Accept and return arrays with predictable shapes, enabling easy
       vectorisation.

In practice, these functions are rarely called directly in production code. Instead, they are
wrapped with :func:`equinox.filter_vmap` to enable efficient batched evaluation over multiple
scenarios or parameter sets.

Note:
    The residual concatenation order in :func:`objective_function` must remain identical to the
    mask concatenation order in :func:`get_active_mask`.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from atmodeller.jax_utils import FloatArray, safe_exp
from atmodeller.parameters import Parameters


def get_active_mask(parameters: Parameters) -> Bool[Array, " dim"]:
    """Gets the mask of active residual quantities.

    Args:
        parameters: Parameters

    Returns:
        Active mask
    """
    activity_mask: Bool[Array, " species"] = jnp.asarray(
        parameters.activity_constraints.active(), dtype=bool
    )
    reactions_mask: Bool[Array, " reactions"] = jnp.ones(
        parameters.reaction_system.number_reactions, dtype=bool
    )
    mass_mask: Bool[Array, " mass_constraints"] = jnp.asarray(
        parameters.mass_constraints.active(), dtype=bool
    )
    stability_mask: Bool[Array, " species"] = jnp.asarray(
        parameters.reaction_system.species.active_stability, dtype=bool
    )

    # jax.debug.print("activity_mask = {out}", out=activity_mask)
    # jax.debug.print("reactions_mask = {out}", out=reactions_mask)
    # jax.debug.print("mass_mask = {out}", out=mass_mask)
    # jax.debug.print("stability_mask = {out}", out=stability_mask)

    active_mask: Bool[Array, " dim"] = jnp.concatenate(
        (activity_mask, reactions_mask, mass_mask, stability_mask), axis=-1
    )
    # jax.debug.print("active_mask = {out}", out=active_mask)

    return active_mask


def get_min_log_elemental_abundance_per_species(
    parameters: Parameters,
) -> Float[Array, "... n_species"]:
    """Gets the elemental mass constraint with the lowest abundance for each species.

    Note:
        This function assumes each species has at least one element present in the formula matrix.
        If a species had no elemental entries, its masked column would be all ``NaN`` and
        :func:`jax.numpy.nanmin` would return ``NaN`` for that species.

    Args:
        parameters: Parameters

    Returns:
        A vector of the minimum log elemental abundance for each species
    """
    formula_matrix: Integer[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix
    )
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    mask: Bool[Array, "n_elements n_species"] = formula_matrix != 0
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    log_abundance: Float[Array, "... n_elements"] = jnp.log(
        parameters.mass_constraints.abundance()
    )
    # jax.debug.print("log_abundance (engine) = {out}", out=log_abundance)

    # Mask log_abundance to nan where element is absent from species, then take min over elements
    # formula_matrix != 0 has shape (elements, species); log_abundance[..., :, None] broadcasts
    # over batch dims and species to give (... n_n_elements n_species)
    masked_abundance: Float[Array, "... n_elements n_species"] = jnp.where(
        mask, log_abundance[..., :, None], jnp.nan
    )
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Find the minimum log abundance per species
    min_abundance_per_species: Float[Array, "... n_species"] = jnp.nanmin(
        masked_abundance, axis=-2
    )
    # jax.debug.print("min_abundance_per_species = {out}", out=min_abundance_per_species)

    return min_abundance_per_species


def get_log_activity(
    parameters: Parameters, solution: Float[Array, "... twice_species"]
) -> Float[Array, "... n_species"]:
    """Gets the log activity of each species.

    Args:
        parameters: Parameters
        solution: Solution array for all species i.e. log number of moles and log stability

    Returns:
        Log activity of each species
    """
    log_number_moles, _ = jnp.split(solution, 2, axis=-1)
    temperature: FloatArray = parameters.state.temperature
    total_pressure: FloatArray = parameters.state.get_pressure(log_number_moles)

    log_activity: Float[Array, "... n_species"] = parameters.reaction_system.get_log_activity(
        log_number_moles, temperature, total_pressure
    )

    return log_activity


def compute_implied_log_stability(
    parameters: Parameters, log_number_moles: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Computes the implied log stability of each species based on the log number of moles.

    This function computes the implied log stability that would exactly satisfy the stability
    constraint for the given log number of moles. This is useful for reconstructing the stability
    value that would yield a zero stability residual.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles for each species

    Returns:
        Implied log stability of each species
    """
    # Ensure the stability residual is exactly zero for the initial number of moles
    log_tau_val: Float[Array, ""] = jnp.log(parameters.solver_parameters.tau)
    min_log_abundance_per_species: Float[Array, "... n_species"] = (
        get_min_log_elemental_abundance_per_species(parameters)
    )
    implied_log_stability: Float[Array, "... n_species"] = (
        min_log_abundance_per_species + log_tau_val - log_number_moles
    )

    # min_log_abundance_per_species propagates NaNs if there is not an imposed elemental mass
    # constraint for a species. This means that stability is not a solution quantity (not used by
    # the solver), and therefore should fall back to an arbitrary non-NaN value. For consistency
    # with the implied logic that the species must be stable (present in the system but no mass
    # constraints), we assign the limit of a stable species, which is log_tau_val.
    implied_log_stability = jnp.where(
        jnp.isnan(implied_log_stability), log_tau_val, implied_log_stability
    )
    # jax.debug.print("implied_log_stability = {out}", out=implied_log_stability)

    return implied_log_stability


@eqx.filter_jit
def objective_function(
    solution: Float[Array, "... twice_species"], parameters: Parameters
) -> Float[Array, "... residual"]:
    """Objective function

    The order of the residual does make a difference to the solution process. More investigations
    are necessary, but justification for the current ordering is as follows:

        1. Activity constraints - fixed target, well conditioned
        2. Reaction constraints - log-linear, physics-based coupling
        3. Mass balance constraints - stiffer
        4. Stability constraints - stiffer still

    Args:
        solution: Solution array for all species i.e. log number of moles and log stability
        parameters: Parameters

    Returns:
        Residual vector over active constraints.
    """
    # jax.debug.print("Starting new objective_function evaluation")
    # jax.debug.print("solution = {out}", out=solution)

    temperature: FloatArray = parameters.state.temperature

    log_number_moles, log_stability = jnp.split(solution, 2, axis=-1)
    # jax.debug.print("log_number_moles = {out}", out=log_number_moles)
    # jax.debug.print("log_stability = {out}", out=log_stability)

    total_pressure: FloatArray = parameters.state.get_pressure(log_number_moles)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)

    log_activity: Float[Array, "... n_species"] = get_log_activity(parameters, solution)
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # Activity constraints residual (dimensionless)
    activity_residual: Float[Array, "... n_species"] = (
        log_activity - parameters.activity_constraints.log_activity(temperature, total_pressure)
    )
    # jax.debug.print("activity_residual = {out}", out=activity_residual)
    # jax.debug.print(
    #     "activity_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(activity_residual),
    #     out2=jnp.nanmax(activity_residual),
    # )
    # jax.debug.print(
    #     "activity_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(activity_residual),
    #     out2=jnp.nanstd(activity_residual),
    # )

    reaction_residual: Float[Array, "... reactions"] = parameters.reaction_system.get_residual(
        log_number_moles, log_activity, log_stability, temperature, total_pressure
    )
    # jax.debug.print("reaction_residual = {out}", out=reaction_residual)

    # jax.debug.print(
    #     "reaction_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(reaction_residual),
    #     out2=jnp.nanmax(reaction_residual),
    # )
    # jax.debug.print(
    #     "reaction_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(reaction_residual),
    #     out2=jnp.nanstd(reaction_residual),
    # )

    # Elemental mass balance residual
    log_element_moles_total: Float[Array, "... n_elements"] = (
        parameters.reaction_system.get_log_element_moles(log_number_moles)
    )
    # jax.debug.print("log_element_moles_total = {out}", out=log_element_moles_total)

    log_target_moles: Float[Array, "... n_elements"] = jnp.log(
        parameters.mass_constraints.abundance()
    )
    # jax.debug.print("log_target_moles = {out}", out=log_target_moles)

    # Dimensionless (ratio error - 1)
    # More robust than log residual for poor initial guesses, which are often the case.
    mass_residual: Float[Array, "... n_elements"] = (
        safe_exp(log_element_moles_total - log_target_moles) - 1
    )
    # Log residual converges fast when near the solution, but can be very large and unstable for
    # poor initial guesses, which are often the case.
    # mass_residual: Float[Array, " elements"] = log_element_moles_total - log_target_moles

    # jax.debug.print("mass_residual = {out}", out=mass_residual)
    # jax.debug.print(
    #     "mass_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(mass_residual),
    #     out2=jnp.nanmax(mass_residual),
    # )
    # jax.debug.print(
    #     "mass_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(mass_residual),
    #     out2=jnp.nanstd(mass_residual),
    # )

    # Stability residual
    log_tau: FloatArray = jnp.log(parameters.solver_parameters.tau)
    # jax.debug.print("log_tau = {out}", out=log_tau)
    log_min_number_moles: Float[Array, "... n_species"] = (
        get_min_log_elemental_abundance_per_species(parameters) + log_tau
    )
    # jax.debug.print("log_min_number_moles = {out}", out=log_min_number_moles)

    # Dimensionless (log-ratio)
    stability_residual: Float[Array, "... n_species"] = (
        log_number_moles + log_stability - log_min_number_moles
    )
    # jax.debug.print("stability_residual = {out}", out=stability_residual)
    # jax.debug.print(
    #     "stability_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(stability_residual),
    #     out2=jnp.nanmax(stability_residual),
    # )
    # jax.debug.print(
    #     "stability_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(stability_residual),
    #     out2=jnp.nanstd(stability_residual),
    # )

    # NOTE: Order must be identical to get_active_mask()
    residual: Float[Array, "... residual"] = jnp.concatenate(
        [activity_residual, reaction_residual, mass_residual, stability_residual], axis=-1
    )
    # jax.debug.print("residual (with nans) = {out}", out=residual)

    # This final masking operation drops nans (unused constraint options)
    active_mask: Bool[Array, "... dim"] = get_active_mask(parameters)
    # jax.debug.print("active_mask = {out}", out=active_mask)
    size: int = parameters.reaction_system.species.number_solution
    # jax.debug.print("size = {out}", out=size)

    active_indices: Integer[Array, " indices"] = jnp.where(active_mask, size=size)[0]
    # jax.debug.print("active_indices = {out}", out=active_indices)

    residual = jnp.take(
        residual, indices=active_indices, unique_indices=True, indices_are_sorted=True
    )
    # jax.debug.print("residual = {out}", out=residual)

    return residual
