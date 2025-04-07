#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""JAX-related functionality for solving the system of equations"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array, jit, lax
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from atmodeller.constants import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Planet,
    SolverParameters,
    SpeciesCollection,
    TracedParameters,
)
from atmodeller.utilities import (
    get_log_number_density_from_log_pressure,
    safe_exp,
    unit_conversion,
)


@eqx.filter_jit
def solve(
    solution_array: Array,
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    solver_parameters: SolverParameters,
    options: dict[str, Any],
) -> tuple[Array, Array, Array]:
    """Solves the system of non-linear equations

    Args:
        solution_array: Solution array
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        solver_parameters: Solver parameters
        options: Options for root find

    Returns:
        The solution array, the status of the solver
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        solver_parameters.solver_instance,
        solution_array,
        args={
            "traced_parameters": traced_parameters,
            "fixed_parameters": fixed_parameters,
        },
        throw=solver_parameters.throw,
        max_steps=solver_parameters.max_steps,
        options=options,
    )

    # jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])
    solver_steps: Array = sol.stats["num_steps"]
    solver_status: Array = sol.result == optx.RESULTS.successful

    return sol.value, solver_status, solver_steps


@jit
def select_valid_solutions(
    sol: Array, solver_status: Array, solver_steps: Array
) -> tuple[Array, Array, Array]:
    """Selects a single valid solution for each simulation from sol, using solver_status.

    Args:
        sol: Solution array of shape (multistarts, simulations, solution) or
            (multistarts, solution)
        solver_status: Status array of shape (multistarts, simulations) or (multistarts,)

    Returns:
        Reduced array of shape (simulations, solution)
        First valid index for each simulation
        Number of steps for each simulation
    """
    # jax.debug.print("sol = {out}", out=sol)
    # jax.debug.print("solver_status = {out}", out=solver_status)
    # jax.debug.print("solver_steps = {out}", out=solver_steps)
    # Check if input is 3D (multistarts, simulations, solution) or 2D (multistarts, solution)
    is_3d: int = sol.ndim == 3

    if is_3d:
        _, simulations, _ = sol.shape  # Get shape
    else:
        _, _ = sol.shape
        simulations: int = 1  # Treat as single simulation for uniform indexing

        # Expand dims for uniform processing
        sol = sol[:, None, :]  # (multistarts, 1, solution)
        solver_status = solver_status[:, None]  # (multistarts, 1)
        solver_steps = solver_steps[:, None]  # (multistarts, 1)

    # Find the first valid index per simulation using jnp.argmax
    # If all multistarts failed for a given case, this will return the first index
    first_valid_index: Array = jnp.argmax(solver_status, axis=0)  # (simulations,)
    # jax.debug.print("first_valid_index = {out}", out=first_valid_index)

    # Gather the selected solutions (simulations, solution_dim)
    selected_solutions: Array = sol[first_valid_index, jnp.arange(simulations)]
    selected_steps: Array = solver_steps[first_valid_index, jnp.arange(simulations)]
    # jax.debug.print("selected_solutions = {out}", out=selected_solutions)
    # jax.debug.print("selected_steps = {out}", out=selected_steps)

    # Find if there is any valid solution. This allows us to maintain the same array size
    # (important for computing the output) whilst at the same time keeping tracking of failed cases
    # with a -1 index. These failed cases can then later be filtered out during output, if the user
    # desires.
    has_valid_solution: Array = jnp.any(solver_status, axis=0)  # (simulations,)
    first_valid_index = jnp.where(has_valid_solution, first_valid_index, -1)
    selected_steps = jnp.where(has_valid_solution, selected_steps, 0)

    # Keep the same dimensions for all output, so below line is commented out
    # if not is_3d:
    #    selected_solutions = selected_solutions.squeeze(0)  # Remove extra dimension for 2D case

    # jax.debug.print("selected_solutions = {out}", out=selected_solutions)

    return selected_solutions, first_valid_index, selected_steps


@jit
def objective_function(solution: Array, kwargs: dict) -> Array:
    """Residual of the reaction network and mass balance

    Args:
        kwargs: Dictionary of pytrees required to compute the residual

    Returns:
        Residual
    """
    return compute_residual(solution, kwargs)


@jit
def get_min_log_elemental_abundance_per_species(
    formula_matrix: Array, mass_constraints: MassConstraints
) -> Array:
    """For each species, find the elemental mass constraint with the lowest abundance.

    If species are present for which there are no elemental mass constraints (for example,
    oxygen if a fugacity constraint is applied instead) the maximum of all elements will be
    returned for that particular species. However, the return array from this function is
    subsequently filtered depending on whether a stability calculation is required for each
    species.

    # TODO: This could be precomputed rather than calculated within the residual calculation.

    Args:
        formula_matrix: Formula matrix
        mass_constraints: Mass constraints

    Returns:
        A vector of the minimum log elemental abundance for each species
    """
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    mask: Array = (formula_matrix != 0).astype(jnp.int_)
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    # Convert to column vector to align for element-wise multiplication
    log_abundance: Array = jnp.expand_dims(
        jnp.array(list(mass_constraints.log_abundance.values())), axis=1
    )
    # jax.debug.print("log_abundance = {out}", out=log_abundance)

    # Element-wise multiplication (broadcasted correctly)
    masked_abundance: Array = mask * log_abundance  # Shape: (n_elements, n_species)
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Replace zeros (or masked-out values) with the max value before taking the min
    masked_abundance = jnp.where(mask != 0, masked_abundance, jnp.max(log_abundance))
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Find the minimum log abundance per species
    min_abundance_per_species: Array = jnp.min(masked_abundance, axis=0)
    # jax.debug.print("min_abundance_per_species = {out}", out=min_abundance_per_species)

    return min_abundance_per_species


@jit
def compute_residual(solution: Array, kwargs: dict) -> Array:
    """Residual of the reaction network and mass balance

    Args:
        solution: Solution array
        kwargs: Dictionary of pytrees required to compute the residual

    Returns:
        Residual
    """
    traced_parameters: TracedParameters = kwargs["traced_parameters"]
    fixed_parameters: FixedParameters = kwargs["fixed_parameters"]
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.temperature
    fugacity_constraints: FugacityConstraints = traced_parameters.fugacity_constraints
    mass_constraints: MassConstraints = traced_parameters.mass_constraints
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)

    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    reaction_stability_matrix: Array = jnp.array(fixed_parameters.reaction_stability_matrix)
    stability_species_indices: Array = jnp.array(fixed_parameters.stability_species_indices)
    fugacity_matrix: Array = jnp.array(fixed_parameters.fugacity_matrix)
    # We only need the formula matrix for elements with mass constraints
    formula_matrix_constraints: Array = jnp.array(fixed_parameters.formula_matrix_constraints)
    # jax.debug.print("Starting new residual evaluation")

    # Species
    if stability_species_indices.size > 0:
        split_idx: int = -len(stability_species_indices)
        log_number_density: Array = solution[:split_idx]
        log_stability: Array = solution[split_idx:]
        # jax.debug.print("log_stability = {out}", out=log_stability)
    else:
        log_number_density = solution

    # jax.debug.print("log_number_density = {out}", out=log_number_density)

    # Stability
    # The reaction_stability_matrix has zero entries so its OK to pad with any value for
    # species without stability since the result is always zero
    log_stability_padded: Array = jnp.zeros_like(log_number_density)
    if stability_species_indices.size > 0:
        log_stability_padded = log_stability_padded.at[stability_species_indices].set(
            log_stability  # type: ignore since log_stability is known
        )
    # jax.debug.print("log_stability_padded = {out}", out=log_stability_padded)

    log_activity: Array = get_log_activity(traced_parameters, fixed_parameters, log_number_density)
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # Based on the definition of the reaction constant we need to convert gas activities
    # (fugacities) from bar to effective number density
    gas_species_mask: Array = jnp.zeros_like(log_activity, dtype=bool)
    gas_species_mask = gas_species_mask.at[gas_species_indices].set(True)
    # jax.debug.print("gas_species_mask = {out}", out=gas_species_mask)

    log_activity_number_density: Array = get_log_number_density_from_log_pressure(
        log_activity, temperature
    )
    log_activity_number_density = jnp.where(
        gas_species_mask, log_activity_number_density, log_activity
    )
    # jax.debug.print("log_activity_number_density = {out}", out=log_activity_number_density)

    # Bulk atmosphere
    log_volume: Array = get_atmosphere_log_volume(fixed_parameters, log_number_density, planet)

    residual: Array = jnp.array([])

    # Reaction network residual
    if reaction_matrix.size > 0:
        log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
            fixed_parameters, temperature
        )
        reaction_residual: Array = (
            reaction_matrix.dot(log_activity_number_density) - log_reaction_equilibrium_constant
        )
        # jax.debug.print("reaction_residual before stability = {out}", out=reaction_residual)

        reaction_residual = reaction_residual - reaction_stability_matrix.dot(
            safe_exp(log_stability_padded)
        )
        # jax.debug.print("reaction_residual after stability = {out}", out=reaction_residual)

        residual = jnp.concatenate([residual, reaction_residual])

    # Fugacity constraints residual
    if fugacity_matrix.size > 0:
        fugacity_species_indices: Array = jnp.array(fixed_parameters.fugacity_species_indices)
        # jax.debug.print("fugacity_species_indices = {out}", out=fugacity_species_indices)
        fugacity_log_activity_number_density: Array = jnp.take(
            log_activity_number_density, fugacity_species_indices
        )
        # jax.debug.print(
        #    "fugacity_log_activity_number_density = {out}",
        #    out=fugacity_log_activity_number_density,
        # )
        # jax.debug.print("fugacity_matrix = {out}", out=fugacity_matrix)
        # TODO: Check if there is a need to consider the stability of the species for which the
        # fugacity constraint is being applied.
        fugacity_residual: Array = fugacity_matrix.dot(fugacity_log_activity_number_density)
        # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
        total_pressure: Array = get_total_pressure(
            fixed_parameters, log_number_density, temperature
        )
        fugacity_residual = fugacity_residual - fugacity_constraints.log_number_density(
            temperature, total_pressure
        )
        # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
        residual = jnp.concatenate([residual, fugacity_residual])

    # Elemental mass balance residual
    if formula_matrix_constraints.size > 0:
        # Number density of elements in the gas or condensed phase
        element_density: Array = get_element_density(
            formula_matrix_constraints, log_number_density
        )
        # jax.debug.print("element_density = {out}", out=element_density)
        element_melt_density: Array = get_element_density_in_melt(
            traced_parameters,
            fixed_parameters,
            formula_matrix_constraints,
            log_number_density,
            log_activity,
            log_volume,
        )
        # jax.debug.print("element_melt_density = {out}", out=element_melt_density)

        # Relative mass error, computed in log-space for numerical stability
        element_density_total: Array = element_density + element_melt_density
        log_element_density_total: Array = jnp.log(element_density_total)
        log_target_density: Array = mass_constraints.log_number_density(log_volume)
        mass_residual: Array = safe_exp(log_element_density_total - log_target_density) - 1
        # jax.debug.print("element_density = {out}", out=mass_residual)

        residual = jnp.concatenate([residual, mass_residual])

        # Stability residual
        if stability_species_indices.size > 0:
            log_min_number_density: Array = (
                get_min_log_elemental_abundance_per_species(
                    formula_matrix_constraints, mass_constraints
                )
                - log_volume
                + jnp.log(fixed_parameters.tau)
            )
            log_min_number_density = jnp.take(log_min_number_density, stability_species_indices)

            log_number_density_stability: Array = jnp.take(
                log_number_density, stability_species_indices
            )
            stability_residual: Array = (
                log_number_density_stability + log_stability - log_min_number_density  # type: ignore
            )
            # jax.debug.print("stability_residual = {out}", out=stability_residual)

            residual = jnp.concatenate([residual, stability_residual])

    # jax.debug.print("residual = {out}", out=residual)

    return residual


@jit
def get_atmosphere_log_molar_mass(
    fixed_parameters: FixedParameters, log_number_density: Array
) -> Array:
    """Gets log molar mass of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log molar mass of the atmosphere
    """
    gas_log_number_density: Array = get_gas_species_data(fixed_parameters, log_number_density)
    gas_molar_mass: Array = get_gas_species_data(
        fixed_parameters, jnp.array(fixed_parameters.molar_masses)
    )
    molar_mass: Array = logsumexp(gas_log_number_density, b=gas_molar_mass) - logsumexp(
        gas_log_number_density
    )

    return molar_mass


@jit
def get_atmosphere_log_volume(
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    planet: Planet,
) -> Array:
    """Gets log volume of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """
    log_volume: Array = (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.temperature)
        - get_atmosphere_log_molar_mass(fixed_parameters, log_number_density)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )

    return log_volume


@jit
def get_total_pressure(
    fixed_parameters: FixedParameters, log_number_density: Array, temperature: ArrayLike
) -> Array:
    """Gets total pressure

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Total pressure
    """
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    pressure: Array = get_pressure_from_log_number_density(log_number_density, temperature)
    gas_pressure: Array = jnp.take(pressure, gas_species_indices)
    # jax.debug.print("gas_pressure = {out}", out=gas_pressure)

    return jnp.sum(gas_pressure)


@jit
def get_element_density(formula_matrix: Array, log_number_density: Array) -> Array:
    """Number density of elements in the gas or condensed phase

    Args:
        formula_matrix: Formula matrix
        log_number_density: Log number density

    Returns:
        Number density of elements in the gas or condensed phase
    """
    element_density: Array = formula_matrix.dot(safe_exp(log_number_density))

    return element_density


@jit
def get_element_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    formula_matrix: Array,
    log_number_density: Array,
    log_activity: Array,
    log_volume: Array,
) -> Array:
    """Gets the number density of elements dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        formula_matrix: Formula matrix
        log_number_density: Log number density
        log_activity: Log activity
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of elements dissolved in melt
    """
    species_melt_density: Array = get_species_density_in_melt(
        traced_parameters, fixed_parameters, log_number_density, log_activity, log_volume
    )
    element_melt_density: Array = formula_matrix.dot(species_melt_density)

    return element_melt_density


@jit
def get_gas_species_data(fixed_parameters: FixedParameters, some_array: ArrayLike) -> Array:
    """Gets the gas species data from an array

    Args:
        fixed_parameters: Fixed parameters
        some_array: Some array to extract gas species data from

    Returns:
        An array with just the gas species data from `some_array`
    """
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    gas_data: Array = jnp.take(some_array, gas_species_indices)

    return gas_data


@jit
def get_log_activity(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
) -> Array:
    """Gets the log activity

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log activity
    """
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.temperature
    species: SpeciesCollection = fixed_parameters.species
    total_pressure: Array = get_total_pressure(fixed_parameters, log_number_density, temperature)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)

    activity_funcs: list[Callable] = [species_.activity.log_activity for species_ in species]

    def apply_activity_function(index: ArrayLike):
        return lax.switch(
            index,
            activity_funcs,
            temperature,
            total_pressure,
        )

    vmap_apply_function: Callable = jax.vmap(apply_activity_function, in_axes=(0,))
    indices: Array = jnp.arange(len(species))
    log_activity_pure_species: Array = vmap_apply_function(indices)
    # jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)
    log_activity = get_log_activity_ideal_mixing(
        fixed_parameters, log_number_density, log_activity_pure_species
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    return log_activity


@jit
def get_log_activity_ideal_mixing(
    fixed_parameters: FixedParameters, log_number_density: Array, log_activity_pure_species: Array
) -> Array:
    """Gets the log activity of species in the atmosphere assuming an ideal mixture

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity_pure_species: Log activity of the pure species

    Returns:
        Log activity of the species assuming ideal mixing in the atmosphere
    """
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    gas_species_mask: Array = jnp.zeros_like(log_activity_pure_species, dtype=bool)
    gas_species_mask = gas_species_mask.at[gas_species_indices].set(True)

    number_density: Array = safe_exp(log_number_density)
    gas_species_number_density: Array = jnp.where(gas_species_mask, number_density, 0)
    atmosphere_log_number_density: Array = jnp.log(jnp.sum(gas_species_number_density))

    log_activity_gas_species: Array = (
        log_activity_pure_species + log_number_density - atmosphere_log_number_density
    )
    log_activity: Array = jnp.where(
        gas_species_mask, log_activity_gas_species, log_activity_pure_species
    )

    return log_activity


@jit
def get_log_pressure_from_log_number_density(
    log_number_density: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets log pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Log pressure
    """
    log_pressure: Array = (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature) + log_number_density
    )

    return log_pressure


@jit
def get_log_Kp(
    species: SpeciesCollection, reaction_matrix: Array, temperature: ArrayLike
) -> Array:
    """Gets log of the equilibrium constant in terms of partial pressures

    Args:
        species: Species
        reaction_matrix: Reaction matrix
        temperature: Temperature

    Returns:
        Log of the equilibrium constant in terms of partial pressures
    """
    gibbs_list: list[ArrayLike] = []
    for species_ in species:
        gibbs: ArrayLike = species_.data.get_gibbs_over_RT(temperature)
        gibbs_list.append(gibbs)

    gibbs_jnp: Array = jnp.array(gibbs_list)
    log_Kp: Array = -1.0 * reaction_matrix.dot(gibbs_jnp)

    return log_Kp


@jit
def get_log_reaction_equilibrium_constant(
    fixed_parameters: FixedParameters,
    temperature: ArrayLike,
) -> Array:
    """Gets the log equilibrium constant of the reactions

    Args:
        fixed_parameters: Fixed parameters
        temperature: Temperature

    Returns:
        Log equilibrium constant of the reactions
    """
    species: SpeciesCollection = fixed_parameters.species
    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)

    log_Kp: Array = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)
    delta_n: Array = jnp.sum(jnp.take(reaction_matrix, gas_species_indices, axis=1), axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)
    log_Kc: Array = log_Kp - delta_n * (jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature))
    # jax.debug.print("log10Kc = {out}", out=log_Kc)

    return log_Kc


@jit
def get_pressure_from_log_number_density(
    log_number_density: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Pressure
    """
    return safe_exp(get_log_pressure_from_log_number_density(log_number_density, temperature))


@jit
def get_species_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    log_activity: Array,
    log_volume: Array,
) -> Array:
    """Gets the number density of species dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity: Log activity
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of species dissolved in melt
    """
    molar_masses: Array = jnp.array(fixed_parameters.molar_masses)
    planet: Planet = traced_parameters.planet

    ppmw: Array = get_species_ppmw_in_melt(
        traced_parameters, fixed_parameters, log_number_density, log_activity
    )

    species_melt_density: Array = (
        ppmw
        * unit_conversion.ppm_to_fraction
        * AVOGADRO
        * planet.melt_mass
        / (molar_masses * safe_exp(log_volume))
    )
    # jax.debug.print("species_melt_density = {out}", out=species_melt_density)

    return species_melt_density


@jit
def get_species_ppmw_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    log_activity: Array,
) -> Array:
    """Gets the ppmw of species dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity: Log activity

    Returns:
        ppmw of species dissolved in melt
    """
    species: SpeciesCollection = fixed_parameters.species
    diatomic_oxygen_index: Array = jnp.array(fixed_parameters.diatomic_oxygen_index)
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.temperature

    fugacity: Array = safe_exp(log_activity)
    total_pressure: Array = get_total_pressure(fixed_parameters, log_number_density, temperature)
    diatomic_oxygen_fugacity: Array = jnp.take(fugacity, diatomic_oxygen_index)

    solubility_funcs: list[Callable] = [
        species_.solubility.jax_concentration for species_ in species
    ]

    def apply_solubility_function(index: ArrayLike, fugacity: ArrayLike):
        return lax.switch(
            index,
            solubility_funcs,
            fugacity,
            temperature,
            total_pressure,
            diatomic_oxygen_fugacity,
        )

    vmap_apply_function: Callable = jax.vmap(apply_solubility_function, in_axes=(0, 0))
    indices: ArrayLike = jnp.arange(len(species))
    species_ppmw: Array = vmap_apply_function(indices, fugacity)
    # jax.debug.print("ppmw = {out}", out=ppmw)

    return species_ppmw
