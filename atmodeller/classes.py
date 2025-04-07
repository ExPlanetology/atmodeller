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
"""Classes"""

from __future__ import annotations

import logging
import pprint
from collections.abc import Mapping
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from atmodeller import TAU
from atmodeller.containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Planet,
    SolutionArguments,
    SolverParameters,
    SpeciesCollection,
    TracedParameters,
)
from atmodeller.engine import select_valid_solutions, solve
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.utilities import partial_rref

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere

    This is the main class that the user interacts with to build interior-atmosphere systems,
    solve them, and retreive the results.

    Args:
        species: Collection of species
        tau: Tau factor for species stability. Defaults to TAU.
    """

    # Save the jit compiled solver for repeat calculations
    _solver: Callable

    def __init__(self, species: SpeciesCollection, tau: float = TAU):
        self.species: SpeciesCollection = species
        self.tau: float = tau
        logger.info("species = %s", [species.name for species in self.species])
        logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def output(self) -> Output:
        return self._output

    def solve(
        self,
        *,
        planet: Planet | None = None,
        initial_log_number_density: ArrayLike | None = None,
        initial_log_stability: ArrayLike | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> None:
        """Solves the system and initialises an Output instance for processing the result

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density: Initial log number density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.
            solver_parameters: Solver parameters. Defaults to None.
        """
        solution_args: SolutionArguments = SolutionArguments.create_with_defaults(
            self.species,
            planet,
            initial_log_number_density,
            initial_log_stability,
            fugacity_constraints,
            mass_constraints,
            solver_parameters,
        )

        # Assemble options dictionary for solve
        options: dict[str, Any] = {
            "lower": self.species.get_lower_bound(),
            "upper": self.species.get_upper_bound(),
            "jac": solution_args.solver_parameters.jac,
        }

        if solution_args.is_batch:
            initial_solution_axes = solution_args.get_initial_solution_vmap()
            traced_parameters_axes = solution_args.get_traced_parameters_vmap()

            # Create an inner vmap callable for batch processing
            inner_solver: Callable = jax.vmap(
                solve,
                in_axes=(initial_solution_axes, traced_parameters_axes, None, None, None),
            )
        else:
            inner_solver = solve  # No batching, use the raw solver

        # Apply an outer vmap over the multistart dimension
        self._solver: Callable = eqx.filter_jit(
            jax.vmap(inner_solver, in_axes=(0, None, None, None, None))
        )

        fixed_parameters: FixedParameters = self.get_fixed_parameters(
            solution_args.fugacity_constraints, solution_args.mass_constraints
        )
        initial_solution_array: Array = solution_args.solution
        traced_parameters: TracedParameters = solution_args.get_traced_parameters()

        solution, solver_status, solver_steps = self._solver(
            initial_solution_array,
            traced_parameters,
            fixed_parameters,
            solution_args.solver_parameters,
            options,
        )

        # Ensure computation is complete before proceeding
        solution.block_until_ready()
        solver_status.block_until_ready()
        solver_steps.block_until_ready()

        valid_solutions, first_valid_index, solver_steps = select_valid_solutions(
            solution, solver_status, solver_steps
        )

        self._output: Output = Output(
            valid_solutions,
            first_valid_index,
            solver_steps,
            solution_args,
            fixed_parameters,
            traced_parameters,
        )

        num_total_models: int = len(first_valid_index)
        num_successful_models: int = jnp.count_nonzero(first_valid_index != -1).item()
        num_failed_models: int = jnp.count_nonzero(first_valid_index == -1).item()
        max_valid_index: int = jnp.max(first_valid_index).item()

        logger.info(f"Attempted to solve {num_total_models} models")

        if num_failed_models > 0:
            logger.warning(
                f"Solve complete: {num_successful_models} successful models but "
                f"{num_failed_models} models failed.\n"
                "Try increasing 'multistart', for example:\n"
                "    solver_parameters = SolverParameters(multistart=5)\n"
                "and then pass solver_parameters to the solve method.\n"
                "You can also adjust 'multistart_perturbation', for example:\n"
                "    solver_parameters = SolverParameters(multistart=5, "
                "multistart_perturbation=20.0)"
            )
        else:
            required_multistarts: int = max(max_valid_index + 1, 1)  # Ensure it's at least 1
            logger.info(
                f"Solve complete: {num_successful_models} successful models\n"
                f"The number of multistarts required was {required_multistarts}, "
                "which can depend on the choice of the random seed"
            )

    def solve_fast(
        self,
        *,
        planet: Planet | None = None,
        initial_log_number_density: Array | None = None,
        initial_log_stability: Array | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        """Solves the system and initialises an Output instance for processing the result

        The idea is that this method is faster than the solve method because it does not recompile
        the solver function each time it is called.

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density: Initial log number density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.
        """
        solution_args: SolutionArguments = SolutionArguments.create_with_defaults(
            self.species,
            planet,
            initial_log_number_density,
            initial_log_stability,
            fugacity_constraints,
            mass_constraints,
        )
        fixed_parameters: FixedParameters = self.get_fixed_parameters(
            solution_args.fugacity_constraints, solution_args.mass_constraints
        )
        initial_solution_array: Array = solution_args.solution
        traced_parameters: TracedParameters = solution_args.get_traced_parameters()

        # Assemble options dictionary for solve
        options: dict[str, Any] = {
            "lower": self.species.get_lower_bound(),
            "upper": self.species.get_upper_bound(),
            "jac": solution_args.solver_parameters.jac,
        }

        solution, solver_status, solver_steps = self._solver(
            initial_solution_array,
            traced_parameters,
            fixed_parameters,
            solution_args.solver_parameters,
            options,
        )

        # Ensure computation is complete before proceeding
        solution.block_until_ready()
        solver_status.block_until_ready()
        solver_steps.block_until_ready()

        valid_solutions, first_valid_index, solver_steps = select_valid_solutions(
            solution, solver_status, solver_steps
        )

        self._output: Output = Output(
            valid_solutions,
            first_valid_index,
            solver_steps,
            solution_args,
            fixed_parameters,
            traced_parameters,
        )

        logger.info("Fast solve complete")

    def get_condensed_species_indices(self) -> tuple[int, ...]:
        """Gets the indices of condensed species

        Returns:
            Indices of the condensed species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.species):
            if species_.data.phase != "g":
                indices.append(nn)

        return tuple(indices)

    def get_diatomic_oxygen_index(self) -> int:
        """Gets the species index corresponding to diatomic oxygen.

        Returns:
            Index of diatomic oxygen, or the first index if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.species):
            if species_.data.hill_formula == "O2":
                logger.debug("Found O2 at index = %d", nn)
                return nn

        # TODO: Bad practice to return the first index because it could be wrong and therefore give
        # rise to spurious results, but an index must be passed to evaluate the species solubility
        # that may depend on fO2.
        return 0

    def get_fixed_parameters(
        self, fugacity_constraints: FugacityConstraints, mass_constraints: MassConstraints
    ) -> FixedParameters:
        """Gets fixed parameters.

        Args:
            fugacity_constraints: Fugacity constraints
            mass_constraints: Mass constraints

        Returns:
            Fixed parameters
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        reaction_stability_matrix: npt.NDArray[np.float_] = self.get_reaction_stability_matrix()
        gas_species_indices: tuple[int, ...] = self.get_gas_species_indices()
        condensed_species_indices: tuple[int, ...] = self.get_condensed_species_indices()
        stability_species_indices: tuple[int, ...] = self.get_stability_species_indices()
        molar_masses: tuple[float, ...] = self.get_molar_masses()
        diatomic_oxygen_index: int = self.get_diatomic_oxygen_index()

        # The complete formula matrix is not required for the calculation but it is used for
        # computing output quantities. So calculate and store it.
        formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix()

        # Formula matrix for elements that are constrained by mass constraints
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
        indices: list[int] = []
        for element in mass_constraints.log_abundance.keys():
            index: int = unique_elements.index(element)
            indices.append(index)
        formula_matrix_constraints: npt.NDArray[np.int_] = formula_matrix.copy()
        formula_matrix_constraints = formula_matrix_constraints[indices, :]

        # Fugacity constraint matrix and indices
        number_fugacity_constraints: int = len(fugacity_constraints.constraints)
        fugacity_species_indices: list[int] = []
        species_names: tuple[str, ...] = self.get_species_names()
        for species_name in fugacity_constraints.constraints.keys():
            index: int = species_names.index(species_name)
            fugacity_species_indices.append(index)
        fugacity_matrix: npt.NDArray[np.float_] = np.identity(number_fugacity_constraints)

        # For fixed parameters all objects must be hashable because it is a static argument
        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=tuple(map(tuple, formula_matrix)),
            formula_matrix_constraints=tuple(map(tuple, formula_matrix_constraints)),
            reaction_matrix=tuple(map(tuple, reaction_matrix)),
            reaction_stability_matrix=tuple(map(tuple, reaction_stability_matrix)),
            stability_species_indices=stability_species_indices,
            fugacity_matrix=tuple(map(tuple, fugacity_matrix)),
            gas_species_indices=gas_species_indices,
            condensed_species_indices=condensed_species_indices,
            fugacity_species_indices=tuple(fugacity_species_indices),
            diatomic_oxygen_index=diatomic_oxygen_index,
            molar_masses=molar_masses,
            tau=self.tau,
        )

        return fixed_parameters

    def get_formula_matrix(self) -> npt.NDArray[np.int_]:
        """Gets the formula matrix.

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            Formula matrix
        """
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
        formula_matrix: npt.NDArray[np.int_] = np.zeros(
            (len(unique_elements), len(self.species)), dtype=jnp.int_
        )

        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(self.species):
                count: int = 0
                try:
                    count = species_.data.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def get_gas_species_indices(self) -> tuple[int, ...]:
        """Gets the indices of gas species

        Returns:
            Indices of the gas species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.species):
            if species_.data.phase == "g":
                indices.append(nn)

        return tuple(indices)

    def get_unique_elements_in_species(self) -> tuple[str, ...]:
        """Gets unique elements.

        Args:
            species: A list of species

        Returns:
            Unique elements in the species ordered alphabetically
        """
        elements: list[str] = []
        for species_ in self.species:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        logger.debug("unique_elements_in_species = %s", sorted_elements)

        return tuple(sorted_elements)

    def get_molar_masses(self) -> tuple[float, ...]:
        """Gets the molar masses of all species.

        Returns:
            Molar masses of all species
        """
        molar_masses: tuple[float, ...] = tuple(
            [species_.data.molar_mass for species_ in self.species]
        )

        logger.debug("molar_masses = %s", molar_masses)

        return molar_masses

    def get_reaction_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction matrix.

        Returns:
            A matrix of linearly independent reactions or an empty array if no reactions
        """
        if len(self.species) == 1:
            logger.debug("Only one species therefore no reactions")
            return np.array([])

        transpose_formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix().T
        reaction_matrix: npt.NDArray[np.float_] = partial_rref(transpose_formula_matrix)

        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def get_reaction_stability_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction stability matrix.

        Returns:
            Reaction stability matrix
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        mask: npt.NDArray[np.bool_] = np.zeros_like(reaction_matrix, dtype=bool)

        if reaction_matrix.size > 0:
            # Find the species to solve for stability
            stability_bool: npt.NDArray[np.bool_] = self.get_stability_species_mask()
            mask[:, stability_bool] = True
            reaction_stability_matrix: npt.NDArray[np.float_] = reaction_matrix * mask
        else:
            reaction_stability_matrix = reaction_matrix

        logger.debug("reaction_stability_matrix = %s", reaction_stability_matrix)

        return reaction_stability_matrix

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        reactions: dict[int, str] = {}
        if reaction_matrix.size != 0:
            for reaction_index in range(reaction_matrix.shape[0]):
                reactants: str = ""
                products: str = ""
                for species_index, species_ in enumerate(self.species):
                    coeff: float = reaction_matrix[reaction_index, species_index].item()
                    if coeff != 0:
                        if coeff < 0:
                            reactants += f"{abs(coeff)} {species_.data.name} + "
                        else:
                            products += f"{coeff} {species_.data.name} + "

                reactants = reactants.rstrip(" + ")
                products = products.rstrip(" + ")
                reaction: str = f"{reactants} = {products}"
                reactions[reaction_index] = reaction

        return reactions

    def get_species_names(self) -> tuple[str, ...]:
        """Gets the names of all species.

        Returns:
            Species names
        """
        return tuple([species_.name for species_ in self.species])

    def get_stability_species_indices(self) -> tuple[int, ...]:
        """Gets the indices of species to solve for stability

        Returns:
            Indices of the species to solve for stability
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.species):
            if species_.solve_for_stability:
                indices.append(nn)

        return tuple(indices)

    def get_stability_species_mask(self) -> npt.NDArray[np.bool_]:
        """Gets the stability species mask

        Returns:
            Mask for the species to solve for the stability
        """
        # Find the species to solve for stability
        stability_bool: npt.NDArray[np.bool_] = np.array(
            [species.solve_for_stability for species in self.species], dtype=np.bool_
        )

        return stability_bool
