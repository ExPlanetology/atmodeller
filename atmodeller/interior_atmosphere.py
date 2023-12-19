"""Interior atmosphere system

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
import pprint
from collections import UserList
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import root
from sklearn.metrics import mean_squared_error

from atmodeller import GAS_CONSTANT, GRAVITATIONAL_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.initial_condition import InitialConditionABC, InitialConditionConstant
from atmodeller.interfaces import (
    ChemicalComponent,
    CondensedSpecies,
    CondensedSpeciesOutput,
    ConstraintABC,
    GasSpecies,
    NoSolubility,
    Solubility,
)
from atmodeller.output import Output
from atmodeller.solubilities import composition_solubilities
from atmodeller.utilities import filter_by_type

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet

    Defines the properties of a planet that are relevant for interior modeling. It provides default
    values suitable for modelling a fully molten Earth-like planet.

    Args:
        mantle_mass: Mass of the planetary mantle. Defaults to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass. Defaults to
            Earth.
        surface_radius: Radius of the planetary surface. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
        melt_composition: Melt composition of the planet. Default is None.

    Attributes:
        mantle_mass: Mass of the planetary mantle.
        mantle_melt_fraction: mass fraction of the mantle that is molten.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass.
        surface_radius: Radius of the planetary surface.
        surface_temperature: Temperature of the planetary surface.
        melt_composition: Melt composition of the planet.
    """

    mantle_mass: float = 4.208261222595111e24  # kg, Earth's mantle mass
    mantle_melt_fraction: float = 1.0  # Completely molten
    core_mass_fraction: float = 0.295334691460966  # Earth's core mass fraction
    surface_radius: float = 6371000.0  # m, Earth's radius
    surface_temperature: float = 2000.0  # K
    melt_composition: str | None = None

    def __post_init__(self):
        logger.info("Creating a new planet")
        logger.info("Mantle mass (kg) = %f", self.mantle_mass)
        logger.info("Mantle melt fraction = %f", self.mantle_melt_fraction)
        logger.info("Core mass fraction = %f", self.core_mass_fraction)
        logger.info("Planetary radius (m) = %f", self.surface_radius)
        logger.info("Planetary mass (kg) = %f", self.planet_mass)
        logger.info("Surface temperature (K) = %f", self.surface_temperature)
        logger.info("Surface gravity (m/s^2) = %f", self.surface_gravity)
        logger.info("Melt Composition = %s", self.melt_composition)

    @property
    def planet_mass(self) -> float:
        """Mass of the planet in SI units"""
        return self.mantle_mass / (1 - self.core_mass_fraction)

    @property
    def surface_area(self) -> float:
        """Surface area of the planet in SI units"""
        return 4.0 * np.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> float:
        """Surface gravity of the planet in SI units"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2


class Species(UserList):
    """Collections of species for an interior-atmosphere system

    A collection of species. It provides methods to filter species based on their phases (gas,
    condensed).

    Args:
        initlist: Initial list of species. Defaults to None.

    Attributes:
        data: List of species contained in the system.
    """

    def __init__(self, initlist: list[ChemicalComponent] | None = None):
        self.data: list[ChemicalComponent]  # For typing.
        super().__init__(initlist)

    @property
    def number(self) -> int:
        """Number of species"""
        return len(self.data)

    @property
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species"""
        return filter_by_type(self, GasSpecies)

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return len(self.gas_species)

    @property
    def condensed_species(self) -> dict[int, CondensedSpecies]:
        """Condensed species."""
        return filter_by_type(self, CondensedSpecies)

    @property
    def number_condensed_species(self) -> int:
        """Number of condensed species"""
        return len(self.condensed_species)

    @property
    def indices(self) -> dict[str, int]:
        """Indices of the species"""
        return {
            chemical_formula: index
            for index, chemical_formula in enumerate(self.chemical_formulas)
        }

    @property
    def chemical_formulas(self) -> list[str]:
        """Chemical formulas of the species"""
        return [species.chemical_formula for species in self.data]

    def conform_solubilities_to_planet_composition(self, planet: Planet) -> None:
        """Ensure that the solubilities of the species are consistent with the planet composition.

        Args:
            planet: A planet.
        """
        if planet.melt_composition is not None:
            msg: str = (
                # pylint: disable=consider-using-f-string
                "Setting solubilities to be consistent with the melt composition (%s)"
                % planet.melt_composition
            )
            logger.info(msg)
            try:
                solubilities: dict[str, Solubility] = composition_solubilities[
                    planet.melt_composition.casefold()
                ]
            except KeyError:
                logger.error("Cannot find solubilities for %s", planet.melt_composition)
                raise

            for species in self.gas_species.values():
                try:
                    species.solubility = solubilities[species.chemical_formula]
                    logger.info(
                        "Found solubility law for %s: %s",
                        species.chemical_formula,
                        species.solubility.__class__.__name__,
                    )
                except KeyError:
                    logger.info("No solubility law for %s", species.chemical_formula)
                    species.solubility = NoSolubility()

    def _species_sorter(self, species: ChemicalComponent) -> tuple[int, str]:
        """Sorter for the species

        Sorts first by species complexity and second by species name.

        Args:
            species: Species

        Returns:
            A tuple to sort first by number of elements and second by species name.
        """
        return (species.formula.atoms, species.chemical_formula)


@dataclass(kw_only=True)
class ReactionNetwork:
    """Determines the necessary reactions to solve a chemical network.

    Args:
        species: Species

    Attributes:
        species: Species
        species_matrix: The stoichiometry matrix of the species in terms of elements
        reaction_matrix: The reaction stoichiometry matrix
    """

    species: Species

    def __post_init__(self):
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self.species.chemical_formulas)
        self.species_matrix: np.ndarray = self.find_matrix()
        self.reaction_matrix: np.ndarray = self.partial_gaussian_elimination()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions))

    @cached_property
    def number_reactions(self) -> int:
        return self.species.number - self.number_unique_elements

    @cached_property
    def number_unique_elements(self) -> int:
        return len(self.unique_elements)

    @cached_property
    def unique_elements(self) -> list[str]:
        elements: list[str] = []
        for species in self.species:
            elements.extend(list(species.formula.composition().keys()))
        unique_elements: list[str] = list(set(elements))
        return unique_elements

    def find_matrix(self) -> np.ndarray:
        """Creates a matrix where species (rows) are split into their element counts (columns).

        Returns:
            For example, self.species = ['CO2', 'H2O'] would return:
                [[0, 1, 2],
                 [2, 0, 1]]
            if the columns represent the elements H, C, and O, respectively.
        """
        matrix: np.ndarray = np.zeros(
            (self.species.number, self.number_unique_elements), dtype=int
        )
        for species_index, species in enumerate(self.species):
            for element_index, element in enumerate(self.unique_elements):
                try:
                    count: int = species.formula.composition()[element].count
                except KeyError:
                    count = 0
                matrix[species_index, element_index] = count
        return matrix

    def partial_gaussian_elimination(self) -> np.ndarray:
        """Performs a partial gaussian elimination to determine the required reactions.

        A copy of `self.species_matrix` is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of species - number of
        elements. These reactions are given in the last r rows of the reduced matrix.

        Returns:
            A matrix of the reaction stoichiometry.
        """
        matrix1: np.ndarray = self.species_matrix
        matrix2: np.ndarray = np.eye(self.species.number)
        augmented_matrix: np.ndarray = np.hstack((matrix1, matrix2))
        logger.debug("augmented_matrix = \n%s", augmented_matrix)

        # Forward elimination.
        for i in range(self.number_unique_elements):  # Note only over the number of elements.
            # Check if the pivot element is zero.
            if augmented_matrix[i, i] == 0:
                # Swap rows to get a non-zero pivot element.
                nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
                augmented_matrix[[i, nonzero_row], :] = augmented_matrix[[nonzero_row, i], :]
            # Perform row operations to eliminate values below the pivot.
            for j in range(i + 1, self.species.number):
                ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("Augmented_matrix after forward elimination = \n%s", augmented_matrix)

        # Backward substitution.
        for i in range(self.number_unique_elements - 1, -1, -1):
            # Normalize the pivot row.
            augmented_matrix[i] /= augmented_matrix[i, i]
            # Eliminate values above the pivot.
            for j in range(i - 1, -1, -1):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("Augmented_matrix after backward substitution = \n%s", augmented_matrix)

        reduced_matrix1: np.ndarray = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix: np.ndarray = augmented_matrix[
            self.number_unique_elements :, matrix1.shape[1] :
        ]
        logger.debug("Reduced_matrix1 = \n%s", reduced_matrix1)
        logger.debug("Reaction_matrix = \n%s", reaction_matrix)

        return reaction_matrix

    @cached_property
    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary"""
        reactions: dict[int, str] = {}
        for reaction_index in range(self.number_reactions):
            reactants: str = ""
            products: str = ""
            for species_index, species in enumerate(self.species):
                coeff: float = self.reaction_matrix[reaction_index, species_index]
                if coeff != 0:
                    if coeff < 0:
                        reactants += f"{abs(coeff)} {species.chemical_formula} + "
                    else:
                        products += f"{coeff} {species.chemical_formula} + "

            reactants = reactants.rstrip(" + ")  # Removes the extra + at the end.
            products = products.rstrip(" + ")  # Removes the extra + at the end.
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

        return reactions

    def get_reaction_log10_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the log10 of the reaction equilibrium constant.

        From the Gibbs free energy, we can calculate logKf as:
        logKf = - G/(ln(10)*R*T)

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature
            pressure: Pressure

        Returns:
            log10 of the reaction equilibrium constant
        """
        gibbs_energy: float = self.get_reaction_gibbs_energy_of_formation(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        equilibrium_constant: float = -gibbs_energy / (np.log(10) * GAS_CONSTANT * temperature)

        return equilibrium_constant

    def get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`
            temperature: Temperature
            pressure: Pressure

        Returns:
            The Gibb's free energy of the reaction
        """
        gibbs_energy: float = 0
        for species_index, species in enumerate(self.species.data):
            assert species.thermodynamic_data is not None
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * species.thermodynamic_data.get_formation_gibbs(
                temperature=temperature, pressure=pressure
            )
        return gibbs_energy

    def get_reaction_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the equilibrium constant of a reaction Kf

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`
            temperature: Temperature
            pressure: Pressure

        Returns:
            The equilibrium constant of the reaction
        """
        equilibrium_constant: float = 10 ** self.get_reaction_log10_equilibrium_constant(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        return equilibrium_constant

    def get_coefficient_matrix(self, *, constraints: SystemConstraints) -> np.ndarray:
        """Builds the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        if nrows == self.species.number:
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.debug(msg)
        else:
            num: int = self.species.number - nrows
            # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
            msg = "%d additional (mass) constraint(s) are necessary " % num
            msg += "to solve the system"
            logger.debug(msg)

        coeff: np.ndarray = np.zeros((nrows, self.species.number))
        coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index: int = self.species.indices[constraint.species]
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("Species = %s", self.species.chemical_formulas)
        logger.debug("Coefficient matrix = \n%s", coeff)

        return coeff

    def assemble_right_hand_side_values(
        self, *, system: InteriorAtmosphereSystem, constraints: SystemConstraints
    ) -> np.ndarray:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            system: Interior atmosphere system
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions
        rhs: np.ndarray = np.zeros(nrows, dtype=float)

        # Reactions
        for reaction_index in range(self.number_reactions):
            logger.debug(
                "Row %02d: Reaction %d: %s",
                reaction_index,
                reaction_index,
                self.reactions[reaction_index],
            )
            rhs[reaction_index] = self.get_reaction_log10_equilibrium_constant(
                reaction_index=reaction_index,
                temperature=system.planet.surface_temperature,
                pressure=system.total_pressure,
            )

        # Constraints
        for index, constraint in enumerate(constraints.reaction_network_constraints):
            row_index: int = self.number_reactions + index
            logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs[row_index] = constraint.get_log10_value(
                temperature=system.planet.surface_temperature, pressure=system.total_pressure
            )
            if constraint.name == "pressure":
                rhs[row_index] += system.log10_fugacity_coefficients_dict[constraint.species]

        logger.debug("RHS vector = %s", rhs)

        return rhs

    def assemble_log_fugacity_coefficients(
        self, *, system: InteriorAtmosphereSystem
    ) -> np.ndarray:
        """Assembles the fugacity coefficient vector on the left-hand side of the equations.

        Args:
            system: Interior atmosphere system

        Returns:
            The log10(fugacity coefficient) vector
        """

        # Initialise to ideal behaviour.
        fugacity_coefficients: np.ndarray = np.ones_like(self.species, dtype=float)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for index, gas_species in self.species.gas_species.items():
            fugacity_coefficients[index] = gas_species.eos.get_value(
                temperature=system.planet.surface_temperature, pressure=system.total_pressure
            )
        log_fugacity_coefficients: np.ndarray = np.log10(fugacity_coefficients)

        logger.debug("Fugacity coefficient vector = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    def get_residual(
        self,
        *,
        system: InteriorAtmosphereSystem,
        constraints: SystemConstraints,
        coefficient_matrix: np.ndarray,
    ) -> np.ndarray:
        """Returns the residual vector of the reaction network.

        Args:
            system: Interior atmosphere system
            constraints: Constraints for the system of equations
            coefficient_matrix: Coefficient matrix

        Returns:
            The residual vector of the reaction network
        """

        rhs: np.ndarray = self.assemble_right_hand_side_values(
            system=system, constraints=constraints
        )
        log_fugacity_coefficients: np.ndarray = self.assemble_log_fugacity_coefficients(
            system=system
        )
        residual_reaction: np.ndarray = (
            coefficient_matrix.dot(log_fugacity_coefficients)
            + coefficient_matrix.dot(system.log_solution)
            - rhs
        )
        logger.debug("Residual_reaction = %s", residual_reaction)

        return residual_reaction


@dataclass(kw_only=True)
class InteriorAtmosphereSystem:
    """An interior-atmosphere system

    Args:
        species: A list of species
        planet: A planet. Defaults to a molten Earth
        initial_condition: Initial condition. Defaults to a constant for all species.

    Attributes:
        species: A list of species
        planet: A planet
        output: All output data for this system. Access the dictionary by (calling) output().
    """

    species: Species
    planet: Planet = field(default_factory=Planet)
    initial_condition: InitialConditionABC = field(default_factory=InitialConditionConstant)
    output: Output = field(init=False)
    _reaction_network: ReactionNetwork = field(init=False)
    # Convenient to set and update on this instance.
    _constraints: SystemConstraints = field(init=False, default_factory=SystemConstraints)
    # The solution is log10 of the partial pressure for gas phases and log10 of the activity for
    # condensed phases. The order aligns with the species.
    _log_solution: np.ndarray = field(init=False)

    def __post_init__(self):
        logger.info("Creating an interior-atmosphere system")
        self.species.conform_solubilities_to_planet_composition(self.planet)
        self._reaction_network = ReactionNetwork(species=self.species)
        self._log_solution = np.zeros_like(self.species, dtype=np.float_)
        self.output = Output(self)

    @property
    def constraints(self) -> SystemConstraints:
        """Constraints"""
        return self._constraints

    @property
    def log_solution(self) -> np.ndarray:
        """Log10 partial pressure for gas phases and log10 activity for condensed phases."""
        return self._log_solution

    @property
    def solution(self) -> np.ndarray:
        """Solution."""
        return 10**self.log_solution

    @property
    def solution_dict(self) -> dict[str, float]:
        """Solution for all species in a dictionary.

        This is convenient for a quick check of the solution, but in general you will want to use
        `self.output()` to return a dictionary of all the data.
        """
        output: dict[str, float] = {}
        for chemical_formula, solution in zip(self.species.chemical_formulas, self.solution):
            output[chemical_formula] = solution

        return output

    @property
    def log10_fugacity_coefficients_dict(self) -> dict[str, float]:
        """Fugacity coefficients (relevant for gas species only) in a dictionary."""
        output: dict[str, float] = {
            species.chemical_formula: species.eos.get_log10_value(
                temperature=self.planet.surface_temperature, pressure=self.total_pressure
            )
            for species in self.species.gas_species.values()
        }
        return output

    @property
    def fugacities_dict(self) -> dict[str, float]:
        """Fugacities of all species in a dictionary."""
        output: dict[str, float] = {
            key: 10**value for key, value in self.log10_fugacities_dict.items()
        }
        return output

    @property
    def log10_fugacities_dict(self) -> dict[str, float]:
        """Log10 fugacities of all species in a dictionary."""
        output: dict[str, float] = {}
        for key, value in self.log10_fugacity_coefficients_dict.items():
            output[key] = np.log10(self.solution_dict[key]) + value
        return output

    @property
    def total_pressure(self) -> float:
        """Total pressure."""
        indices: list[int] = list(self.species.gas_species.keys())
        return sum(float(self.solution[index]) for index in indices)

    @property
    def atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere."""
        mu_atmosphere: float = 0
        for index, species in self.species.gas_species.items():
            mu_atmosphere += species.molar_mass * self.solution[index]
        mu_atmosphere /= self.total_pressure

        return mu_atmosphere

    def isclose(
        self, target_dict: dict[str, float], rtol: float = 1.0e-5, atol: float = 1.0e-8
    ) -> np.bool_:
        """Determines if the solution pressures are close to target values within a tolerance.

        Args:
            target_dict: Dictionary of species and their target values
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            True if the solution is close to the target, otherwise False
        """

        if len(self.solution_dict) != len(target_dict):
            return np.bool_(False)

        target_pressures: np.ndarray = np.array(
            [target_dict[species.formula.formula] for species in self.species]
        )
        isclose: np.bool_ = np.isclose(target_pressures, self.solution, rtol=rtol, atol=atol).all()

        return isclose

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        initial_solution: list[float] | np.ndarray | None = None,
        method: str = "hybr",
        tol: float | None = None,
        **options,
    ) -> None:
        """Solves the system to determine the activities and partial pressures with constraints.

        Args:
            constraints: Constraints for the system of equations
            initial_solution: Initial guess for the solution. Defaults to None.
            method: Type of solver. Defaults to 'hybr'.
            tol: Tolerance for termination. Defaults to None.
            **options: Keyword arguments for solver options. Available keywords depend on method.
        """
        self.set_constraints(constraints)
        self._log_solution = self._solve(
            initial_solution=initial_solution,
            method=method,
            tol=tol,
            **options,
        )

        # Recompute quantities that depend on the solution, since species.mass is not called for
        # the reaction network but this sets the solution for the gas phase.
        for species in self.species.gas_species.values():
            species.mass(
                planet=self.planet,
                system=self,
            )
        for species in self.species.condensed_species.values():
            species.output = CondensedSpeciesOutput(
                activity=species.activity.get_value(
                    temperature=self.planet.surface_temperature, pressure=self.total_pressure
                )
            )

        self.output.add(constraints)

        self.initial_condition.update(self.output)

        logger.info(pprint.pformat(self.solution_dict))

    def set_constraints(self, constraints: SystemConstraints) -> None:
        """Combines user-prescribed constraints with intrinsic (activity) constraints.

        Args;
            constraints: Constraints for the system of equations
        """
        logger.info("Set constraints")
        self._constraints = constraints

        for condensed_species in self.species.condensed_species.values():
            self._constraints.append(condensed_species.activity)
        logger.debug("Constraints: %s", pprint.pformat(self._constraints))

    def _conform_initial_solution_to_constraints(self) -> None:
        """Conforms the initial solution (estimate) to activity, pressure and fugacity constraints.

        Pressure and fugacity constraints can be imposed directly on the initial solution
        estimate. For simplicity we impose both as pressure constraints.
        """
        for constraint in self.constraints.reaction_network_constraints:
            index: int = self.species.indices[constraint.species]
            logger.debug("Setting %s %d", constraint.species, index)
            self._log_solution[index] = np.log10(
                constraint.get_value(
                    temperature=self.planet.surface_temperature, pressure=self.total_pressure
                )
            )
        logger.debug("total_pressure = %s", self.total_pressure)
        logger.debug("Conforming initial solution to constraints = %s", self._log_solution)

    def _solve(
        self,
        *,
        initial_solution: list[float] | np.ndarray | None = None,
        method: str,
        tol: float | None,
        **options,
    ) -> np.ndarray:
        """Solves the non-linear system of equations.

        Args:
            initial_solution: Initial guess for the solution
            method: Type of solver
            tol: Tolerance for termination
            **options: Keyword arguments for solver options. Available keywords depend on method.

        Returns:
            The solution array
        """

        # Manual override trumps self.initial_condition
        if initial_solution is not None:
            self._log_solution = np.log10(np.array(initial_solution))
        else:
            constraints_eval: np.ndarray = self.constraints.evaluate_log10_values(self)
            log10_value: float | np.ndarray = self.initial_condition.get_log10_value(
                constraints_eval
            )
            self._log_solution = log10_value * np.ones(self.species.number)

        self._conform_initial_solution_to_constraints()

        # For outputting to the logger below.
        initial_guess: np.ndarray = self._log_solution.copy()

        coefficient_matrix: np.ndarray = self._reaction_network.get_coefficient_matrix(
            constraints=self.constraints
        )

        sol = root(
            self._objective_func,
            self._log_solution,
            args=(coefficient_matrix,),
            method=method,
            tol=tol,
            options=options,
        )

        logger.info(sol["message"])
        logger.debug("sol = %s", sol)

        # TODO: Raise a more descriptive exception.
        if not sol.success:
            raise SystemExit()

        sol = sol.x

        logger.debug("Initial guess solution = %s", initial_guess)
        logger.debug("Actual solution = %s", sol)
        error: np.ndarray = np.sqrt(mean_squared_error(sol, initial_guess))
        logger.info(
            "%s: RMSE (actual vs initial) = %s", self.initial_condition.__class__.__name__, error
        )

        return sol

    def _objective_func(
        self,
        log_solution: np.ndarray,
        coefficient_matrix: np.ndarray,
    ) -> np.ndarray:
        """Objective function for the non-linear system.

        Args:
            log_solution: Log10 of the activities and pressures of each species
            coefficient_matrix: Coefficient matrix

        Returns:
            The solution, which is the log10 of the activities and pressures for each species
        """
        logger.debug("log_solution = %s", log_solution)
        self._log_solution = log_solution

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = self._reaction_network.get_residual(
            system=self, constraints=self.constraints, coefficient_matrix=coefficient_matrix
        )

        # Compute residual for the mass balance (if relevant).
        residual_mass: np.ndarray = np.zeros(
            len(self.constraints.mass_constraints), dtype=np.float_
        )
        for constraint_index, constraint in enumerate(self.constraints.mass_constraints):
            for species in self.species.gas_species.values():
                residual_mass[constraint_index] += species.mass(
                    planet=self.planet,
                    system=self,
                    element=constraint.species,
                )
            # Mass values are constant so no need to pass any arguments to get_value().
            residual_mass[constraint_index] -= constraint.get_value()
            # Normalise by target mass to compute a relative residual.
            residual_mass[constraint_index] /= constraint.get_value()
        logger.debug("Residual_mass = %s", residual_mass)

        # Compute residual for the total pressure (if relevant).
        residual_total_pressure: np.ndarray = np.zeros(
            len(self.constraints.total_pressure_constraint), dtype=np.float_
        )
        if len(self.constraints.total_pressure_constraint):
            constraint: ConstraintABC = self.constraints.total_pressure_constraint[0]
            residual_total_pressure[0] += (
                np.log10(self.total_pressure) - constraint.get_log10_value()
            )

        # Combined residual.
        residual: np.ndarray = np.concatenate(
            (residual_reaction, residual_mass, residual_total_pressure)
        )
        logger.debug("Residual = %s", residual)

        return residual
