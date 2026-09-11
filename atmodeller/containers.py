# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core container classes for species collections and solver configuration."""

import logging
from collections.abc import Iterable, Iterator
from typing import Any, Generic, Optional, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float
from molmass import Formula

from atmodeller.constants import (
    DISSOLVED_STATE,
    GAS_STATE,
    LOG_NUMBER_MOLES_LOWER,
    LOG_NUMBER_MOLES_UPPER,
    LOG_STABILITY_LOWER,
    LOG_STABILITY_UPPER,
    SOLID_STATE,
    TAU,
)
from atmodeller.eos.core import IdealGas
from atmodeller.interfaces import (
    ActivityProtocol,
    ChemicalSpeciesData,
    SolubilityProtocol,
    SpeciesProtocol,
    TSpecies_co,
)
from atmodeller.jax_utils import NpFloat, NpInt, RootFindParameters, as_j64
from atmodeller.sci_utils import unit_conversion
from atmodeller.solubility.core import NoSolubility
from atmodeller.thermodata import ActivityCoefficient, thermodynamic_data_source
from atmodeller.thermodata.core import (
    ThermodynamicCoefficients,
    thermodynamic_coefficients_dictionary,
)

logger: logging.Logger = logging.getLogger(__name__)


class ChemicalSpecies(eqx.Module):
    """Chemical species that participate in reactions

    Args:
        data: Chemical species data
        activity: Activity
        solve_for_stability: Solve for stability
        number_solution: Number of solution quantities
        thermo: Thermodynamic coefficients
        include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
            fraction aggregations.
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solve_for_stability: bool
    number_solution: int
    thermo: ThermodynamicCoefficients
    include_in_phase_mass: bool

    @classmethod
    def create(
        cls,
        formula: str,
        state: str,
        activity: ActivityProtocol,
        solve_for_stability: bool,
        number_solution: int,
        include_in_phase_mass: bool,
    ) -> Self:
        """Creates an instance.

        Args:
            formula: Formula
            state: State of aggregation, as typically defined by JANAF
            activity: Activity
            solve_for_stability: Solve for stability
            number_solution: Number of solution quantities
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations.

        Returns:
            An instance
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, state)

        try:
            thermo: ThermodynamicCoefficients = thermodynamic_coefficients_dictionary[
                species_data.name
            ]
        except KeyError:
            raise KeyError(
                f"{species_data.name} not available. "
                f"Available species are {thermodynamic_data_source.available_species()}"
            )

        return cls(
            species_data,
            activity,
            solve_for_stability,
            number_solution,
            thermo,
            include_in_phase_mass,
        )

    @classmethod
    def create_condensed(
        cls,
        formula: str,
        *,
        state: str = SOLID_STATE,
        activity: ActivityProtocol = ActivityCoefficient(),
        solve_for_stability: bool = True,
        include_in_phase_mass: bool = True,
    ) -> Self:
        """Creates a condensed species with some default values.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :const:`~atmodeller.constants.SOLID_STATE`.
            activity: Activity. Defaults to unity activity.
            solve_for_stability: Solve for stability. Defaults to ``True``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            A condensed species
        """
        # Either both the number of moles and stability are solved for, or alternatively stability
        # can be enforced in which case the number of moles is irrelevant and there is nothing to
        # solve for.
        number_solution: int = 2 if solve_for_stability else 0

        return cls.create(
            formula, state, activity, solve_for_stability, number_solution, include_in_phase_mass
        )

    @classmethod
    def create_gas(
        cls,
        formula: str,
        *,
        state: str = GAS_STATE,
        activity: ActivityProtocol = IdealGas(),
        solve_for_stability: bool = False,
        include_in_phase_mass: bool = True,
    ) -> Self:
        """Creates a gas species with some default values.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :const:`~atmodeller.constants.GAS_STATE`.
            activity: Activity. Defaults to an ideal gas.
            solve_for_stability: Solve for stability. Defaults to ``False``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            A gas species
        """
        # The number of moles is always solved for, and stability can be if desired, although
        # this is not typically done for gas species because they are usually stable over the range
        # of conditions considered and truncating the abundance can severely distort the results,
        # notably for O2.
        number_solution: int = 2 if solve_for_stability else 1

        return cls.create(
            formula, state, activity, solve_for_stability, number_solution, include_in_phase_mass
        )

    @property
    def name(self) -> str:
        return self.data.name

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Array:
        """Gets Gibbs energy over RT

        Args:
            temperature: Temperature (K)

        Returns:
            Gibbs energy over RT
        """
        return self.thermo.get_gibbs_over_RT(temperature)

    def __str__(self) -> str:
        return f"{self.data.name}: {self.activity.__class__.__name__}"


class ReservoirSpecies(eqx.Module):
    """Reservoir species

    A species that is not part of the reaction network but can exchange with it. For example, this
    can represent a volatile species dissolved in a melt that can exchange with the gas phase but
    is not explicitly included in the reaction network.

    Args:
        data: Chemical species data
        activity: Activity
        solubility: Solubility
        number_solution: Number of solution quantities
        include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
            fraction aggregations.
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solubility: SolubilityProtocol
    number_solution: int
    include_in_phase_mass: bool

    @classmethod
    def create_dissolved(
        cls,
        formula: str,
        *,
        activity: ActivityProtocol = ActivityCoefficient(),
        solubility: Optional[SolubilityProtocol] = None,
        include_in_phase_mass: bool = True,
    ) -> Self:
        """Creates a dissolved species with some default values.

        Args:
            formula: Formula
            activity: Activity. Defaults to unity activity.
            solubility: Solubility. Defaults to no solubility.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            A dissolved species
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, state=DISSOLVED_STATE)

        if solubility is None:
            solubility = NoSolubility()
            number_solution: int = 0
        else:
            number_solution = 1

        return cls(species_data, activity, solubility, number_solution, include_in_phase_mass)

    @property
    def solve_for_stability(self) -> bool:
        """Always ``False`` — reservoir species have no stability variable."""
        return False

    def __str__(self) -> str:
        return f"{self.data.name}: {self.solubility.__class__.__name__}"


class SpeciesCollection(eqx.Module, Generic[TSpecies_co]):
    """Container of species and metadata"""

    species: tuple[TSpecies_co, ...]
    """Species in the collection"""
    species_names: tuple[str, ...]
    """Unique names of all species"""
    number_solution: int
    """Number of solution quantities, which cannot depend on traced quantities"""
    unique_elements_map: dict[str, int]
    """Mapping of unique element name to index in the unique elements array"""

    def __init__(self, species: Iterable[TSpecies_co]):
        self.species = tuple(species)
        self.species_names = tuple(species_.data.name for species_ in self)

        # Ensure number_solution is static
        self.number_solution = sum(species.number_solution for species in self)

        # Unique elements in species in alphabetical order
        elements: list[str] = []
        for species_ in self.species:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        self.unique_elements_map = {
            element: index for index, element in enumerate(sorted(unique_elements))
        }

        logger.debug(
            "Creating %s: %s", self.__class__.__name__, tuple(str(species) for species in self)
        )

    @property
    def active_stability(self) -> Bool[Array, " n_species"]:
        """Active stability mask"""
        return jnp.array([species.solve_for_stability for species in self], dtype=bool)

    @property
    def element_molar_masses(self) -> NpFloat:
        """Element molar masses for the unique elements in the species"""
        element_molar_masses: list[float] = []

        for element_ in self.unique_elements:
            mformula: Formula = Formula(element_)
            molar_mass: float = mformula.mass * unit_conversion.g_to_kg
            element_molar_masses.append(molar_mass)

        return np.array(element_molar_masses, dtype=float)

    @property
    def molar_masses(self) -> Float[Array, " n_species"]:
        """Molar masses for all species in the collection"""
        return jnp.array([species_.data.molar_mass for species_ in self], dtype=float)

    @property
    def number_elements(self) -> int:
        """Number of unique elements in the species"""
        return len(self.unique_elements)

    @property
    def number_species(self) -> int:
        """Number of species"""
        return len(self)

    @property
    def phase_mass_mask(self) -> Bool[Array, "..."]:
        """Mask for species included in phase-level mass, mole, and fraction aggregations"""
        return jnp.array([species.include_in_phase_mass for species in self], dtype=bool)

    @property
    def reaction_species(self) -> "SpeciesCollection[ChemicalSpecies]":
        """Reaction species collection"""
        return SpeciesCollection(
            [species for species in self if isinstance(species, ChemicalSpecies)]
        )

    @property
    def reaction_species_mask(self) -> Bool[Array, "..."]:
        """Mask for reaction species in the collection"""
        return jnp.array([isinstance(species_, ChemicalSpecies) for species_ in self], dtype=bool)

    @property
    def reservoir_species(self) -> "SpeciesCollection[ReservoirSpecies]":
        """Reservoir species collection"""
        return SpeciesCollection(
            [species for species in self if isinstance(species, ReservoirSpecies)]
        )

    @property
    def reservoir_species_mask(self) -> Bool[Array, "..."]:
        """Mask for reservoir species in the collection"""
        return jnp.array([isinstance(species_, ReservoirSpecies) for species_ in self], dtype=bool)

    @property
    def unique_elements(self) -> tuple[str, ...]:
        """Unique elements in species in alphabetical order"""
        return tuple(self.unique_elements_map.keys())

    def get_element_index(self, element: str) -> int:
        """Get the index of an element in the unique-elements map.

        Returns ``-1`` when the element is not present.
        """
        # TODO: Returning a non-existent element with an index of -1 is a bit hacky.
        return self.unique_elements_map.get(element, -1)

    def __getitem__(self, index: int) -> TSpecies_co:
        return self.species[index]

    def __iter__(self) -> Iterator[TSpecies_co]:
        return iter(self.species)

    def __len__(self) -> int:
        return len(self.species)

    def __str__(self) -> str:
        return str(tuple(str(species) for species in self.species))


def get_formula_matrix(species: SpeciesCollection[SpeciesProtocol]) -> NpInt:
    """Gets the formula matrix.

    Elements are given in rows and species in columns following the convention in :cite:t:`LKS17`.

    Args:
        species: Species collection

    Returns:
        Formula matrix
    """
    formula_matrix: NpInt = np.zeros(
        (len(species.unique_elements), species.number_species), dtype=int
    )

    for element_index, element in enumerate(species.unique_elements):
        for species_index, species_ in enumerate(species):
            count: int = 0
            try:
                count = species_.data.composition[element][0]
            except KeyError:
                count = 0
            formula_matrix[element_index, species_index] = count

    # logger.debug("formula_matrix = %s", formula_matrix)

    return formula_matrix


class SolverParameters(RootFindParameters):  # pragma: no cover
    """Solver parameters

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
        max_starts: Maximum number of starts. Defaults to ``10``.
        tau: Tau factor for species stability. Defaults to :const:`~atmodeller.constants.TAU`.
    """

    max_starts: int = 10
    """Maximum number of starts"""
    tau: Array = eqx.field(converter=as_j64, default=TAU)
    """Tau factor for species stability"""

    def get_options(self, number_species: int) -> dict[str, Any]:
        """Gets the solver options.

        Args:
            number_species: Number of species

        Returns:
            Solver options
        """
        options: dict[str, Any] = {
            "lower": self._get_lower_bound(number_species),
            "upper": self._get_upper_bound(number_species),
            "jac": self.jac,
        }

        return options

    def _get_lower_bound(self, number_species: int) -> Float[Array, " dim"]:
        """Gets the lower bound for truncating the solution during the solve.

        Args:
            number_species: Number of species

        Returns:
            Lower bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(
            number_species, LOG_NUMBER_MOLES_LOWER, LOG_STABILITY_LOWER
        )

    def _get_upper_bound(self, number_species: int) -> Float[Array, " dim"]:
        """Gets the upper bound for truncating the solution during the solve.

        Args:
            number_species: Number of species

        Returns:
            Upper bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(
            number_species, LOG_NUMBER_MOLES_UPPER, LOG_STABILITY_UPPER
        )

    def _get_hypercube_bound(
        self, number_species: int, log_number_moles_bound: float, stability_bound: float
    ) -> Float[Array, " dim"]:
        """Gets the bound on the hypercube.

        Args:
            number_species: Number of species
            log_number_moles_bound: Bound on the log number of moles
            stability_bound: Bound on the stability

        Returns:
            Bound on the hypercube that contains the root
        """
        bound: Array = jnp.concatenate(
            (
                log_number_moles_bound * jnp.ones(number_species),
                stability_bound * jnp.ones(number_species),
            )
        )

        return bound
