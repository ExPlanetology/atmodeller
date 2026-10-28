# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Core container classes for species collections and solver configuration."""

import logging
from collections.abc import Iterable, Iterator
from typing import Any, Generic, Optional, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float

from atmodeller.constants import (
    DISSOLVED_SUFFIX,
    GAS_PHASE_COMPONENT,
    GAS_SUFFIX,
    LIQUID_PHASE_COMPONENT,
    LIQUID_SUFFIX,
    LOG_NUMBER_MOLES_LOWER,
    LOG_NUMBER_MOLES_UPPER,
    LOG_STABILITY_LOWER,
    LOG_STABILITY_UPPER,
    TAU,
    TAU_MAX,
    TAU_NUM,
)
from atmodeller.database import DataBase
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
from atmodeller.thermodata import ActivityCoefficient, ThermodynamicProperties

logger: logging.Logger = logging.getLogger(__name__)


class ChemicalSpecies(eqx.Module):
    """Chemical species that participate in reactions

    Args:
        data: Chemical species data
        activity: Activity
        solve_for_stability: Solve for stability
        number_solution: Number of solution quantities
        thermo: Thermodynamic properties, which provide the heat capacity, enthalpy, entropy and
            Gibbs energy, e.g. ``species.thermo.enthalpy(temperature)``
        include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
            fraction aggregations.
        is_gas: Whether the species is a gas, which can only be in the gas phase
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solve_for_stability: bool
    number_solution: int
    thermo: ThermodynamicProperties
    include_in_phase_mass: bool
    is_gas: bool

    @classmethod
    def _create(
        cls,
        formula: str,
        suffix: str,
        activity: ActivityProtocol,
        solve_for_stability: bool,
        number_solution: int,
        include_in_phase_mass: bool,
        thermo: ThermodynamicProperties,
        is_gas: bool,
        extrapolate_below: bool = True,
        extrapolate_above: bool = True,
    ) -> Self:
        """Creates an instance.

        Args:
            formula: Formula
            suffix: Suffix of the name, e.g. ``g`` for ``H2O_g``
            activity: Activity
            solve_for_stability: Solve for stability
            number_solution: Number of solution quantities
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations.
            thermo: Thermodynamic properties
            is_gas: Whether the species is a gas
            extrapolate_below: Whether to extrapolate the thermodynamic data below their
                temperature range, otherwise the species is not stable there. Defaults to
                ``True``.
            extrapolate_above: Whether to extrapolate the thermodynamic data above their
                temperature range, otherwise the species is not stable there. Defaults to
                ``True``.

        Returns:
            An instance
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, suffix)

        if not (extrapolate_below and extrapolate_above):
            thermo = thermo.with_extrapolation(extrapolate_below, extrapolate_above)

        return cls(
            species_data,
            activity,
            solve_for_stability,
            number_solution,
            thermo,
            include_in_phase_mass,
            is_gas,
        )

    @classmethod
    def from_component(
        cls,
        component: str,
        *,
        formula: str | None = None,
        activity: ActivityProtocol = ActivityCoefficient(),
        solve_for_stability: bool = True,
        include_in_phase_mass: bool = True,
        database: DataBase | None = None,
        extrapolate_below: bool = False,
        extrapolate_above: bool = False,
    ) -> Self:
        """Creates a condensed species from a phase component

        This creates a species of a pure phase, such as a mineral, or of a mixed phase, such as
        the silicate melt or metal, of which the phase component is an end-member. The species is
        named from the phase component, e.g. ``O2Si_cristobalite-beta``, or ``H2O_l`` for the
        liquid. To stitch several phase components together for a pure phase, use
        :meth:`~atmodeller.phases.PurePhase.from_formula` or
        :meth:`~atmodeller.phases.PurePhase.from_components`.

        By default, the thermodynamic data are not extrapolated outside their temperature range,
        where the species is therefore not stable. A condensed phase is often bounded by a phase
        boundary, such as a melting point, so the same set of condensates can be used at any
        temperature.

        Args:
            component: Name of the phase component, e.g. ``cristobalite-beta`` or ``liquid``
            formula: Formula. Defaults to ``None``, which takes the formula of a mineral, and is
                needed for ``liquid``, which is a phase component of several formulas.
            activity: Activity. Defaults to unity activity.
            solve_for_stability: Solve for stability. Defaults to ``True``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.
            database: Database. Defaults to ``None``, which uses the default database
                (:meth:`~atmodeller.database.DataBase.default`).
            extrapolate_below: Whether to extrapolate the thermodynamic data below their
                temperature range, otherwise the species is not stable there. Defaults to
                ``False``.
            extrapolate_above: Whether to extrapolate the thermodynamic data above their
                temperature range, otherwise the species is not stable there. Defaults to
                ``False``.

        Returns:
            A condensed species

        Raises:
            KeyError: If the phase component is not available for the formula
            ValueError: If the phase component is the gas, or is shared by several formulas and no
                formula is given
        """
        if component == GAS_PHASE_COMPONENT:
            raise ValueError(
                "The gas is not a condensed phase component. Create it with create_gas."
            )
        database = database or DataBase.default()
        if formula is None:
            formula = database.formula_of(component)

        return cls._create_condensed(
            formula,
            LIQUID_SUFFIX if component == LIQUID_PHASE_COMPONENT else component,
            database.get_phase_component(formula, component),
            activity=activity,
            solve_for_stability=solve_for_stability,
            include_in_phase_mass=include_in_phase_mass,
            extrapolate_below=extrapolate_below,
            extrapolate_above=extrapolate_above,
        )

    @classmethod
    def _create_condensed(
        cls,
        formula: str,
        suffix: str,
        thermo: ThermodynamicProperties,
        *,
        activity: ActivityProtocol,
        solve_for_stability: bool,
        include_in_phase_mass: bool,
        extrapolate_below: bool,
        extrapolate_above: bool,
    ) -> Self:
        """Creates a condensed species from its thermodynamic properties

        Used by :meth:`from_component` and :meth:`~atmodeller.phases.PurePhase.from_components`,
        which take the thermodynamic properties from a database.
        """
        # Either both the number of moles and stability are solved for, or alternatively stability
        # can be enforced in which case the number of moles is irrelevant and there is nothing to
        # solve for.
        number_solution: int = 2 if solve_for_stability else 0

        return cls._create(
            formula,
            suffix,
            activity,
            solve_for_stability,
            number_solution,
            include_in_phase_mass,
            thermo,
            False,
            extrapolate_below,
            extrapolate_above,
        )

    @classmethod
    def create_gas(
        cls,
        formula: str,
        *,
        activity: ActivityProtocol = IdealGas(),
        solve_for_stability: bool = False,
        include_in_phase_mass: bool = True,
        database: DataBase | None = None,
    ) -> Self:
        """Creates a gas species with some default values.

        Args:
            formula: Formula
            activity: Activity. Defaults to an ideal gas.
            solve_for_stability: Solve for stability. Defaults to ``False``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.
            database: Database. Defaults to ``None``, which uses the default database
                (:meth:`~atmodeller.database.DataBase.default`).

        Returns:
            A gas species

        Raises:
            KeyError: If the gas is not available for the formula
        """
        thermo: ThermodynamicProperties = (database or DataBase.default()).get_phase_component(
            formula, GAS_PHASE_COMPONENT
        )

        # The number of moles is always solved for, and stability can be if desired, although
        # this is not typically done for gas species because they are usually stable over the range
        # of conditions considered and truncating the abundance can severely distort the results,
        # notably for O2.
        number_solution: int = 2 if solve_for_stability else 1

        return cls._create(
            formula,
            GAS_SUFFIX,
            activity,
            solve_for_stability,
            number_solution,
            include_in_phase_mass,
            thermo,
            True,
        )

    @property
    def name(self) -> str:
        return self.data.name

    def standard_gibbs_over_RT(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Gets the standard Gibbs energy over RT

        See :meth:`~atmodeller.thermodata.core.ThermodynamicProperties.standard_gibbs_over_RT`
        for the standard state.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Standard Gibbs energy over RT
        """
        return self.thermo.standard_gibbs_over_RT(temperature, pressure)

    def __str__(self) -> str:
        return f"{self.data.name}: {self.activity.__class__.__name__}"


class ReservoirSpecies(eqx.Module):
    """Reservoir species

    A species that is not part of the reaction network but can exchange with it. For example, this
    can represent a volatile species dissolved in a melt that can exchange with the gas phase but
    is not explicitly included in the reaction network.

    This is separate from :class:`ChemicalSpecies` because dissolution is given by empirically
    calibrated solubility laws, which relate the concentration of the dissolved species to the
    fugacity of its gas, rather than by the Gibbs energy of a reaction. A reservoir species
    therefore has a solubility law instead of thermodynamic data, and only enters the mass balance.

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
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, DISSOLVED_SUFFIX)

        if solubility is None:  # pragma: no cover
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
    """Container of species and metadata

    Args:
        species: Species, whose names must be unique

    Raises:
        ValueError: If two species have the same name
    """

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
        duplicates: list[str] = sorted(
            {name for name in self.species_names if self.species_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"Species names must be unique, but {duplicates} appear more than once"
            )

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
        """Element molar masses for the unique elements in the species

        Each is the mass per atom of the element in the composition of a species that contains it,
        so the electron (the charge of an ion, element ``e-``) is included like any other element.
        """
        element_molar_masses: dict[str, float] = {}

        for species_ in self:
            for element_, (count, mass, _) in species_.data.composition.items():
                element_molar_masses.setdefault(element_, mass / count * unit_conversion.g_to_kg)

        return np.array(
            [element_molar_masses[element_] for element_ in self.unique_elements], dtype=float
        )

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
        tau: Tau factor for species stability, i.e. the final tau of the tau sweep. Defaults to
            :data:`~atmodeller.constants.TAU`.
        tau_max: Largest tau, at the start of the tau sweep for systems with species stability.
            Defaults to :data:`~atmodeller.constants.TAU_MAX`.
        tau_num: Number of tau values in the tau sweep, from ``tau_max`` to ``tau`` inclusive.
            Defaults to :data:`~atmodeller.constants.TAU_NUM`.
        log_number_moles_lower: Lower bound on the log number of moles of a species. Defaults to
            :data:`~atmodeller.constants.LOG_NUMBER_MOLES_LOWER`.
        log_number_moles_upper: Upper bound on the log number of moles of a species. Defaults to
            :data:`~atmodeller.constants.LOG_NUMBER_MOLES_UPPER`.
        log_stability_lower: Lower bound on the log stability of a species. Defaults to
            :data:`~atmodeller.constants.LOG_STABILITY_LOWER`.
        log_stability_upper: Upper bound on the log stability of a species. Defaults to
            :data:`~atmodeller.constants.LOG_STABILITY_UPPER`.
    """

    max_starts: int = 10
    """Maximum number of starts"""
    tau: Array = eqx.field(converter=as_j64, default=TAU)
    """Tau factor for species stability, i.e. the final tau of the tau sweep"""
    tau_max: float = TAU_MAX
    """Largest tau, at the start of the tau sweep"""
    tau_num: int = TAU_NUM
    """Number of tau values in the tau sweep, from ``tau_max`` to ``tau`` inclusive"""
    log_number_moles_lower: float = LOG_NUMBER_MOLES_LOWER
    """Lower bound on the log number of moles of a species"""
    log_number_moles_upper: float = LOG_NUMBER_MOLES_UPPER
    """Upper bound on the log number of moles of a species"""
    log_stability_lower: float = LOG_STABILITY_LOWER
    """Lower bound on the log stability of a species"""
    log_stability_upper: float = LOG_STABILITY_UPPER
    """Upper bound on the log stability of a species"""

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
            number_species, self.log_number_moles_lower, self.log_stability_lower
        )

    def _get_upper_bound(self, number_species: int) -> Float[Array, " dim"]:
        """Gets the upper bound for truncating the solution during the solve.

        Args:
            number_species: Number of species

        Returns:
            Upper bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(
            number_species, self.log_number_moles_upper, self.log_stability_upper
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
