# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reaction network API.

Defines reaction network components used to assemble the full thermodynamic equilibrium system.

This module provides:

- Construction of element-species formula matrices,
- Core chemical reaction networks derived from elemental constraints,
- Dissolution reaction networks coupling gas and condensed species,
- Aggregation of reaction blocks into a unified :class:`ReactionSystem`,
- Evaluation of equilibrium constants and nonlinear residuals.

The reaction system combines stoichiometric matrices, thermodynamic data, and phase-specific
constraints into a form suitable for JAX-compiled nonlinear solvers. Both core reactions and
dissolution reactions are represented in a unified matrix form with associated stability handling.

Primary entry point:
    :class:`ReactionSystem`
"""

import logging
import pprint
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Optional, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from atmodeller.constants import DISSOLUTION_PPMW_FLOOR, GAS_STATE, SILICATE_MELT_PHASE_INDEX
from atmodeller.containers import SpeciesCollection, get_formula_matrix
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.jax_utils import (
    FloatArray,
    NpArray,
    NpBool,
    NpFloat,
    NpInt,
    masked_logsumexp,
    partial_rref,
    safe_exp,
    to_hashable,
)
from atmodeller.phases import BasePhase, GasPhase
from atmodeller.thermodata import thermodynamic_data_source

logger: logging.Logger = logging.getLogger(__name__)


def get_reaction_dictionary(
    reaction_matrix: NpFloat, species_names: Iterable[str]
) -> dict[int, str]:
    """Gets reactions as a dictionary.

    Args:
        reaction_matrix: Reaction matrix of shape (number_reactions, number_species)
        species_names: Species names corresponding to the columns of the reaction matrix

    Returns:
        Reactions as a dictionary
    """
    reactions: dict[int, str] = {}

    if reaction_matrix.size != 0:
        for reaction_index in range(reaction_matrix.shape[0]):
            reactants: str = ""
            products: str = ""
            for species_index, name in enumerate(species_names):
                coeff: float = reaction_matrix[reaction_index, species_index].item()
                if coeff != 0:
                    if coeff < 0:
                        reactants += f"{abs(coeff)} {name} + "
                    else:
                        products += f"{coeff} {name} + "

            reactants = reactants.rstrip(" + ")
            products = products.rstrip(" + ")
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

    return reactions


class PhaseIndex(eqx.Module):
    """Stores start and stop indices of a phase in the full species collection.

    Args:
        start: Starting index of the phase in the full species collection
        stop: Stopping index of the phase in the full species collection
    """

    start: int
    stop: int

    @property
    def slice(self) -> slice:
        """Slice object for indexing arrays."""
        return slice(self.start, self.stop)

    def mask(self, n_total: int) -> NpArray:
        """Boolean mask for this phase

        Args:
            n_total: Total number of species in the full species collection

        Returns:
            Boolean mask for this phase
        """
        mask: NpBool = np.zeros(n_total, dtype=bool)
        mask[self.start : self.stop] = True
        return mask

    def __len__(self) -> int:  # pragma: no cover
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"PhaseIndex(start={self.start}, stop={self.stop})"


class PhaseSystem(eqx.Module):
    """A phase system containing multiple phases.

    This class represents a complete set of coexisting phases.

    Args:
        gas: Gas phase
        other_phases: Tuple of other phases

    Attributes:
        phases: All phases, where the first phase is always the gas phase by construction
        species: All species
    """

    phases: tuple[BasePhase, ...]
    """all phases, where the first phase is always the gas phase by construction"""
    species: SpeciesCollection[SpeciesProtocol]
    """All species"""
    _phase_indices: dict[int, PhaseIndex]
    """Phase indices for slicing species arrays"""

    def __init__(self, gas: GasPhase, *, other_phases: Optional[tuple[BasePhase, ...]] = None):
        other_phases_ = tuple() if other_phases is None else other_phases
        self.phases = (gas,) + other_phases_

        other_species: tuple[SpeciesProtocol, ...] = tuple(
            species for phase in other_phases_ for species in phase.species.species
        )
        all_species: tuple[SpeciesProtocol, ...] = self.gas.species.species + other_species

        self.species = SpeciesCollection(all_species)

        # Phase indexing
        start: int = 0
        self._phase_indices = {}

        for nn, phase in enumerate(self.phases):
            n: int = len(phase)
            self._phase_indices[nn] = PhaseIndex(start, start + n)
            start += n

    @property
    def condensates_species_mask(self) -> Bool[Array, " n_species"]:
        """Boolean mask for condensate species for generating the auto initial condition"""
        return jnp.logical_and(~self.gas_species_mask, self.species.reaction_species_mask)

    @property
    def gas(self) -> GasPhase:
        """Gas phase"""
        return cast(GasPhase, self.phases[0])

    @property
    def gas_slice(self) -> slice:
        """Gas is always the first phase by construction"""
        return self.phase_slice(0)

    @property
    def gas_species_mask(self) -> NpBool:
        """Gas is always the first phase by construction"""
        return self.phase_mask(0)

    def phase_slice(self, phase_index: int) -> slice:
        """Slice object for a given phase index

        Args:
            phase_index: Index of the phase

        Returns:
            Slice object for the phase
        """
        return self._phase_indices[phase_index].slice

    def phase_mask(self, phase_index: int) -> NpBool:
        """Boolean mask for a given phase index

        Args:
            phase_index: Index of the phase

        Returns:
            Boolean mask for the phase
        """
        return self._phase_indices[phase_index].mask(len(self.species))


class BaseReactionBlock(eqx.Module):
    """Base reaction block"""

    species: eqx.AbstractVar[SpeciesCollection[SpeciesProtocol]]
    """Species collection"""

    @property
    @abstractmethod
    def number_reactions(self) -> int:
        """Number of reactions in the reaction block"""

    @abstractmethod
    def get_log_Kp(self, temperature: FloatArray, *args, **kwargs) -> FloatArray:
        """Gets log of the equilibrium constant of each reaction in the reaction block"""

    @abstractmethod
    def get_matrix(self) -> NpFloat:
        """Gets the full reaction matrix of the reaction block"""

    @abstractmethod
    def get_stability_matrix(self) -> NpFloat:
        """Gets the full stability matrix of the reaction block"""

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.get_matrix(), self.species.species_names)

    def output_to_logger(self):
        """Outputs the reaction block to the logger"""
        logger.debug("%s matrix = %s", self.__class__.__name__, self.get_matrix())
        logger.info(
            "%s network = %s",
            self.__class__.__name__,
            pprint.pformat(self.get_reaction_dictionary(), sort_dicts=False),
        )


class ReactionNetwork(BaseReactionBlock):
    """Handles core chemical reactions.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    reaction_matrix: NpFloat
    """Reaction matrix"""
    reaction_matrix_full: NpFloat
    """Reaction matrix expanded to full species space"""
    reaction_stability_mask_full: NpBool
    """Stability mask for reaction matrix expanded to full species space"""
    reaction_stability_matrix_full: NpFloat
    """Reaction stability matrix expanded to full species space"""
    vmap_gibbs: Callable
    """Vectorised Gibbs free energy function for reaction species"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)

        # Reaction matrix of linearly independent reactions
        transpose_formula_matrix: NpInt = get_formula_matrix(self.species.reaction_species).T
        self.reaction_matrix: NpFloat = partial_rref(transpose_formula_matrix)

        # Reaction matrix expanded to full species space
        self.reaction_matrix_full: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )
        # Insert reduced matrix into correct columns
        self.reaction_matrix_full[:, self.species.reaction_species_mask] = self.reaction_matrix

        self.reaction_stability_mask_full = np.broadcast_to(
            self.species.active_stability, self.reaction_matrix_full.shape
        )
        self.reaction_stability_matrix_full = (
            self.reaction_matrix_full * self.reaction_stability_mask_full
        )

        gibbs_funcs: list[Callable] = [
            to_hashable(species_.get_gibbs_over_RT) for species_ in self.species.reaction_species
        ]

        def apply_gibbs(index: Integer[Array, ""], temperature: FloatArray) -> FloatArray:
            return lax.switch(index, gibbs_funcs, temperature)

        self.vmap_gibbs = eqx.filter_vmap(apply_gibbs, in_axes=(0, None), out_axes=-1)

        self.output_to_logger()

        temperature_min, temperature_max = self.get_temperature_range()
        logger.info(
            "Thermodynamic data requires temperatures between %d K and %d K",
            np.ceil(temperature_min),
            np.floor(temperature_max),
        )

    @classmethod
    def available_species(cls) -> tuple[str, ...]:  # pragma: no cover
        return thermodynamic_data_source.available_species()

    @property
    def number_reactions(self) -> int:
        """Number of core reactions"""
        return max(
            0,
            self.species.reaction_species.number_species
            - len(self.species.reaction_species.unique_elements),
        )

    def get_log_Kp(self, temperature: FloatArray) -> Float[Array, "... n_reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Args:
            temperature: Temperature (K)

        Returns:
            Log of the equilibrium constant of each reaction
        """
        gibbs_values: Float[Array, "... n_species"] = self.vmap_gibbs(
            jnp.arange(self.species.reaction_species.number_species), temperature
        )
        # jax.debug.print("gibbs_values = {out}", out=gibbs_values)
        reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(self.reaction_matrix)
        # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)
        log_Kp: Float[Array, "... n_reactions"] = -jnp.einsum(
            "rs,...s->...r", reaction_matrix, gibbs_values
        )
        # jax.debug.print("log_Kp = {out}", out=log_Kp)

        return log_Kp

    def get_matrix(self) -> NpFloat:
        return self.reaction_matrix_full

    def get_stability_matrix(self) -> NpFloat:
        return self.reaction_stability_matrix_full

    def get_temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range of the thermodynamic data for the species

        Returns:
            Minimum and maximum temperature that is valid for the species
        """
        temperature_min: list[float] = [
            min(species.thermo.T_min) for species in self.species.reaction_species
        ]
        temperature_max: list[float] = [
            max(species.thermo.T_max) for species in self.species.reaction_species
        ]

        return max(temperature_min), min(temperature_max)


class DissolutionNetwork(BaseReactionBlock):
    """Handles all reactions where a species dissolves into or exchanges with a phase.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    reaction_indices_map: NpInt
    """Mapping of dissolution species to corresponding reaction species"""
    dissolution_matrix: NpFloat
    """Dissolution reaction matrix"""
    vmap_solubility: Callable
    """Vectorized solubility function for dissolution reactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)

        # Most direct to construct the dissolution matrix in full species space
        dissolution_matrix: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )

        # Gas indices in the full species collection corresponding to each dissolution species
        reaction_indices_map: list[int] = []

        for reaction_index, dissolution_species_ in enumerate(self.species.reservoir_species):
            # Reservoir mask
            reservoir_mask: NpBool = np.array(
                [s is dissolution_species_ for s in self.species],
                dtype=bool,
            )
            # Gas species index
            name: str = f"{dissolution_species_.data.hill_formula}_{GAS_STATE}"
            gas_idx: int = self.species.species_names.index(name)

            reaction_indices_map.append(gas_idx)

            dissolution_matrix[reaction_index, reservoir_mask] = 1.0
            dissolution_matrix[reaction_index, gas_idx] = -1.0

        self.reaction_indices_map = np.array(reaction_indices_map, dtype=int)
        self.dissolution_matrix = dissolution_matrix

        solubility_funcs: list[Callable] = [
            to_hashable(species_.solubility.jax_concentration)
            for species_ in self.species.reservoir_species
        ]

        def apply_solubility(
            index: Integer[Array, ""],
            fugacity_val: FloatArray,
            temp: FloatArray,
            press: FloatArray,
            o2_fug: FloatArray,
        ) -> FloatArray:
            return lax.switch(index, solubility_funcs, fugacity_val, temp, press, o2_fug)

        self.vmap_solubility: Callable = eqx.filter_vmap(
            apply_solubility, in_axes=(0, -1, None, None, None), out_axes=-1
        )

        self.output_to_logger()

    @property
    def number_reactions(self) -> int:
        """Number of dissolution reactions"""
        return self.species.reservoir_species.number_species

    def get_log_Kp(
        self,
        temperature: FloatArray,
        gas_species_activity: Float[Array, "... n_gas_species"],
        pressure: FloatArray,
        fO2: FloatArray,
        log_solvent_molar_mass: FloatArray,
    ) -> Float[Array, "... n_reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Solubility concentrations in ppmw are clamped to a minimum of
        :data:`~atmodeller.constants.DISSOLUTION_PPMW_FLOOR` before taking the logarithm to
        prevent ``log(0)`` in underflow-prone regimes.

        Args:
            temperature: Temperature (K)
            gas_species_activity: Gas species activity regulating dissolution reactions
            pressure: Pressure (bar)
            fO2: Oxygen fugacity (bar)
            log_solvent_molar_mass: Log of solvent molar mass (kg mol\\ :sup:`-1`)

        Returns:
            Log of the equilibrium constant of each reaction
        """
        # Return empty array if no dissolution species
        if self.number_reactions == 0:
            return jnp.zeros((*jnp.shape(temperature), 0), dtype=float)

        species_ppmw_concentration: Float[Array, "... n_species"] = self.vmap_solubility(
            jnp.arange(self.number_reactions), gas_species_activity, temperature, pressure, fO2
        )
        species_ppmw_concentration = jnp.maximum(
            species_ppmw_concentration, DISSOLUTION_PPMW_FLOOR
        )
        # jax.debug.print("solvent_molar_mass = {out}", out=jnp.exp(log_solvent_molar_mass))

        # Convert from ppmw to mole fraction of melt, which is the activity basis for dissolution
        # reactions.
        log_Kp: Float[Array, "... reactions"] = (
            jnp.log(species_ppmw_concentration)
            - jnp.log(1e6)
            + log_solvent_molar_mass
            - jnp.log(self.species.molar_masses[self.reaction_indices_map])
            - jnp.log(gas_species_activity)
        )
        # jax.debug.print("log_Kp = {out}", out=log_Kp)

        return log_Kp

    def get_matrix(self) -> NpFloat:
        return self.dissolution_matrix

    def get_stability_matrix(self) -> NpFloat:
        """Dissolution reactions do not directly affect stability, so return zero"""
        return np.zeros_like(self.dissolution_matrix, dtype=float)


class ReactionSystem(BaseReactionBlock):
    """Unified reaction system for core chemical reactions and dissolution reactions.

    Handles phase indexing, assembly of reaction matrices, stability, and evaluation of equilibrium
    constants and residuals in a JAX-compatible way.

    Args:
        phase_system: Phase system containing all phases
    """

    phase_system: PhaseSystem
    """Phase system"""
    formula_matrix: NpInt
    """Formula matrix of the full species collection"""
    reaction: ReactionNetwork
    """Core chemical reaction network"""
    dissolution: DissolutionNetwork
    """Dissolution reaction network"""
    matrix: NpFloat
    """Reaction matrix of the full reaction system"""
    stability_matrix: NpFloat
    """Stability matrix of the full reaction system"""
    _O2_index: NpInt
    _has_O2: NpBool
    _log_stoich_matrix: NpFloat

    def __init__(self, phase_system: PhaseSystem):
        self.phase_system = phase_system
        self.formula_matrix = get_formula_matrix(self.species)

        with np.errstate(divide="ignore"):
            self._log_stoich_matrix = np.where(
                self.formula_matrix > 0, np.log(self.formula_matrix), -np.inf
            )
        self.reaction = ReactionNetwork(self.species)
        self.dissolution = DissolutionNetwork(self.species)
        self.matrix = np.vstack([block.get_matrix() for block in self.blocks])
        self.stability_matrix = np.vstack([block.get_stability_matrix() for block in self.blocks])

        # Could be an integer (but represented as a float) or np.nan
        self._O2_index = np.nan_to_num(self.phase_system.gas.O2_index, nan=0).astype(int)
        self._has_O2 = ~np.isnan(self.phase_system.gas.O2_index)

        self.output_to_logger()

    @property
    def species(self) -> SpeciesCollection[SpeciesProtocol]:  # pyright: ignore - Equinox override
        """Species collection of the full reaction system"""
        return self.phase_system.species

    @property
    def blocks(self) -> tuple[BaseReactionBlock, ...]:
        """Reaction blocks"""
        return self.reaction, self.dissolution

    @property
    def number_reactions(self):
        """Number of reactions"""
        return sum(block.number_reactions for block in self.blocks)

    def get_log_activity(
        self,
        log_number_moles: Float[Array, "... n_species"],
        temperature: FloatArray,
        pressure: FloatArray,
    ) -> FloatArray:
        """Gets log activity of each species.

        Args:
            log_number_moles: Log number of moles of each species
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log activity of each species
        """
        log_activities = []

        for phase_index, phase in enumerate(self.phase_system.phases):
            # jax.debug.print("phase.name = {out}", out=phase.name)
            phase_slice: slice = self.phase_system.phase_slice(phase_index)
            log_activity: Float[Array, "... n_phase_species"] = phase.get_log_activity(
                log_number_moles[..., phase_slice], temperature, pressure
            )
            log_activities.append(log_activity)

        log_activity: Float[Array, "... n_species"] = jnp.concatenate(log_activities, axis=-1)
        # jax.debug.print("log_activity = {out}", out=log_activity)

        return log_activity

    def get_log_Kp(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_activity: Float[Array, "... n_species"],
        temperature: FloatArray,
        pressure: FloatArray,
    ) -> Float[Array, "... n_reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Assembles the log Kps from the reaction and dissolution networks, which may require
        different inputs.

        Args:
            log_number_moles: Log number of moles of each species
            log_activity: Log activity of each species
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log of the equilibrium constant of each reaction
        """
        log_Kp_reaction: Float[Array, "... n_core_reactions"] = self.reaction.get_log_Kp(
            temperature
        )
        # jax.debug.print("log_Kp_reaction = {out}", out=log_Kp_reaction)

        # Log activity of chemical species regulating dissolution
        log_activity_dissolution: Float[Array, "... n_dissolution_species"] = jnp.take(
            log_activity,
            indices=self.dissolution.reaction_indices_map,
            unique_indices=True,
            indices_are_sorted=True,
            axis=-1,
        )
        # jax.debug.print("log_activity_dissolution = {out}", out=log_activity_dissolution)

        activity_dissolution: Float[Array, "... n_dissolution_species"] = jnp.exp(
            log_activity_dissolution
        )

        # Get fO2 or nan if not present
        fO2: FloatArray = jnp.where(
            self._has_O2, jnp.take(jnp.exp(log_activity), self._O2_index, axis=-1), jnp.nan
        )
        # jax.debug.print("fO2 = {out}", out=fO2)

        # TODO: This could probably be refactored to obtain the index directly from
        # self.phase_system(?)
        melt_slice: slice = self.phase_system.phase_slice(SILICATE_MELT_PHASE_INDEX)
        melt_phase: BasePhase = self.phase_system.phases[SILICATE_MELT_PHASE_INDEX]

        log_solvent_molar_mass: FloatArray = melt_phase.get_log_phase_molar_mass(
            log_number_moles[..., melt_slice]
        )
        # jax.debug.print("log_solvent_molar_mass = {out}", out=log_solvent_molar_mass)

        log_Kp_dissolution: Float[Array, "... n_dissolution_reactions"] = (
            self.dissolution.get_log_Kp(
                temperature, activity_dissolution, pressure, fO2, log_solvent_molar_mass
            )
        )
        # jax.debug.print("log_Kp_dissolution = {out}", out=log_Kp_dissolution)

        # Broadcast log_Kp_reaction to match batch shape of log_Kp_dissolution.
        # e.g. scalar T: log_Kp_reaction (3,), log_Kp_dissolution (2, 1) -> broadcast to (2, 3)
        # e.g. batched T: log_Kp_reaction (2, 3), log_Kp_dissolution (2, 1) -> already compatible
        batch_shape: tuple[int, ...] = log_Kp_dissolution.shape[:-1]
        log_Kp_reaction = jnp.broadcast_to(
            log_Kp_reaction, batch_shape + (log_Kp_reaction.shape[-1],)
        )

        return jnp.concatenate([log_Kp_reaction, log_Kp_dissolution], axis=-1)

    def get_matrix(self) -> NpFloat:
        return self.matrix

    def get_stability_matrix(self) -> ArrayLike:
        return self.stability_matrix

    def get_log_element_moles(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... num_elements"]:
        """Gets log number of moles of each element.

        Args:
            log_number_moles: Log number of moles of each species

        Returns:
            Log number of moles of each element
        """
        log_terms: Float[Array, "... element species"] = (
            log_number_moles[..., None, :] + self._log_stoich_matrix
        )

        return masked_logsumexp(log_terms, axis=-1, keepdims=False)

    def apply_stability(
        self,
        residual: Float[Array, "... n_reactions"],
        log_stability: Float[Array, "... n_species"],
    ) -> Float[Array, "... n_reactions"]:
        """Subtracts the stability contribution from residual.

        Args:
            residual: Residual of the reaction network before applying stability
            log_stability: Log stability of each species

        Returns:
            Residual of the reaction network after applying stability
        """
        return residual - jnp.einsum(
            "rs,...s->...r", self.get_stability_matrix(), safe_exp(log_stability)
        )

    def get_residual(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_activity: Float[Array, "... n_species"],
        log_stability: Float[Array, "... n_species"],
        temperature: FloatArray,
        pressure: FloatArray,
    ) -> Float[Array, "... n_reactions"]:
        """Gets the residual of the reaction network.

        Args:
            log_number_moles: Log number of moles of each species
            log_activity: Log activity of each species
            log_stability: Log stability of each species
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Residual of the reaction network
        """
        log_Kp: Float[Array, "... n_reactions"] = self.get_log_Kp(
            log_number_moles, log_activity, temperature, pressure
        )
        # jax.debug.print("log_Kp = {out}", out=log_Kp)
        residual: Float[Array, "... n_reactions"] = (
            jnp.einsum("rs,...s->...r", self.matrix, log_activity) - log_Kp
        )
        # jax.debug.print("reaction residual before stability = {out}", out=residual)

        residual = self.apply_stability(residual, log_stability)
        # jax.debug.print("reaction residual after stability = {out}", out=residual)

        return residual
