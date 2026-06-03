# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase container classes for thermodynamic equilibrium calculations.

This module defines the core phase abstractions and four concrete phase types used in the
equilibrium solver.

The available phase types are:

- :class:`GasPhase`: Multicomponent gas mixture, supporting both ideal and non-ideal (real gas)
  behavior. Tracks the O2 index for redox calculations and provides gas-specific outputs (e.g.,
  partial pressures, volume).
- :class:`MeltPhase`: Multicomponent silicate melt, optionally with dissolved volatiles and/or
  condensed species treated as additional to the solvent. Phase-level properties (mass, moles,
  molar mass, and derived fractions) can be computed relative to just the solvent mass, while
  per-species properties such as activity are always computed from raw species amounts.
- :class:`SolidPhase`: Multicomponent silicate solid, with similar background handling as the
  melt phase.
- :class:`MetalPhase`: Multicomponent metal, with similar background handling as the melt phase.
- :class:`PurePhase`: Single-species, unity-activity phase (e.g., a pure mineral, ice, or
  liquid). Only one species is permitted, and its activity is fixed at unity.

All phases are JAX-compatible :class:`equinox.Module` subclasses and wrap a
:class:`~atmodeller.containers.SpeciesCollection` of thermodynamic species (Hill formula +
aggregation state). Species are constructed from their Hill formulas and assigned an aggregation
state consistent with the JANAF/NASA convention:

- ``"g"`` : gas
- ``"l"`` : liquid
- ``"s"`` : solid

Quantities are accumulated in log-space throughout for numerical stability. Many methods accept an
optional background component (e.g., the silicate melt mass) that contributes to phase totals but
is not tracked as an explicit species in the solver.

All methods that accept number of moles (e.g., ``log_number_moles``) and other batchable inputs are
designed to accommodate batched arrays, with shapes broadcast-compatible with the batch dimension.
This enables consistent handling of both single and batched calculations, as required by the
:class:`PhaseOutput` class.

The module also provides a general :class:`PhaseOutput` class for output-friendly, broadcasted
access to phase and species properties, as well as a :class:`GasPhaseOutput` for gas-specific
outputs.
"""

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import Formula

from atmodeller import override
from atmodeller.containers import ChemicalSpecies, SpeciesCollection, get_formula_matrix
from atmodeller.interfaces import RedoxBufferProtocol, SpeciesProtocol, TSpecies_co
from atmodeller.jax_utils import FloatArray, as_j64, masked_logsumexp, safe_exp, to_hashable
from atmodeller.sci_utils import GAS_CONSTANT_BAR
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer

# Due to a Pyright bug (#4965). See Equinox documentation.
if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar

DEFAULT_BACKGROUND_MASS: float = 0.0
"""Default background mass in kg (i.e., no background mass)"""
DUMMY_MOLAR_MASS: float = 1.0
"""Default molar mass of the background component (kg mol\\ :sup:`-1`) (only meaningful when
background mass is non-zero)"""

logger: logging.Logger = logging.getLogger(__name__)


class BasePhase(eqx.Module, Generic[TSpecies_co]):
    """Base class for all phases

    This class defines the physical and model state of a phase, including its species, background
    properties, and core logic for thermodynamic calculations. It is intentionally kept separate
    from :class:`PhaseOutput`, which handles output formatting and broadcasting for batch results.

    Note:
        For all methods, temperature and pressure should be provided as scalars or 1-D arrays
        matching the batch dimension. All inputs will be broadcast as needed, but providing
        compatible shapes avoids shape mismatch errors and ensures correct broadcasting behavior.

        All methods that accept number of moles (e.g., ``log_number_moles``) are designed to
        accommodate batched inputs, i.e., arrays with a leading batch dimension. This is required
        since these methods are called by the :class:`PhaseOutput` class, which handles both single
        and batched calculations.

    Args:
        name: Phase name
        species: An iterable of species in the phase
        background_mass: Mass of the background component (kg). Should be a scalar or a 1-D array
            matching the batch dimension if batching is used. Defaults to
            :data:`DEFAULT_BACKGROUND_MASS`.
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`). Should
            be a scalar or a 1-D array matching the batch dimension if batching is used. Defaults
            to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not zero.
    """

    name: str
    """Phase name"""
    species: SpeciesCollection[TSpecies_co]
    """Collection of species in the phase"""
    background_mass: FloatArray
    """Mass of the background component"""
    background_molar_mass: FloatArray
    """Molar mass of the background component"""
    vmap_log_activity: Callable
    """Vectorized log activity functions for each species in the phase"""
    output_class: AbstractClassVar[type["PhaseOutput"]]
    """Output class for the phase"""

    def __init__(
        self,
        name: str,
        species: Iterable[TSpecies_co] = (),
        background_mass: ArrayLike = DEFAULT_BACKGROUND_MASS,
        background_molar_mass: ArrayLike = DUMMY_MOLAR_MASS,
    ):
        self.name = name
        self.species = SpeciesCollection(species)
        self.background_mass = as_j64(background_mass)
        self.background_molar_mass = as_j64(background_molar_mass)

        log_activity_funcs: list[Callable] = [
            to_hashable(species_.activity.log_activity) for species_ in self.species
        ]

        def apply_log_activity(
            index: Integer[Array, ""],
            temperature: FloatArray,
            pressure: FloatArray,
            mole_fractions: FloatArray,
        ) -> FloatArray:
            return lax.switch(index, log_activity_funcs, temperature, pressure, mole_fractions)

        self.vmap_log_activity = eqx.filter_vmap(
            apply_log_activity, in_axes=(0, None, None, None), out_axes=-1
        )

        logger.info(
            "Creating %s: %s",
            self.__class__.__name__,
            tuple(str(species) for species in self.species),
        )

    @classmethod
    def empty(cls) -> "BasePhase":
        """Creates an empty phase instance with no species and zero background mass.

        Returns:
            An empty phase instance
        """
        return cls("empty")

    @property
    def is_empty(self) -> bool:
        """Indicates whether the phase contains no species"""
        return self.species.number_species == 0

    @property
    def log_background_mass(self) -> FloatArray:
        """Log mass of the background component (kg)"""
        return jnp.log(self.background_mass)

    @property
    def log_background_moles(self) -> FloatArray:
        """Log moles of the background component (mol)"""
        return self.log_background_mass - self.log_background_molar_mass

    @property
    def log_background_molar_mass(self) -> FloatArray:
        r"""Log molar mass of the background component (kg mol\ :sup:`-1`)"""
        return jnp.log(self.background_molar_mass)

    @property
    def species_names(self) -> tuple[str, ...]:
        """List of species names in the phase"""
        return self.species.species_names

    def get_log_activity(
        self,
        log_number_moles: Float[Array, "... n_species"],
        temperature: FloatArray,
        pressure: FloatArray,
    ) -> Float[Array, "... n_species"]:
        """Gets the log activity of each species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log activity of each species in the phase
        """
        if self.is_empty:
            return jnp.zeros_like(log_number_moles)

        log_mole_fractions: Float[Array, "... n_species"] = self.get_log_mole_fraction(
            log_number_moles
        )
        # jax.debug.print("before log_activity: log_mole_fractions = {out}", out=log_mole_fractions)

        # Log activity coefficient of pure species
        log_activity: Float[Array, "... n_species"] = self.vmap_log_activity(
            jnp.arange(self.species.number_species),
            temperature,
            pressure,
            jnp.exp(log_mole_fractions),
        )
        log_activity: Float[Array, "... n_species"] = log_activity + log_mole_fractions

        return log_activity

    def get_log_activity_with_stability(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_stability: Float[Array, "... n_species"],
        temperature: FloatArray,
        pressure: FloatArray,
    ) -> Float[Array, "... n_species"]:
        """Gets the log activity of each species in the phase, accounting for stability.

        Unstable species are assigned a log activity of negative infinity.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_stability: Log stability of each species in the phase
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log activity of each species in the phase, with unstable species set to negative
                infinity
        """
        log_activity: Float[Array, "... n_species"] = self.get_log_activity(
            log_number_moles, temperature, pressure
        )
        log_stability_masked: Float[Array, "... n_species"] = jnp.where(
            self.species.active_stability, log_stability, -jnp.inf
        )
        log_activity_with_stability: Float[Array, "... n_species"] = log_activity - safe_exp(
            log_stability_masked
        )

        return log_activity_with_stability

    def apply_phase_mass_mask(
        self, log_array: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Zeros out (in log-space) species that do not contribute to phase-level aggregations.

        Species with ``include_in_phase_mass=False`` are replaced with ``-inf``; all others are
        passed through unchanged.

        Args:
            log_array: Log-space values whose entries for non-contributing species are to be masked

        Returns:
            The input array with non-contributing species set to ``-inf``.
        """
        log_mask: Float[Array, " n_species"] = jnp.where(
            self.species.phase_mass_mask, 0.0, -jnp.inf
        )

        return log_array + log_mask

    def get_log_mass(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Gets the log mass of each species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mass of each species in the phase (kg)
        """
        log_mass: Float[Array, "... n_species"] = log_number_moles + jnp.log(
            self.species.molar_masses
        )

        return log_mass

    def get_log_phase_mass(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... 1"]:
        """Gets the log mass of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mass of the phase (kg)
        """
        log_mass: Float[Array, "... n_species"] = self.get_log_mass(log_number_moles)
        log_mass = self.apply_phase_mass_mask(log_mass)

        # Broadcast the background scalar to the batch shape and concatenate along the species axis
        # so the result is (... n_species+1).
        log_background_mass: Float[Array, "... 1"] = jnp.broadcast_to(
            self.log_background_mass, log_mass.shape[:-1]
        )[..., None]
        log_mass_with_background: Float[Array, "... n_species_plus_one"] = jnp.concatenate(
            [log_mass, log_background_mass], axis=-1
        )

        return masked_logsumexp(log_mass_with_background, axis=-1, keepdims=True)

    def get_log_mass_fraction(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Gets the log mass fraction of the species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mass fraction of each species in the phase (kg kg\\ :sup:`-1`)
        """
        log_mass: Float[Array, "... n_species"] = self.get_log_mass(log_number_moles)
        log_phase_mass: Float[Array, "... 1"] = self.get_log_phase_mass(log_number_moles)
        log_mass_fraction: Float[Array, "... n_species"] = log_mass - log_phase_mass
        # jax.debug.print("log_mass_fraction = {out}", out=log_mass_fraction)

        return log_mass_fraction

    def get_log_phase_moles(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... 1"]:
        """Gets the log moles of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log moles of the phase (mol)
        """
        log_number_moles = self.apply_phase_mass_mask(log_number_moles)

        # Broadcast the background scalar to the batch shape and concatenate along the species axis
        # so the result is (... n_species+1).
        log_background_moles: Float[Array, "... 1"] = jnp.broadcast_to(
            self.log_background_moles, log_number_moles.shape[:-1]
        )[..., None]
        log_moles_with_background: Float[Array, "... n_species_plus_one"] = jnp.concatenate(
            [log_number_moles, log_background_moles], axis=-1
        )

        return masked_logsumexp(log_moles_with_background, axis=-1, keepdims=True)

    def get_log_mole_fraction(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Gets the log mole fraction of the species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mole fraction of each species in the phase (mol mol\\ :sup:`-1`)
        """
        log_phase_moles: Float[Array, "... 1"] = self.get_log_phase_moles(log_number_moles)
        log_mole_fraction: Float[Array, "... n_species"] = log_number_moles - log_phase_moles

        return log_mole_fraction

    def get_log_phase_molar_mass(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... 1"]:
        r"""Gets the log molar mass of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log molar mass of the phase (kg mol\ :sup:`-1`)
        """
        log_phase_mass: Float[Array, "... 1"] = self.get_log_phase_mass(log_number_moles)
        log_number_total: Float[Array, "... 1"] = self.get_log_phase_moles(log_number_moles)

        return log_phase_mass - log_number_total

    def output(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_stability: Float[Array, "... n_species"],
        temperature: FloatArray,
        pressure: FloatArray,
    ) -> "PhaseOutput[Self]":
        r"""Constructs an output helper object for phase-level and species-level properties.

        Args:
            log_number_moles: Log number of moles of each species in the phase (mol)
            log_stability: Log stability of each species in the phase
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            An output helper object for accessing and manipulating output quantities
        """
        output: PhaseOutput[Self] = self.output_class(
            self,
            log_number_moles,
            log_stability,
            temperature,
            pressure,
            self.background_mass,
            self.background_molar_mass,
        )

        return output

    def __len__(self) -> int:
        return len(self.species)


TPhase_co = TypeVar("TPhase_co", bound=BasePhase, covariant=True)


class PhaseOutput(eqx.Module, Generic[TPhase_co]):
    r"""Output helper class for phase-level and species-level results.

    This class provides a broadcasted, output-friendly view of the results from a phase instance.
    It is intentionally kept separate from :class:`BasePhase` to clearly distinguish between the
    phase's physical/model state and the output representation, especially regarding broadcasting
    and batch handling.

    All methods and properties return arrays that honour the input batching: the leading dimension
    always corresponds to the batch (number of solutions or calculations), so outputs are shaped
    (#n_batch, ...), or are otherwise broadcast-compatible with the batch dimension. Arrays are not
    strictly expanded to the batch dimension, but are shaped to be broadcast compatible as needed.
    This ensures consistent handling of both single and batched calculations.

    Note:
        Attributes (e.g., _log_number_moles, _temperature, etc.) that are private have internal
        shapes and must be broadcast or processed before being suitable for output. Always use the
        provided properties to access output quantities with correct shapes and broadcasting.

    Args:
        phase: The phase instance associated with this output
        log_number_moles: Log number of moles for each species in the phase
        log_stability: Log stability for each species in the phase
        temperature: Temperature (K)
        pressure: Pressure (bar)
        background_mass: Mass of the background component (kg)
        background_molar_mass: Molar mass of the background component (kg mol\ :sup:`-1`)
    """

    phase: TPhase_co
    _log_number_moles: Float[Array, "... n_species"]
    _log_stability: Float[Array, "... n_species"]
    _temperature: FloatArray
    _pressure: FloatArray
    _background_mass: FloatArray
    _background_molar_mass: FloatArray

    @property
    def batch_size(self) -> int:
        return self.log_number_moles.shape[0]

    def _reshape_to_batched_species(
        self, x: Float[Array, "... n_species"]
    ) -> Float[Array, "#n_batch n_species"]:
        """Normalizes phase arrays to shape ``(n_batch, n_species)`` with a generic reshape."""
        return jnp.reshape(x, (-1, self.phase.species.number_species))

    @property
    def log_number_moles(self) -> Float[Array, "#n_batch n_species"]:
        return self._reshape_to_batched_species(self._log_number_moles)

    @property
    def log_stability(self) -> Float[Array, "#n_batch n_species"]:
        return self._reshape_to_batched_species(self._log_stability)

    @property
    def temperature(self) -> Float[Array, "... 1"]:
        return jnp.atleast_1d(self._temperature)[..., None]

    @property
    def pressure(self) -> Float[Array, "... 1"]:
        return jnp.atleast_1d(self._pressure)[..., None]

    @property
    def background_mass(self) -> Float[Array, "... 1"]:
        return jnp.atleast_1d(self._background_mass)[..., None]

    @property
    def log_background_mass(self) -> Float[Array, "... 1"]:
        return jnp.log(self.background_mass)

    @property
    def background_molar_mass(self) -> Float[Array, "... 1"]:
        return jnp.atleast_1d(self._background_molar_mass)[..., None]

    @property
    def log_background_molar_mass(self) -> Float[Array, "... 1"]:
        return jnp.log(self.background_molar_mass)

    @property
    def log_background_number_moles(self) -> Float[Array, "... 1"]:
        return self.log_background_mass - self.log_background_molar_mass

    @property
    def background_number_moles(self) -> Float[Array, "... 1"]:
        return jnp.exp(self.log_background_number_moles)

    @property
    def is_empty(self) -> bool:
        """Indicates whether the phase contains no species."""
        return self.phase.is_empty

    @property
    def include_in_phase_mass(self) -> Bool[Array, "1 n_species"]:
        """Boolean mask indicating which species to include in phase-level mass and derived
        aggregations."""
        return jnp.atleast_2d(self.phase.species.phase_mass_mask)

    @property
    def formula_matrix(self) -> Integer[Array, "n_elements n_species"]:
        return jnp.asarray(get_formula_matrix(self.phase.species))

    @property
    def log_stoich_matrix(self) -> Float[Array, "n_element n_species"]:
        formula_matrix: Integer[Array, "n_elements n_species"] = self.formula_matrix
        return jnp.where(formula_matrix > 0, jnp.log(formula_matrix), -jnp.inf)

    @property
    def log_element_number_moles(self) -> Float[Array, "#n_batch n_elements"]:
        log_terms: Array = self.log_number_moles[..., None, :] + self.log_stoich_matrix
        result: Float[Array, "#n_batch n_elements"] = masked_logsumexp(
            log_terms, axis=-1, keepdims=False
        )

        return result

    @property
    def element_number_moles(self) -> Float[Array, "#n_batch n_elements"]:
        return jnp.exp(self.log_element_number_moles)

    @property
    def element_mass(self) -> Float[Array, "#n_batch n_elements"]:
        return jnp.exp(
            self.log_element_number_moles + jnp.log(self.phase.species.element_molar_masses)
        )

    # Phase outputs
    @property
    def phase_species_number_moles(self) -> Float[Array, "#n_batch 1"]:
        """Total phase moles from all constituent species and inert background"""
        return jnp.exp(self.phase.get_log_phase_moles(self.log_number_moles))

    @property
    def phase_element_number_moles(self) -> Float[Array, "#n_batch 1"]:
        """TODO: This ignores the background component, because without specifying its Hill
        formula we cannot determine its elemental composition. This is a limitation of the current
        design, and we may want to revisit this in the future. Currently, however, this property is
        only used for a metallicity calculation for the gas phase, where the background is almost
        always zero since all species are tracked."""
        total_element_moles: Float[Array, "#n_batch 1"] = jnp.sum(
            self.element_number_moles, axis=-1, keepdims=True
        )
        return total_element_moles

    @property
    def log_phase_mass(self) -> Float[Array, "#n_batch 1"]:
        return self.phase.get_log_phase_mass(self.log_number_moles)

    @property
    def phase_mass(self) -> Float[Array, "#n_batch 1"]:
        return jnp.exp(self.log_phase_mass)

    @property
    def phase_molar_mass(self) -> Float[Array, "#n_batch 1"]:
        return jnp.exp(self.phase.get_log_phase_molar_mass(self.log_number_moles))

    @property
    def species_to_phase_mass_ratio(self) -> Float[Array, "#n_batch 1"]:
        """Mass of tracked species divided by total phase mass.

        Sum of all species mass (no background, no mask), for comparison with the phase total.
        For a true dilute system, this provides a check that the dissolved content is in the dilute
        approximation. Otherwise, depending on the assumptions and objectives of the user, it
        provides a metric of how much the tracked species contribute to the total phase mass.
        """
        log_species_mass_sum: Float[Array, "#n_batch 1"] = masked_logsumexp(
            self.phase.get_log_mass(self.log_number_moles), axis=-1, keepdims=True
        )
        log_phase_mass: Float[Array, "#n_batch 1"] = self.phase.get_log_phase_mass(
            self.log_number_moles
        )
        log_species_to_phase_mass_ratio: Float[Array, "#n_batch 1"] = (
            log_species_mass_sum - log_phase_mass
        )
        return jnp.exp(log_species_to_phase_mass_ratio)

    # Species outputs
    @property
    def species_activity(self) -> Float[Array, "#n_batch n_species"]:
        log_activity: Float[Array, "#n_batch n_species"] = (
            self.phase.get_log_activity_with_stability(
                self.log_number_moles, self.log_stability, self._temperature, self._pressure
            )
        )
        return jnp.exp(log_activity)

    @property
    def species_mass_fraction(self) -> Float[Array, "#n_batch n_species"]:
        log_mass_fraction: Float[Array, "#n_batch n_species"] = self.phase.get_log_mass_fraction(
            self.log_number_moles
        )
        return jnp.exp(log_mass_fraction)

    @property
    def species_mass(self) -> Float[Array, "#n_batch n_species"]:
        log_mass: Float[Array, "#n_batch n_species"] = self.phase.get_log_mass(
            self.log_number_moles
        )
        return jnp.exp(log_mass)

    @property
    def species_mole_fraction(self) -> Float[Array, "#n_batch n_species"]:
        log_mole_fraction: Float[Array, "#n_batch n_species"] = self.phase.get_log_mole_fraction(
            self.log_number_moles
        )
        return jnp.exp(log_mole_fraction)

    @property
    def species_number_moles(self) -> Float[Array, "#n_batch n_species"]:
        return jnp.exp(self.log_number_moles)


class GasPhaseOutput(PhaseOutput["GasPhase"]):
    """Output helper class for GasPhase-specific properties."""

    @property
    def log10dIW_1_bar(self) -> Float[Array, "#n_batch 1"]:
        """Log10 of the oxygen fugacity relative to the IW buffer at 1 bar."""
        O2_index: ArrayLike = self.phase.O2_index

        def no_oxygen() -> Float[Array, "#n_batch 1"]:
            return jnp.full((self.batch_size, 1), jnp.nan)

        def with_oxygen() -> Float[Array, "#n_batch 1"]:
            log10_fugacity = jnp.log10(self.species_activity[..., O2_index.astype(int)])
            buffer: RedoxBufferProtocol = IronWustiteBuffer()
            buffer_at_one_bar = buffer.log10_fugacity(self._temperature, 1.0)
            log10_shift_at_one_bar = log10_fugacity - buffer_at_one_bar
            return jnp.expand_dims(log10_shift_at_one_bar, axis=-1)

        return lax.cond(jnp.isnan(O2_index), no_oxygen, with_oxygen)

    @property
    def log10dIW_P(self) -> Float[Array, "#n_batch 1"]:
        """Log10 of the oxygen fugacity relative to the IW buffer at the pressure of interest."""
        O2_index: ArrayLike = self.phase.O2_index

        def no_oxygen() -> Float[Array, "#n_batch 1"]:
            return jnp.full((self.batch_size, 1), jnp.nan)

        def with_oxygen() -> Float[Array, "#n_batch 1"]:
            log10_fugacity = jnp.log10(self.species_activity[..., O2_index.astype(int)])
            buffer: RedoxBufferProtocol = IronWustiteBuffer(evaluation_pressure=None)
            buffer_at_P = buffer.log10_fugacity(self._temperature, self._pressure)
            log10_shift_at_P = log10_fugacity - buffer_at_P
            return jnp.expand_dims(log10_shift_at_P, axis=-1)

        return lax.cond(jnp.isnan(O2_index), no_oxygen, with_oxygen)

    @property
    def species_partial_pressure(self) -> Float[Array, "#n_batch n_species"]:
        """Partial pressure of each species (bar)"""
        return self.pressure * self.species_mole_fraction

    @property
    def volume(self) -> Float[Array, "#n_batch 1"]:
        r"""Volume of the gas phase (m\ :sup:`3`)"""
        return (
            self.phase_species_number_moles * GAS_CONSTANT_BAR * self.temperature / self.pressure
        )


class GasPhase(BasePhase[ChemicalSpecies]):
    """Multicomponent gas mixture

    Models gas species as an ideal mixture of (potentially) non-ideal pure gases, where each pure
    species contributes an activity based on its own (potentially real gas) equation of state.

    Args:
        species: An iterable of species in the phase
        name: Name of the phase. Defaults to ``gas``.
        background_mass: Mass of the background component (kg). Should be a scalar or a 1-D array
            matching the batch dimension if batching is used. Defaults to
            :data:`DEFAULT_BACKGROUND_MASS`.
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`). Should
            be a scalar or a 1-D array matching the batch dimension if batching is used. Defaults
            to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not zero.
    """

    output_class: ClassVar[type[PhaseOutput]] = GasPhaseOutput
    """Output class for the phase"""

    # Without an override pylance gets confused, throwing missing argument warnings even though
    # defaults are provided in the base class. This avoids reporting this false-alarm to the user.
    @override
    def __init__(
        self,
        species: Iterable[ChemicalSpecies] = (),
        name: str = "gas",
        background_mass: ArrayLike = DEFAULT_BACKGROUND_MASS,
        background_molar_mass: ArrayLike = DUMMY_MOLAR_MASS,
    ):
        super().__init__(name, species, background_mass, background_molar_mass)

    @property
    def O2_index(self) -> Float[Array, ""]:
        """Gets the species index corresponding to diatomic oxygen.

        Note:
            This returns a float array for type consistency.

        Returns:
            Index of diatomic oxygen, or ``np.nan`` if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.species):
            if species_.data.hill_formula == "O2":
                # logger.debug("Found O2 at index = %d", nn)
                return jnp.array(nn, dtype=float)

        return jnp.array(jnp.nan, dtype=float)


class CondensedPhase(BasePhase[SpeciesProtocol]):
    """Multicomponent condensed phase (e.g., silicate melt or solid)

    A condensed phase can optionally treat dissolved and/or condensed species as additional to the
    solvent (the bulk melt mass that species dissolve into). When enabled, these species are
    included on top of the solvent when computing phase-level totals (mass, moles, molar mass,
    and derived fractions). This is useful in the dilute limit, where their contribution to the
    total phase mass is negligible, or when the background component already accounts for them.

    Note that this only affects phase-level aggregations — per-species thermodynamic properties
    (e.g., activity) are always computed from the raw species amounts and are unaffected. Total
    mass and mole conservation is therefore always maintained by the solver.

    Args:
        species: An iterable of species in the phase
        name: Name of the phase. Defaults to ``condensed``.
        background_mass: Mass of the background component (kg). Should be a scalar or a 1-D array
            matching the batch dimension if batching is used. Defaults to
            :data:`DEFAULT_BACKGROUND_MASS`.
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`). Should
            be a scalar or a 1-D array matching the batch dimension if batching is used. Defaults
            to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not zero.
    """

    output_class: ClassVar[type["PhaseOutput"]] = PhaseOutput[Self]
    """Output class for the phase"""

    # Without an override pylance gets confused, throwing missing argument warnings even though
    # defaults are provided in the base class. This avoids reporting this false-alarm to the user.
    @override
    def __init__(
        self,
        species: Iterable[TSpecies_co] = (),
        name: str = "condensed_phase",
        background_mass: ArrayLike = DEFAULT_BACKGROUND_MASS,
        background_molar_mass: ArrayLike = DUMMY_MOLAR_MASS,
    ):
        super().__init__(name, species, background_mass, background_molar_mass)


class PurePhase(CondensedPhase):
    """Single-species, unity-activity phase (e.g., a pure mineral, ice, or liquid)

    Args:
        species: An iterable of species in the phase
        name: Name of the phase. Defaults to ``pure_phase``.
        background_mass: Mass of the background component (kg). Should be a scalar or a 1-D array
            matching the batch dimension if batching is used. Defaults to
            :data:`DEFAULT_BACKGROUND_MASS`.
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`). Should
            be a scalar or a 1-D array matching the batch dimension if batching is used. Defaults
            to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not zero.
    """

    # Without an override pylance gets confused, throwing missing argument warnings even though
    # defaults are provided in the base class. This avoids reporting this false-alarm to the user.
    @override
    def __init__(
        self,
        species: Iterable[ChemicalSpecies] = (),
        name: str = "pure_phase",
        background_mass: ArrayLike = DEFAULT_BACKGROUND_MASS,
        background_molar_mass: ArrayLike = DUMMY_MOLAR_MASS,
    ):
        super().__init__(species, name, background_mass, background_molar_mass)

    def __check_init__(self):
        if self.species.number_species != 1:
            raise ValueError("A pure phase must contain exactly one species.")

    @classmethod
    def from_species(cls, species: str | Iterable[str] = (), **kwargs) -> Self:
        r"""Creates a phase instance from species names using the factory class.

        Args:
            species: A single species name or an iterable of names
            **kwargs: Arbitrary keyword arguments to pass to the factory class when constructing
                species instances.

        Returns:
            An instance of the phase with the specified species and background properties
        """
        if isinstance(species, str):
            species = [species]

        species_list: list[ChemicalSpecies] = []

        for species_ in species:
            hill_formula: str = Formula(species_).formula
            species_list.append(ChemicalSpecies.create_condensed(hill_formula, **kwargs))

        # Pure phase name should always correspond to the single species it contains
        phase_name: str = species_list[0].name

        return cls(species_list, phase_name)
