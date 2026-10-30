# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Phase container classes for thermodynamic equilibrium calculations.

This module defines the core phase abstractions and three concrete phase types used in the
equilibrium solver.

The available phase types are:

- :class:`GasPhase`: Multicomponent gas mixture, supporting both ideal and non-ideal (real gas)
  behaviour. Tracks the O2 index for redox calculations and provides gas-specific outputs (e.g.,
  partial pressures, volume).
- :class:`CondensedPhase`: Multicomponent condensed phase, used for the silicate melt, the
  silicate solid and the metal, optionally with dissolved volatiles and/or condensed species
  treated as additional to the solvent. Phase-level properties (mass, moles, molar mass, and
  derived fractions) can be computed relative to just the solvent mass, while per-species
  properties such as activity are always computed from raw species amounts.
- :class:`PurePhase`: Single-species, unity-activity phase (e.g., a pure mineral, ice, or
  liquid). Only one species is permitted, and its activity is fixed at unity.

All phases are JAX-compatible :class:`equinox.Module` subclasses and wrap a
:class:`~atmodeller.containers.SpeciesCollection` of thermodynamic species, each named from its Hill
formula and a suffix (see :class:`~atmodeller.interfaces.ChemicalSpeciesData`):

- ``"g"`` : gas
- ``"l"`` : liquid
- the end-member of a mineral, e.g. ``"quartz-alpha"``
- ``"cd"`` : the condensed end-members of a formula stitched together
- ``"s"`` : the solid end-members of a formula stitched together, without the liquid

Quantities are accumulated in log-space throughout for numerical stability. Many methods accept an
optional background component (e.g., the silicate melt mass) that contributes to phase totals but
is not tracked as an explicit species in the solver. This background mass is treated as chemically
inert bulk material: it has no thermodynamic activity of its own, but it still contributes to
physical, mass-dependent quantities, most notably (1) sizing the solvent reservoir that dissolved
species partition into (e.g., a condensed phase's dilution/concentration calculations scale with
the solvent mass), and (2) the planet's total mass and, in turn, derived quantities like surface
gravity (see :meth:`~atmodeller.state.BasePlanet.get_planet_mass` and
:meth:`~atmodeller.state.BasePlanet.get_surface_gravity`).

All methods that accept number of moles (e.g., ``log_number_moles``) and other batchable inputs are
designed to accommodate batched arrays, with shapes broadcast-compatible with the batch dimension.
This enables consistent handling of both single and batched calculations, as required by the
:class:`PhaseOutput` class.

The module also provides a general :class:`PhaseOutput` class for output-friendly, broadcasted
access to phase and species properties, as well as a :class:`GasPhaseOutput` for gas-specific
outputs.
"""

import logging
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar, cast

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from atmodeller import override
from atmodeller.containers import (
    ChemicalSpecies,
    ReservoirSpecies,
    SpeciesCollection,
    get_formula_matrix,
)
from atmodeller.constants import (
    EXTRAPOLATION_CONSTANT_CP,
    EXTRAPOLATION_OFF,
    Extrapolation,
)
from atmodeller.database import DataBase
from atmodeller.interfaces import (
    ActivityProtocol,
    RedoxBufferProtocol,
    SpeciesProtocol,
    TSpecies_co,
)
from atmodeller.jax_utils import FloatArray, as_j64, masked_logsumexp, safe_exp
from atmodeller.sci_utils import GAS_CONSTANT_BAR
from atmodeller.thermodata import ActivityCoefficient
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer

# Due to a Pyright bug (#4965). See Equinox documentation.
if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar

DEFAULT_BACKGROUND_MASS: float = 0.0
"""Default background mass (kg), i.e. no background mass

See :attr:`BasePhase.background_mass` for what "background" mass means (the portion of a phase's
mass that is not modelled by explicit chemical species)."""
DUMMY_MOLAR_MASS: float = 1.0
"""Default molar mass of the background component (kg mol\\ :sup:`-1`) (only meaningful when
background mass is non-zero)"""

logger: logging.Logger = logging.getLogger(__name__)


def _describe(item: object) -> str:
    """Describes a species, or another object given in its place, for an error message"""
    if isinstance(item, ChemicalSpecies):
        return f"{item.name} ({'gas' if item.is_gas else 'condensed'})"
    if isinstance(item, ReservoirSpecies):
        return f"{item.data.name} (dissolved)"

    return type(item).__name__


def _resolve_extrapolation(species: TSpecies_co, extrapolation: Extrapolation) -> TSpecies_co:
    """Sets the extrapolation of a condensed species where it is decided by its phase (``None``)

    Args:
        species: Species
        extrapolation: Extrapolation decided by the phase

    Returns:
        The species with its extrapolation set
    """
    if not isinstance(species, ChemicalSpecies):
        return species
    thermo = species.thermo
    if None not in (thermo.extrapolate_below, thermo.extrapolate_above):
        return species
    thermo = thermo.with_extrapolation(
        extrapolation if thermo.extrapolate_below is None else thermo.extrapolate_below,
        extrapolation if thermo.extrapolate_above is None else thermo.extrapolate_above,
    )

    return cast(TSpecies_co, replace(species, thermo=thermo))


class BasePhase(eqx.Module, Generic[TSpecies_co]):
    """Base class for all phases

    This class defines the physical and model state of a phase, including its species, background
    properties, and core logic for thermodynamic calculations. It is intentionally kept separate
    from :class:`PhaseOutput`, which handles output formatting and broadcasting for batch results.

    Note:
        For all methods, temperature and pressure should be provided as scalars or 1-D arrays
        matching the batch dimension. All inputs will be broadcast as needed, but providing
        compatible shapes avoids shape mismatch errors and ensures correct broadcasting behaviour.

        All methods that accept number of moles (e.g., ``log_number_moles``) are designed to
        accommodate batched inputs, i.e., arrays with a leading batch dimension. This is required
        since these methods are called by the :class:`PhaseOutput` class, which handles both single
        and batched calculations.

    Note:
        A phase's total mass is the sum of two independent contributions:

        1. ``background_mass``: a fixed quantity, set once at construction time, representing
           mass that is *not* modelled by any explicit chemical species (e.g., the bulk of a
           planet's iron core when only trace species dissolved in it are tracked reactively).
           It never changes during a solve.
        2. The mass of any species in ``species`` for which ``include_in_phase_mass`` is
           ``True`` — this is *solved for* and adds on top of ``background_mass``.

        So ``background_mass`` alone is **not** the phase's total mass unless every species has
        ``include_in_phase_mass=False`` (the default for gas species is ``True``, so this mostly
        matters for condensed phases). The self-consistent total, combining both contributions,
        is what :meth:`get_log_phase_mass` (and downstream, ``PhaseOutput.phase_mass``) returns.
        :attr:`PhaseOutput.species_to_phase_mass_ratio` exists specifically to show how much of
        that total the tracked species account for.

        ``background_mass`` is treated as chemically inert: it has no thermodynamic activity of
        its own, but it still matters physically, since it (1) sizes the solvent reservoir that
        dissolved species partition into (see :class:`CondensedPhase`), and (2) contributes to
        mass-dependent quantities computed downstream, e.g. a planet's total mass and surface
        gravity (see :class:`~atmodeller.state.BasePlanet`).

    Args:
        name: Phase name
        species: An iterable of species in the phase
        background_mass: Mass of the background component (kg), i.e. the untracked portion of the
            phase's mass described above. Should be a scalar or a 1-D array matching the batch
            dimension if batching is used. Defaults to :data:`DEFAULT_BACKGROUND_MASS`.
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`). Should
            be a scalar or a 1-D array matching the batch dimension if batching is used. Defaults
            to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not zero.
    """

    name: str
    """Phase name"""
    species: SpeciesCollection[TSpecies_co]
    """Collection of species in the phase"""
    background_mass: FloatArray
    """Mass of the background (untracked) component; see the class docstring for how this relates
    to the phase's actual total mass"""
    background_molar_mass: FloatArray
    """Molar mass of the background (untracked) component"""
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

        logger.info(
            "Creating %s: %s",
            self.__class__.__name__,
            tuple(str(species) for species in self.species),
        )

    @property
    def is_empty(self) -> bool:
        """Indicates whether the phase contains no species"""
        return self.species.number_species == 0

    @property
    def log_background_mass(self) -> FloatArray:
        """Log mass of the background (untracked) component (kg)

        See the class docstring: this is only part of the phase's total mass, not the whole of
        it, whenever any species has ``include_in_phase_mass=True``.
        """
        return jnp.log(self.background_mass)

    @property
    def log_background_moles(self) -> FloatArray:
        """Log moles of the background (untracked) component (mol)"""
        return self.log_background_mass - self.log_background_molar_mass

    @property
    def log_background_molar_mass(self) -> FloatArray:
        r"""Log molar mass of the background (untracked) component (kg mol\ :sup:`-1`)"""
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

        Each activity model is applied directly, in a loop that is unrolled at trace time. The
        models are read from :attr:`species` when called, so their parameters are part of the
        pytree and can be traced and differentiated. This also avoids :func:`jax.lax.switch` under
        :func:`jax.vmap`, which would evaluate every activity model for every species.

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
        mole_fractions: Float[Array, "... n_species"] = jnp.exp(log_mole_fractions)
        batch_shape: tuple[int, ...] = jnp.broadcast_shapes(
            jnp.shape(temperature), jnp.shape(pressure), log_mole_fractions.shape[:-1]
        )
        log_activity: Float[Array, "... n_species"] = jnp.stack(
            [
                jnp.broadcast_to(
                    species_.activity.log_activity(temperature, pressure, mole_fractions),
                    batch_shape,
                )
                for species_ in self.species
            ],
            axis=-1,
        )
        log_activity = log_activity + log_mole_fractions

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
        passed through unchanged. See :class:`BasePhase`'s docstring for how the surviving
        (masked-in) species mass combines with ``background_mass`` to give the phase total.

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

        This is the self-consistent total mass of the phase: ``background_mass`` (the untracked
        component, fixed at construction) plus the mass of every species with
        ``include_in_phase_mass=True`` (solved for). See :class:`BasePhase`'s docstring.

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

        The denominator is the phase total from :meth:`get_log_phase_mass`, which includes the
        untracked ``background_mass`` component, so these fractions sum to less than 1 whenever
        ``background_mass`` is non-zero.

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
        background_mass: Mass of the background (untracked) component (kg); see
            :class:`BasePhase`'s docstring for how this relates to the phase's actual total mass
            (``phase_mass``), which also includes the mass of any species with
            ``include_in_phase_mass=True``
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
        """Normalises phase arrays to shape ``(n_batch, n_species)`` with a generic reshape."""
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
        """Mass of the background (untracked) component; not the phase's total mass on its own
        (see :attr:`phase_mass` and :class:`BasePhase`'s docstring)"""
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
        """Total phase moles from the species included in the phase mass
        (``include_in_phase_mass``) and the inert background"""
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

        The numerator is the sum of *every* species' mass, regardless of its
        ``include_in_phase_mass`` setting (no masking); the denominator is the phase total from
        :meth:`~BasePhase.get_log_phase_mass`, i.e. ``background_mass`` plus only the species
        whose ``include_in_phase_mass`` is ``True``. So this ratio is:

        - exactly ``1`` only when ``background_mass == 0`` *and* every species in the phase has
          ``include_in_phase_mass=True`` (the mask is then a no-op and numerator == denominator);
        - ``< 1`` whenever ``background_mass > 0`` (there's untracked mass the species don't
          account for);
        - ``> 1`` if any species has ``include_in_phase_mass=False`` (it counts in the numerator
          but not the denominator).

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
        background_mass: Mass of the background (untracked) component (kg); see
            :class:`BasePhase`'s docstring. Defaults to :data:`DEFAULT_BACKGROUND_MASS` (zero).
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`).
            Defaults to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not
            zero.
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
        species = tuple(species)
        not_gas: list[str] = [
            _describe(species_)
            for species_ in species
            if not (isinstance(species_, ChemicalSpecies) and species_.is_gas)
        ]
        if not_gas:
            raise ValueError(
                f"The {name} phase can only have gas species, created with "
                f"ChemicalSpecies.create_gas, not {not_gas}"
            )
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
        name: Name of the phase. Defaults to ``condensed_phase``.
        background_mass: Mass of the background (untracked) component (kg); see
            :class:`BasePhase`'s docstring. Defaults to :data:`DEFAULT_BACKGROUND_MASS` (zero).
        background_molar_mass: Molar mass of the background component (kg mol\\ :sup:`-1`).
            Defaults to :data:`DUMMY_MOLAR_MASS`; only meaningful when ``background_mass`` is not
            zero.
        dissolved: Whether the phase can have dissolved species, which is only the case for the
            silicate melt, the solvent of their solubility laws. Defaults to ``False``.

    Raises:
        ValueError: If a species is a gas, or is dissolved and ``dissolved`` is ``False``
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
        dissolved: bool = False,
    ):
        species = tuple(species)
        not_condensed: list[str] = [
            _describe(species_)
            for species_ in species
            if not isinstance(species_, (ChemicalSpecies, ReservoirSpecies))
            or (isinstance(species_, ChemicalSpecies) and species_.is_gas)
        ]
        if not_condensed:
            raise ValueError(
                f"The {name} phase can only have condensed or dissolved species, created with "
                "ChemicalSpecies.from_end_member or ReservoirSpecies.create_dissolved, not "
                f"{not_condensed}"
            )
        if not dissolved:
            misplaced: list[str] = [
                _describe(species_)
                for species_ in species
                if isinstance(species_, ReservoirSpecies)
            ]
            if misplaced:
                raise ValueError(
                    f"The {name} phase cannot have dissolved species, since their solubility laws "
                    f"are for the silicate melt, not {misplaced}. Use condensed species, with an "
                    "activity model if needed."
                )
        # A species of a solution is needed where it is metastable as a pure phase. A pure
        # phase has already set its extrapolation.
        species = tuple(
            _resolve_extrapolation(species_, EXTRAPOLATION_CONSTANT_CP) for species_ in species
        )
        super().__init__(name, species, background_mass, background_molar_mass)


class PurePhase(CondensedPhase):
    """Single-species phase (e.g., a pure mineral, ice, or liquid)

    A pure phase has the name of its species, under which it appears in the output, and no
    background. Its activity is that of its species, which is unity by default
    (:class:`~atmodeller.thermodata.core.ActivityCoefficient`). A species with an activity model
    for a solution belongs in a :class:`CondensedPhase` instead.

    Args:
        species: Species of the phase
    """

    # Without an override pylance gets confused, throwing missing argument warnings even though
    # defaults are provided in the base class. This avoids reporting this false-alarm to the user.
    @override
    def __init__(self, species: ChemicalSpecies):
        if not isinstance(species, ChemicalSpecies) or species.is_gas:
            raise ValueError(
                "A pure phase has a condensed species, created with "
                f"ChemicalSpecies.from_end_member, not {_describe(species)}"
            )
        # A pure phase is not stable outside the temperature range of its data, which is often
        # bounded by a phase boundary, such as a melting point
        super().__init__((_resolve_extrapolation(species, EXTRAPOLATION_OFF),), species.name)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        *,
        liquid: bool = True,
        database: DataBase | None = None,
        activity: ActivityProtocol = ActivityCoefficient(),
        solve_for_stability: bool = True,
        include_in_phase_mass: bool = True,
        extrapolate_below: Extrapolation | None = None,
        extrapolate_above: Extrapolation | None = None,
    ) -> Self:
        """Creates a pure phase from all the condensed end-members of a formula

        The condensed end-members that can be stitched together
        (:meth:`~atmodeller.database.DataBase.stitchable_end_members`), e.g. quartz to
        cristobalite to liquid for ``SiO2``, are stitched together in order of temperature
        (:meth:`~atmodeller.database.DataBase.stitch`), and the species is named with
        :data:`~atmodeller.constants.CONDENSED_SUFFIX`, e.g. ``O2Si_cd``. Without the liquid,
        only the solids are stitched together, e.g. quartz to cristobalite for ``SiO2``, and the
        species is named with :data:`~atmodeller.constants.SOLID_SUFFIX`, e.g. ``O2Si_s``. See
        :meth:`~atmodeller.containers.ChemicalSpecies.from_formula`.

        Args:
            formula: Formula
            liquid: Whether to include the liquid. Defaults to ``True``.
            database: Database. Defaults to ``None``, which uses the default database.
            activity: Activity. Defaults to unity activity.
            solve_for_stability: Solve for stability. Defaults to ``True``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.
            extrapolate_below: Extrapolation of the thermodynamic data below their temperature
                range (``"off"``, ``"data"`` or ``"constant_cp"``). Defaults to ``None``, which
                is ``"off"``, so the phase is not stable there.
            extrapolate_above: Extrapolation of the thermodynamic data above their temperature
                range. Defaults to ``None``, which is ``"off"``.

        Returns:
            A pure phase, which has the name of its species

        Raises:
            KeyError: If the formula has no condensed end-members, or no solid end-members
                without the liquid
            ValueError: If the end-members cannot be stitched together
        """
        species: ChemicalSpecies = ChemicalSpecies.from_formula(
            formula,
            liquid=liquid,
            database=database,
            activity=activity,
            solve_for_stability=solve_for_stability,
            include_in_phase_mass=include_in_phase_mass,
            extrapolate_below=extrapolate_below,
            extrapolate_above=extrapolate_above,
        )

        return cls(species)

    @classmethod
    def from_end_members(
        cls,
        end_members: Iterable[str],
        *,
        name: str,
        formula: str | None = None,
        database: DataBase | None = None,
        activity: ActivityProtocol = ActivityCoefficient(),
        solve_for_stability: bool = True,
        include_in_phase_mass: bool = True,
        extrapolate_below: Extrapolation | None = None,
        extrapolate_above: Extrapolation | None = None,
    ) -> Self:
        """Creates a pure phase by stitching end-members together

        For example, ``from_end_members(("quartz-alpha", "quartz-beta"), name="quartz")`` gives
        ``O2Si_quartz``. The end-members must be of one formula, and are stitched together in
        order of temperature (:meth:`~atmodeller.database.DataBase.stitch`). See
        :meth:`~atmodeller.containers.ChemicalSpecies.from_end_members`.

        Args:
            end_members: Names of the end-members
            name: Suffix of the species name
            formula: Formula. Defaults to ``None``, which takes the formula of the minerals, and
                is only needed when the end-members are all shared by several formulas, such as
                ``liquid``.
            database: Database. Defaults to ``None``, which uses the default database.
            activity: Activity. Defaults to unity activity.
            solve_for_stability: Solve for stability. Defaults to ``True``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.
            extrapolate_below: Extrapolation of the thermodynamic data below their temperature
                range (``"off"``, ``"data"`` or ``"constant_cp"``). Defaults to ``None``, which
                is ``"off"``, so the phase is not stable there.
            extrapolate_above: Extrapolation of the thermodynamic data above their temperature
                range. Defaults to ``None``, which is ``"off"``.

        Returns:
            A pure phase, which has the name of its species

        Raises:
            KeyError: If an end-member is not available for the formula
            ValueError: If the end-members cannot be stitched together, or no formula is
                given for end-members that are shared by several formulas
        """
        species: ChemicalSpecies = ChemicalSpecies.from_end_members(
            end_members,
            name=name,
            formula=formula,
            database=database,
            activity=activity,
            solve_for_stability=solve_for_stability,
            include_in_phase_mass=include_in_phase_mass,
            extrapolate_below=extrapolate_below,
            extrapolate_above=extrapolate_above,
        )

        return cls(species)

    @classmethod
    def from_end_member(
        cls,
        end_member: str,
        *,
        formula: str | None = None,
        database: DataBase | None = None,
        activity: ActivityProtocol = ActivityCoefficient(),
        solve_for_stability: bool = True,
        include_in_phase_mass: bool = True,
        extrapolate_below: Extrapolation | None = None,
        extrapolate_above: Extrapolation | None = None,
    ) -> Self:
        """Creates a pure phase from one end-member

        For example, ``from_end_member("cristobalite-beta")`` gives ``O2Si_cristobalite-beta`` and
        ``from_end_member("liquid", formula="H2O")`` gives ``H2O_l``. See
        :meth:`~atmodeller.containers.ChemicalSpecies.from_end_member`.

        Args:
            end_member: Name of the end-member
            formula: Formula. Defaults to ``None``, which takes the formula of a mineral, and is
                needed for ``liquid``, which is an end-member of several formulas.
            database: Database. Defaults to ``None``, which uses the default database.
            activity: Activity. Defaults to unity activity.
            solve_for_stability: Solve for stability. Defaults to ``True``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.
            extrapolate_below: Extrapolation of the thermodynamic data below their temperature
                range (``"off"``, ``"data"`` or ``"constant_cp"``). Defaults to ``None``, which
                is ``"off"``, so the phase is not stable there.
            extrapolate_above: Extrapolation of the thermodynamic data above their temperature
                range. Defaults to ``None``, which is ``"off"``.

        Returns:
            A pure phase, which has the name of its species
        """
        species: ChemicalSpecies = ChemicalSpecies.from_end_member(
            end_member,
            formula=formula,
            database=database,
            activity=activity,
            solve_for_stability=solve_for_stability,
            include_in_phase_mass=include_in_phase_mass,
            extrapolate_below=extrapolate_below,
            extrapolate_above=extrapolate_above,
        )

        return cls(species)
