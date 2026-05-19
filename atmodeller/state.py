# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Thermodynamic state and planetary state models

The hierarchy is:

- :class:`BaseThermodynamicState`: shared interface and phase-mass helpers
- :class:`ThermodynamicState`: generic state with fixed pressure
- :class:`BasePlanet`: common planetary quantities (surface area, gravity, mass)
- :class:`ThinAtmospherePlanet`: pressure from thin-atmosphere mechanical balance when pressure is
  not specified
- :class:`PressureScalingLawPlanet`: pressure from a scaling law
  :cite:p:`Schlichting2022{Equation 8}` when pressure is not specified
"""

from abc import abstractmethod
from collections.abc import Iterable
from typing import Optional, Self, cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float
from scipy.constants import gravitational_constant

from atmodeller import override
from atmodeller.constants import (
    GAS_PHASE_INDEX,
    METAL_PHASE_INDEX,
    SILICATE_MELT_PHASE_INDEX,
    SILICATE_SOLID_PHASE_INDEX,
)
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.phases import BasePhase, CondensedPhase, GasPhase, PurePhase
from atmodeller.reactions import PhaseSystem, ReactionSystem
from atmodeller.sci_utils import FE_MOLAR_MASS, SIO2_MOLAR_MASS, earth, unit_conversion


class BaseThermodynamicState(eqx.Module):
    reaction_system: ReactionSystem
    """Reaction system representing the thermodynamic state of the system"""
    temperature: FloatArray
    """Temperature (K)"""
    pressure: FloatArray
    """Pressure (bar). Should not be used directly; use :meth:`get_pressure` instead"""

    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    @property
    def phase_system(self) -> PhaseSystem:
        """Phase system representing the thermodynamic state of the planetary body"""
        return self.reaction_system.phase_system

    @property
    def gas_phase_index(self) -> int:
        return GAS_PHASE_INDEX

    @property
    def silicate_melt_phase_index(self) -> int:
        return SILICATE_MELT_PHASE_INDEX

    @property
    def silicate_solid_phase_index(self) -> int:
        return SILICATE_SOLID_PHASE_INDEX

    @property
    def metal_phase_index(self) -> int:
        return METAL_PHASE_INDEX

    @abstractmethod
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Pressure (bar)"""
        ...

    @abstractmethod
    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary representation."""
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> Self:
        """Updates the state."""
        ...

    def get_silicate_solid_mass(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> FloatArray:
        """Gets the silicate solid mass.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Silicate solid mass (kg)
        """
        solid: BasePhase = self.phase_system.phases[self.silicate_solid_phase_index]
        solid_slice: slice = self.phase_system.phase_slice(self.silicate_solid_phase_index)

        log_number_moles_solid: Float[Array, "... solid_species"] = log_number_moles[
            ..., solid_slice
        ]
        solid_mass: Float[Array, "... 1"] = jnp.exp(
            solid.get_log_phase_mass(log_number_moles_solid)
        )
        solid_mass_squeeze: FloatArray = jnp.squeeze(solid_mass, axis=-1)

        return solid_mass_squeeze

    def get_silicate_melt_mass(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> FloatArray:
        """Gets the silicate melt mass.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Silicate melt mass (kg)
        """
        melt: BasePhase = self.phase_system.phases[self.silicate_melt_phase_index]
        melt_slice: slice = self.phase_system.phase_slice(self.silicate_melt_phase_index)

        log_number_moles_melt: Float[Array, "... melt_species"] = log_number_moles[..., melt_slice]
        melt_mass: Float[Array, "... 1"] = jnp.exp(melt.get_log_phase_mass(log_number_moles_melt))
        melt_mass_squeeze: FloatArray = jnp.squeeze(melt_mass, axis=-1)

        return melt_mass_squeeze

    def get_metal_mass(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the metal mass.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Metal mass (kg)
        """
        metal: BasePhase = self.phase_system.phases[self.metal_phase_index]
        metal_slice: slice = self.phase_system.phase_slice(self.metal_phase_index)

        log_number_moles_metal: Float[Array, "... metal_species"] = log_number_moles[
            ..., metal_slice
        ]
        metal_mass: Float[Array, "... 1"] = jnp.exp(
            metal.get_log_phase_mass(log_number_moles_metal)
        )
        metal_mass_squeeze: FloatArray = jnp.squeeze(metal_mass, axis=-1)

        return metal_mass_squeeze

    def get_silicate_melt_fraction(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> FloatArray:
        """Gets the silicate melt fraction.

        .. math::
            X_{\\rm melt} = \\frac{m_{\\rm melt}}{m_{\\rm melt} + m_{\\rm solid}}

        where :math:`m_{\\rm melt}` is the melt mass and :math:`m_{\\rm solid}` is the solid mass.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Melt fraction (between 0 and 1) (kgkg\\ :sup:`-1`)
        """
        silicate_melt_mass: FloatArray = self.get_silicate_melt_mass(log_number_moles)
        silicate_solid_mass: FloatArray = self.get_silicate_solid_mass(log_number_moles)

        melt_fraction: FloatArray = silicate_melt_mass / (silicate_melt_mass + silicate_solid_mass)

        return melt_fraction


class ThermodynamicState(BaseThermodynamicState):
    """A generic thermodynamic state

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the system
        temperature: Temperature (K)
        pressure: Pressure (bar)
    """

    reaction_system: ReactionSystem
    """Reaction system representing the thermodynamic state of the system"""
    temperature: FloatArray
    """Temperature (K)"""
    pressure: FloatArray
    """Pressure (bar). Should not be used directly; use :meth:`get_pressure` instead"""

    # For helpful typing information since eqx.field(converter=as_j64) confuses the type checker
    @override
    def __init__(
        self, reaction_system: ReactionSystem, temperature: ArrayLike, pressure: ArrayLike
    ):
        self.reaction_system = reaction_system
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)

    @classmethod
    def from_species(
        cls,
        gas_species: Iterable[ChemicalSpecies],
        pressure: ArrayLike,
        *,
        mass: ArrayLike = 1.0,
        melt_fraction: ArrayLike = 1.0,
        temperature: ArrayLike = 2000,
        background_silicate_molar_mass: ArrayLike = SIO2_MOLAR_MASS,
        background_metal_mass: ArrayLike = 0.0,
        background_metal_molar_mass: ArrayLike = FE_MOLAR_MASS,
        silicate_melt_species: Iterable[SpeciesProtocol] = (),
        silicate_solid_species: Iterable[SpeciesProtocol] = (),
        metal_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
    ) -> Self:
        """Builds an instance from phase-species inputs.

        Args:
            gas_species: Iterable of species in the gas phase
            pressure: Pressure (bar)
            mass: Mass (kg). Defaults to ``1`` (i.e., reference unit mass).
            melt_fraction: Melt fraction. Defaults to ``1`` (i.e., fully molten).
            temperature: Temperature (K). Defaults to ``2000``.
            background_silicate_molar_mass: Background molar mass of a silicate phase. Defaults to
                :data:`~atmodeller.sci_utils.SIO2_MOLAR_MASS`.
            background_metal_mass: Background mass of the metal phase, Defaults to ``0``.
            background_metal_molar_mass: Background molar mass of the metal phase. Defaults
                to :data:`~atmodeller.sci_utils.FE_MOLAR_MASS`.
            silicate_melt_species: Iterable of species in the silicate melt phase. Defaults to an
                empty tuple.
            silicate_solid_species: Iterable of species in the silicate solid phase. Defaults to an
                empty tuple.
            metal_species: Iterable of species in the metal phase. Defaults to an empty tuple.
            condensates: Iterable of pure phases representing condensates in the system. Defaults
                to an empty tuple.

        Returns:
            An instance
        """
        background_silicate_melt_mass: ArrayLike = mass * melt_fraction
        background_silicate_solid_mass: ArrayLike = mass * (1 - melt_fraction)

        gas: GasPhase = GasPhase(gas_species)
        silicate_melt: CondensedPhase = CondensedPhase(
            silicate_melt_species,
            "silicate_melt",
            background_silicate_melt_mass,
            background_silicate_molar_mass,
        )
        silicate_solid: CondensedPhase = CondensedPhase(
            silicate_solid_species,
            "silicate_solid",
            background_silicate_solid_mass,
            background_silicate_molar_mass,
        )
        metal: CondensedPhase = CondensedPhase(
            metal_species, "metal", background_metal_mass, background_metal_molar_mass
        )
        # NOTE: This ordering must be maintained
        other_phases: tuple[BasePhase, ...] = (silicate_melt, silicate_solid, metal) + tuple(
            condensates
        )
        phase_system: PhaseSystem = PhaseSystem(gas, other_phases=other_phases)
        reaction_system: ReactionSystem = ReactionSystem(phase_system)

        return cls(reaction_system, temperature, pressure)

    @override
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        Returns:
            Pressure (bar)
        """
        del log_number_moles

        return self.pressure

    @override
    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary representation.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            A dictionary of the values
        """
        base_dict = {}
        base_dict["silicate_melt_mass"] = self.get_silicate_melt_mass(log_number_moles)
        base_dict["silicate_solid_mass"] = self.get_silicate_solid_mass(log_number_moles)
        base_dict["silicate_melt_fraction"] = self.get_silicate_melt_fraction(log_number_moles)
        base_dict["silicate_mass"] = (
            base_dict["silicate_melt_mass"] + base_dict["silicate_solid_mass"]
        )
        base_dict["temperature"] = self.temperature
        base_dict["pressure"] = self.pressure

        return base_dict

    @override
    def update(
        self, temperature: Optional[ArrayLike] = None, pressure: Optional[ArrayLike] = None
    ) -> Self:
        """Updates the state.

        New values are assumed to be broadcastable to the shapes of the existing fields. Keeping
        leaf shapes stable helps avoid unnecessary JAX recompilation, including in jitted
        workflows.

        Args:
            temperature: Temperature (K). Defaults to ``None``.
            pressure: Pressure (bar). Defaults to ``None``.

        Returns:
            Updated state
        """
        state_updated: ThermodynamicState = self

        if temperature is not None:
            temperature = jnp.broadcast_to(as_j64(temperature), self.temperature.shape)
            state_updated = eqx.tree_at(lambda s: s.temperature, state_updated, temperature)

        if pressure is not None:
            pressure = jnp.broadcast_to(as_j64(pressure), self.pressure.shape)
            state_updated = eqx.tree_at(lambda s: s.pressure, state_updated, pressure)

        return cast(Self, state_updated)


class BasePlanet(BaseThermodynamicState):
    """A planet

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface (m)
        temperature: Temperature (K)
        pressure: Pressure (bar)
        background_planet_mass: Planet mass (kg) from only the background melt and solid mass,
            plus the metallic core mass. This value will only be equal to the actual planet mass if
            ``include_in_phase_mass`` is ``False`` for all species in the melt and solid phases.
    """

    reaction_system: ReactionSystem
    """Reaction system representing the thermodynamic state of the planetary body"""
    surface_radius: FloatArray
    """Radius of the surface (m)"""
    temperature: FloatArray
    """Temperature (K)"""
    pressure: FloatArray
    """Pressure (bar)"""
    background_planet_mass: FloatArray
    """Planet mass (kg) from only the background melt and solid mass, plus the core mass"""

    # For helpful typing information since eqx.field(converter=as_j64) confuses the type checker
    @override
    def __init__(
        self,
        reaction_system: ReactionSystem,
        surface_radius: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        background_planet_mass: ArrayLike,
    ):
        self.reaction_system = reaction_system
        self.surface_radius = as_j64(surface_radius)
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.background_planet_mass = as_j64(background_planet_mass)

    @classmethod
    def from_species(
        cls,
        gas_species: Iterable[ChemicalSpecies],
        *,
        planet_mass: ArrayLike = earth.mass,
        core_mass_fraction: ArrayLike = earth.core_mass_fraction,
        mantle_melt_fraction: ArrayLike = 1.0,
        surface_radius: ArrayLike = earth.radius,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = jnp.nan,
        background_silicate_molar_mass: ArrayLike = SIO2_MOLAR_MASS,
        background_metal_molar_mass: ArrayLike = FE_MOLAR_MASS,
        silicate_melt_species: Iterable[SpeciesProtocol] = (),
        silicate_solid_species: Iterable[SpeciesProtocol] = (),
        metal_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
    ) -> Self:
        """Builds a new instance from phase-species inputs.

        Default values are for a fully molten Earth.

        Args:
            gas_species: Iterable of species in the gas phase
            planet_mass: Mass of the planet (kg). Defaults to Earth.
            core_mass_fraction: Mass fraction of the iron core relative to the planetary mass
                (kg kg\\ :sup:`-1`). Defaults to Earth.
            mantle_melt_fraction: Mantle melt fraction (kg kg\\ :sup:`-1`). Defaults to ``1.0``.
            surface_radius: Radius of the planetary surface (m). Defaults to Earth.
            temperature: Temperature (K). Defaults to ``2000``.
            pressure: Pressure (bar). Defaults to ``NaN`` to solve for the mechanical pressure
                balance at the surface.
            background_silicate_molar_mass: Background molar mass of a silicate phase. Defaults to
                :data:`~atmodeller.sci_utils.SIO2_MOLAR_MASS`.
            background_metal_molar_mass: Background molar mass of the metal phase. Defaults
                to :data:`~atmodeller.sci_utils.FE_MOLAR_MASS`.
            silicate_melt_species: Iterable of species in the silicate melt phase. Defaults to an
                empty tuple.
            silicate_solid_species: Iterable of species in the silicate solid phase. Defaults to an
                empty tuple.
            metal_species: Iterable of species in the metal phase. Defaults to an empty tuple.
            condensates: Iterable of pure phases representing condensates in the system. Defaults
                to an empty tuple.

        Returns:
            An instance
        """
        mantle_mass: ArrayLike = planet_mass * (1 - core_mass_fraction)
        background_melt_mass: ArrayLike = mantle_mass * mantle_melt_fraction
        background_solid_mass: ArrayLike = mantle_mass * (1 - mantle_melt_fraction)
        metallic_core_mass: ArrayLike = planet_mass * core_mass_fraction

        gas: GasPhase = GasPhase(gas_species)
        silicate_melt: CondensedPhase = CondensedPhase(
            silicate_melt_species,
            "silicate_melt",
            background_melt_mass,
            background_silicate_molar_mass,
        )
        silicate_solid: CondensedPhase = CondensedPhase(
            silicate_solid_species,
            "silicate_solid",
            background_solid_mass,
            background_silicate_molar_mass,
        )
        metal: CondensedPhase = CondensedPhase(
            metal_species, "metal", metallic_core_mass, background_metal_molar_mass
        )
        # NOTE: This ordering must be maintained
        other_phases: tuple[BasePhase, ...] = (silicate_melt, silicate_solid, metal) + tuple(
            condensates
        )
        phase_system = PhaseSystem(gas, other_phases=other_phases)
        reaction_system: ReactionSystem = ReactionSystem(phase_system)

        return cls(reaction_system, surface_radius, temperature, pressure, planet_mass)

    @property
    def background_metallic_core_mass(self) -> FloatArray:
        """Mass of the metallic core from only the background metal mass (kg)

        This value will only be equal to the actual metallic core mass if ``include_in_phase_mass``
        is ``False`` for all species in the metal phase.
        """
        return self.phase_system.phases[self.metal_phase_index].background_mass

    @property
    def background_mantle_mass(self) -> FloatArray:
        """Mass of the mantle from only the background melt and solid mass (kg)

        This value will only be equal to the actual mantle mass if ``include_in_phase_mass`` is
        ``False`` for all species in the melt and solid phases.
        """
        return (
            self.phase_system.phases[self.silicate_melt_phase_index].background_mass
            + self.phase_system.phases[self.silicate_solid_phase_index].background_mass
        )

    @property
    def surface_area(self) -> FloatArray:
        """Surface area (m\\ :sup:`2`)

        .. math::
            A_{\\rm surf} = 4 \\pi R_{\\rm surf}^2

        where :math:`R_{\\rm surf}` is the surface radius.
        """
        return 4.0 * jnp.pi * jnp.square(self.surface_radius)

    def get_planet_mass(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the planet mass.

        Computes the planet mass from the mass of the condensed phases and the metallic core.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Planet mass (kg)
        """
        mantle_melt_mass: FloatArray = self.get_silicate_melt_mass(log_number_moles)
        mantle_solid_mass: FloatArray = self.get_silicate_solid_mass(log_number_moles)
        metal_mass: FloatArray = self.get_metal_mass(log_number_moles)
        planet_mass = mantle_melt_mass + mantle_solid_mass + metal_mass

        return planet_mass

    def get_surface_gravity(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the surface gravity.

        .. math::
            g_{\\rm surf} = \\frac{GM_{\\rm p}}{R_{\\rm surf}^2}

        where :math:`G` is the gravitational constant, :math:`M_{\\rm p}` is the planet mass, and
        :math:`R_{\\rm surf}` is the surface radius.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Surface gravity (ms\\ :sup:`-2`)
        """
        planet_mass: FloatArray = self.get_planet_mass(log_number_moles)
        # jax.debug.print("planet_mass = {out}", out=planet_mass_squeeze)

        surface_gravity: FloatArray = (
            gravitational_constant * planet_mass / jnp.square(self.surface_radius)
        )
        # jax.debug.print("surface_gravity = {out}", out=surface_gravity)

        return surface_gravity

    @override
    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary representation.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            A dictionary of the values
        """
        base_dict = {}
        base_dict["mantle_melt_mass"] = self.get_silicate_melt_mass(log_number_moles)
        base_dict["mantle_solid_mass"] = self.get_silicate_solid_mass(log_number_moles)
        base_dict["mantle_mass"] = base_dict["mantle_melt_mass"] + base_dict["mantle_solid_mass"]
        base_dict["mantle_melt_fraction"] = self.get_silicate_melt_fraction(log_number_moles)
        base_dict["metallic_core_mass"] = self.get_metal_mass(log_number_moles)
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.get_surface_gravity(log_number_moles)
        base_dict["surface_radius"] = self.surface_radius
        base_dict["planet_mass"] = self.get_planet_mass(log_number_moles)
        base_dict["temperature"] = self.temperature
        base_dict["pressure"] = self.get_pressure(log_number_moles)

        return base_dict

    @override
    def update(
        self,
        planet_mass: Optional[ArrayLike] = None,
        core_mass_fraction: Optional[ArrayLike] = None,
        mantle_melt_fraction: Optional[ArrayLike] = None,
        surface_radius: Optional[ArrayLike] = None,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
    ) -> Self:
        """Updates the state.

        New values are assumed to be broadcastable to the shapes of the existing fields. Keeping
        leaf shapes stable helps avoid unnecessary JAX recompilation, including in jitted
        workflows.

        Args:
            planet_mass: Mass of the planet (kg). Defaults to ``None``.
            core_mass_fraction: Mass fraction of the iron core relative to the planetary mass
                (kgkg\\ :sup:`-1`). Defaults to ``None``.
            mantle_melt_fraction: Mantle melt fraction. Defaults to ``None``.
            surface_radius: Radius of the planetary surface (m). Defaults to ``None``.
            temperature: Temperature (K). Defaults to ``None``.
            pressure: Pressure (bar). Defaults to ``None``.

        Returns:
            Updated state
        """
        state_updated: BasePlanet = self

        if planet_mass is not None:
            planet_mass = jnp.broadcast_to(as_j64(planet_mass), self.background_planet_mass.shape)
            state_updated = eqx.tree_at(
                lambda s: s.background_planet_mass, state_updated, planet_mass
            )

        if core_mass_fraction is not None:
            core_mass_fraction = jnp.broadcast_to(
                as_j64(core_mass_fraction), self.background_metallic_core_mass.shape
            )
            state_updated = eqx.tree_at(
                lambda s: (
                    s.reaction_system.phase_system.phases[self.metal_phase_index].background_mass
                ),
                state_updated,
                state_updated.background_planet_mass * core_mass_fraction,
            )

        if mantle_melt_fraction is not None:
            mantle_mass: FloatArray = (
                state_updated.background_planet_mass - state_updated.background_metallic_core_mass
            )
            state_updated = eqx.tree_at(
                lambda s: (
                    s.reaction_system.phase_system.phases[
                        self.silicate_melt_phase_index
                    ].background_mass
                ),
                state_updated,
                mantle_mass * mantle_melt_fraction,
            )
            state_updated = eqx.tree_at(
                lambda s: (
                    s.reaction_system.phase_system.phases[
                        self.silicate_solid_phase_index
                    ].background_mass
                ),
                state_updated,
                mantle_mass * (1 - mantle_melt_fraction),
            )

        if surface_radius is not None:
            surface_radius = jnp.broadcast_to(as_j64(surface_radius), self.surface_radius.shape)
            state_updated = eqx.tree_at(lambda s: s.surface_radius, state_updated, surface_radius)

        if temperature is not None:
            temperature = jnp.broadcast_to(as_j64(temperature), self.temperature.shape)
            state_updated = eqx.tree_at(lambda s: s.temperature, state_updated, temperature)

        if pressure is not None:
            pressure = jnp.broadcast_to(as_j64(pressure), self.pressure.shape)
            state_updated = eqx.tree_at(lambda s: s.pressure, state_updated, pressure)

        return cast(Self, state_updated)


class ThinAtmospherePlanet(BasePlanet):
    """A planet with a thin atmosphere

    In this context, "thin atmosphere" means the atmosphere is shallow compared with the planetary
    radius, so gravity is approximated as constant throughout the atmospheric column. Surface
    gravity is therefore set by the planetary body (condensed phases plus metallic core), and the
    atmosphere is treated as non-self-gravitating in the pressure balance.

    .. math::
        P_{\\rm surf} = \\frac{m_{\\rm atm} g_{\\rm surf}}{A_{\\rm surf}}

    where :math:`m_{\\rm atm}` is the atmospheric mass, :math:`g_{\\rm surf}` is the surface
    gravity, and :math:`A_{\\rm surf}` is the surface area.

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface (m)
        metallic_core_mass: Metallic core mass (kg)
        temperature: Temperature (K)
        pressure: Pressure (bar)
        background_planet_mass: Planet mass (kg) from only the background melt and solid mass,
            plus the metallic core mass. This value will only be equal to the actual planet mass if
            ``include_in_phase_mass`` is ``False`` for all species in the melt and solid phases.
    """

    @override
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        A pressure is used if specified, otherwise the default behaviour is to compute the
        pressure from the mechanical pressure balance at the planetary surface assuming the thin
        atmosphere approximation. That is, gravity is computed from the planetary body and treated
        as constant throughout the atmospheric column.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Pressure (bar)
        """
        pressure_specified: Bool[Array, "..."] = ~jnp.isnan(self.pressure)

        log_number_moles_gas: Float[Array, "... gas_species"] = log_number_moles[
            ..., self.phase_system.gas_slice
        ]
        # jax.debug.print("log_number_moles_gas = {out}", out=log_number_moles_gas)
        gas_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.gas.get_log_phase_mass(log_number_moles_gas)
        )
        gas_mass_squeeze: FloatArray = jnp.squeeze(gas_mass, axis=-1)
        # jax.debug.print("gas_mass = {out}", out=gas_mass_squeeze)

        surface_gravity: FloatArray = self.get_surface_gravity(log_number_moles)
        mechanical_pressure: FloatArray = (
            gas_mass_squeeze * surface_gravity / self.surface_area * unit_conversion.Pa_to_bar
        )
        # jax.debug.print("mechanical_pressure = {out}", out=mechanical_pressure)

        pressure: FloatArray = jnp.where(pressure_specified, self.pressure, mechanical_pressure)
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure


class PressureScalingLawPlanet(BasePlanet):
    """A planet with a scaling law for the atmospheric pressure

    A pressure is used if specified, otherwise it is computed from the scaling law
    :cite:p:`Schlichting2022{Equation 8}`:

    .. math::
        P_{\\text{surf}} = 1 \\times 10^6 \\frac{M_{\\text{atm}}}{M_{\\text{p}}} \\left( \\frac{M_\\text{p}}{M_{\\text{Earth}}}\\right)^{2/3}

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface (m)
        metallic_core_mass: Metallic core mass (kg)
        temperature: Temperature (K)
        pressure: Pressure (bar)
        background_planet_mass: Planet mass (kg) from only the background melt and solid mass, plus
            the metallic core mass. This value will only be equal to the actual planet mass if
            ``include_in_phase_mass`` is ``False`` for all species in the melt and solid phases.
    """

    @override
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        A pressure is used if specified, otherwise it is computed from the scaling law.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Pressure (bar)
        """
        pressure_specified: Bool[Array, "..."] = ~jnp.isnan(self.pressure)

        log_number_moles_gas: Float[Array, "... gas_species"] = log_number_moles[
            ..., self.phase_system.gas_slice
        ]
        # jax.debug.print("log_number_moles_gas = {out}", out=log_number_moles_gas)
        gas_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.gas.get_log_phase_mass(log_number_moles_gas)
        )
        gas_mass_squeeze: FloatArray = jnp.squeeze(gas_mass, axis=-1)

        planet_mass: FloatArray = self.get_planet_mass(log_number_moles)

        scaling_law: FloatArray = (
            1e6 * (gas_mass_squeeze / planet_mass) * (planet_mass / earth.mass) ** (2 / 3)
        )

        pressure: FloatArray = jnp.where(pressure_specified, self.pressure, scaling_law)
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure


# The default planet model is the thin atmosphere model
Planet = ThinAtmospherePlanet
