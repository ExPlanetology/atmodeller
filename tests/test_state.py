# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for atmodeller.state"""

import logging

import pytest

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.phases import PurePhase
from atmodeller.solubility import get_solubility_models
from atmodeller.state import BasePlanet, PressureScalingLawPlanet, ThermodynamicState

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")


def test_thermodynamic_state_update() -> None:
    """Tests ThermodynamicState.update with temperature and pressure kwargs."""
    state: ThermodynamicState = ThermodynamicState.from_species(
        (H2O_g,), pressure=2.0, temperature=1000.0
    )

    updated_state: ThermodynamicState = state.update(temperature=1500.0, pressure=5.0)

    assert float(updated_state.temperature) == pytest.approx(1500.0)
    assert float(updated_state.pressure) == pytest.approx(5.0)
    # Original state is unaffected (immutability)
    assert float(state.temperature) == pytest.approx(1000.0)
    assert float(state.pressure) == pytest.approx(2.0)


def _background_masses(planet: BasePlanet) -> tuple[float, float, float]:
    """Background core, melt and solid masses (kg)"""
    phases = planet.phase_system.phases
    return (
        float(planet.background_metallic_core_mass),
        float(phases[planet.silicate_melt_phase_index].background_mass),
        float(phases[planet.silicate_solid_phase_index].background_mass),
    )


def test_base_planet_update() -> None:
    """BasePlanet.update keeps the planet mass and the background core, melt and solid masses
    consistent, whichever of planet_mass, core_mass_fraction and mantle_melt_fraction changes."""
    planet: BasePlanet = PressureScalingLawPlanet.from_species(
        (H2O_g,), temperature=1000.0, core_mass_fraction=0.3, mantle_melt_fraction=0.4
    )
    mass: float = float(planet.background_planet_mass)

    def check(updated: BasePlanet, planet_mass: float, core_fraction: float, melt: float) -> None:
        core, melt_mass, solid = _background_masses(updated)
        assert float(updated.background_planet_mass) == pytest.approx(planet_mass)
        assert core + melt_mass + solid == pytest.approx(planet_mass)
        assert core == pytest.approx(planet_mass * core_fraction)
        assert melt_mass / (melt_mass + solid) == pytest.approx(melt)

    # Only the planet mass: the core and mantle are rescaled, keeping both fractions
    check(planet.update(planet_mass=2 * mass), 2 * mass, 0.3, 0.4)
    # Only the core fraction: the mantle takes the rest, keeping its melt fraction
    check(planet.update(core_mass_fraction=0.5), mass, 0.5, 0.4)
    # Only the melt fraction
    check(planet.update(mantle_melt_fraction=0.9), mass, 0.3, 0.9)
    # All together
    check(planet.update(planet_mass=6.0e24, core_mass_fraction=0.2), 6.0e24, 0.2, 0.4)


def test_dissolved_species_raise() -> None:
    """A dissolved species raises without its gas species, or outside the silicate melt"""
    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=get_solubility_models()["H2O_basalt_dixon95"]
    )
    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")

    with pytest.raises(ValueError, match="H2O_d needs the gas species H2O_g"):
        BasePlanet.from_species((H2_g,), silicate_melt_species=(H2O_d,))
    with pytest.raises(ValueError, match="H2O_d in metal_species"):
        BasePlanet.from_species((H2O_g,), metal_species=(H2O_d,))


def test_duplicate_names_raise() -> None:
    """Two species, or two phases, with the same name raise"""
    with pytest.raises(ValueError, match="Species names must be unique"):
        BasePlanet.from_species((H2O_g, H2O_g))
    with pytest.raises(ValueError, match="Phase names must be unique"):
        graphite: PurePhase = PurePhase.from_component("graphite")
        BasePlanet.from_species((H2O_g,), condensates=(graphite, graphite))
