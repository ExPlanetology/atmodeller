# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for atmodeller.state"""

import logging

import pytest

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies
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


def test_base_planet_update() -> None:
    """Tests BasePlanet.update with planet_mass and core_mass_fraction kwargs."""
    planet: BasePlanet = PressureScalingLawPlanet.from_species((H2O_g,), temperature=1000.0)

    new_mass: float = 6.0e24
    new_core_mass_fraction: float = 0.3
    updated_planet: BasePlanet = planet.update(
        planet_mass=new_mass, core_mass_fraction=new_core_mass_fraction
    )

    assert float(updated_planet.background_planet_mass) == pytest.approx(new_mass)
    assert float(updated_planet.background_metallic_core_mass) == pytest.approx(
        new_mass * new_core_mass_fraction
    )
