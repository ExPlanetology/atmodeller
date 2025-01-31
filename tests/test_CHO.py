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
"""Tests for ideal C-H-O interior-atmosphere systems"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    PressureConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies
from atmodeller.eos.holland import get_holland_eos_models
from atmodeller.eos.interfaces import RealGas
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
from atmodeller.solubility.hydrogen_species import (
    H2_basalt_hirschmann,
    H2O_peridotite_sossi,
)
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.thermodata.redox_buffers2 import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_holland: dict[str, RealGas] = get_holland_eos_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_H2O() -> None:
    """Tests H2O (a single species)"""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())

    species: Species = Species([H2O_g])

    oceans: float = 2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 1.0312913336898137,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.2570770067190733,
        "H2_g": 0.24964688044710354,
        "O2_g": 8.838043080858959e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_holland() -> None:
    """Tests H2-H2O at the IW buffer using thermodynamic data from Holland and Powell"""

    H2O_g: GasSpecies = GasSpecies(
        "H2O",
        solubility=H2O_peridotite_sossi(),
        thermodata_dataset=ThermodynamicDatasetHollandAndPowell(),
    )
    H2_g: GasSpecies = GasSpecies(
        "H2",
        thermodata_dataset=ThermodynamicDatasetHollandAndPowell(),
    )
    O2_g: GasSpecies = GasSpecies(
        "O2",
        thermodata_dataset=ThermodynamicDatasetHollandAndPowell(),
    )

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25706291267455456,
        "H2_g": 0.2519202361466346,
        "O2_g": 8.838045400824612e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_basalt_melt() -> None:
    """Tests H2-H2O at the IW buffer."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet(melt_composition="basalt")
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.09442361772602827,
        "H2_g": 0.0916964417272344,
        "O2_g": 8.837679290584522e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_plus() -> None:
    """Tests H2-H2O at the IW buffer+2.

    A comparable test has been run for FastChem, by tweaking the oxygen logarithmic abundance from
    11.3067 (this test) to 11.65854 to ensure that the oxygen fugacity (8.7E-6) is the same to
    allow fair comparison. The FastChem element abundance file is (copy-paste):

    # test_H_fO2_plus from atmodeller
    e-  0.0
    H   12.00
    O   11.65854
    """

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(2)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25822630157632576,
        "H2_g": 0.025076641442856793,
        "O2_g": 8.837799444728465e-06,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_minus() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.23441348219541092,
        "H2_g": 2.2761616637646966,
        "O2_g": 8.839768586501877e-10,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 5
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 6.259516638093527,
        "H2_g": 6.075604219756987,
        "O2_g": 8.846766776792243e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_1500K() -> None:
    """Tests H2-H2O at a different temperature."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    planet.surface_temperature = 1500.0  # K

    target: dict[str, float] = {
        "H2O_g": 0.25666635568842355,
        "H2_g": 0.31320683835217444,
        "O2_g": 2.3940728554564946e-12,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "CO2_g": 13.500258901609417,
        "CO_g": 59.61099658675477,
        "H2O_g": 0.25824632142741644,
        "H2_g": 0.2501021517329365,
        "O2_g": 8.886185271201372e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C_total_pressure() -> None:
    """Tests H2-H2O and CO-CO2 with a total pressure constraint."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint(element="H", value=h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            TotalPressureConstraint(value=100),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "CO2_g": 18.38547187555166,
        "CO_g": 81.10641004993212,
        "H2O_g": 0.25824733320813686,
        "H2_g": 0.24987064773894369,
        "O2_g": 8.90272867718254e-08,
    }

    system.solve(constraints, factor=1)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_pH2_fO2_real_gas() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the partial pressure of H2.
    """

    H2O_g: GasSpecies = GasSpecies(
        "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
    )
    H2_g: GasSpecies = GasSpecies("H2", eos=eos_holland["H2"])
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            PressureConstraint(H2_g, value=1000),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 1466.9613852210507,
        "H2_g": 1000.0,
        "O2_g": 1.0453574209588085e-07,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_fH2_fO2_real_gas() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """

    H2O_g: GasSpecies = GasSpecies(
        "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
    )
    H2_g: GasSpecies = GasSpecies("H2", eos=eos_holland["H2"])
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=1000),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 1001.131462103614,
        "H2_g": 755.5960144468955,
        "O2_g": 9.96495147231471e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C_real_gas() -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann(), eos=eos_holland["H2"])
    H2O_g: GasSpecies = GasSpecies(
        "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
    )
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies(formula="CO", eos=eos_holland["CO"])
    CO2_g: GasSpecies = GasSpecies(
        formula="CO2", solubility=CO2_basalt_dixon(), eos=eos_holland["CO2"]
    )
    CH4_g: GasSpecies = GasSpecies(formula="CH4", eos=eos_holland["CH4"])

    species: Species = Species([H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g])

    oceans: float = 10
    planet: Planet = Planet()
    planet.surface_temperature = 2000
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=958),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("C", value=c_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "CH4_g": 10.300421855316944,
        "CO2_g": 67.62567996145735,
        "CO_g": 277.9018480265112,
        "H2O_g": 953.4324527154502,
        "H2_g": 694.3036008172556,
        "O2_g": 1.0132255325169718e-07,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)
