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
"""Tests with sulfur"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
from atmodeller.solubility.hydrogen_species import (
    H2_basalt_hirschmann,
    H2O_basalt_dixon,
)
from atmodeller.solubility.other_species import N2_basalt_libourel
from atmodeller.solubility.sulfur_species import (
    S2_basalt_boulliung,
    S2_sulfate_basalt_boulliung,
    S2_sulfide_basalt_boulliung,
)
from atmodeller.thermodata.redox_buffers2 import IronWustiteBuffer

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_S2_SO_Sulfide_IW() -> None:
    """Tests S2 and SO Sulfide Solubilities at the IW buffer, 2000 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_sulfide_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([OS_g, S2_g, O2_g])

    planet: Planet = Planet()
    S2_fugacity: float = 1e-5
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(S2_g, S2_fugacity),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "O2_g": 8.837305999112288e-08,
        "OS_g": 6.065941014324351e-05,
        "S2_g": 1.0000000000000021e-05,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_Sulfide_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide Solubility at the IW buffer, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_sulfide_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "O2S_g": 0.0051503430108147475,
        "O2_g": 1.0704618217953718e-06,
        "OS_g": 0.003118667795320716,
        "S2_g": 0.0038617334921464104,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_Sulfate_IW() -> None:
    """Tests SO, S2, and SO2 Sulfate Solubility at the IW buffer, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_sulfate_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 2e-4 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "O2S_g": 1.063284666606797,
        "O2_g": 1.0822622113773915e-06,
        "OS_g": 0.6403270621279517,
        "S2_g": 161.02246440252563,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at the IW buffer, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "O2S_g": 0.005150342505310183,
        "O2_g": 1.07046182179468e-06,
        "OS_g": 0.0031186674892254495,
        "S2_g": 0.003861732734095561,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IWp3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW+3, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(3)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "O2S_g": 100.69672945635124,
        "O2_g": 0.0010781611515592554,
        "OS_g": 1.9212862681250629,
        "S2_g": 1.4551766876176104,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IWm3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW-3, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-3)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "O2S_g": 1.6287452019154868e-07,
        "O2_g": 1.0704521154076894e-09,
        "OS_g": 3.118803982220825e-06,
        "S2_g": 3.862105789686752e-06,
    }

    system.solve(constraints, factor=1)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_HOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, O and S species at IW, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_basalt_dixon())
    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann())

    species: Species = Species([H2O_g, H2_g, OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            ElementMassConstraint("H", mass_H),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.9903915306258877,
        "H2_g": 0.9222606922132459,
        "O2S_g": 0.00515074947616005,
        "O2_g": 1.0706933686571881e-06,
        "OS_g": 0.003118576656225403,
        "S2_g": 0.0038606727019204617,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_CHONS_Species_IW_MixConstraints() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 2173 K,
    Mix of Fugacity and Mass Constraints."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_basalt_dixon())
    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann())
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())
    N2_g: GasSpecies = GasSpecies("N2", solubility=N2_basalt_libourel())

    species: Species = Species([H2O_g, H2_g, OS_g, S2_g, O2_g, O2S_g, CO_g, CO2_g, N2_g])

    planet: Planet = Planet(surface_temperature=2173)
    S2_fugacity: float = 5e-3
    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_N: float = 0.0000028 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(S2_g, S2_fugacity),
            ElementMassConstraint("H", mass_H),
            ElementMassConstraint("C", mass_C),
            ElementMassConstraint("N", mass_N),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "CO2_g": 48.16030372489886,
        "CO_g": 229.1163201944724,
        "H2O_g": 0.9961865678678682,
        "H2_g": 0.9191849917932919,
        "N2_g": 2.3529190100178394,
        "O2S_g": 0.0059702568124798475,
        "O2_g": 1.090521391161828e-06,
        "OS_g": 0.0035817434180847162,
        "S2_g": 0.004999999999999994,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_COS_Species_IW() -> None:
    """Tests Sulfur Solubility with C, O and S species at IW, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g, CO_g, CO2_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            ElementMassConstraint("C", mass_C),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "CO2_g": 48.40004714798208,
        "CO_g": 230.27789863253048,
        "O2S_g": 0.005294061041603201,
        "O2_g": 1.0903222121820412e-06,
        "OS_g": 0.0031763625559195945,
        "S2_g": 0.003932969025751321,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_CHOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW-3, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_basalt_dixon())
    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann())
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, OS_g, S2_g, O2_g, O2S_g, CO_g, CO2_g])

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            ElementMassConstraint("H", mass_H),
            ElementMassConstraint("C", mass_C),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "CO2_g": 48.18987265370282,
        "CO_g": 229.2728192546971,
        "H2O_g": 0.9961810965321554,
        "H2_g": 0.9192434070113206,
        "O2S_g": 0.0052944124929951136,
        "O2_g": 1.0903708193614053e-06,
        "OS_g": 0.0031765026174872956,
        "S2_g": 0.003933140539967806,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)
