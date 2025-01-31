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
"""Comparisons with FactSage 8.2 and FastChem 3.1.1"""

# Want to use chemistry symbols so pylint: disable=invalid-name

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
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.thermodata.redox_buffers2 import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_kg

logger: logging.Logger = debug_logger()

TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage"""


def test_H_O(helper) -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint.

    The FastChem element abundance file is:

    # test_H_O from atmodeller
    e-  0.0
    H   12.00
    O   11.40541658
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    o_kg: float = 6.25774e20

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("O", o_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    fastchem_result: dict[str, float] = {
        "H2O_g": 76.45861543,
        "H2_g": 73.84378192,
        "O2_g": 8.91399329e-08,
    }

    system.solve(constraints)
    assert helper.isclose(system, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_reduced(helper) -> None:
    """C-H-O system at IW-2

    Similar to :cite:p:`BHS22{Table E, row 1}`
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 175.5,
        "H2O_g": 13.8,
        "CO_g": 6.21,
        "CO2_g": 0.228,
        "CH4_g": 38.07,
        "O2_g": 1.25e-15,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_IW(helper) -> None:
    """C-H-O system at IW+0.5

    Similar to :cite:p:`BHS22{Table E, row 2}`.

    The FastChem element abundance file is:

    # test_CHO_IW from atmodeller
    e-  0.0
    H   12.00
    O   11.54211516
    C   10.92386535
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
            # FugacityConstraint(O2_g, 1.01633868e-14),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 28.66,
        "CO2_g": 30.88,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "H2_g": 236.98,
        "O2_g": 4.11e-13,
    }

    fastchem_result: dict[str, float] = {
        "CH4_g": 29.61919788,
        "CO2_g": 29.82548282,
        "CO_g": 45.94958264,
        "H2O_g": 332.03616807,
        "H2_g": 236.73845646,
        "O2_g": 3.96475584e-13,
    }

    system.solve(constraints)

    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert helper.isclose(system, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_oxidised(helper) -> None:
    """C-H-O system at IW+2

    Similar to :cite:p:`BHS22{Table E, row 3}`
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 0.1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 0.00129,
        "CO2_g": 3.25,
        "CO_g": 0.873,
        "H2O_g": 218.48,
        "H2_g": 27.40,
        "O2_g": 1.29e-11,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_highly_oxidised(helper) -> None:
    """C-H-O system at IW+4

    Similar to :cite:p:`BHS22{Table E, row 4}`
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 5 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(4)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 7.13e-05,
        "CO2_g": 357.23,
        "CO_g": 10.21,
        "H2O_g": 432.08,
        "H2_g": 5.78,
        "O2_g": 1.14e-09,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_middle_temperature(helper) -> None:
    """C-H-O system at 873 K"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 59.066,
        "H2O_g": 18.320,
        "CO_g": 8.91e-4,
        "CO2_g": 7.48e-4,
        "CH4_g": 19.548,
        "O2_g": 1.27e-25,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_low_temperature(helper) -> None:
    """C-H-O system at 450 K"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 450
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            PressureConstraint(H2O_g, 8),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 55.475,
        "H2O_g": 8.0,
        "CO_g": 2.12e-16,
        "CO2_g": 1.24e-14,
        "CH4_g": 16.037,
        "O2_g": 7.85e-54,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_condensed(helper) -> None:
    """Graphite stable with around 50% condensed C mass fraction"""

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr])

    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 5 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "O2_g": 1.27e-25,
        "H2_g": 14.564,
        "CO_g": 0.07276,
        "H2O_g": 4.527,
        "CO2_g": 0.061195,
        "CH4_g": 96.74,
        "C_cr": 1.0,
        "mass_C_cr": 3.54162e20,
    }

    system.solve(constraints, factor=10)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_unstable(helper) -> None:
    """C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g, C_cr])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "O2_g": 4.11e-13,
        "H2_g": 236.98,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "CO2_g": 30.88,
        "CH4_g": 28.66,
        "C_cr": 0.12202,
        # FactSage also predicts no C, so these values are set close to the atmodeller output so
        # the test knows to pass.
        "mass_C_cr": 1.1e-15,
    }

    system.solve(constraints, factor=1)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_water_condensed(helper) -> None:
    """Condensed water at 10 bar"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")

    species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

    planet: Planet = Planet()
    planet.surface_temperature = 411.75
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=7),
            ElementMassConstraint("H", h_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "H2O_l": 1.0,
        "mass_H2O_l": 1.23802e21,
    }

    system.solve(constraints, factor=10)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_water_condensed_O_abundance(helper) -> None:
    """Condensed water at 10 bar

    This is the same test as above, but this time constraining the total pressure and oxygen
    abundance. This test reveals a discrepancy in the moles of H and O in condensed H2O, which
    do not appear to agree with stoichiometry.
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")

    species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

    planet: Planet = Planet()
    planet.surface_temperature = 411.75
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    o_kg: float = 1.14375e21

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("O", o_kg),
            ElementMassConstraint("H", h_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "H2O_l": 1.0,
        "mass_H2O_l": 1.247201e21,
    }

    system.solve(constraints, factor=10)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_water_condensed_O_abundance(helper) -> None:
    """Condensed water at 10 bar

    This is the same test as above, but this time constraining the total pressure and oxygen
    abundance. This test reveals a discrepancy in the moles of H and O in condensed H2O, which
    do not appear to agree with stoichiometry.
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")

    species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

    planet: Planet = Planet()
    planet.surface_temperature = 411.75
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    o_kg: float = 1.14375e21

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("O", o_kg),
            ElementMassConstraint("H", h_kg),
            TotalPressureConstraint(9.924344608),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "H2O_l": 1.0,
        "degree_of_condensation_H": 0.893755,
    }

    system.solve(constraints)
    system.output(to_excel=True)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_water_condensed(helper, graphite_water_condensed) -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    system = graphite_water_condensed
    # system.output(to_excel=True)

    factsage_result: dict[str, float] = {
        "CH4_g": 0.3241,
        "CO2_g": 4.3064,
        "CO_g": 2.77e-6,
        "C_cr": 1.0,
        "H2O_g": 5.3672,
        "H2O_l": 1.0,
        "H2_g": 0.0023,
        "O2_g": 4.74e-48,
        "mass_C_cr": 8.75101e19,
        "mass_H2O_l": 2.74821e21,
    }

    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
