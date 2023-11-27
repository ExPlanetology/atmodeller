"""Tests for the Holland and Powell EOS models

See the LICENSE file for licensing information.
"""

import logging
from typing import Type

import numpy as np

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.eos.holland import (
    CO2_CORK_simple_HP91,
    CO2_MRK_simple_HP91,
    get_holland_eos_models,
)
from atmodeller.interfaces import (
    GasSpecies,
    IdealGas,
    NoSolubility,
    RealGasABC,
    ThermodynamicData,
    ThermodynamicDataBase,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltDixonCO2, BasaltH2, PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

logger: logging.Logger = debug_logger()

thermodynamic_data: Type[ThermodynamicDataBase] = ThermodynamicData

eos_models: dict[str, RealGasABC] = get_holland_eos_models()

rtol: float = 1.0e-8
atol: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_MRKCO2(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, CO2_MRK_simple_HP91, 9.80535714428564)


def test_CorkH2(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, eos_models["H2"], 4.672042007568433)


def test_CorkCO(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, eos_models["CO"], 7.737070657107842)


def test_CorkCH4(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, eos_models["CH4"], 8.013532244610671)


def test_simple_CorkCO2(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, CO2_CORK_simple_HP91, 7.120242298956865)


def test_CorkCO2_at_P0(check_values) -> None:
    """Below P0 so virial contribution excluded."""
    check_values.fugacity_coefficient(2000, 2e3, eos_models["CO2"], 1.6063624424808558)


def test_CorkCO2_above_P0(check_values) -> None:
    """Above P0 so virial contribution included."""
    check_values.fugacity_coefficient(2000, 10e3, eos_models["CO2"], 7.4492345831832525)


def test_CorkH2O_above_Tc_below_P0(check_values) -> None:
    """Above Tc and below P0."""
    check_values.fugacity_coefficient(2000, 1e3, eos_models["H2O"], 1.048278616058322)


def test_CorkH2O_above_Tc_above_P0(check_values) -> None:
    """Above Tc and above P0."""
    check_values.fugacity_coefficient(2000, 5e3, eos_models["H2O"], 1.3444013638026706)


def test_CorkH2O_below_Tc_below_Psat(check_values) -> None:
    """Below Tc and below Psat."""
    # Psat = 0.118224 kbar at T = 600 K.
    check_values.fugacity_coefficient(600, 0.1e3, eos_models["H2O"], 0.7910907770688191)


def test_CorkH2O_below_Tc_above_Psat(check_values) -> None:
    """Below Tc and above Psat."""
    # Psat = 0.118224 kbar at T = 600 K.
    check_values.fugacity_coefficient(600, 1e3, eos_models["H2O"], 0.13704706029361396)


def test_CorkH2O_below_Tc_above_P0(check_values) -> None:
    """Below Tc and above P0."""
    check_values.fugacity_coefficient(600, 10e3, eos_models["H2O"], 0.39074941260585533)


def test_H2_with_cork() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_class=thermodynamic_data,
                eos=IdealGas(),  # This is the default if nothing specified
            ),
            GasSpecies(
                chemical_formula="H2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                eos=eos_models["H2"],
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                eos=IdealGas(),  # This is the default if nothing specified
            ),
        ]
    )

    # oceans: float = 1
    planet: Planet = Planet(surface_temperature=2000)
    # h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=1e3),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 747.5737656770727,
        "H2O": 1072.4328856736947,
        "O2": 9.76211086495026e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures)


def test_non_ideal() -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2",
                solubility=BasaltH2(),
                thermodynamic_class=thermodynamic_data,
                eos=eos_models["H2"],
            ),
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_class=thermodynamic_data,
                eos=eos_models["H2O"],
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
            ),
            GasSpecies(
                chemical_formula="CO",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                eos=eos_models["CO"],
            ),
            GasSpecies(
                chemical_formula="CO2",
                solubility=BasaltDixonCO2(),
                thermodynamic_class=thermodynamic_data,
                eos=eos_models["CO2"],
            ),
            GasSpecies(
                chemical_formula="CH4",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                eos=eos_models["CH4"],
            ),
        ]
    )

    oceans: float = 10
    planet: Planet = Planet()
    planet.surface_temperature = 2000
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=958),
            IronWustiteBufferConstraintHirschmann(),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CH4": 10.402516906435752,
        "CO": 275.92114750956955,
        "CO2": 64.23450582493761,
        "H2": 696.9742997262443,
        "H2O": 933.3320305098492,
        "O2": 9.862052864392796e-08,
    }

    # Initial solution (i.e. estimates) must correspond by position to the order in the species
    # You don't actually need to specify initial estimates for this test to find a solution, but
    # it is here to show the user how to implement them if desired. Remember that the initial
    initial_solution: np.ndarray = np.array([1000, 1000, 1e-7, 100, 10, 1])

    system.solve(constraints, factor=1, initial_solution=initial_solution)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)
