"""Tests for the Holland and Powell EOS models

See the LICENSE file for licensing information.
"""

import logging

import numpy as np

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.eos.holland import (
    CH4_CORK_HP91,
    CO2_CORK_HP91,
    CO_CORK_HP91,
    H2_CORK_HP91,
    H2O_CORK_HP91,
    CO2_CORK_simple_HP91,
    CO2_MRK_simple_HP91,
    get_holland_eos_models,
)
from atmodeller.interfaces import (
    GasSpecies,
    IdealGas,
    NoSolubility,
    RealGasABC,
    ThermodynamicDataset,
    ThermodynamicDatasetABC,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltDixonCO2, BasaltH2, PeridotiteH2O
from atmodeller.utilities import UnitConversion, earth_oceans_to_kg

logger: logging.Logger = debug_logger()

thermodynamic_dataset: ThermodynamicDatasetABC = ThermodynamicDataset()

eos_models: dict[str, RealGasABC] = get_holland_eos_models()

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# Figure 2a, Holland and Powell (1991)
def test_CORK_H2O_volume_1kbar(check_values) -> None:
    expected: float = 47.502083040419844
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(873, 1000, H2O_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


# Figure 7, Holland and Powell (1991)
def test_CORK_CO2_volume_1kbar(check_values) -> None:
    expected: float = 96.13326116472262
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(873, 1000, CO2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


# Figure 8a, Holland and Powell (1991)
def test_CORK_CO_volume_1kbar(check_values) -> None:
    expected: float = 131.475184896045
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1173, 1000, CO_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CO_volume_2kbar(check_values) -> None:
    expected: float = 71.32153159834933
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(973, 2000, CO_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CO_volume_4kbar(check_values) -> None:
    expected: float = 62.22167162862537
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1473, 4000, CO_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


# Figure 8b, Holland and Powell (1991)
def test_CORK_CH4_volume_1kbar(check_values) -> None:
    expected: float = 131.6743085645421
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1173, 1000, CH4_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CH4_volume_2kbar(check_values) -> None:
    expected: float = 72.14376119913776
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(973, 2000, CH4_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CH4_volume_4kbar(check_values) -> None:
    expected: float = 63.106094264549
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1473, 4000, CH4_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


# Figure 8c, Holland and Powell (1991)
def test_CORK_H2_volume_500bar(check_values) -> None:
    expected: float = 149.1657987388235
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(773, 500, H2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_H2_volume_1800bar(check_values) -> None:
    expected: float = 55.04174839002075
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(773, 1800, H2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_2_volume_10kb(check_values) -> None:
    expected: float = 20.67497630046999
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(773, 10000, H2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_MRKCO2(check_values) -> None:
    check_values.fugacity_coefficient(
        2000, 10e3, CO2_MRK_simple_HP91, 9.80535714428564, rtol=RTOL, atol=ATOL
    )


def test_CorkH2(check_values) -> None:
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["H2"], 4.672042007568433, rtol=RTOL, atol=ATOL
    )


def test_CorkCO(check_values) -> None:
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["CO"], 7.737070657107842, rtol=RTOL, atol=ATOL
    )


def test_CorkCH4(check_values) -> None:
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["CH4"], 8.013532244610671, rtol=RTOL, atol=ATOL
    )


def test_simple_CorkCO2(check_values) -> None:
    check_values.fugacity_coefficient(
        2000, 10e3, CO2_CORK_simple_HP91, 7.120242298956865, rtol=RTOL, atol=ATOL
    )


def test_CorkCO2_at_P0(check_values) -> None:
    """Below P0 so virial contribution excluded."""
    check_values.fugacity_coefficient(
        2000, 2e3, eos_models["CO2"], 1.5754570751655304, rtol=RTOL, atol=ATOL
    )


def test_CorkCO2_above_P0(check_values) -> None:
    """Above P0 so virial contribution included."""
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["CO2"], 7.144759853226838, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_above_Tc_below_P0(check_values) -> None:
    """Above Tc and below P0."""
    check_values.fugacity_coefficient(
        2000, 1e3, eos_models["H2O"], 1.048278616058322, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_above_Tc_above_P0(check_values) -> None:
    """Above Tc and above P0."""
    check_values.fugacity_coefficient(
        2000, 5e3, eos_models["H2O"], 1.3444013638026706, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_below_Tc_below_Psat(check_values) -> None:
    """Below Tc and below Psat."""
    # Psat = 0.118224 kbar at T = 600 K.
    check_values.fugacity_coefficient(
        600, 0.1e3, eos_models["H2O"], 0.7910907770688191, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_below_Tc_above_Psat(check_values) -> None:
    """Below Tc and above Psat."""
    # Psat = 0.118224 kbar at T = 600 K.
    check_values.fugacity_coefficient(
        600, 1e3, eos_models["H2O"], 0.13704706029361396, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_below_Tc_above_P0(check_values) -> None:
    """Below Tc and above P0."""
    check_values.fugacity_coefficient(
        600, 10e3, eos_models["H2O"], 0.39074941260585533, rtol=RTOL, atol=ATOL
    )


def test_H2_with_cork() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_dataset=thermodynamic_dataset,
                eos=IdealGas(),  # This is the default if nothing specified
            ),
            GasSpecies(
                chemical_formula="H2",
                solubility=NoSolubility(),
                thermodynamic_dataset=thermodynamic_dataset,
                eos=eos_models["H2"],
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_dataset=thermodynamic_dataset,
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

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_non_ideal() -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2",
                solubility=BasaltH2(),
                thermodynamic_dataset=thermodynamic_dataset,
                eos=eos_models["H2"],
            ),
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_dataset=thermodynamic_dataset,
                eos=eos_models["H2O"],
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_dataset=thermodynamic_dataset,
            ),
            GasSpecies(
                chemical_formula="CO",
                solubility=NoSolubility(),
                thermodynamic_dataset=thermodynamic_dataset,
                eos=eos_models["CO"],
            ),
            GasSpecies(
                chemical_formula="CO2",
                solubility=BasaltDixonCO2(),
                thermodynamic_dataset=thermodynamic_dataset,
                eos=eos_models["CO2"],
            ),
            GasSpecies(
                chemical_formula="CH4",
                solubility=NoSolubility(),
                thermodynamic_dataset=thermodynamic_dataset,
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
        "CH4": 10.374185001822335,
        "CO": 275.1721430723753,
        "CO2": 65.30951836095214,
        "H2": 696.9435961399346,
        "H2O": 933.3255567030764,
        "O2": 9.862215541051231e-08,
    }

    # Initial solution (i.e. estimates) must correspond by position to the order in the species
    # You don't actually need to specify initial estimates for this test to find a solution, but
    # it is here to show the user how to implement them if desired. Remember that the initial
    initial_solution: np.ndarray = np.array([1000, 1000, 1e-7, 100, 10, 1])

    system.solve(constraints, factor=1, initial_solution=initial_solution)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)
