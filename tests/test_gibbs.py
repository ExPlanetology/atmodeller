"""Tests for simple CHO interior-atmosphere systems.

See the LICENSE file for licensing information.

Tests using the JANAF data for simple CHO interior-atmosphere systems.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.interfaces import GasSpecies, NoSolubility
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltDixonCO2, PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

rtol: float = 1.0e-8
atol: float = 1.0e-8

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 0.3822360385284366,
        "H2O": 0.39052430401952754,
        "O2": 8.699910873114417e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_basalt_melt() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet(melt_composition="basalt")
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 0.09234539760208682,
        "H2O": 0.09434603056360964,
        "O2": 8.699588020866791e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_fO2_plus() -> None:
    """Tests H2-H2O at the IW buffer+2."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 0.03848650454730284,
        "H2O": 0.3932060714737498,
        "O2": 8.6997229898027e-06,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_fO2_minus() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 3.333411504869784,
        "H2O": 0.34060048172302065,
        "O2": 8.701509284136011e-10,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 5
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 9.176991833746184,
        "H2O": 9.381263432996723,
        "O2": 8.70971420519968e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_1500K() -> None:
    """Tests H2-H2O at a different temperature."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    planet.surface_temperature = 1500.0  # K

    target_pressures: dict[str, float] = {
        "H2": 0.4653066317980525,
        "H2O": 0.38971035064383824,
        "O2": 2.500733096628671e-12,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 59.520234541503754,
        "CO2": 13.368489839161574,
        "H2": 0.3840114011180472,
        "H2O": 0.39324466188613644,
        "O2": 8.740159677986617e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_H_and_C_total_pressure() -> None:
    """Tests H2-H2O and CO-CO2 with a total pressure constraint."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
            TotalPressureConstraint(species="None", value=100),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 81.01220583060697,
        "CO2": 18.210852074534948,
        "H2": 0.3836944531352654,
        "H2O": 0.3932478266315712,
        "O2": 8.754746041914438e-08,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)
