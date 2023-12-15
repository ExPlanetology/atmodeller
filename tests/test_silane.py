"""Tests for simple SiHO interior-atmosphere systems

See the LICENSE file for licensing information.

Tests using the JANAF data for simple SiHO interior-atmosphere systems.
"""

import logging

from atmodeller import __version__, debug_file_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.eos.holland import get_holland_eos_models
from atmodeller.interfaces import GasSpecies, LiquidSpecies, NoSolubility, RealGasABC
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltH2, PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

eos_models: dict[str, RealGasABC] = get_holland_eos_models()

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_file_logger()
# logger.setLevel(logging.INFO)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_Si_O_H_gas_mass() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    oceans: float = 1
    sih_ratio: float = 8.41
    h_kg: float = earth_oceans_to_kg(oceans)
    si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="Si", value=si_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 251.71684790878402,
        "H2O": 26.69334671405076,
        "H4Si": 0.004941388093389984,
        "O2": 0.0002521142481275591,
        "OSi": 168.06091497713342,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_Si_O_H_gas_liquid_fugacity() -> None:
    """Tests H2-H2O and SiO2-SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O"),
            GasSpecies(chemical_formula="H2"),
            GasSpecies(chemical_formula="O2"),
            GasSpecies(chemical_formula="OSi"),
            GasSpecies(chemical_formula="H4Si"),
            LiquidSpecies(
                chemical_formula="O2Si",
                name_in_thermodynamic_data="O2Si(l)",
            ),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2O", value=26.69334671405076),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 251.71644802599758,
        "H2O": 26.693346714050723,
        "H4Si": 0.004943884250250349,
        "O2": 0.00025211504915767703,
        "O2Si": 1.0,
        "OSi": 168.14661280978274,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_Si_O_H_gas_liquid_totalpressure() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
            LiquidSpecies(
                chemical_formula="O2Si",
                name_in_thermodynamic_data="O2Si(l)",
            ),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    constraints: SystemConstraints = SystemConstraints(
        [
            TotalPressureConstraint(value=446.5616035491304),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 251.7164480259466,
        "H2O": 26.693346714045386,
        "H4Si": 0.004943884250248337,
        "O2": 0.00025211504915767833,
        "O2Si": 1.0,
        "OSi": 168.14661280978257,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_Si_O_H_gas_liquid_mixed_nonideality() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=NoSolubility(), eos=eos_models["H2O"]),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility(), eos=eos_models["H2"]),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
            LiquidSpecies(
                chemical_formula="O2Si",
                name_in_thermodynamic_data="O2Si(l)",
            ),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    oceans: float = 1
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 251.20543518593024,
        "H2O": 27.537256752879383,
        "H4Si": 0.0053801913454906555,
        "O2": 0.0002521181696420575,
        "O2Si": 1.0,
        "OSi": 168.14557222531593,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_Si_O_H_gas_liquid_mixed_solubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
            LiquidSpecies(
                chemical_formula="O2Si",
                name_in_thermodynamic_data="O2Si(l)",
            ),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    oceans: float = 1
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 1.0350426722398558,
        "H2O": 0.10919784748492806,
        "H4Si": 8.445645240340669e-08,
        "O2": 0.0002495327483804572,
        "O2Si": 1.0,
        "OSi": 169.01440984074077,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_Si_O_H_gas_liquid_mixed_solubility_nonideality() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O(), eos=eos_models["H2O"]),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2(), eos=eos_models["H2"]),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
            LiquidSpecies(
                chemical_formula="O2Si",
                name_in_thermodynamic_data="O2Si(l)",
            ),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    oceans: float = 1
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 1.0194633278892928,
        "H2O": 0.10874845128110579,
        "H4Si": 8.446481184634864e-08,
        "O2": 0.00024953259986850617,
        "O2Si": 1.0,
        "OSi": 169.0144601360852,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    # test_Si_O_H_gas_mass()
    # test_Si_O_H_gas_liquid_fugacity()
    # test_Si_O_H_gas_liquid_totalpressure()
    test_Si_O_H_gas_liquid_mixed_nonideality()
    # test_Si_O_H_gas_liquid_mixed_solubility()
    # test_Si_O_H_gas_liquid_mixed_solubility_nonideality()
