"""Tests with sulfur.

See the LICENSE file for licensing information.

Tests using the JANAF data for simple interior-atmosphere systems that include sulfur.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.interfaces import GasSpecies, NoSolubility
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import (
    BasaltDixonCO2,
    BasaltDixonH2O,
    BasaltH2,
    BasaltLibourelN2,
    BasaltS2,
    BasaltS2_Sulfate,
    BasaltS2_Sulfide,
)

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

rtol: float = 1.0e-8
atol: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_S2_SO_Sulfide_IW() -> None:
    """Tests S2 and SO Sulfide Solubilities at the IW buffer, 2000 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2_Sulfide()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet()
    S2_fugacity: float = 1e-5
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="S2", value=S2_fugacity),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 1e-05,
        "OS": 6.018454943818516e-05,
        "O2": 8.699485217915599e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_Sulfide_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2_Sulfide()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.003704873878513222,
        "OS": 0.002991983415707734,
        "O2S": 0.004839725408460836,
        "O2": 1.0269757432683765e-06,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_Sulfate_IW() -> None:
    """Tests SO, S2, and SO2 Sulfate Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2_Sulfate()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 2e-4 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "OS": 0.6267850732546694,
        "S2": 161.05514086015452,
        "O2": 1.0367593216540548e-06,
        "O2S": 1.0186830187658191,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_TotalSolubility_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.0037048732091408782,
        "OS": 0.00299198314542162,
        "O2S": 0.0048397249712554885,
        "O2": 1.0269757432682946e-06,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_TotalSolubility_IWp3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW+3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(log10_shift=3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 1.466040005579648,
        "OS": 1.8876140158988506,
        "O2S": 96.83727235805357,
        "O2": 0.0010329892821750164,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_TotalSolubility_IWm3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW-3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(log10_shift=-3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 3.7052440315568952e-06,
        "OS": 2.9921318707665795e-06,
        "O2S": 1.5305309773280557e-07,
        "O2": 1.0269750531288816e-09,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_HOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, O and S species at IW, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="H", value=mass_H),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.9880831835505409,
        "H2": 0.9394385103704318,
        "OS": 0.002991611571529395,
        "S2": 0.0037035369886239495,
        "O2": 1.0270911157002842e-06,
        "O2S": 0.004839395737226402,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHONS_Species_IW_MixConstraints() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 2173 K,
    Mix of Fugacity and Mass Constraints."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
            GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    S2_fugacity: float = 5e-3
    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_N: float = 0.0000028 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="S2", value=S2_fugacity),
            MassConstraint(species="H", value=mass_H),
            MassConstraint(species="C", value=mass_C),
            MassConstraint(species="N", value=mass_N),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.9938673900925166,
        "H2": 0.9372852480839629,
        "OS": 0.0035043975980956635,
        "S2": 0.004999999999999999,
        "O2": 1.0439314797530534e-06,
        "O2S": 0.005715191906614295,
        "CO": 229.4990247811397,
        "CO2": 47.19901534042048,
        "N2": 2.3491951394429105,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_COS_Species_IW() -> None:
    """Tests Sulfur Solubility with C, O and S species at IW, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="C", value=mass_C),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "OS": 0.003040722714933701,
        "S2": 0.0037650388586196962,
        "O2": 1.0437570555403361e-06,
        "O2S": 0.004958587456124795,
        "CO": 230.67647327936177,
        "CO2": 47.437207165120746,
    }

    # initial_solution: np.ndarray = np.array([0.003, 0.003, 1e-6, 0.005, 230, 47])
    system.solve(constraints, factor=1)  # , initial_solution=initial_solution)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW-3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="H", value=mass_H),
            MassConstraint(species="C", value=mass_C),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.9938624694394833,
        "H2": 0.9373397768144074,
        "OS": 0.0030408454906003597,
        "S2": 0.0037651891167604728,
        "O2": 1.04379968829635e-06,
        "O2S": 0.0049588889406645496,
        "CO": 229.65516731392094,
        "CO2": 47.22814633265304,
    }

    # initial_solution: np.ndarray = np.array([1, 1, 1e-3, 1e-3, 1e-6, 1e-3, 1e2, 1])
    system.solve(constraints, factor=1)  # , initial_solution=initial_solution)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)
