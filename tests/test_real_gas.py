# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for systems with real gases"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import (
    ActivityConstraintProtocol,
    ActivityProtocol,
    SolubilityProtocol,
    SpeciesProtocol,
)
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()
eos_models: Mapping[str, ActivityProtocol] = get_eos_models()

# Gas Species
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_models["H2_chabrier21"])
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
SiO_g: ChemicalSpecies = ChemicalSpecies.create_gas("OSi")
H4Si_g: ChemicalSpecies = ChemicalSpecies.create_gas("H4Si")
gas_species_subneptune: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g, SiO_g, H4Si_g)

# Force SiO2 to have unity activity, as per previous tests. Eventually can be relaxed to allow for
# a self-consistent activity, but this is fine for testing the real gas EOS.
O2Si_l: PurePhase = PurePhase.from_species("O2Si", state="l")
condensates_subneptune: tuple[PurePhase, ...] = (O2Si_l,)


def test_fO2_holley() -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`"""

    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_beattie_holley58"]
    )
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)

    # Temperature is within the range of the Holley model
    planet: BaseThermodynamicState = Planet.from_species(gas_species, temperature=1000)

    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 32.77037875523393,
                    "H2_g": 71.50338102110962,
                    "O2_g": 1.525466019972294e-21,
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    planet: Planet = Planet.from_species(
        gas_species_subneptune, temperature=3400, condensates=condensates_subneptune
    )
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = (
        0.1459 * planet.background_planet_mass
    )  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 7250.890001201146,
                    "H2_g": 11625.21660262108,
                    "H4Si_g": 67590.85056516326,
                    "O2_g": 1.791165364818965e-05,
                    "OSi_g": 625.5604897933299,
                },
                "activity": {
                    "H2O_g": 7250.89000120113,
                    "H2_g": 251640.8508698987,
                    "H4Si_g": 67590.85056516311,
                    "O2_g": 1.791165364818961e-05,
                    "OSi_g": 625.5604897933285,
                },
            }
        },
        "condensates": {"activity": {"O2Si_l": 1.0}},
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune

    This case effectively saturates the maximum allowable log number density at a value of 70
    based on the default hypercube that brackets the solution (see LOG_NUMBER_MOLES_UPPER).
    This is fine for a test, but this test is not physically realistic because solubilities are
    ignored, which would greatly lower the pressure and hence the number density.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet.from_species(
        gas_species_subneptune,
        condensates=condensates_subneptune,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = (
        0.1459 * planet.background_planet_mass
    )  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 429507.1704491525,
                    "H2_g": 2.927180242775929,
                    "H4Si_g": 0.000704240746855,
                    "O2_g": 10.394369851620414,
                    "OSi_g": 0.821179444164847,
                },
                "activity": {
                    "H2O_g": 429507.1704491535,
                    "H2_g": 19567.21447690322,
                    "H4Si_g": 0.000704240746855,
                    "O2_g": 10.394369851620437,
                    "OSi_g": 0.821179444164849,
                },
            }
        },
        "condensates": {"activity": {"O2Si_l": 1.0}},
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune_batch() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet.from_species(
        gas_species_subneptune,
        condensates=condensates_subneptune,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = (
        0.1459 * planet.background_planet_mass
    )  # Si = 14.59 wt% Kargel & Lewis (1993)
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    # output.to_excel(file_prefix="test_chabrier_subNeptune_batch", output_format="named_arrays")

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [4.477789711513712e05, 4.785890592398898e05, 5.039107471956282e05]
                    ),
                    "H2_g": np.array(
                        [3.463824822645956e-02, 7.208115634579626e-03, 2.129125602157067e-03]
                    ),
                    "O2_g": np.array(
                        [2.597033179470946e04, 8.263153509596182e04, 1.447811285078976e05]
                    ),
                },
                "activity": {
                    "H2_g": np.array(
                        [4.081150539627139e02, 2.445386584856476e02, 1.945159917637966e02]
                    )
                },
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_pH2_fO2_real_gas() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2O", activity=eos_models["H2O_cork_holland98"]
    )
    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_cork_cs_holland91"]
    )
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g)
    melt_species: tuple[ReservoirSpecies, ...] = (H2O_d,)

    planet: Planet = Planet.from_species(gas_species, silicate_melt_species=melt_species)

    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(0.072885576196744)
    }

    mass_constraints: dict[str, ArrayLike] = {"H": 1.47126255324872e22}

    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 1470.2567650857518,
                    "H2_g": 999.9971214963639,
                    "O2_g": 1.045357420958815e-07,
                }
            }
        }
    }

    # output.to_excel("pH2_fO2_real_gas")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_subNeptune_melt_phase() -> None:
    """Tests a more realistic sub-Neptune with computed melt-phase activities.

    This is an extension of the model setup in :cite:t:`Hakim2026`.

    The melt phase consists of a chemically-reactive component SiO2(l) and a dissolved component
    H2O(l). Here, the activities of both species in the melt phase are calculated
    self-consistently.
    """

    # The species we specify in the melt should be considered as already included in the
    # "background" melt mass, so we set include_in_phase_mass=False for both species
    O2Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed(
        "O2Si", state="l", include_in_phase_mass=False
    )
    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )
    melt_species: tuple[SpeciesProtocol, ...] = (O2Si_l, H2O_d)

    # Temperature must be compatible with the choice of species, i.e. chemically-reactive species
    # must have thermodynamic data available at the specified temperature. Here, we are limited by
    # O2Si(l).
    surface_temperature = 3400  # K
    planet_mass = 4.6 * earth.mass  # kg
    surface_radius = 1.5 * earth.radius  # m

    planet: Planet = Planet.from_species(
        gas_species_subneptune,
        silicate_melt_species=melt_species,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )

    # The previous mass constraints are still OK, because we are not allowing the melt species to
    # contribute additionally to the planet mass. So these calculations are still exact.
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = (
        0.1459 * planet.background_planet_mass
    )  # Si = 14.59 wt% Kargel & Lewis (1993)
    # o_kg: ArrayLike = 6.74717e24
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array([34154.093778660186, 34646.96658299584, 34773.24084650488]),
                    "H2_g": np.array([1.950644666178021, 0.164768512975864, 0.026732588634311]),
                    "O2_g": np.array([34689.06957751295, 118754.57192278373, 205353.69545182592]),
                },
                "activity": {
                    "H2_g": np.array([26.934201849530517, 14.767185065795928, 11.270717825878766])
                },
            }
        },
        "melt": {
            "species": {
                "activity": {
                    "H2O_d": np.array([0.398791191345742, 0.401658333667266, 0.4023896095421]),
                    "O2Si_l": np.array([0.442946540896139, 0.442946543278513, 0.442946543844512]),
                }
            }
        },
    }

    # output.to_excel(file_prefix="test_subNeptune_melt_phase", output_format="named_arrays")

    # We can also dump a summary of the solver stats to the logger for debugging purposes
    output.solver_stats_to_logger()

    assert output.compare(target, rtol=RTOL, atol=ATOL)
