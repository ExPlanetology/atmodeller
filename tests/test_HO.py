# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for H-O systems"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from jaxtyping import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import ActivityConstraintProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.jax_utils import NpFloat
from atmodeller.output import Output
from atmodeller.parameters import FixedActivityConstraint, Parameters
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
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

# Gas species
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")

# Melt species
H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
)


def test_version() -> None:
    """Test version."""
    assert __version__ == "2.0.0"


def test_H2O() -> None:
    """Tests a single species (H2O)."""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g,)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d,)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )

    oceans: ArrayLike = 2
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {"species": {"partial_pressure": {"H2O_g": 1.031222661909767}}}
    }

    # output.to_excel("test_H2O")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_O() -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint."""
    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)

    planet: BaseThermodynamicState = Planet.from_species(gas_species)

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(oceans)
    o_kg: ArrayLike = 6.25774e20
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "O": o_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    fastchem_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 76.45861543,
                    "H2_g": 73.84378192,
                    "O2_g": 8.91399329e-08,
                }
            }
        }
    }

    # output.to_excel("test_H_O")

    assert output.compare(fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility."""
    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d,)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )

    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
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
                    "H2O_g": 0.25706302966676686,
                    "H2_g": 0.24913460758554615,
                    "O2_g": 8.83851351689605e-08,
                }
            }
        }
    }

    # output.to_excel("test_H_fO2")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_fH2() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility and mixed fugacity constraints."""
    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d,)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )

    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "H2_g": FixedActivityConstraint(np.array([1.0e-8, 1.0e-7, 1.0e-6])),
        "O2_g": IronWustiteBuffer(np.array([-1, 0, 1])),
    }

    oceans: float = 1
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
                    "H2O_g": np.array(
                        [3.262913506271090e-09, 1.031823848794260e-07, 3.262913506271089e-06]
                    ),
                    "H2_g": np.array(
                        [1.000000000000005e-08, 9.999999999999959e-08, 1.000000000000000e-06]
                    ),
                    "O2_g": np.array(
                        [8.838513516896060e-09, 8.838513516896038e-08, 8.838513516896018e-07]
                    ),
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_temperature() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of surface temperatures."""

    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)
    melt_species = (H2O_d,)

    # Number of surface temperatures is different to number of species to test array shapes work.
    temperature = np.array([1500, 2000, 2500, 3000])

    planet: Planet = Planet.from_species(
        gas_species, silicate_melt_species=melt_species, temperature=temperature
    )

    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(),
        "H2_g": FixedActivityConstraint(np.nan),
    }

    oceans: float = 1
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
                    "H2O_g": np.array(
                        [
                            2.5664832779576274e-01,
                            2.570630296667669e-01,
                            2.5720073676678623e-01,
                            2.572575286659322e-01,
                        ]
                    ),
                    "H2_g": np.array(
                        [
                            3.133425134391215e-01,
                            2.491346075855462e-01,
                            2.2655541184547373e-01,
                            2.1993754236733598e-01,
                        ]
                    ),
                    "O2_g": np.array(
                        [
                            2.394194493859145e-12,
                            8.838513516896051e-08,
                            4.544970468047982e-05,
                            2.7394226348237743e-03,
                        ]
                    ),
                }
            }
        }
    }

    # output.to_excel("test_H_fO2_batch_temperature")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_fO2_shift() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of fO2 shifts."""

    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d,)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )

    # Set up a range of fO2 shifts
    num: int = 4
    fO2_shifts: NpFloat = np.linspace(-10, 10, num, dtype=float)
    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(fO2_shifts),
    }

    oceans: float = 1
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
                    "H2O_g": np.array(
                        [
                            2.9750112030316336e-04,
                            1.609211286541073e-01,
                            2.582847658398107e-01,
                            2.585229792463592e-01,
                        ]
                    ),
                    "H2_g": np.array(
                        [
                            2.883254933977426e01,
                            7.238926631964266e00,
                            5.392952102069516e-03,
                            2.5054952892245634e-06,
                        ]
                    ),
                    "O2_g": np.array(
                        [
                            8.838513516896015e-18,
                            4.1024745645760365e-11,
                            1.904200012911654e-04,
                            8.838513516896026e02,
                        ]
                    ),
                }
            }
        }
    }

    # output.to_excel("test_H_fO2_batch_fO2_shift")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_H_mass() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of H budgets."""

    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d,)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )

    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(oceans)
    # Set up a range of H masses
    mass_constraints: dict[str, ArrayLike] = {"H": np.array([h_kg, 10 * h_kg, 100 * h_kg])}

    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [2.570630296667669e-01, 2.42596290006555e01, 1.6102351856877922e03]
                    ),
                    "H2_g": np.array(
                        [2.491346075855462e-01, 2.3511405584397032e01, 1.560571785164134e03]
                    ),
                    "O2_g": np.array(
                        [8.838513516896051e-08, 8.83851351689605e-08, 8.838513516896051e-08]
                    ),
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)
