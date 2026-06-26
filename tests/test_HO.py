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
"""Tests for H-O systems"""

import logging
from typing import Mapping

import numpy as np
from jaxtyping import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, FixedFugacityConstraint, Planet, SpeciesNetwork
from atmodeller.interfaces import FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.output import Output
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.type_aliases import NpFloat
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas(
    "H2O", solubility=solubility_models["H2O_peridotite_sossi23"]
)
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
species: SpeciesNetwork = SpeciesNetwork((H2O_g, H2_g, O2_g))

gas_HO_model: EquilibriumModel = EquilibriumModel(species)


def test_version():
    """Test version."""
    assert __version__ == "1.0.2"


def test_H2O(helper) -> None:
    """Tests a single species (H2O)."""

    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    species: SpeciesNetwork = SpeciesNetwork((H2O_g,))
    planet: Planet = Planet()
    model: EquilibriumModel = EquilibriumModel(species)

    oceans: ArrayLike = 2
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    model.solve(state=planet, mass_constraints=mass_constraints, solver="basic")
    output: Output = model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {"H2O_g": 1.0312913336898137}

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_O(helper) -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint."""

    species: SpeciesNetwork = SpeciesNetwork.create(("H2_g", "H2O_g", "O2_g"))

    planet: Planet = Planet()
    model: EquilibriumModel = EquilibriumModel(species)

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: ArrayLike = 6.25774e20
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "O": o_kg}

    model.solve(state=planet, mass_constraints=mass_constraints, solver="basic")
    output: Output = model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    fastchem_result: dict[str, float] = {
        "H2O_g": 76.45861543,
        "H2_g": 73.84378192,
        "O2_g": 8.91399329e-08,
    }

    output.to_excel("test_H_O")

    assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_H_fO2(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility."""

    planet: Planet = Planet()
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(),
    }
    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    gas_HO_model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver="basic",
    )
    output: Output = gas_HO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 0.2570800742364775,
        "H2_g": 0.2491511264610601,
        "O2_g": 8.838513516896038e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_fH2(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility and mixed fugacity constraints."""

    planet: Planet = Planet()
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(np.array([-1, 0, 1])),
        "H2_g": FixedFugacityConstraint(np.array([1.0e-8, 1.0e-7, 1.0e-6])),
    }

    gas_HO_model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        solver="basic",
        solver_recompile=True,
    )
    output: Output = gas_HO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([3.262913506271090e-09, 1.031823848794260e-07, 3.262913506271089e-06]),
        "H2_g": np.array([1.000000000000005e-08, 9.999999999999959e-08, 1.000000000000000e-06]),
        "O2_g": np.array([8.838513516896060e-09, 8.838513516896038e-08, 8.838513516896018e-07]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_temperature(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of surface temperatures."""

    # Number of surface temperatures is different to number of species to test array shapes work.
    surface_temperatures: NpFloat = np.array([1500, 2000, 2500, 3000])
    planet: Planet = Planet(temperature=surface_temperatures)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(),
        "H2_g": FixedFugacityConstraint(np.nan),
    }
    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    gas_HO_model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver="basic",
        solver_recompile=True,
    )
    output: Output = gas_HO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array(
            [
                2.566653037020448e-01,
                2.570800742364757e-01,
                2.572178041535549e-01,
                2.572746043480848e-01,
            ]
        ),
        "H2_g": np.array(
            [
                3.133632393608037e-01,
                2.491511264610584e-01,
                2.265704456625875e-01,
                2.199521409043987e-01,
            ]
        ),
        "O2_g": np.array(
            [
                2.394194493859141e-12,
                8.838513516896038e-08,
                4.544970468047975e-05,
                2.739422634823809e-03,
            ]
        ),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_fO2_shift(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of fO2 shifts."""

    planet: Planet = Planet()
    # Set up a range of fO2 shifts
    num: int = 4
    fO2_shifts: NpFloat = np.linspace(-10, 10, num, dtype=np.float64)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(fO2_shifts),
    }
    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    gas_HO_model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver="basic",
        solver_recompile=True,
    )
    output: Output = gas_HO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array(
            [
                2.974916728388850e-04,
                1.609262626310640e-01,
                2.583020136413676e-01,
                2.585402668639946e-01,
            ]
        ),
        "H2_g": np.array(
            [
                2.883163373152490e01,
                7.239157580397396e00,
                5.393312233908341e-03,
                2.505662833497315e-06,
            ]
        ),
        "O2_g": np.array(
            [
                8.838513516896005e-18,
                4.102474564576031e-11,
                1.904200012911665e-04,
                8.838513516896137e02,
            ]
        ),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_H_mass(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of H budgets."""

    planet: Planet = Planet()
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(),
    }
    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    # Set up a range of H masses
    mass_constraints: dict[str, ArrayLike] = {"H": np.array([h_kg, 10 * h_kg, 100 * h_kg])}

    gas_HO_model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver="basic",
        solver_recompile=True,
    )
    output: Output = gas_HO_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([2.570800742364757e-01, 2.426110356931991e01, 1.610286613431932e03]),
        "H2_g": np.array([2.491511264610584e-01, 2.351283467393216e01, 1.560621626756960e03]),
        "O2_g": np.array([8.838513516896038e-08, 8.838513516896038e-08, 8.838513516896102e-08]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)
