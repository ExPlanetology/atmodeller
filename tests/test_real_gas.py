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
"""Tests for systems with real gases"""

import logging
from typing import Mapping

import numpy as np
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, Planet, SpeciesNetwork
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import ActivityProtocol, FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.output import Output
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()
eos_models: Mapping[str, ActivityProtocol] = get_eos_models()

H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_models["H2_chabrier21"])
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
SiO_g: ChemicalSpecies = ChemicalSpecies.create_gas("OSi")
H4Si_g: ChemicalSpecies = ChemicalSpecies.create_gas("H4Si")
O2Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed("O2Si")
species: SpeciesNetwork = SpeciesNetwork((H2_g, H2O_g, O2_g, H4Si_g, SiO_g, O2Si_l))
subneptune_model: EquilibriumModel = EquilibriumModel(species)


def test_fO2_holley(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`"""

    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_beattie_holley58"]
    )
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    species: SpeciesNetwork = SpeciesNetwork((H2_g, H2O_g, O2_g))
    # Temperature is within the range of the Holley model
    planet: Planet = Planet(temperature=1000)
    model: EquilibriumModel = EquilibriumModel(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
    }

    model.solve(
        state=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        solver="basic",
    )
    output: Output = model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 32.77037875523393,
        "H2_g": 71.50338102110962,
        "O2_g": 1.525466019972294e-21,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    planet: Planet = Planet(temperature=3400)
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    subneptune_model.solve(state=planet, mass_constraints=mass_constraints, solver="basic")
    output: Output = subneptune_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 7250.890001201146,
        "H2O_g_activity": 7250.89000120113,
        "H2_g": 11625.21660262108,
        "H2_g_activity": 251640.8508698987,
        "H4Si_g": 67590.85056516326,
        "H4Si_g_activity": 67590.85056516311,
        "O2Si_l": 9.311489514762553e04,
        "O2Si_l_activity": 1.0,
        "O2_g": 1.791165364818965e-05,
        "O2_g_activity": 1.791165364818961e-05,
        "OSi_g": 625.5604897933299,
        "OSi_g_activity": 625.5604897933285,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune

    This case effectively saturates the maximum allowable log number density at a value of 70
    based on the default hypercube that brackets the solution (see LOG_NUMBER_MOLES_UPPER).
    This is fine for a test, but this test is not physically realistic because solubilities are
    ignored, which would greatly lower the pressure and hence the number density.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        temperature=surface_temperature, planet_mass=planet_mass, surface_radius=surface_radius
    )
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    subneptune_model.solve(state=planet, mass_constraints=mass_constraints)
    output: Output = subneptune_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 429507.1704491525,
        "H2O_g_activity": 429507.1704491535,
        "H2_g": 2.927180242775929,
        "H2_g_activity": 19567.21447690322,
        "H4Si_g": 0.000704240746855,
        "H4Si_g_activity": 0.000704240746855,
        "O2Si_l": 4.497910721606553e05,
        "O2Si_l_activity": 1.0,
        "O2_g": 10.394369851620414,
        "O2_g_activity": 10.394369851620437,
        "OSi_g": 0.821179444164847,
        "OSi_g_activity": 0.821179444164849,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune_batch(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        temperature=surface_temperature, planet_mass=planet_mass, surface_radius=surface_radius
    )
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    subneptune_model.solve(
        state=planet,
        mass_constraints=mass_constraints,
        solver="basic",
        solver_recompile=True,
    )
    output: Output = subneptune_model.output
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([4.477789711513712e05, 4.785890592398898e05, 5.039107471956282e05]),
        "H2_g": np.array([3.463824822645956e-02, 7.208115634579626e-03, 2.129125602157067e-03]),
        "H2_g_activity": np.array(
            [4.081150539627139e02, 2.445386584856476e02, 1.945159917637966e02]
        ),
        "O2_g": np.array([2.597033179470946e04, 8.263153509596182e04, 1.447811285078976e05]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_pH2_fO2_real_gas(helper) -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2O",
        solubility=solubility_models["H2O_peridotite_sossi23"],
        activity=eos_models["H2O_cork_holland98"],
    )
    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_cork_cs_holland91"]
    )
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    species: SpeciesNetwork = SpeciesNetwork((H2O_g, H2_g, O2_g))
    planet: Planet = Planet()
    model: EquilibriumModel = EquilibriumModel(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(0.072885576196744)
    }

    mass_constraints: dict[str, ArrayLike] = {"H": 1.47126255324872e22}

    model.solve(
        state=planet,
        mass_constraints=mass_constraints,
        fugacity_constraints=fugacity_constraints,
        solver="basic",
        # Guide the solver with an improved initial guess, otherwise use solver="robust".
        initial_log_number_moles=np.array([54, 54, 31]),
    )
    output: Output = model.output

    # output.to_excel("pH2_fO2_real_gas")
    solution: dict[str, ArrayLike] = output.quick_look()

    # logger.debug("solution = %s", pprint.pformat(solution))

    target: dict[str, float] = {
        "H2O_g": 1470.2567650857518,
        "H2_g": 999.9971214963639,
        "O2_g": 1.045357420958815e-07,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)
