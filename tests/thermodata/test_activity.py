# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests activity models"""

import logging
from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import ArrayLike
from pytest import approx

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import ActivityProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.jax_utils import FloatArray
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import Planet
from atmodeller.thermodata.activity_models import Si_iron_badro15

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


def test_Si_iron_badro15() -> None:
    """Tests the activity model for Si in Fe :cite:p:`B15`"""
    # Here, must specify the index of Si in the mole fraction array for the metal species. In this
    # case, Fe is at index 0 and Si is at index 1.
    model: Si_iron_badro15 = Si_iron_badro15(gamma=1.0, Si_index=1)

    log_activity: FloatArray = model.log_activity(
        temperature=2000, pressure=1e5, mole_fractions=jnp.array([[0.9, 0.1], [0.8, 0.2]])
    )

    expected: list[float] = [jnp.log(1.0 * 0.3 * 0.1).item(), jnp.log(1.0 * 0.3 * 0.2).item()]

    assert jnp.asarray(log_activity).tolist() == approx(expected)


def test_subNeptune_melt_phase_with_si_activity() -> None:
    """Tests a more realistic sub-Neptune with Si activity in the metal phase.

    Includes a metal phase with Fe and Si, where the activity of Si is calculated using the Badro
    2015 model. The base of this test is the same as `test_subNeptune_melt_phase` in
    `test_real_gas.py`.

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

    # Create a metal phase with Fe and Si, where the activity of Si is calculated using the Badro
    # 2015 model
    # TODO: For activity model include_in_phase_mass might have to be True?
    Fe_l: ChemicalSpecies = ChemicalSpecies.create_condensed(
        "Fe", state="l", include_in_phase_mass=False
    )
    # NOTE: The index of Si in the mole fraction array for the metal species is 1, since Fe is at
    # index 0 in metal_species. You must manually specify the correct index!
    Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed(
        "Si", state="l", include_in_phase_mass=False, activity=Si_iron_badro15(Si_index=1)
    )
    metal_species: tuple[SpeciesProtocol, ...] = (Fe_l, Si_l)

    # Temperature must be compatible with the choice of species, i.e. chemically-reactive species
    # must have thermodynamic data available at the specified temperature. Here, we are limited by
    # O2Si(l).
    surface_temperature = 3400  # K
    planet_mass = 4.6 * earth.mass  # kg
    surface_radius = 1.5 * earth.radius  # m

    planet: Planet = Planet.from_species(
        gas_species_subneptune,
        silicate_melt_species=melt_species,
        metal_species=metal_species,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )

    # # The previous mass constraints are still OK, because we are not allowing the melt species to
    # # contribute additionally to the planet mass. So these calculations are still exact.
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = (
        0.1459 * planet.background_planet_mass
    )  # Si = 14.59 wt% Kargel & Lewis (1993)
    # o_kg: ArrayLike = 6.74717e24
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * 7  # * np.array([7.0, 7.5, 8.0])

    # Add some Fe to the system, which can only be dumped in the metal core
    fe_kg: ArrayLike = 0.25 * planet.background_planet_mass

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)
    logger.info("fe_kg = %s", fe_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg, "Fe": fe_kg}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    # target: dict[str, Any] = {
    #     "gas": {
    #         "species": {
    #             "partial_pressure": {
    #                 "H2O_g": np.array([34154.093778660186, 34646.96658299584, 34773.24084650488]),
    #                 "H2_g": np.array([1.950644666178021, 0.164768512975864, 0.026732588634311]),
    #                 "O2_g": np.array([34689.06957751295, 118754.57192278373, 205353.69545182592]),
    #             },
    #             "activity": {
    #                 "H2_g": np.array([26.934201849530517, 14.767185065795928, 11.270717825878766])
    #             },
    #         }
    #     },
    #     "melt": {
    #         "species": {
    #             "activity": {
    #                 "H2O_d": np.array([0.398791191345742, 0.401658333667266, 0.4023896095421]),
    #                 "O2Si_l": np.array([0.442946540896139, 0.442946543278513, 0.442946543844512]),
    #             }
    #         }
    #     },
    # }

    output.to_excel(file_prefix="test_subNeptune_melt_phase_activity")

    # We can also dump a summary of the solver stats to the logger for debugging purposes
    output.solver_stats_to_logger()

    # assert output.compare(target, rtol=RTOL, atol=ATOL)
