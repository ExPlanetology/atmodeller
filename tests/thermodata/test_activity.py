# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests activity models"""

import logging
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import ArrayLike
from pytest import approx

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
from atmodeller.jax_utils import FloatArray
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import Planet
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer
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
    """Tests the activity model for Si in Fe :cite:p:`Badro2015`

    Checks ``log_activity`` against a direct evaluation of
    :math:`\\ln \\gamma_{\\rm Si} = \\ln \\gamma_{\\rm Si}^0(T) - \\varepsilon_{\\rm Si}^{\\rm
    Si}(T) \\ln(1 - X_{\\rm Si})`, using the reference activity coefficient and self-interaction
    parameter tabulated at 1873 K in :cite:t:`Badro2015`'s Table S1.
    """
    # Here, must specify the index of Si in the mole fraction array for the metal species. In this
    # case, Si is at index 1.
    model: Si_iron_badro15 = Si_iron_badro15(gamma=1.0, Si_index=1)

    temperature: float = 2000.0
    mole_fractions = jnp.array([[0.9, 0.1], [0.8, 0.2]])

    log_activity: FloatArray = model.log_activity(
        temperature=temperature, pressure=1e5, mole_fractions=mole_fractions
    )

    T0: float = 1873.0
    ln_gamma_si0: float = -6.65 * T0 / temperature
    epsilon: float = 12.41 * T0 / temperature
    x_si = mole_fractions[..., 1]
    expected = ln_gamma_si0 - epsilon * jnp.log(1 - x_si)

    assert jnp.asarray(log_activity).tolist() == approx(jnp.asarray(expected).tolist())


def test_subNeptune_melt_phase_with_si_activity() -> None:
    """Tests a sub-Neptune with Si activity in the metal phase.

    Includes a metal phase (i.e. metallic core) with Fe and Si, and an Si activity model.

    The silicate melt phase consists of a chemically-reactive component SiO2(l) and a dissolved
    component H2O(l). Here, the activities of both species in the melt phase are calculated
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

    # Create a metal phase with Fe and Si, where Si additionally has an activity model
    Fe_l: ChemicalSpecies = ChemicalSpecies.create_condensed(
        "Fe", state="l", include_in_phase_mass=True
    )
    # NOTE: The index of Si in the mole fraction array for the metal species is 1, since Fe is at
    # index 0 in metal_species. You must manually specify the correct index.
    Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed(
        "Si", state="l", include_in_phase_mass=True, activity=Si_iron_badro15(Si_index=1)
    )
    # Since Si_index=1 above, it must appear in the correct order in metal_species
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
        # None means the metal phase's mass comes entirely from tracked metal_species
        core_mass_fraction=None,
    )

    # Recall that `background_planet_mass` is only the mass of the background melt phase because
    # the metal phase is not tracked in the background.
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = 0.4 * planet.background_planet_mass

    # Add some Fe to the system, which can only be dumped in the metal core (no oxides specified
    # in the melt phase).
    # Again, since `background_planet_mass` excludes the core, we work backwards to compute a
    # reasonable mass of Fe to add. This only accounts for Fe, so the resulting core mass fraction
    # ends up a bit higher than the 0.34 implied here once the solved Si mass is added on top.
    core_fe_mass_fraction: float = 0.34
    fe_kg: ArrayLike = (
        core_fe_mass_fraction / (1 - core_fe_mass_fraction) * planet.background_planet_mass
    )

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("fe_kg = %s", fe_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "Fe": fe_kg}

    # Imposing O as a mass constraint is tricky since O is stoichiometrically constrained to H2O
    # and SiO2, but now the metal phase can also compete for Si. In practice, imposing an fO2 is
    # a more consistent approach to enable the model to find a solution.
    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer(-2.0)}

    parameters: Parameters = Parameters(
        planet, mass_constraints=mass_constraints, activity_constraints=activity_constraints
    )

    model: EquilibriumModel = EquilibriumModel(parameters)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 15104.470712482722,
                    "H2_g": 22051.2608663181,
                    "O2_g": 0.000292407835799,
                },
                "activity": {"H2_g": 129738.44187124078},
            }
        },
        "silicate_melt": {
            "species": {"activity": {"H2O_d": 0.265202082056106, "O2Si_l": 0.625714953627593}}
        },
        "metal": {"species": {"activity": {"Fe_l": 0.70731567726206, "Si_l": 0.080081162752299}}},
    }

    # output.to_excel(file_prefix="test_subNeptune_melt_phase_with_si_activity", output_format="named_arrays")

    # We can also dump a summary of the solver stats to the logger for debugging purposes
    output.solver_stats_to_logger()

    assert output.compare(target, rtol=RTOL, atol=ATOL)
