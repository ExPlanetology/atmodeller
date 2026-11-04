# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests activity models"""

import logging
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import ArrayLike
from pytest import approx

from atmodeller import console_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import (
    ActivityConstraintProtocol,
    ActivityProtocol,
    SolubilityProtocol,
    SpeciesProtocol,
)
from atmodeller.jax_utils import FloatArray, NpFloat
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import Planet
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer
from atmodeller.thermodata.activity_models import Si_iron_badro15

logger: logging.Logger = console_logger()
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


@pytest.fixture(scope="module")
def subneptune_output() -> Output:
    """A sub-Neptune with a silicate melt and a metal phase with Fe and Si, compiled and solved once
    for every case as one batch, each solved on its first attempt so the rows do not affect each
    other.

    Row 0 imposes the fO2, and rows 1 to 3 impose several O masses instead. The silicate melt
    phase consists of a chemically-reactive component SiO2(l) and a dissolved component H2O(l),
    and the metal phase of Fe and Si, where Si has an activity model. The activities of the
    species in the melt and metal phases are calculated self-consistently. This is an extension of
    the model setup in :cite:t:`Hakim2026`.
    """
    # The species we specify in the melt should be considered as already included in the
    # "background" melt mass, so we set include_in_phase_mass=False for both species
    O2Si_l: ChemicalSpecies = ChemicalSpecies.from_end_member(
        "liquid", formula="SiO2", include_in_phase_mass=False
    )
    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )
    melt_species: tuple[SpeciesProtocol, ...] = (O2Si_l, H2O_d)

    # Create a metal phase with Fe and Si, where Si additionally has an activity model
    Fe_l: ChemicalSpecies = ChemicalSpecies.from_end_member(
        "liquid", formula="Fe", include_in_phase_mass=True
    )
    # NOTE: The index of Si in the mole fraction array for the metal species is 1, since Fe is at
    # index 0 in metal_species. You must manually specify the correct index.
    Si_l: ChemicalSpecies = ChemicalSpecies.from_end_member(
        "liquid", formula="Si", include_in_phase_mass=True, activity=Si_iron_badro15(Si_index=1)
    )
    # Since Si_index=1 above, it must appear in the correct order in metal_species
    metal_species: tuple[SpeciesProtocol, ...] = (Fe_l, Si_l)

    # The melt and metal end-members are extended with a constant heat capacity outside their
    # data, so any temperature works. At this temperature all are inside their data.
    planet: Planet = Planet.from_species(
        gas_species_subneptune,
        silicate_melt_species=melt_species,
        metal_species=metal_species,
        temperature=3400,
        planet_mass=4.6 * earth.mass,
        surface_radius=1.5 * earth.radius,
        # None means the metal phase's mass comes entirely from tracked metal_species
        core_mass_fraction=None,
    )

    # Recall that `background_planet_mass` is only the mass of the background melt phase because
    # the metal phase is not tracked in the background.
    background_mass: float = float(np.asarray(planet.background_planet_mass))
    h_kg: NpFloat = np.full(4, 0.01 * background_mass)
    # Si = 14.59 wt% Kargel & Lewis (1993) for the cases with an O mass
    si_kg: NpFloat = np.array([0.4 * background_mass] + [0.1459 * background_mass] * 3)

    # Again, since `background_planet_mass` excludes the core, we work backwards to compute a
    # reasonable mass of Fe to add. This only accounts for Fe, so the resulting core mass fraction
    # ends up a bit higher than the 0.34 implied here once the solved Si mass is added on top.
    core_fe_mass_fraction: float = 0.34
    fe_kg: NpFloat = np.full(
        4, core_fe_mass_fraction / (1 - core_fe_mass_fraction) * background_mass
    )

    # Imposing O as a mass constraint is tricky since O is stoichiometrically constrained to H2O
    # and SiO2, but now the metal phase can also compete for Si. Row 0 instead imposes an fO2, a
    # more consistent approach, and its O mass is inactive (NaN), as is the fO2 of the other rows.
    o_kg: NpFloat = np.array([np.nan, 7.0e24, 7.5e24, 8.0e24])
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "Fe": fe_kg, "O": o_kg}
    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(np.array([-2.0, np.nan, np.nan, np.nan]))
    }

    parameters: Parameters = Parameters(
        planet, mass_constraints=mass_constraints, activity_constraints=activity_constraints
    )
    output: Output = EquilibriumModel(parameters).solve_with_default()

    # We can also dump a summary of the solver stats to the logger for debugging purposes
    output.solver_stats_to_logger()
    assert np.all(np.asarray(output.checked_solution.successful_attempt) == 1)

    return output


def _compare_rows(output: Output, target: dict[str, Any], rows: slice) -> bool:
    """Compares a target with some rows of a batched output, as :meth:`Output.compare` does for
    every row"""
    out: dict[str, Any] = output.to_dict(expand_to_batch=True, ravel=True, to_numpy=True)

    def compare(target_: dict[str, Any], out_: dict[str, Any]) -> bool:
        return all(
            compare(value, out_[key])
            if isinstance(value, dict)
            else bool(np.allclose(value, np.asarray(out_[key])[rows], rtol=RTOL, atol=ATOL))
            for key, value in target_.items()
        )

    return compare(target, out)


def test_subNeptune_melt_phase_with_si_activity(subneptune_output: Output) -> None:
    """Tests a sub-Neptune with Si activity in the metal phase, at an imposed fO2 (row 0)."""
    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 15057.702229615414,
                    "H2_g": 22319.434606884446,
                    "O2_g": 0.0002924078357985892,
                },
                "activity": {"H2_g": 129336.72835135741},
            }
        },
        "silicate_melt": {
            "species": {"activity": {"H2O_d": 0.2647911866598453, "O2Si_l": 0.6257140779609305}}
        },
        "metal": {
            "species": {"activity": {"Fe_l": 0.7073157842386129, "Si_l": 0.08008105068145352}}
        },
    }

    assert _compare_rows(subneptune_output, target, slice(0, 1))


def test_subNeptune_melt_phase(subneptune_output: Output) -> None:
    """Tests a sub-Neptune with computed melt-phase activities for several O masses (rows 1 to 3).

    The melt phase consists of a chemically-reactive component SiO2(l) and a dissolved component
    H2O(l), and the planet also has a metal phase (see ``subneptune_output``).
    """
    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array([18407.507470817447, 18450.38188829789, 18459.51355243942]),
                    "H2_g": np.array(
                        [0.641845615687295, 0.02511762505160897, 0.0024370511634242546]
                    ),
                    "O2_g": np.array([60188.08181318284, 191933.18537929462, 324297.0848929098]),
                },
                "activity": {
                    "H2_g": np.array([11.020401158306242, 6.185687757328123, 4.761090091777577])
                },
            }
        },
        "silicate_melt": {
            "species": {
                "activity": {
                    "H2O_d": np.array(
                        [0.29276669247057885, 0.2931074474812595, 0.2931799724752679]
                    ),
                    "O2Si_l": np.array(
                        [0.3121290589103115, 0.31212906225115455, 0.31212906291056275]
                    ),
                }
            }
        },
    }

    assert _compare_rows(subneptune_output, target, slice(1, 4))
