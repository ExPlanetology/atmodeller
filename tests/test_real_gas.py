# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for systems with real gases"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller import console_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.jax_utils import NpFloat
from atmodeller.interfaces import (
    ActivityConstraintProtocol,
    ActivityProtocol,
    SolubilityProtocol,
)
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

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

# Force SiO2 to have unity activity, as per previous tests. Eventually can be relaxed to allow for
# a self-consistent activity, but this is fine for testing the real gas EOS.
O2Si_l: PurePhase = PurePhase.from_end_member("liquid", formula="SiO2")
condensates_subneptune: tuple[PurePhase, ...] = (O2Si_l,)


def test_fO2_holley() -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`

    This is the only test in which an EOS that solves for its volume with its own root find, as
    the Beattie-Bridgeman EOS does, is evaluated inside the solve, where the inner root find is
    vmapped and differentiated within the Jacobian. The other models in these tests use EOS whose
    volume is tabulated or analytic.
    """

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
                    "H2O_g": 32.77150764086231,
                    "H2_g": 71.50578586144273,
                    "O2_g": 1.5254660199722744e-21,
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def chabrier_output() -> Output:
    """The H2 EOS from :cite:t:`CD21` with silicate species, compiled and solved once for every
    case as one batch: an Earth-sized planet (row 0), and a sub-Neptune (rows 1 to 4) for several O
    masses. Every row is solved on the first attempt, so the rows do not affect each other."""
    earth_mass: float = 5.9722e24  # kg
    earth_radius: float = 6371000  # m
    planet_mass: NpFloat = np.array([earth_mass] + [4.6 * 5.97224e24] * 4)
    surface_radius: NpFloat = np.array([earth_radius] + [1.5 * 6371000] * 4)
    planet: Planet = Planet.from_species(
        gas_species_subneptune,
        condensates=condensates_subneptune,
        temperature=3400,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    h_kg: ArrayLike = 0.01 * planet.background_planet_mass
    si_kg: ArrayLike = (
        0.1459 * planet.background_planet_mass
    )  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: NpFloat = np.array([10 * float(np.asarray(h_kg)[0]), 6.74717e24, 7.0e24, 7.5e24, 8.0e24])
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    output: Output = EquilibriumModel(
        Parameters(planet, mass_constraints=mass_constraints)
    ).solve_with_default()
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


def test_chabrier_earth(chabrier_output: Output) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 7108.302860803897,
                    "H2_g": 11857.477242560706,
                    "H4Si_g": 67368.89566345635,
                    "O2_g": 1.758832698936969e-05,
                    "OSi_g": 631.2841521604278,
                },
                "activity": {
                    "H2O_g": 7108.3028608039,
                    "H2_g": 248949.53960147503,
                    "H4Si_g": 67368.89566345638,
                    "O2_g": 1.75883269893697e-05,
                    "OSi_g": 631.2841521604281,
                },
            }
        },
        "O2Si_l": {"species": {"activity": {"O2Si_l": 1.0}}},
    }

    assert _compare_rows(chabrier_output, target, slice(0, 1))


def test_chabrier_subNeptune(chabrier_output: Output) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune

    This test is not physically realistic because solubilities are ignored, which would greatly
    lower the pressure and hence the number of moles in the atmosphere.
    """
    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 429506.981900562,
                    "H2_g": 3.049981984494761,
                    "H4Si_g": 0.000696153160247318,
                    "O2_g": 10.454569410545353,
                    "OSi_g": 0.8188117708669286,
                },
                "activity": {
                    "H2O_g": 429506.9819005624,
                    "H2_g": 19510.788558096978,
                    "H4Si_g": 0.0006961531602473188,
                    "O2_g": 10.454569410545362,
                    "OSi_g": 0.8188117708669294,
                },
            }
        },
        "O2Si_l": {"species": {"activity": {"O2Si_l": 1.0}}},
    }

    assert _compare_rows(chabrier_output, target, slice(1, 2))


def test_chabrier_subNeptune_batch(chabrier_output: Output) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """
    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [4.477789711513712e05, 4.785890592398898e05, 5.039107471956282e05]
                    ),
                    "H2_g": np.array(
                        [3.625304904527265e-02, 7.563302360493786e-03, 2.239339401376818e-03]
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

    assert _compare_rows(chabrier_output, target, slice(2, 5))


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
                    "H2O_g": 1470.2003824504136,
                    "H2_g": 999.9680041351174,
                    "O2_g": 1.0453574209588052e-07,
                }
            }
        }
    }

    # output.to_excel("pH2_fO2_real_gas")

    assert output.compare(target, rtol=RTOL, atol=ATOL)
