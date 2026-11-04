# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for systems with stable or unstable condensates

C-H-O gas with graphite is tested with the C-H-O comparisons in ``test_CHO.py``.
"""

import logging
from typing import Any

import equinox as eqx
import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth
from atmodeller.state import BaseThermodynamicState, Planet, ThermodynamicState

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

# Gas species
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")
N2_g: ChemicalSpecies = ChemicalSpecies.create_gas("N2")
CHN_g: ChemicalSpecies = ChemicalSpecies.create_gas("CHN")
H_g: ChemicalSpecies = ChemicalSpecies.create_gas("H")

# Condensates
graphite: PurePhase = PurePhase.from_end_member("graphite")
diamond: PurePhase = PurePhase.from_end_member("diamond")
water: PurePhase = PurePhase.from_end_member("liquid", formula="H2O")


def test_water_stable() -> None:
    """Condensed water at 10 bar"""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g)
    condensates: tuple[PurePhase, ...] = (water,)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, temperature=411.75, condensates=condensates
    )

    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(1)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "O": 1.14375e21}

    parameters: Parameters = Parameters(planet, mass_constraints=mass_constraints)

    output: Output = EquilibriumModel(parameters).solve_with_default()

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {"partial_pressure": {"H2O_g": 3.3596, "H2_g": 6.5604, "O2_g": 5.6433e-58}}
        },
        "H2O_l": {"species": {"activity": {"H2O_l": 1.0}, "mass": {"H2O_l": 1.247201e21}}},
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.fixture(scope="module")
def graphite_water_model() -> EquilibriumModel:
    """C-H-O gas with graphite and liquid water, compiled once and shared by the tests that
    update its temperature and mass constraints"""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)
    condensates: tuple[PurePhase, ...] = (water, graphite)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, temperature=430, condensates=condensates
    )
    mass_constraints: dict[str, ArrayLike] = {"C": 1.08e20, "H": 3.10e20, "O": 2.48298883581636e21}

    return EquilibriumModel(Parameters(planet, mass_constraints=mass_constraints))


def test_graphite_water_stable(graphite_water_model: EquilibriumModel) -> None:
    """Tests C and water in equilibrium at 430 K and 10 bar"""
    output: Output = graphite_water_model.solve_with_default()

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 0.3241,
                    "CO2_g": 4.3064,
                    "CO_g": 2.77e-6,
                    "H2_g": 0.0023,
                    "O2_g": 4.74e-48,
                    "H2O_g": 5.3672,
                }
            }
        },
        "C_graphite": {
            "species": {"activity": {"C_graphite": 1.0}, "mass": {"C_graphite": 8.75101e19}}
        },
        "H2O_l": {"species": {"activity": {"H2O_l": 1.0}, "mass": {"H2O_l": 2.74821e21}}},
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert _stable(output) == {"C_graphite": True, "H2O_l": True}


def _stable(output: Output) -> dict[str, bool]:
    """Whether each condensate of the graphite-water model is stable"""
    out: dict[str, Any] = output.to_dict()
    return {
        name: bool(out[name]["species"]["stable"][name].item()) for name in ("C_graphite", "H2O_l")
    }


def test_graphite_water_oxidised(graphite_water_model: EquilibriumModel) -> None:
    """Tests water and unstable graphite under an O2-rich atmosphere at 368 K

    O exceeds what H2O and CO2 can hold, so the excess is O2 gas and CH4 is about 1e-117 bar, far
    below the former default lower bound on the log number of moles (-200). Atmodeller result.
    """
    mass_constraints: dict[str, ArrayLike] = {
        "C": 1.5630593225639497e20,
        "H": 4.694481522139993e20,
        "O": 5.571195263063157e21,
    }
    model: EquilibriumModel = graphite_water_model.update_state(
        temperature=367.8167944988913
    ).update_constraints(mass_constraints=mass_constraints)

    output: Output = model.solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 1.2986814831972477e-117,
                    "CO2_g": 86.87677822347462,
                    "CO_g": 1.0923958934052829e-35,
                    "H2_g": 4.663683521768718e-34,
                    "O2_g": 298.1046811843703,
                    "H2O_g": 0.8243073846580263,
                }
            }
        },
        "H2O_l": {
            "species": {"activity": {"H2O_l": 1.0}, "mass": {"H2O_l": 4.1930821668734045e21}}
        },
    }

    assert bool(np.all(output.checked_solution.success))
    assert output.compare(target, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert _stable(output) == {"C_graphite": False, "H2O_l": True}


def test_impose_stable() -> None:
    """Tests a user-imposed stable condensate"""
    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, N2_g, CH4_g, CHN_g, H_g)

    # Since in this example we do not provide carbon in the injected gas stream, we cannot solve
    # for the stability of any carbon-bearing products because in order to do so requires
    # specification of the mass of carbon in the system.
    imposed_graphite: PurePhase = PurePhase.from_end_member("graphite", solve_for_stability=False)
    condensates: tuple[PurePhase, ...] = (imposed_graphite,)

    state: ThermodynamicState = ThermodynamicState.from_species(
        gas_species, pressure=1, temperature=1773.15, condensates=condensates
    )

    # Mole fractions of the input gases
    mole_fractions: dict[str, ArrayLike] = {"H2": 0.5, "N2": 0.5}

    parameters: Parameters = Parameters(state, mass_constraints=mole_fractions, mass_units="moles")

    output: Output = EquilibriumModel(parameters).solve_with_default()

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 0.000194708,
                    "H_g": 0.000201266,
                    "H2_g": 0.49807992,
                    "N2_g": 0.49866269,
                }
            }
        },
        "C_graphite": {"species": {"activity": {"C_graphite": 1.0}}},
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)

    # Without a mass constraint on C, the amount of the imposed graphite is not determined
    graphite: dict[str, Any] = output.to_dict()["C_graphite"]["species"]
    assert bool(graphite["stable"]["C_graphite"].item())
    assert np.isnan(graphite["number_moles"]["C_graphite"]).all()
    assert np.isnan(graphite["mass"]["C_graphite"]).all()


def test_graphite_diamond() -> None:
    """Tests that the solver selects graphite at low pressure and diamond at high pressure

    At 1500 K the graphite-diamond transition is around 45 kbar. Both pressures are solved in one
    batch. The unstable polymorph has a negligible amount and an activity less than unity.
    """
    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, CH4_g)
    condensates: tuple[PurePhase, ...] = (graphite, diamond)

    state: ThermodynamicState = ThermodynamicState.from_species(
        gas_species, pressure=np.array([1.0, 1.0e5]), temperature=1500, condensates=condensates
    )

    parameters: Parameters = Parameters(
        state, mass_constraints={"C": 1.0, "H": 1.0}, mass_units="moles"
    )

    output: Output = EquilibriumModel(parameters).solve_with_default()
    out: dict[str, Any] = output.to_dict(to_numpy=True)

    # Graphite stable at 1 bar (row 0) and diamond stable at 1e5 bar (row 1)
    for row, (stable, unstable) in enumerate(
        (("C_graphite", "C_diamond"), ("C_diamond", "C_graphite"))
    ):
        stable_species: dict[str, Any] = out[stable]["species"]
        unstable_species: dict[str, Any] = out[unstable]["species"]

        assert np.isclose(stable_species["activity"][stable][row], 1.0)
        assert stable_species["number_moles"][stable][row] > 0.1
        assert unstable_species["activity"][unstable][row] < 1.0
        assert unstable_species["number_moles"][unstable][row] < 1.0e-20


def test_water_one_component() -> None:
    """Water vapour and liquid water have one component, so H alone is constrained, at the
    saturation pressure. Constraining H and O as well raises, as do H and O alongside N2, which
    are the right number but not independent."""

    def model(
        gas_species: tuple[ChemicalSpecies, ...], mass: dict[str, ArrayLike]
    ) -> EquilibriumModel:
        planet: Planet = Planet.from_species(
            gas_species, condensates=(water,), temperature=300, mantle_melt_fraction=0
        )
        return EquilibriumModel(Parameters(planet, mass_constraints=mass))

    # One water model for every case, with O inactive (NaN) unless it is given
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(1)
    water_model: EquilibriumModel = model((H2O_g,), {"H": h_kg, "O": np.nan})

    output: Output = water_model.solve_with_default()
    np.testing.assert_allclose(output.to_dict()["gas"]["phase"]["pressure"], 0.0353362, rtol=1e-5)

    with pytest.raises(ValueError, match="2 constraints"):
        water_model.update_constraints(
            mass_constraints={"H": h_kg, "O": 8 * h_kg}
        ).solve_with_default()
    with pytest.raises(ValueError, match="not independent"):
        model((H2O_g, N2_g), {"H": h_kg, "O": 8 * h_kg}).solve_with_default()

    # Inside JIT the constraints are traced, so the check is skipped, and the model fails instead
    @eqx.filter_jit
    def solve(model_: EquilibriumModel, o_kg: ArrayLike) -> Output:
        return model_.update_constraints(
            mass_constraints={"H": h_kg, "O": o_kg}
        ).solve_with_default()

    for o_kg, success in ((np.nan, True), (8 * h_kg, False)):
        assert bool(solve(water_model, o_kg).to_dict()["solver"]["success"].item()) is success
