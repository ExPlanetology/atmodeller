# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for H-O and C-H-O systems

Cases that share species are rows of one batched model, so that the solver is compiled once per
set of species. The C-H-O comparisons with FactSage and FastChem are rows of a model of C-H-O gas
with graphite, which is unstable in the gas-only comparisons.
"""

import importlib.metadata
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import ActivityConstraintProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.jax_utils import NpFloat
from atmodeller.output import Output
from atmodeller.output_base import OutputElementsSpeciesDict, OutputNamedArraysDict
from atmodeller.parameters import FixedActivityConstraint, Parameters
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
CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "CO2", solubility=solubility_models["CO2_basalt_dixon95"], include_in_phase_mass=False
)

OCEAN: float = float(np.asarray(earth.oceans_to_hydrogen_mass(1)))
"""Mass of H in one Earth ocean (kg)"""


def test_version() -> None:
    """The package version matches the installed metadata, i.e. pyproject.toml"""
    assert __version__ == importlib.metadata.version("atmodeller")


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


# Each case is (temperature, fO2 shift relative to IW, H oceans, H2 fugacity, and the expected
# partial pressures of H2O, H2 and O2). A NaN H2 fugacity is unconstrained.
HO_CASES: dict[str, tuple[float, float, float, float, tuple[float, float, float]]] = {
    "fH2=1e-8, IW-1": (
        2000, -1, 1, 1.0e-8,
        (3.262913506271090e-09, 1.000000000000005e-08, 8.838513516896060e-09),
    ),
    "fH2=1e-7, IW": (
        2000, 0, 1, 1.0e-7,
        (1.031823848794260e-07, 9.999999999999959e-08, 8.838513516896038e-08),
    ),
    "fH2=1e-6, IW+1": (
        2000, 1, 1, 1.0e-6,
        (3.262913506271089e-06, 1.000000000000000e-06, 8.838513516896018e-07),
    ),
    "1500 K": (
        1500, 0, 1, np.nan,
        (2.5664832779576274e-01, 3.133425134391215e-01, 2.394194493859145e-12),
    ),
    "2000 K": (
        2000, 0, 1, np.nan,
        (2.570630296667669e-01, 2.491346075855462e-01, 8.838513516896051e-08),
    ),
    "2500 K": (
        2500, 0, 1, np.nan,
        (2.5720073676678623e-01, 2.2655541184547373e-01, 4.544970468047982e-05),
    ),
    "3000 K": (
        3000, 0, 1, np.nan,
        (2.572575286659322e-01, 2.1993754236733598e-01, 2.7394226348237743e-03),
    ),
    "IW-10": (
        2000, -10, 1, np.nan,
        (2.9750112030316336e-04, 2.883254933977426e01, 8.838513516896015e-18),
    ),
    "IW-3.33": (
        2000, -10 / 3, 1, np.nan,
        (1.609211286541073e-01, 7.238926631964266e00, 4.1024745645760365e-11),
    ),
    "IW+3.33": (
        2000, 10 / 3, 1, np.nan,
        (2.582847658398107e-01, 5.392952102069516e-03, 1.904200012911654e-04),
    ),
    "IW+10": (
        2000, 10, 1, np.nan,
        (2.585229792463592e-01, 2.5054952892245634e-06, 8.838513516896026e02),
    ),
    "10 oceans": (
        2000, 0, 10, np.nan,
        (2.42596290006555e01, 2.3511405584397032e01, 8.83851351689605e-08),
    ),
    "100 oceans": (
        2000, 0, 100, np.nan,
        (1.6102351856877922e03, 1.560571785164134e03, 8.838513516896051e-08),
    ),
}  # fmt: skip


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility, in one batch

    The batch varies the temperature, fO2 shift, H budget and an imposed H2 fugacity.
    """
    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d,)

    def column(index: int) -> NpFloat:
        return np.array([case[index] for case in HO_CASES.values()], dtype=float)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species, temperature=column(0)
    )
    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(column(1)),
        "H2_g": FixedActivityConstraint(column(3)),
    }
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(column(2))
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    output: Output = EquilibriumModel(parameters).solve_with_default()

    expected: NpFloat = np.array([case[-1] for case in HO_CASES.values()])
    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    name: expected[:, i] for i, name in enumerate(("H2O_g", "H2_g", "O2_g"))
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


# Each case is (temperature, fO2 shift relative to IW, H kg, C kg, O kg, references). A NaN fO2
# shift or O mass is unconstrained. References are FactSage, unless labelled FastChem.
CHO_CASES: dict[str, tuple[float, float, float, float, float, list[dict[str, Any]]]] = {
    # Similar to BHS22, Table E, row 1
    "IW-2": (
        1400, -2, 3 * OCEAN, 3 * OCEAN, np.nan,
        [{"partial_pressure": {"H2_g": 175.5, "H2O_g": 13.8, "CO_g": 6.21, "CO2_g": 0.228,
                               "CH4_g": 38.07, "O2_g": 1.25e-15}}],
    ),
    # Similar to BHS22, Table E, row 2. Graphite is unstable.
    "IW+0.5": (
        1400, 0.5, 3 * OCEAN, 3 * OCEAN, np.nan,
        [
            {"partial_pressure": {"CH4_g": 28.66, "CO2_g": 30.88, "CO_g": 46.42,
                                  "H2O_g": 337.16, "H2_g": 236.98, "O2_g": 4.11e-13},
             "C_s_activity": 0.12202},
            # FastChem
            {"partial_pressure": {"CH4_g": 29.61919788, "CO2_g": 29.82548282,
                                  "CO_g": 45.94958264, "H2O_g": 332.03616807,
                                  "H2_g": 236.73845646, "O2_g": 3.96475584e-13}},
        ],
    ),
    # Similar to BHS22, Table E, row 3
    "IW+2": (
        1400, 2, OCEAN, 0.1 * OCEAN, np.nan,
        [{"partial_pressure": {"CH4_g": 0.00129, "CO2_g": 3.25, "CO_g": 0.873, "H2O_g": 218.48,
                               "H2_g": 27.40, "O2_g": 1.29e-11}}],
    ),
    # Similar to BHS22, Table E, row 4
    "IW+4": (
        1400, 4, OCEAN, 5 * OCEAN, np.nan,
        [{"partial_pressure": {"CH4_g": 7.13e-05, "CO2_g": 357.23, "CO_g": 10.21,
                               "H2O_g": 432.08, "H2_g": 5.78, "O2_g": 1.14e-09}}],
    ),
    "873K": (
        873, 0, OCEAN, OCEAN, np.nan,
        [{"partial_pressure": {"H2_g": 59.066, "H2O_g": 18.320, "CO_g": 8.91e-4,
                               "CO2_g": 7.48e-4, "CH4_g": 19.548, "O2_g": 1.27e-25}}],
    ),
    "450K": (
        450, np.nan, OCEAN, OCEAN, 1.02999e20,
        [{"partial_pressure": {"H2_g": 55.475, "H2O_g": 8.0, "CO2_g": 1.24e-14,
                               "O2_g": 7.85e-54, "CH4_g": 16.037, "CO_g": 2.12e-16}}],
    ),
    # Graphite stable with around 50% condensed C mass fraction
    "graphite_stable": (
        873, np.nan, OCEAN, 5 * OCEAN, 2.73159e19,
        [{"partial_pressure": {"O2_g": 1.27e-25, "H2_g": 14.564, "CO_g": 0.07276,
                               "H2O_g": 4.527, "CO2_g": 0.061195, "CH4_g": 96.74},
          "C_s_activity": 1.0, "C_s_mass": 3.54162e20}],
    ),
}  # fmt: skip


@pytest.fixture(scope="module")
def CHO_solution() -> Output:
    """Solution of the C-H-O model with graphite, solved for all cases in one batch"""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)
    condensates: tuple[PurePhase, ...] = (PurePhase.from_species("C", state="s"),)

    def column(index: int) -> ArrayLike:
        return np.array([case[index] for case in CHO_CASES.values()], dtype=float)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, temperature=column(0), condensates=condensates
    )
    activity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(column(1))
    }
    mass_constraints: dict[str, ArrayLike] = {"H": column(2), "C": column(3), "O": column(4)}
    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    output: Output = EquilibriumModel(parameters).solve_with_default()
    assert bool(np.all(output.multi_attempt_solution.success))

    return output


@pytest.fixture(scope="module")
def CHO_output(CHO_solution: Output) -> dict[str, Any]:
    """Output of the C-H-O model with graphite in the named arrays format"""
    return OutputNamedArraysDict(
        CHO_solution.parameters, CHO_solution.multi_attempt_solution, CHO_solution.initial_guess
    ).to_dict(expand_to_batch=True, ravel=True, to_numpy=True)


def test_CHO_elements_species(CHO_solution: Output, CHO_output: dict[str, Any]) -> None:
    """In the elements and species format, the graphite phase, which has the name of its species,
    keeps the species entry beside its phase entry, with the same values as the named arrays"""
    out: dict[str, Any] = OutputElementsSpeciesDict(
        CHO_solution.parameters, CHO_solution.multi_attempt_solution, CHO_solution.initial_guess
    ).to_dict(expand_to_batch=True, ravel=True, to_numpy=True)
    graphite_mass: NpFloat = CHO_output["C_s"]["species"]["mass"]["C_s"]

    assert {"C_s", "phase"} <= set(out["C_s"])
    np.testing.assert_array_equal(out["C_s"]["C_s"]["mass"], graphite_mass)
    np.testing.assert_array_equal(out["C"]["C_s"]["mass"], graphite_mass)


@pytest.mark.parametrize("case", CHO_CASES)
def test_CHO(CHO_output: dict[str, Any], case: str) -> None:
    """Partial pressures and graphite match FactSage and FastChem"""
    row: int = list(CHO_CASES).index(case)
    graphite: dict[str, Any] = CHO_output["C_s"]["species"]

    for reference in CHO_CASES[case][-1]:
        computed: dict[str, float] = {
            name: CHO_output["gas"]["species"]["partial_pressure"][name][row]
            for name in reference["partial_pressure"]
        }
        expected: dict[str, float] = dict(reference["partial_pressure"])
        if "C_s_activity" in reference:
            computed["C_s_activity"] = graphite["activity"]["C_s"][row]
            expected["C_s_activity"] = reference["C_s_activity"]
        if "C_s_mass" in reference:
            computed["C_s_mass"] = graphite["mass"]["C_s"][row]
            expected["C_s_mass"] = reference["C_s_mass"]

        np.testing.assert_allclose(
            np.log10(list(computed.values())),
            np.log10(list(expected.values())),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg=f"{case}: {list(expected)}",
        )


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2 with H2O and CO2 solubility."""
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d, CO2_d)

    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )
    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}
    mass_constraints: dict[str, ArrayLike] = {"C": OCEAN, "H": OCEAN}
    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    output: Output = EquilibriumModel(parameters).solve_with_default()

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 0.2582458752325180,
                    "H2_g": 0.2502809714412906,
                    "O2_g": 8.838513516896038e-08,
                    "CO_g": 59.65835224848439,
                    "CO2_g": 13.43793686555727,
                }
            }
        }
    }

    assert output.compare(target, rtol=TOLERANCE, atol=TOLERANCE, log=False)
