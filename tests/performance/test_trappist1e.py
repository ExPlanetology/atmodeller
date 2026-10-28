# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling the solver performance and memory usage using Trappist-1e as a test case."""

import copy
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

from atmodeller import (
    ChemicalSpecies,
    EquilibriumModel,
    Planet,
    ReservoirSpecies,
    debug_logger,
    earth,
)
from atmodeller.interfaces import ActivityConstraintProtocol, SolubilityProtocol
from atmodeller.jax_utils import NpFloat
from atmodeller.output import Output
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import bulk_silicate_earth_abundances
from atmodeller.sci_utils import trappist1e as trappist1e_parameters
from atmodeller.solubility import get_solubility_models
from atmodeller.state import BaseThermodynamicState
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

# For no particular reason, use 24 as the random seed
RANDOM_SEED: int = 24

# Gas species
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")
N2_g: ChemicalSpecies = ChemicalSpecies.create_gas("N2")
H3N_g: ChemicalSpecies = ChemicalSpecies.create_gas("H3N")
S2_g: ChemicalSpecies = ChemicalSpecies.create_gas("S2")
H2S_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2S")
O2S_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2S")
OS_g: ChemicalSpecies = ChemicalSpecies.create_gas("OS")
Cl2_g: ChemicalSpecies = ChemicalSpecies.create_gas("Cl2")
ClH_g: ChemicalSpecies = ChemicalSpecies.create_gas("ClH")

gas_species: tuple[ChemicalSpecies, ...] = (
    H2_g,
    H2O_g,
    O2_g,
    CO_g,
    CO2_g,
    CH4_g,
    N2_g,
    H3N_g,
    S2_g,
    H2S_g,
    O2S_g,
    OS_g,
    Cl2_g,
    ClH_g,
)


def trappist1e_elemental_abundances() -> dict:
    """Returns the elemental abundances for TRAPPIST-1e used in :cite:t:`BTH25`."""

    earth_bse: dict[str, dict[str, float]] = bulk_silicate_earth_abundances()

    trappist1e_bse: dict[str, dict[str, float]] = copy.deepcopy(earth_bse)
    mass_scale_factor: float = trappist1e_parameters.mantle_mass / earth.mantle_mass

    for element, values in trappist1e_bse.items():
        trappist1e_bse[element] = {key: value * mass_scale_factor for key, value in values.items()}  # type: ignore

    return trappist1e_bse


def trappist1e_constraints(number_of_realisations: int) -> tuple[dict, dict]:
    """Returns the constraints for TRAPPIST-1e used in :cite:t:`BTH25`.

    Args:
        number_of_realisations: Number of realisations to generate for the constraints

    Returns:
        Tuple of (fugacity_constraints, mass_constraints)
    """

    np.random.seed(RANDOM_SEED)
    # Log uniform sampling
    log10_number_oceans: NpFloat = np.random.uniform(-1, 1, number_of_realisations)
    log10_ch_ratios: NpFloat = np.random.uniform(-1, 1, number_of_realisations)
    fO2_log10_shifts: NpFloat = np.random.uniform(-5, 5, number_of_realisations)

    h_kg = earth.oceans_to_hydrogen_mass(10**log10_number_oceans)
    c_kg = h_kg * 10**log10_ch_ratios

    fugacity_constraints: dict[str, ActivityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(fO2_log10_shifts)
    }

    trappist1e_bse: dict[str, dict[str, float]] = trappist1e_elemental_abundances()

    mass_constraints: dict[str, Any] = {
        "H": h_kg,
        "C": c_kg,
        "N": trappist1e_bse["N"]["mean"],
        "S": trappist1e_bse["S"]["mean"],
        "Cl": trappist1e_bse["Cl"]["mean"],
    }

    return fugacity_constraints, mass_constraints


@pytest.fixture(scope="module")
def trappist1e_with_solubility(request) -> EquilibriumModel:
    """Sets up a model of TRAPPIST-1e with solubility constraints.

    Pass the number_of_realisations via pytest's indirect parametrisation.
    """
    number_of_realisations: int = request.param
    logger.info(
        "creating TRAPPIST-1e model with solubility constraints for %d realisation(s)",
        number_of_realisations,
    )

    # Melt species
    include_in_phase_mass: bool = False

    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O",
        solubility=solubility_models["H2O_basalt_dixon95"],
        include_in_phase_mass=include_in_phase_mass,
    )
    H2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2",
        solubility=solubility_models["H2_basalt_hirschmann12"],
        include_in_phase_mass=include_in_phase_mass,
    )
    CO_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CO",
        solubility=solubility_models["CO_basalt_yoshioka19"],
        include_in_phase_mass=include_in_phase_mass,
    )
    CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CO2",
        solubility=solubility_models["CO2_basalt_dixon95"],
        include_in_phase_mass=include_in_phase_mass,
    )
    CH4_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CH4",
        solubility=solubility_models["CH4_basalt_ardia13"],
        include_in_phase_mass=include_in_phase_mass,
    )
    N2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "N2",
        solubility=solubility_models["N2_basalt_libourel03"],
        include_in_phase_mass=include_in_phase_mass,
    )
    S2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "S2",
        solubility=solubility_models["S2_basalt_boulliung23"],
        include_in_phase_mass=include_in_phase_mass,
    )
    Cl2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "Cl2",
        solubility=solubility_models["Cl2_basalt_thomas21"],
        include_in_phase_mass=include_in_phase_mass,
    )
    melt_species: tuple[ReservoirSpecies, ...] = (
        H2O_d,
        H2_d,
        CO_d,
        CO2_d,
        CH4_d,
        N2_d,
        S2_d,
        Cl2_d,
    )

    # Condensates
    C_graphite: PurePhase = PurePhase.from_end_member("graphite")
    condensates_graphite_only: tuple[PurePhase, ...] = (C_graphite,)

    magma_ocean_temperature: float = 1800.0
    # In the paper we perform simulations at 0.1 and 1.0 melt fraction
    mantle_melt_fraction: float = 1.0

    trappist1e_magma_ocean: BaseThermodynamicState = Planet.from_species(
        gas_species,
        temperature=magma_ocean_temperature,
        planet_mass=trappist1e_parameters.mass,
        surface_radius=trappist1e_parameters.radius,
        mantle_melt_fraction=mantle_melt_fraction,
        silicate_melt_species=melt_species,
        condensates=condensates_graphite_only,
    )

    # Constraints
    fugacity_constraints, mass_constraints = trappist1e_constraints(number_of_realisations)

    model: EquilibriumModel = EquilibriumModel.from_state(
        trappist1e_magma_ocean,
        mass_constraints=mass_constraints,
        activity_constraints=fugacity_constraints,
    )

    return model


def time_jax_compile_and_run(
    f: Callable[..., Output], n_exec: int = 5, *args, **kwargs
) -> tuple[float, float, Output, list[Output]]:
    """Times JAX compilation and averages execution time over n_exec runs of a solve.

    Args:
        f: The solve to call and time, which returns an output
        n_exec: Number of execution runs to average. Defaults to ``5``.
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (compile_time, avg_exec_time, result1, results)
    """
    # Compilation + first call
    start = time.perf_counter()
    result1 = f(*args, **kwargs)
    result1.solution.block_until_ready()
    end = time.perf_counter()
    compile_time = end - start

    # Execution only (average over n_exec)
    exec_times = []
    results = []
    for _ in range(n_exec):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        result.solution.block_until_ready()
        end = time.perf_counter()
        exec_times.append(end - start)
        results.append(result)

    avg_exec_time = sum(exec_times) / n_exec

    logger.info(f"JAX compile+first call time: {compile_time:.4f} s")
    logger.info(f"JAX execution (call) avg time over {n_exec} runs: {avg_exec_time:.4f} s")

    return compile_time, avg_exec_time, result1, results


# Parametrise the fixture with different numbers of realisations
@pytest.mark.performance
@pytest.mark.parametrize("trappist1e_with_solubility", [1, 100, 1000], indirect=True)
# Extend to 10,000 (or 30,000) realisations for testing the limits of memory usage and performance
# @pytest.mark.parametrize(
#    "trappist1e_with_solubility", [1, 100, 1000, 10000, 20000, 30000], indirect=True
# )
def test_trappist1e_with_solubility_batch(trappist1e_with_solubility: EquilibriumModel) -> None:
    """Profiles JAX compilation and execution time for TRAPPIST-1e with solubility constraints,
    and checks that every realisation is solved."""
    compile_time, avg_exec_time, first_result, _ = time_jax_compile_and_run(
        trappist1e_with_solubility.solve_with_default
    )
    logger.info(
        "TRAPPIST-1e with solubility constraints (batch size %d): compile time = %.4f s,"
        " avg execution time = %.4f s",
        trappist1e_with_solubility.parameters.batch_size,
        compile_time,
        avg_exec_time,
    )

    # The timings are logged rather than asserted, since they depend on the machine. The sampled
    # conditions span a wide range, so every realisation being solved checks robustness at scale.
    assert np.all(np.asarray(first_result.multi_attempt_solution.success))
