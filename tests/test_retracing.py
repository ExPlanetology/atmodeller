# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests JAX retracing and recompilation behaviour for workflows.

These tests verify two complementary expectations:
1. Common parameter/state updates keep pytree structure stable and do not retrace the solver.
2. Rebuilding the solver changes model static structure and therefore retraces.
"""

import logging
from collections.abc import Mapping
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import ActivityConstraintProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.jax_utils import NpFloat
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
from atmodeller.solubility.library import get_solubility_models
from atmodeller.state import Planet
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

# The upstream error text currently includes the grammatical typo "it is has".
RETRACE_ERROR_SUBSTRING: str = "it is has now been traced 2 times"


@pytest.fixture(scope="module")
def equilibrium_model() -> EquilibriumModel:
    """Builds a deterministic multi-batch C-H-O model for trace-stability tests.

    Returns:
        Equilibrium model
    """
    # Gas species
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
    CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
    CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
    CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")
    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    # Melt species
    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )
    CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CO2", solubility=solubility_models["CO2_basalt_dixon95"], include_in_phase_mass=False
    )
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d, CO2_d)

    # Thermodynamic state
    planet: Planet = Planet.from_species(
        gas_species=gas_species, silicate_melt_species=melt_species
    )

    # Activity constraints
    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    # Mass constraints: this establishes the model batch size used in trace checks.
    n_models: int = 10

    rng = np.random.default_rng(42)
    oceans: NpFloat = np.exp(rng.uniform(np.log(0.1), np.log(10), size=n_models))
    ch_ratio: float = 1
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = ch_ratio * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    parameters: Parameters = Parameters(
        state=planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    return EquilibriumModel(parameters)


@pytest.fixture(scope="module")
def value_update_workflow() -> Callable[[EquilibriumModel], bool]:
    """Creates a workflow that applies structure-preserving model updates.

    The wrapped ``call_solver`` function is guarded with ``assert_max_traces(max_traces=1)`` to
    ensure updates that preserve pytree structure do not trigger retracing.

    Returns:
        Callable that executes several solve/update steps and returns ``True`` on completion.
    """

    @eqx.filter_jit
    # Assert that the solver is only traced once. If a test fails, it likely means that the
    # solver is being retraced on every call, which would be a major performance issue.
    @eqx.debug.assert_max_traces(max_traces=1)
    def call_solver(model: EquilibriumModel, initial_solution: Array) -> Output:
        """Wraps ``model.solve`` with a single-trace assertion."""
        return model.solve(initial_solution)

    def workflow(model: EquilibriumModel) -> bool:
        # Keep shape and dtype explicit so this input is trace-stable across calls.
        prev_solution: Array = jnp.full(
            (model.parameters.batch_size, model.parameters.species.number_species * 2),
            jnp.nan,
            dtype=float,
        )

        # First call to the solver. This should trigger compilation.
        initial_output: Output = call_solver(model, prev_solution)
        # jax.debug.print("First call to solver complete.")

        # Second call with same model structure should not retrace.
        warm_start_output: Output = call_solver(model, initial_output.solution)
        # jax.debug.print("Second call to solver complete.")

        # Update constraints and call the solver again. This should NOT trigger recompilation
        # because the pytree structure is unchanged, even though the values are
        # different.
        model = model.update_constraints(
            mass_constraints={"H": model.parameters.mass_constraints.abundance_dict["H"] * 2}
        )
        mass_updated_output: Output = call_solver(model, warm_start_output.solution)

        # Species keys are accepted at construction but not when updating, which raises
        with pytest.raises(ValueError, match="H2O"):
            model.update_constraints(mass_constraints={"H2O": 1.0e21})
        # jax.debug.print("Third call to solver complete.")

        # Replace oxygen mass constraint with a fugacity-style activity constraint.
        # This still should not retrace because only values change.
        model = model.update_constraints(
            mass_constraints={"O": jnp.nan}, activity_constraints={"O2_g": IronWustiteBuffer(1.0)}
        )
        activity_updated_output: Output = call_solver(model, mass_updated_output.solution)
        # jax.debug.print("Fourth call to solver complete.")

        # Update scalar planetary state values; structure remains unchanged.
        model = model.update_state(surface_radius=earth.radius * 2, mantle_melt_fraction=0.5)
        _ = call_solver(model, activity_updated_output.solution)
        # jax.debug.print("Fifth call to solver complete.")

        return True

    return workflow


@pytest.fixture(scope="module")
def jitted_value_update_workflow(
    value_update_workflow: Callable[[EquilibriumModel], bool],
) -> Callable[[EquilibriumModel], bool]:
    """Creates a JIT-compiled wrapper around the full workflow.

    This adds a second trace guard around the outer workflow itself, in addition to the inner
    ``call_solver`` guard defined in ``value_update_workflow``.

    Returns:
        Jitted callable that should remain trace-stable across repeated calls.
    """

    @eqx.filter_jit
    # Assert that the outer workflow is only traced once. Failures here indicate retracing at
    # the workflow level, independent of the inner solver-level guard.
    @eqx.debug.assert_max_traces(max_traces=1)
    def jitted_workflow(model: EquilibriumModel) -> bool:
        """Runs the non-jitted workflow inside an outer JIT trace guard."""
        return value_update_workflow(model)

    return jitted_workflow


def test_workflow_does_not_retrace_solver(
    equilibrium_model: EquilibriumModel, value_update_workflow: Callable[[EquilibriumModel], bool]
) -> None:
    """Ensures non-jitted workflow reuse does not retrace ``call_solver``."""

    assert value_update_workflow(equilibrium_model)


def test_jitted_workflow_does_not_retrace(
    equilibrium_model: EquilibriumModel,
    jitted_value_update_workflow: Callable[[EquilibriumModel], bool],
) -> None:
    """Ensures jitted workflow reuse does not retrace the workflow or the solver."""

    assert jitted_value_update_workflow(equilibrium_model)


def test_repeated_jitted_workflow_calls_do_not_retrace(
    equilibrium_model: EquilibriumModel,
    jitted_value_update_workflow: Callable[[EquilibriumModel], bool],
) -> None:
    """Ensures repeated jitted workflow calls stay trace-stable across iterations."""
    for i in range(10):
        logger.debug("Loop iteration %d", i)
        assert jitted_value_update_workflow(equilibrium_model)


def test_rebuilding_solver_retraces_call_solver(
    equilibrium_model: EquilibriumModel, value_update_workflow: Callable[[EquilibriumModel], bool]
) -> None:
    """Ensures changing model static structure triggers a retrace error.

    ``rebuild_solver`` swaps in a new solver callable, changing static pytree content. This test
    uses the non-jitted workflow so the failure is isolated to solver-level retracing.
    """
    # The first successful call consumes the single allowed trace for ``call_solver``.
    value_update_workflow(equilibrium_model)

    equilibrium_model = equilibrium_model.rebuild_solver()

    with pytest.raises(RuntimeError) as exc_info:
        value_update_workflow(equilibrium_model)

    # Keep exact substring match for the trace-count regression signal.
    assert RETRACE_ERROR_SUBSTRING in str(exc_info.value)
