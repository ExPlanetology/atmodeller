# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the alternate solver constructors in :mod:`atmodeller.solvers`.

:func:`atmodeller.solvers.make_solver_with_jit_single_path` is exercised elsewhere (e.g.
``tests/test_retracing.py`` via :class:`~atmodeller.classes.EquilibriumModel`), but the other
public constructors (:func:`~atmodeller.solvers.make_solver`,
:func:`~atmodeller.solvers.make_solver_with_jit_dual_path`, and
:func:`~atmodeller.solvers.make_solver_with_jit_batch_only`) are not. This module exercises those
directly, bypassing :class:`~atmodeller.classes.EquilibriumModel` since it is hardwired to
:data:`atmodeller.solvers.make_solver_with_jit`. The result reported by the tau sweep is tested
with a stub retry solver.
"""

import logging
from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from equinox._enum import EnumerationItem
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies, SolverParameters
from atmodeller.interfaces import ActivityConstraintProtocol
from atmodeller.jax_utils import MultiAttemptSolution
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth
from atmodeller.solvers import (
    make_solver,
    make_solver_with_jit_batch_only,
    make_solver_with_jit_dual_path,
    make_tau_sweep_solver,
)
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

# Gas species shared by all parameter sets below (mirrors tests/test_condensates.py)
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")
GAS_SPECIES: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

# Graphite condensate: created with the default `solve_for_stability=True`, so including it in a
# model is what flips `active_stability` to True for one or more species.
graphite: PurePhase = PurePhase.from_component("graphite")


def _build_parameters(*, with_condensate: bool, batch_size: int = 1) -> Parameters:
    """Builds Parameters for a C-H-O system, optionally with a stability-active graphite phase.

    Args:
        with_condensate: Whether to include a graphite condensate (activates stability solving)
        batch_size: Number of batch realisations to construct via array-valued mass constraints

    Returns:
        Parameters instance
    """
    condensates: tuple[PurePhase, ...] = (graphite,) if with_condensate else ()

    planet: BaseThermodynamicState = Planet.from_species(
        GAS_SPECIES, temperature=1200, condensates=condensates
    )

    activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.0)}

    oceans: ArrayLike = np.linspace(0.5, 2.0, batch_size) if batch_size > 1 else 1.0
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 2.0 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    parameters: Parameters = Parameters(
        state=planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )

    return parameters


def _base_solution_array(parameters: Parameters) -> Array:
    """Builds a NaN placeholder initial guess, matching EquilibriumModel.solve_with_default."""
    return jnp.full(
        (parameters.batch_size, parameters.reaction_system.species.number_species * 2),
        jnp.nan,
    )


def _assert_valid_output(output: Output, parameters: Parameters) -> None:
    """Common assertions that a solver call produced a well-formed, finite Output."""
    assert isinstance(output, Output)
    solution: Array = output.solution
    assert solution.shape == (
        parameters.batch_size,
        parameters.reaction_system.species.number_species * 2,
    )
    assert jnp.all(jnp.isfinite(solution))


def test_make_solver() -> None:
    """Exercises `make_solver` without JIT, executing the batch-retry branch of `dispatch_solver`.

    With no active-stability species the generic batch-retry path is used. The tau-sweep branch is
    executed under JIT by `test_make_solver_with_jit_dual_path`. `lax.cond` traces both branches
    in either case.
    """
    parameters: Parameters = _build_parameters(with_condensate=False)
    assert not bool(jnp.any(parameters.reaction_system.species.active_stability).item())

    dispatch_solver: Callable = make_solver(parameters)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    base_solution_array: Array = _base_solution_array(parameters)

    output: Output = dispatch_solver(parameters, key, base_solution_array)

    _assert_valid_output(output, parameters)
    assert jnp.all(output.multi_attempt_solution.success)


def test_make_solver_with_jit_dual_path() -> None:
    """Exercises `make_solver_with_jit_dual_path`, which JIT-compiles both dispatch branches.

    `make_solver_with_jit_dual_path` is `eqx.filter_jit(make_solver(parameters))`. With a graphite
    condensate present (`solve_for_stability=True` by default) the tau-sweep branch is executed,
    complementing `test_make_solver`, which executes the batch-retry branch without JIT.
    """
    parameters: Parameters = _build_parameters(with_condensate=True)
    assert bool(jnp.any(parameters.reaction_system.species.active_stability).item())

    solver: Callable = make_solver_with_jit_dual_path(parameters)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    base_solution_array: Array = _base_solution_array(parameters)

    output: Output = solver(parameters, key, base_solution_array)

    _assert_valid_output(output, parameters)
    assert jnp.all(output.multi_attempt_solution.success)


def test_make_solver_with_jit_batch_only() -> None:
    """Exercises `make_solver_with_jit_batch_only` with an explicitly batched Parameters.

    This constructor uses only the vmapped single-pass batch solver (no retry, no tau sweep), so
    an explicit batch dimension (mirroring the convention in
    tests/test_retracing.py and tests/performance/test_trappist1e.py) confirms the vmapping axes
    are derived and applied correctly.
    """
    batch_size: int = 4
    parameters: Parameters = _build_parameters(with_condensate=False, batch_size=batch_size)
    assert parameters.batch_size == batch_size

    solver: Callable = make_solver_with_jit_batch_only(parameters)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    base_solution_array: Array = _base_solution_array(parameters)

    output: Output = solver(parameters, key, base_solution_array)

    _assert_valid_output(output, parameters)
    assert jnp.all(output.multi_attempt_solution.success)


def _stub_solution(
    value: Float[Array, "batch solution"], converged: Bool[Array, " batch"], larger_tau: ArrayLike
) -> MultiAttemptSolution:
    """Stub solution that reports more steps and attempts for a solve at a larger tau and, as the
    batch retry solver, zero steps and attempts for a model that did not converge"""
    result_value: Array = jnp.full(
        converged.shape,
        optx.RESULTS.successful._value,  # pyright: ignore
    )
    result: optx.RESULTS = cast(
        optx.RESULTS,
        EnumerationItem(result_value, optx.RESULTS),  # pyright: ignore
    )
    num_steps: Array = jnp.where(converged, jnp.where(larger_tau, 5, 3), 0)
    attempts: Array = jnp.where(converged, jnp.where(larger_tau, 2, 1), 0)

    return MultiAttemptSolution(
        optx.Solution(value, result, None, {"num_steps": num_steps}, None), attempts
    )


@pytest.mark.parametrize(
    "solver_parameters",
    [SolverParameters(), SolverParameters(tau=1.0e-20, tau_max=1.0e-2, tau_num=3)],
    ids=["default_tau", "custom_tau"],
)
@pytest.mark.parametrize(
    "warm_start_converges, expected_steps, expected_attempts",
    [(True, [3, 3], [1, 1]), (False, [3, 0], [1, 0])],
    ids=["converged_at_final_tau", "failed_at_final_tau"],
)
def test_tau_sweep_reports_final_tau(
    solver_parameters: SolverParameters,
    warm_start_converges: bool,
    expected_steps: list[int],
    expected_attempts: list[int],
) -> None:
    """The tau sweep reports the solve at the final tau when one of two models fails the first
    solve, with the tau settings of ``SolverParameters``.

    The first model always converges, so it keeps the result of the first solve. The second model
    converges at a larger tau, but at the final tau only from a warm start and only if
    ``warm_start_converges``, so it reports the final step of the sweep. Each converged solve adds
    one to the initial guess, so the second model gains one per converged step of the
    ``tau_num``-step sweep. A solve below the final tau fails for both models, so a sweep that
    ignored ``solver_parameters.tau`` would fail the test.
    """
    final_tau: float = float(solver_parameters.tau)

    def batch_retry_solver(
        initial_guess: Array, parameters: Parameters, *args: object, **kwargs: object
    ) -> MultiAttemptSolution:
        tau: Array = parameters.solver_parameters.tau
        larger_tau: Array = tau > final_tau
        warm_start: Array = initial_guess[1, 0] > 0
        converged: Array = jnp.stack(
            [jnp.array(True), larger_tau | (warm_start & warm_start_converges)]
        ) & (tau >= final_tau)
        value: Array = jnp.where(converged[:, None], initial_guess + 1.0, initial_guess)

        return _stub_solution(value, converged, larger_tau)

    def batch_solver(initial_guess: Array, parameters: Parameters) -> MultiAttemptSolution:
        """Polishes without changing the solution"""
        return _stub_solution(initial_guess, jnp.ones(2, dtype=bool), False)

    parameters: Parameters = eqx.tree_at(
        lambda p: p.solver_parameters, _build_parameters(with_condensate=True), solver_parameters
    )
    tau_sweep_solver: Callable = make_tau_sweep_solver(batch_retry_solver, batch_solver)

    multi_sol: MultiAttemptSolution = tau_sweep_solver(
        jnp.zeros((2, 2)), parameters, jax.random.PRNGKey(0)
    )

    # The second model converges at every step before the last, and at the last if it converges
    sweep_steps: int = solver_parameters.tau_num - (0 if warm_start_converges else 1)
    np.testing.assert_array_equal(multi_sol.value[:, 0], [1.0, sweep_steps])
    np.testing.assert_array_equal(multi_sol.num_steps, expected_steps)
    np.testing.assert_array_equal(multi_sol.attempts, expected_attempts)
