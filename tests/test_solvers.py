# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the alternate solver constructors in :mod:`atmodeller.solvers`.

:func:`atmodeller.solvers.make_solver_with_jit_single_path` is exercised elsewhere (e.g.
``tests/test_retracing.py`` via :class:`~atmodeller.classes.EquilibriumModel`), but the other
public constructors (:func:`~atmodeller.solvers.make_solver`,
:func:`~atmodeller.solvers.make_solver_with_jit_dual_path`, and
:func:`~atmodeller.solvers.make_solver_with_jit_batch_only`) are not. This module exercises those
directly, bypassing :class:`~atmodeller.classes.EquilibriumModel` since it is hardwired to
:data:`atmodeller.solvers.make_solver_with_jit`.
"""

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import ActivityConstraintProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth
from atmodeller.solvers import (
    make_solver,
    make_solver_with_jit_batch_only,
    make_solver_with_jit_dual_path,
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
graphite: PurePhase = PurePhase.from_species("C")


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


@pytest.mark.parametrize("with_condensate", [False, True])
def test_make_solver_dispatch_branches(with_condensate: bool) -> None:
    """Exercises both `lax.cond` branches inside `dispatch_solver` via `make_solver`.

    With no active-stability species the generic batch-retry path is used; with a graphite
    condensate present (`solve_for_stability=True` by default) the tau-sweep path is used.
    """
    parameters: Parameters = _build_parameters(with_condensate=with_condensate)

    has_active_stability: bool = bool(
        jnp.any(parameters.reaction_system.species.active_stability).item()
    )
    assert has_active_stability is with_condensate

    dispatch_solver: Callable = make_solver(parameters)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    base_solution_array: Array = _base_solution_array(parameters)

    output: Output = dispatch_solver(parameters, key, base_solution_array)

    _assert_valid_output(output, parameters)
    assert jnp.all(output.multi_attempt_solution.solver_success)


def test_make_solver_with_jit_dual_path() -> None:
    """Exercises `make_solver_with_jit_dual_path`, which JIT-compiles both dispatch branches.

    `make_solver_with_jit_dual_path` is just `eqx.filter_jit(make_solver(parameters))`, so its
    `dispatch_solver` source is identical to what `test_make_solver_dispatch_branches` already
    exercises for both the stability and no-stability paths (`lax.cond` traces both branches
    regardless of which is taken at runtime). This test only needs to confirm the `filter_jit`
    wrapping itself works; the no-condensate case is the cheaper of the two to construct.
    """
    parameters: Parameters = _build_parameters(with_condensate=False)

    solver: Callable = make_solver_with_jit_dual_path(parameters)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    base_solution_array: Array = _base_solution_array(parameters)

    output: Output = solver(parameters, key, base_solution_array)

    _assert_valid_output(output, parameters)
    assert jnp.all(output.multi_attempt_solution.solver_success)


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
    # make_batch_solver always reports a single attempt, so `solver_success` (from the underlying
    # optimistix Solution) is the meaningful per-element convergence signal here.
    assert jnp.all(output.multi_attempt_solution.solution.result == optx.RESULTS.successful)
    assert jnp.all(output.multi_attempt_solution.solver_success)
