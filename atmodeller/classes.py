# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Equilibrium model API

Provides :class:`EquilibriumModel`, the user-facing entry point for running thermodynamic
equilibrium calculations from a prepared :class:`~atmodeller.parameters.Parameters` object.

The module couples immutable model parameters to a compiled solver callable, then exposes small
wrapper methods for solving and parameter updates while preserving Equinox tree semantics.

Typical usage:

.. code-block:: python

    from atmodeller.classes import EquilibriumModel
    from atmodeller.parameters import Parameters

    # Create Parameters object (not shown)
    parameters = Parameters(...)
    model = EquilibriumModel(parameters)
    output = model.solve_with_default()
"""

from collections.abc import Callable, Mapping
from typing import Literal, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from atmodeller.containers import SolverParameters
from atmodeller.interfaces import ActivityConstraintProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import BaseThermodynamicState

SolverCallable = Callable[
    [Parameters, PRNGKeyArray, Float[Array, "#n_batch twice_species"]], Output
]


class EquilibriumModel(eqx.Module):
    """Main user-facing API to run equilibrium calculations

    The model bundles immutable :class:`~atmodeller.parameters.Parameters`, a compiled solver
    closure, and a random key used by stochastic solver routines.

    Args:
        parameters: Fully prepared model parameters, including state, constraints, and solver
            settings.
    """

    parameters: Parameters
    solver: SolverCallable
    key: PRNGKeyArray

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.solver = make_solver_with_jit(parameters)
        self.key = jax.random.PRNGKey(0)

    @classmethod
    def from_state(
        cls,
        state: BaseThermodynamicState,
        *,
        activity_constraints: Mapping[str, ActivityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        mass_units: Literal["mass", "moles"] = "mass",
        solver_parameters: SolverParameters | None = None,
    ) -> Self:
        """Builds a model by constructing :class:`~atmodeller.parameters.Parameters` internally.

        This is a convenience constructor equivalent to creating ``Parameters(...)`` first and
        then passing that object to :class:`EquilibriumModel`.

        Args:
            state: Thermodynamic state
            activity_constraints: Optional activity/fugacity constraint mapping. Defaults to
                ``None``.
            mass_constraints: Optional mass-constraint mapping. Defaults to ``None``.
            mass_units: Units used for ``mass_constraints``. Defaults to ``"mass"``.
            solver_parameters: Optional solver parameter object. Defaults to ``None``.

        Returns:
            A new :class:`EquilibriumModel` instance
        """
        parameters: Parameters = Parameters(
            state=state,
            activity_constraints=activity_constraints,
            mass_constraints=mass_constraints,
            mass_units=mass_units,
            solver_parameters=solver_parameters,
        )

        return cls(parameters)

    def solve(self, base_solution_array: Float[Array, "#n_batch twice_species"]) -> Output:
        """Runs the solver and returns the output state.

        Note:
            This method is intentionally a thin Python wrapper. The heavy numerical path is
            compiled inside ``self.solver``
            (created by :func:`~atmodeller.solvers.make_solver_with_jit`).
            The same key is reused for repeat calls, so stochastic branches remain deterministic.

        Args:
            base_solution_array: Initial guess for the solver, typically a broadcasted array of
                log number moles and log stabilities.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        return self.solver(self.parameters, self.key, base_solution_array)

    def solve_with_default(self) -> Output:
        """Runs the solver with a default initial guess.

        Note:
            Like :meth:`solve`, this method is a lightweight non-jitted wrapper around the
            already-jitted ``self.solver`` callable.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        base_solution_array: Float[Array, "#n_batch twice_species"] = jnp.full(
            (
                self.parameters.batch_size,
                self.parameters.reaction_system.species.number_species * 2,
            ),
            jnp.nan,
        )

        return self.solver(self.parameters, self.key, base_solution_array)

    def rebuild_solver(self) -> Self:
        """Rebuilds the compiled solver from the model's current parameters.

        Use this after parameter updates that may have changed array-leaf shapes or batching in a
        way that invalidates the vmapping axes captured by the existing solver closure, or that
        invalidates the original leaf-shape broadcasting assumptions preserved by the
        ``update`` methods on the parameter containers.

        Returns:
            A new instance of :class:`EquilibriumModel` with a rebuilt solver
        """
        solver_rebuilt: SolverCallable = make_solver_with_jit(self.parameters)
        model_rebuilt: EquilibriumModel = eqx.tree_at(lambda m: m.solver, self, solver_rebuilt)

        return cast(Self, model_rebuilt)

    def update_constraints(
        self,
        *,
        activity_constraints: Mapping[str, ActivityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        mass_units: Literal["mass", "moles"] = "mass",
    ) -> Self:
        """Returns a model with updated mass and activity/fugacity constraints.

        This is a convenience wrapper around
        :meth:`atmodeller.parameters.Parameters.update_constraints` that preserves the immutable
        Equinox tree semantics of the model.

        Args:
            activity_constraints: New activity/fugacity constraint mapping. Defaults to ``None``.
            mass_constraints: New mass-constraint mapping. Defaults to ``None``.
            mass_units: Units used for ``mass_constraints``. Defaults to ``"mass"``.

        Returns:
            A new model instance with updated parameters
        """
        parameters_updated: Parameters = self.parameters.update_constraints(
            activity_constraints=activity_constraints,
            mass_constraints=mass_constraints,
            mass_units=mass_units,
        )
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m.parameters, self, parameters_updated
        )

        return cast(Self, model_updated)

    def update_state(self, *args: object, **kwargs: object) -> Self:
        """Returns a model with an updated thermodynamic state.

        Arguments are forwarded to ``self.parameters.update_state``, which in turn forwards them
        to the underlying thermodynamic state's ``update`` implementation.

        Args:
            *args: Positional arguments accepted by the state ``update`` method
            **kwargs: Keyword arguments accepted by the state ``update`` method

        Returns:
            A new model instance with updated parameters
        """
        parameters_updated: Parameters = self.parameters.update_state(*args, **kwargs)
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m.parameters, self, parameters_updated
        )

        return cast(Self, model_updated)
