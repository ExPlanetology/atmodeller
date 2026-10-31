# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

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

import logging
from collections.abc import Callable, Mapping
from typing import Literal, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from atmodeller.containers import SolverParameters, SpeciesCollection, get_formula_matrix
from atmodeller.initializer import ElementPotentialInitializer, InitializerProtocol
from atmodeller.interfaces import ActivityConstraintProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import BaseThermodynamicState

logger: logging.Logger = logging.getLogger(__name__)

SolverCallable = Callable[
    [Parameters, PRNGKeyArray, Float[Array, "n_batch twice_species"]], Output
]


class EquilibriumModel(eqx.Module):
    """Main user-facing API to run equilibrium calculations

    The model bundles immutable :class:`~atmodeller.parameters.Parameters`, an initializer for
    the solver's initial guess, a compiled solver closure, and a random key used by stochastic
    solver routines.

    Args:
        parameters: Fully prepared model parameters, including state, constraints, and solver
            settings.
        initializer: Initializer for the cold initial guess, used for batch elements without an
            initial guess. Defaults to
            :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`.
    """

    parameters: Parameters
    initializer: InitializerProtocol = eqx.field(static=True)
    solver: SolverCallable
    key: PRNGKeyArray

    def __init__(self, parameters: Parameters, initializer: InitializerProtocol | None = None):
        self.parameters = parameters
        self.initializer = ElementPotentialInitializer() if initializer is None else initializer
        self.solver = make_solver_with_jit(parameters, self.initializer)
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
        initializer: InitializerProtocol | None = None,
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
            initializer: Optional initializer for the cold initial guess. Defaults to ``None``.

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

        return cls(parameters, initializer)

    def solve(self, base_solution_array: Float[Array, "n_batch twice_species"]) -> Output:
        """Runs the solver and returns the output state.

        Note:
            This method is intentionally a thin Python wrapper. The heavy numerical path is
            compiled inside ``self.solver``
            (created by :func:`~atmodeller.solvers.make_solver_with_jit`).
            The same key is reused for repeat calls, so stochastic branches remain deterministic.

        Args:
            base_solution_array: Initial guess for the solver, typically a broadcasted array of
                log number moles and log stabilities. Rows are used as given, and rows that
                contain ``NaN`` are replaced by the cold guess of the model's initializer.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        self._check_constraints()
        self._log_initial_guess(base_solution_array)

        return self.solver(self.parameters, self.key, base_solution_array)

    def _check_constraints(self) -> None:
        """Checks that the constraints of each model match the components of the species

        The activity and mass constraints together must number the components of the species,
        which is the rank of the formula matrix, less the condensates that are imposed stable,
        since the activity of each of these is fixed, and the mass constraints must be
        independent. The number of components is fewer than the number of elements when elements
        occur in fixed ratios in every species, e.g. H and O when the only species are ``H2O_g``
        and ``H2O_l``. Nothing is checked if the constraints are traced (inside JIT).

        Raises:
            ValueError: If the constraints of a model do not match the components
        """
        try:
            activity_active: np.ndarray = np.asarray(
                self.parameters.activity_constraints.active(), dtype=bool
            )
            mass_active: np.ndarray = np.asarray(
                self.parameters.mass_constraints.active(), dtype=bool
            )
        except jax.errors.TracerArrayConversionError:
            return

        batch_size: int = self.parameters.batch_size
        activity_active = np.broadcast_to(activity_active, (batch_size, activity_active.shape[-1]))
        mass_active = np.broadcast_to(mass_active, (batch_size, mass_active.shape[-1]))

        species: SpeciesCollection = self.parameters.mass_constraints.species
        species_names: tuple[str, ...] = self.parameters.activity_constraints.species.species_names
        elements: tuple[str, ...] = species.unique_elements
        formula_matrix: np.ndarray = get_formula_matrix(species)
        number_components: int = species.number_components
        # A condensate that is imposed stable has no unknowns, and its fixed activity constrains
        # the system like an activity constraint
        imposed: list[str] = [
            species_.name for species_ in species.reaction_species if species_.number_solution == 0
        ]
        number_needed: int = number_components - len(imposed)

        # Models with the same active constraints are checked once
        patterns, first_model = np.unique(
            np.concatenate((activity_active, mass_active), axis=-1), axis=0, return_index=True
        )
        for pattern, model in zip(patterns, first_model):
            activity_row, mass_row = pattern[: len(species_names)], pattern[len(species_names) :]
            constrained: list[str] = [
                name for name, active in zip(species_names, activity_row) if active
            ] + [element for element, active in zip(elements, mass_row) if active]
            if len(constrained) != number_needed:
                imposed_note: str = (
                    f", less the {len(imposed)} condensates imposed stable {imposed}"
                    if imposed
                    else ""
                )
                raise ValueError(
                    f"Model {model} has {len(constrained)} constraints {constrained}, but "
                    f"{number_needed} are needed: the {number_components} components of the "
                    f"species{imposed_note}. The number of components is the number of elements "
                    f"{elements} unless elements occur in fixed ratios in every species, which "
                    "then need only one constraint between them."
                )
            mass_elements: list[str] = [e for e, active in zip(elements, mass_row) if active]
            if np.linalg.matrix_rank(formula_matrix[mass_row]) != len(mass_elements):
                raise ValueError(
                    f"Model {model} has mass constraints on {mass_elements} that are not "
                    "independent, since some of these elements occur in fixed ratios in every "
                    "species. Constrain only one element of each fixed ratio."
                )

    def _log_initial_guess(self, base_solution_array: ArrayLike) -> None:
        """Logs how many models start from the initializer's cold guess and how many from the
        given initial guess. Nothing is logged if the initial guess is traced (inside JIT).

        Args:
            base_solution_array: Initial guess for the solver
        """
        try:
            guess: np.ndarray = np.atleast_2d(np.asarray(base_solution_array))
        except jax.errors.TracerArrayConversionError:
            return

        batch_size: int = self.parameters.batch_size
        cold: np.ndarray = np.broadcast_to(np.any(np.isnan(guess), axis=-1), (batch_size,))
        logger.info(
            "Initial guess: %d model(s) from %s, %d model(s) from the given initial guess",
            np.count_nonzero(cold),
            type(self.initializer).__name__,
            batch_size - np.count_nonzero(cold),
        )

    def solve_with_default(self) -> Output:
        """Runs the solver with a default initial guess.

        Note:
            Like :meth:`solve`, this method is a lightweight non-jitted wrapper around the
            already-jitted ``self.solver`` callable.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        base_solution_array: Float[Array, "n_batch twice_species"] = jnp.full(
            (
                self.parameters.batch_size,
                self.parameters.reaction_system.species.number_species * 2,
            ),
            jnp.nan,
        )

        return self.solve(base_solution_array)

    def rebuild_solver(self) -> Self:
        """Rebuilds the compiled solver from the model's current parameters.

        Use this after parameter updates that may have changed array-leaf shapes or batching in a
        way that invalidates the vmapping axes captured by the existing solver closure, or that
        invalidates the original leaf-shape broadcasting assumptions preserved by the
        ``update`` methods on the parameter containers.

        Returns:
            A new instance of :class:`EquilibriumModel` with a rebuilt solver
        """
        solver_rebuilt: SolverCallable = make_solver_with_jit(self.parameters, self.initializer)
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
