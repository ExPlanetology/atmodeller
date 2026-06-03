# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameter containers for thermodynamic calculations.

This module defines immutable, JAX-friendly parameter objects used by the solver:

- :class:`ActivityConstraintSet`: stores per-species activity/fugacity constraints.
- :class:`MassConstraintSet`: stores elemental abundance constraints in moles.
- :class:`Parameters`: bundles state, constraints, and solver settings into one object.

Factory methods validate and normalize user inputs, while ``update`` methods return new instances
with leaf shapes kept stable for efficient JAX transformations, also within jitted workflows.

Note:
    Construct parameter containers once outside ``jit`` (or other JAX transforms), then use
    ``update`` methods to preserve leaf signatures and avoid unnecessary retracing.
"""

from collections.abc import Callable, Mapping
from typing import Literal, Optional, Self, cast

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import Composition, CompositionItem, Formula

from atmodeller.containers import SolverParameters, SpeciesCollection
from atmodeller.interfaces import ActivityConstraintProtocol, SpeciesProtocol
from atmodeller.jax_utils import FloatArray, as_j64, get_batch_size, stack_broadcast, to_hashable
from atmodeller.reactions import ReactionSystem
from atmodeller.state import BaseThermodynamicState

VALID_MASS_UNITS: tuple[str, str] = ("mass", "moles")


def _validate_mass_units(units: str) -> Literal["mass", "moles"]:
    """Validates and returns supported mass constraint units.

    Args:
        units: Units of mass constraints, either "mass" or "moles".

    Returns:
        Validated units

    Raises:
        ValueError: If the provided units are not supported.
    """
    if units not in VALID_MASS_UNITS:
        raise ValueError(f"Invalid units '{units}'. Expected one of {VALID_MASS_UNITS}.")
    return cast(Literal["mass", "moles"], units)


class FixedActivityConstraint(eqx.Module):
    """A fixed activity constraint

    This must adhere to :class:`~atmodeller.interfaces.ActivityConstraintProtocol`.

    Args:
        activity: Activity (dimensionless) or fugacity referenced to 1 bar for gaseous species.
            Defaults to :data:`jax.numpy.nan` to indicate no constraint.
    """

    activity: Array = eqx.field(converter=as_j64, default=jnp.nan)
    """Activity"""

    def active(self) -> Bool[Array, "..."]:
        """Active activity constraint

        Returns:
            ``True`` if the activity constraint is active, otherwise ``False``
        """
        return ~jnp.isnan(self.activity)

    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Log activity

        Note:
            This method is designed to be fully compatible with both :func:`jax.vmap` and explicit
            batched input. It supports broadcasting of ``temperature`` and ``pressure`` to match
            the batch size, as required by output routines, while also working with per-instance
            vectorization as used by the engine and solver.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            - Log activity (dimensionless) for condensed species, or
            - Log fugacity referenced to 1 bar for gaseous species, or
            - :data:`jax.numpy.nan` if the constraint is not active
        """
        broadcast_shape: tuple[int, ...] = jnp.broadcast_shapes(
            jnp.shape(self.activity), jnp.shape(temperature), jnp.shape(pressure)
        )
        # jax.debug.print("broadcast_shape = {out}", out=broadcast_shape)

        return jnp.broadcast_to(jnp.log(self.activity), broadcast_shape)


class ActivityConstraintSet(eqx.Module):
    """Activity/fugacity constraints applied to species in the system

    Prefer constructing this object once outside :func:`jax.jit` and applying :meth:`update` inside
    workflows.

    Args:
        species: Species collection
        activity_constraints: Mapping of a species name and an activity constraint. Defaults to
            ``None``.
    """

    species: SpeciesCollection
    """Species collection"""
    constraints_dict: dict[str, ActivityConstraintProtocol]
    """Activity constraints dictionary mapping species name to activity constraint"""

    def __init__(
        self,
        species: SpeciesCollection,
        activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
    ):
        activity_constraints_: Mapping[str, ActivityConstraintProtocol] = (
            activity_constraints if activity_constraints is not None else {}
        )

        constraints_dict: dict[str, ActivityConstraintProtocol] = {}

        for species_name in species.species_names:
            if species_name in activity_constraints_:
                constraints_dict[species_name] = activity_constraints_[species_name]
            else:
                # No imposed activity/fugacity since NaNs are returned by default.
                constraints_dict[species_name] = FixedActivityConstraint()

        # jax.debug.print("constraints_dict = {out}", out=constraints_dict)

        self.species = species
        self.constraints_dict = constraints_dict

    @property
    def ordered_constraints(self) -> tuple[ActivityConstraintProtocol, ...]:
        """Activity constraints in the canonical species order of the species collection

        This explicit ordering is required for stable internal JAX operations. Relying on
        dictionary iteration for semantic species alignment is fragile across transformed code
        paths, so constraints are always materialized in ``species.species_names`` order.
        """
        return tuple(
            self.constraints_dict[species_name] for species_name in self.species.species_names
        )

    def active(self) -> Bool[Array, "... species"]:
        """Active activity constraints

        Returns:
            Mask indicating whether activity constraints are active or not
        """
        arrays: list[Array] = [constraint.active() for constraint in self.ordered_constraints]
        active_constraints: Bool[Array, "... species"] = stack_broadcast(arrays)
        # jax.debug.print("active_constraints = {out}", out=active_constraints)

        return active_constraints

    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> Float[Array, "... species"]:
        """Log activity

        Note:
            This method is designed to be fully compatible with both :func:`jax.vmap` and explicit
            batched input. It supports broadcasting of ``temperature`` and ``pressure`` to match
            the batch size, as required by output routines that call
            :func:`atmodeller.initial_solution.auto_initial_guess` with batched input outside of
            :func:`jax.vmap`, while also working with per-instance vectorization as used by the
            engine and solver.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log activity (dimensionless) or log fugacity referenced to 1 bar for gaseous species
        """
        # Compute the broadcast shape for all constraints, temperature, and pressure
        arrays: list[Array] = [constraint.active() for constraint in self.ordered_constraints]
        temp_shape: tuple[int, ...] = jnp.shape(temperature)
        pres_shape: tuple[int, ...] = jnp.shape(pressure)
        broadcast_shape: tuple[int, ...] = jnp.broadcast_shapes(
            *[arr.shape for arr in arrays], temp_shape, pres_shape
        )
        # jax.debug.print("broadcast_shape = {out}", out=broadcast_shape)

        def make_broadcasting_log_activity(log_activity_func: Callable) -> Callable:
            """Wrapper to ensure output is shape consistent for :func:`jax.lax.switch`"""

            def wrapped_log_activity(temperature: ArrayLike, pressure: ArrayLike) -> Array:
                out = log_activity_func(temperature, pressure)
                return jnp.broadcast_to(out, broadcast_shape)

            return wrapped_log_activity

        activity_funcs: list[Callable] = [
            to_hashable(make_broadcasting_log_activity(constraint.log_activity))
            for constraint in self.ordered_constraints
        ]
        # jax.debug.print("activity_funcs = {out}", out=activity_funcs)

        # TODO: this is likely legacy code, but in case something imminently breaks it is kept here
        # for now - DJB, 15/04/2026.
        # Temperature must be a float array to ensure branches have identical types
        # temperature = as_j64(temperature)

        def apply_activity(index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike) -> Array:
            # jax.debug.print("index = {out}", out=index)
            return lax.switch(index, activity_funcs, temperature, pressure)

        indices: Integer[Array, " species"] = jnp.arange(len(self.ordered_constraints))
        vmap_activity: Callable = eqx.filter_vmap(
            apply_activity, in_axes=(0, None, None), out_axes=-1
        )
        log_activity: Float[Array, "... species"] = vmap_activity(indices, temperature, pressure)
        # jax.debug.print("log_activity = {out}", out=log_activity)

        return log_activity

    def update(self, new_constraints: Mapping[str, ActivityConstraintProtocol]) -> Self:
        """Updates the activity/fugacity constraints with new values from a dictionary.

        Note:
            Prefer this method over re-instantiation when reusing a compiled JAX function.
            It enforces leaf shape/dtype stability preventing unnecessary retracing. It also
            supports partial updates, leaving unspecified species unchanged. Constraint activation
            and deactivation are determined by the concrete
            :class:`~atmodeller.interfaces.ActivityConstraintProtocol` implementation. For
            example, implementations such as :class:`~atmodeller.parameters.FixedActivityConstraint`
            treat internal ``NaN`` values as inactive.

        Args:
            new_constraints: Dictionary with new constraint values for some or all species. The
                keys should be species names and the values should be the new constraint values.
                Original constraints that are not included in the ``new_constraints`` dictionary
                will be retained. Unknown species keys are ignored.

        Returns:
            An instance with updated activity/fugacity constraints
        """
        constraints_dict: dict[str, ActivityConstraintProtocol] = dict(self.constraints_dict)

        for species_name, new_value in new_constraints.items():
            if species_name not in constraints_dict:
                continue

            original_value: ActivityConstraintProtocol = constraints_dict[species_name]

            original_dynamic, _ = eqx.partition(original_value, eqx.is_array)
            new_dynamic, new_static = eqx.partition(new_value, eqx.is_array)

            # Keep leaf signatures stable to avoid unnecessary retracing under JAX transformations.
            new_dynamic_stable = jtu.tree_map(
                lambda new_leaf, original_leaf: jnp.broadcast_to(
                    jnp.asarray(new_leaf, dtype=original_leaf.dtype), original_leaf.shape
                ),
                new_dynamic,
                original_dynamic,
            )
            constraints_dict[species_name] = eqx.combine(new_dynamic_stable, new_static)

        activity_constraint_set_updated: ActivityConstraintSet = eqx.tree_at(
            lambda c: c.constraints_dict, self, constraints_dict
        )

        return cast(Self, activity_constraint_set_updated)


class MassConstraintSet(eqx.Module):
    """Mass/abundance constraints applied to elements in the system

    Prefer constructing this object once outside ``jit`` and applying :meth:`update` inside
    workflows.

    Args:
        species: Species collection
        mass_constraints: Dictionary mapping element or species names to mass/abundance arrays.
            Defaults to ``None``.
        units: Units of ``mass_constraints``. Defaults to ``mass``.
        squeeze_input: Whether to squeeze input arrays to remove any singleton dimensions
            since this class is designed to store per-element abundances as 1D arrays. This allows
            users to input mass constraints direct from output dictionaries without having to
            manually squeeze batch dimensions. Defaults to ``True``.
    """

    species: SpeciesCollection
    """Species collection"""
    abundance_dict: dict[str, FloatArray]
    """Mapping of an element name to an abundance constraint (mol)"""

    def __init__(
        self,
        species: SpeciesCollection,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        units: Literal["mass", "moles"] = "mass",
        squeeze_input: bool = True,
    ):
        units = _validate_mass_units(units)

        mass_constraints_: Mapping[str, ArrayLike] = (
            mass_constraints if mass_constraints is not None else {}
        )

        abundance_dict: dict[str, Array] = {}

        for element in species.unique_elements:
            element_sum: ArrayLike = 0
            has_constraint: bool = False
            # This accommodates mass constraints given as mass or moles of elements or species.
            for species_, value_ in mass_constraints_.items():
                composition: Composition = Formula(species_).composition()
                if element in composition:
                    element_composition: CompositionItem = composition[element]
                    has_constraint = True
                    # Always convert to moles for storage
                    if units == "mass":
                        # To get moles: (mass of element in species) / (molar mass of element)
                        # But here, value_ is the mass of the species, so:
                        # moles of element = (mass of species * element_composition.fraction) /
                        #   element molar mass
                        element_index: int = species.get_element_index(element)
                        element_molar_mass: float = species.element_molar_masses[element_index]
                        scale: float = element_composition.fraction / element_molar_mass
                    else:
                        # element_composition.count is the atom count
                        # value_ is in moles of species, so moles of element = count * value_
                        scale = element_composition.count
                    element_sum += scale * value_

            element_sum = as_j64(element_sum)

            if squeeze_input:
                element_sum = jnp.squeeze(element_sum)

            # All elements must be included as keys in the abundance dictionary, even if they
            # are not present in any constraints. In the latter case, the abundance is set to
            # NaN to indicate that the constraint is inactive.
            abundance_dict[element] = element_sum if has_constraint else as_j64(jnp.nan)

        self.species = species
        self.abundance_dict = abundance_dict

    def abundance(self) -> Float[Array, "... n_elements"]:
        """Abundance array constructed from the abundance dictionary

        Returns:
            Abundance array constructed from the abundance dictionary
        """
        arrays: list[Array] = [
            self.abundance_dict[element] for element in self.species.unique_elements
        ]
        abundance_array: Float[Array, "... n_elements"] = stack_broadcast(arrays)
        # jax.debug.print("abundance_array = {out}", out=abundance_array)

        return abundance_array

    def abundance_mol(self, batch_size: int = 1) -> Float[Array, "#n_batch n_elements"]:
        """Abundance by moles for all elements with broadcasting to a specified batch size.

        Args:
            batch_size: Batch size to broadcast the abundance arrays to. Defaults to ``1``.

        Returns:
            Abundance by moles for all elements
        """
        arrays: list[Array] = []

        for element in self.species.unique_elements:
            arr: Array = self.abundance_dict[element]
            arr = jnp.broadcast_to(arr, (batch_size,) + arr.shape[1:])
            arrays.append(arr)

        abundance_array: Float[Array, "... n_elements"] = jnp.stack(arrays, axis=-1)
        # jax.debug.print("abundance_array = {out}", out=abundance_array)

        return abundance_array

    def abundance_mass(self, batch_size: int = 1) -> Float[Array, "#n_batch n_elements"]:
        """Abundance by mass for all elements with broadcasting to a specified batch size.

        Args:
            batch_size: Batch size to broadcast the abundance arrays to. Defaults to ``1``.

        Returns:
            Abundance by mass for all elements
        """
        return self.abundance_mol(batch_size) * self.species.element_molar_masses

    def update(
        self,
        new_abundances: Mapping[str, ArrayLike],
        units: Literal["mass", "moles"] = "mass",
        squeeze_input: bool = True,
    ) -> Self:
        """Updates the abundance constraints with new values from a dictionary.

        Note:
            Prefer this method over re-instantiation when reusing a compiled JAX function.
            It enforces leaf shape/dtype stability preventing unnecessary retracing. It also
            supports partial updates, leaving unspecified elements unchanged. Previously active
            constraints can be turned off by setting the mapping value to ``NaN`` and conversely
            previously inactive constraints can be turned on by setting the mapping value to a
            non-``NaN`` value.

        Args:
            new_abundances: Dictionary with new abundance values for some or all elements. The keys
                should be element names and the values should be the new abundance values. Original
                abundances that are not included in the ``new_abundance`` dictionary will be
                retained. Unknown element keys are ignored.
            units: Units of ``new_abundances``. Defaults to ``mass``.
            squeeze_input: Whether to squeeze input arrays to remove any singleton dimensions
                since this class is designed to store per-element abundances as 1D arrays. This
                allows users to input mass constraints direct from output dictionaries without
                having to manually squeeze batch dimensions. Defaults to ``True``.

        Returns:
            An instance with updated abundance constraints
        """
        units = _validate_mass_units(units)

        abundance_dict: dict[str, Array] = dict(self.abundance_dict)

        for element, new_value in new_abundances.items():
            if element not in abundance_dict:
                continue

            original_value: Array = abundance_dict[element]
            # Keep leaf signatures stable to avoid unnecessary retracing under JAX transforms.
            value_array: FloatArray = jnp.asarray(new_value)
            if units == "mass":
                element_index: int = self.species.get_element_index(element)
                element_molar_mass: float = self.species.element_molar_masses[element_index]
                value_array = value_array / element_molar_mass
            # Cast once at the end to ensure we match the original_value dtype after any arithmetic
            value_array = jnp.asarray(value_array, dtype=original_value.dtype)

            if squeeze_input:
                value_array = jnp.squeeze(value_array)

            abundance_dict[element] = jnp.broadcast_to(value_array, original_value.shape)

        mass_constraint_set_updated: MassConstraintSet = eqx.tree_at(
            lambda c: c.abundance_dict, self, abundance_dict
        )

        return cast(Self, mass_constraint_set_updated)

    def active(self) -> Bool[Array, "... elements"]:
        """Active mass constraints

        Returns:
            Mask indicating whether elemental mass constraints are active or not
        """
        return ~jnp.isnan(self.abundance())


class Parameters(eqx.Module):
    """Parameters

    Prefer constructing this object once outside ``jit`` and applying :meth:`update_constraints`
    and :meth:`update_state` inside workflows.

    Args:
        state: Thermodynamic state
        activity_constraints: Mapping of a species name and an activity constraint. Defaults to a
            new instance of :class:`ActivityConstraintSet`.
        mass_constraints: Mapping of element name and mass/abundance constraint. Defaults to a
            new instance of :class:`MassConstraintSet`.
        mass_units: Units of ``mass_constraints``. Defaults to ``mass``.
        solver_parameters: Solver parameters. Defaults to a new instance of
            :class:`~atmodeller.containers.SolverParameters`.
    """

    state: BaseThermodynamicState
    """Thermodynamic state"""
    activity_constraints: ActivityConstraintSet
    """Activity constraints"""
    mass_constraints: MassConstraintSet
    """Mass constraints"""
    solver_parameters: SolverParameters
    """Solver parameters"""
    batch_size: int = 1
    """Batch size"""

    def __init__(
        self,
        state: BaseThermodynamicState,
        activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        mass_units: Literal["mass", "moles"] = "mass",
        solver_parameters: Optional[SolverParameters] = None,
    ):
        activity_constraint_set: ActivityConstraintSet = ActivityConstraintSet(
            state.reaction_system.phase_system.species, activity_constraints
        )
        mass_constraints_set: MassConstraintSet = MassConstraintSet(
            state.reaction_system.phase_system.species, mass_constraints, mass_units
        )
        batch_size: int = get_batch_size((state, activity_constraint_set, mass_constraints_set))
        # jax.debug.print("batch_size (parameters) = {out}", out=batch_size)

        solver_parameters_: SolverParameters = (
            SolverParameters() if solver_parameters is None else solver_parameters
        )

        self.state = state
        self.activity_constraints = activity_constraint_set
        self.mass_constraints = mass_constraints_set
        self.solver_parameters = solver_parameters_
        self.batch_size = batch_size

    @property
    def reaction_system(self) -> ReactionSystem:
        """Reaction system representing the thermodynamic state of the planetary body"""
        return self.state.reaction_system

    @property
    def species(self) -> SpeciesCollection[SpeciesProtocol]:
        """Species in the system"""
        return self.reaction_system.phase_system.species

    @property
    def species_names(self) -> tuple[str, ...]:
        """Species names in the system"""
        return self.species.species_names

    @property
    def element_names(self) -> tuple[str, ...]:
        """Unique elements in the system"""
        return self.reaction_system.species.unique_elements

    def update_constraints(
        self,
        *,
        activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        mass_units: Literal["mass", "moles"] = "mass",
    ) -> Self:
        """Updates the mass and activity/fugacity constraints of the parameters.

        New values are assumed to be broadcastable to the shapes of the existing fields. Keeping
        leaf shapes stable helps avoid unnecessary JAX recompilation.

        Args:
            activity_constraints: New activity/fugacity constraints. Defaults to ``None``.
            mass_constraints: New mass constraints. Defaults to ``None``.
            mass_units: Units of ``mass_constraints``. Defaults to ``mass``.

        Returns:
            Updated parameters
        """
        parameters_updated: Parameters = self

        if mass_constraints is not None:
            mass_constraints_set_updated: MassConstraintSet = self.mass_constraints.update(
                mass_constraints, units=mass_units
            )
            parameters_updated = eqx.tree_at(
                lambda p: p.mass_constraints, parameters_updated, mass_constraints_set_updated
            )

        if activity_constraints is not None:
            activity_constraints_set_updated: ActivityConstraintSet = (
                self.activity_constraints.update(activity_constraints)
            )
            parameters_updated = eqx.tree_at(
                lambda p: p.activity_constraints,
                parameters_updated,
                activity_constraints_set_updated,
            )

        return cast(Self, parameters_updated)

    def update_state(self, *args, **kwargs) -> Self:
        """Updates the thermodynamic state of the parameters.

        New values are assumed to be broadcastable to the shapes of the existing fields. Keeping
        leaf shapes stable helps avoid unnecessary JAX recompilation.

        Args:
            *args: Positional arguments to pass to the ``update`` method of the thermodynamic state
            **kwargs: Keyword arguments to pass to the ``update`` method of the thermodynamic state

        Returns:
            Updated parameters
        """
        state_updated: BaseThermodynamicState = self.state.update(*args, **kwargs)
        parameters_updated = eqx.tree_at(lambda p: p.state, self, state_updated)

        return cast(Self, parameters_updated)
