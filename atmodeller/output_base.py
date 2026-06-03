# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base output infrastructure.

This module contains shared output infrastructure used by higher-level output interfaces.
It provides:

- Abstract base classes for output dictionary representations
- Shared PyTree utilities for flattening, merging, expanding, raveling, and array conversion
- Common typing and phase-to-dictionary conversion contracts

This file is intentionally focused on reusable foundations rather than the public master output
API and export facade.
"""

import logging
from abc import abstractmethod
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree

from atmodeller import override
from atmodeller.containers import MultiAttemptSolution
from atmodeller.initial_solution import generate_auto_initial_guess
from atmodeller.jax_utils import FloatArray, stack_broadcast
from atmodeller.parameters import ActivityConstraintSet, MassConstraintSet, Parameters
from atmodeller.phases import BasePhase, GasPhaseOutput, PhaseOutput

logger: logging.Logger = logging.getLogger(__name__)


def flatten_dictionary(d: dict, parent_key: str = "") -> dict[str, Any]:
    """Recursively flattens a nested dictionary, joining keys with "." to form column names.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used during recursion)

    Returns:
        Flat dictionary with dot-joined keys
    """
    items: dict[str, Any] = {}

    for k, v in d.items():
        new_key: str = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dictionary(v, new_key))
        else:
            items[new_key] = v

    return items


def recursively_merge_dictionaries(d1: dict, d2: dict) -> dict[Any, Any]:
    """Recursively merges two dictionaries.

    Args:
        d1: The first dictionary
        d2: The second dictionary, which will overwrite values in the first dictionary if there are
            duplicate keys

    Returns:
        The merged dictionary
    """
    out: dict[str, Any] = dict(d1)

    for k, v in d2.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = recursively_merge_dictionaries(out[k], v)
            else:
                out[k] = v
        else:
            out[k] = v

    return out


def expand_jax_arrays_in_pytree_to_batch(pytree: PyTree, batch_size: int) -> PyTree:
    """Expands all arrays in a PyTree to the batch size.

    Note:
        :func:`jax.tree_util.tree_map` does not preserve the insertion order of standard Python
        dictionaries. Dictionary keys are treated as an unordered set and are typically processed
        in a canonical (sorted) order. If preserving key order is important, consider using
        :class:`collections.OrderedDict`.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to expand
        batch_size: Batch size to expand to

    Returns:
        PyTree with arrays expanded to batch size
    """

    def expand(x: Any) -> Any:
        if isinstance(x, jnp.ndarray):
            x = jnp.atleast_1d(x)
            # Always broadcast if shape[0] != batch_size
            if x.shape[0] != batch_size:
                x = jnp.broadcast_to(x, (batch_size,) + x.shape[1:])
            return x
        return x

    return tree_map(expand, pytree)


def ravel_jax_arrays_in_pytree(pytree: PyTree) -> PyTree:
    """Ravels all JAX arrays in a PyTree to 1-D arrays.

    Note:
        :func:`jax.tree_util.tree_map` does not preserve the insertion order of standard Python
        dictionaries. Dictionary keys are treated as an unordered set and are typically processed
        in a canonical (sorted) order. If preserving key order is important, consider using
        :class:`collections.OrderedDict`.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to ravel

    Returns:
        PyTree with all JAX arrays raveled to 1-D
    """

    def ravel(x: Any) -> Any:
        if isinstance(x, jnp.ndarray):
            return jnp.ravel(x)
        return x

    return tree_map(ravel, pytree)


def convert_jax_arrays_in_pytree_to_numpy(pytree: PyTree) -> PyTree:
    """Converts all JAX arrays in a PyTree to NumPy arrays.

    Note:
        :func:`jax.tree_util.tree_map` does not preserve the insertion order of standard Python
        dictionaries. Dictionary keys are treated as an unordered set and are typically processed
        in a canonical (sorted) order. If preserving key order is important, consider using
        :class:`collections.OrderedDict`.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to convert

    Returns:
        PyTree with JAX arrays converted to NumPy arrays
    """
    return tree_map(lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, pytree)


class BaseOutputDict(eqx.Module):
    """Represents the output of a model as a nested dictionary.

    This is a base class specification (explicit interface) for output dictionary representations.

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    parameters: Parameters
    multi_attempt_solution: MultiAttemptSolution

    @abstractmethod
    def _phase_output_to_dict(self, phase_output: PhaseOutput[BasePhase]) -> dict[str, Any]:
        """Dictionary representation of the phase output

        Returns:
            A dictionary
        """

    @abstractmethod
    def to_dict(
        self, *, expand_to_batch: bool = False, to_numpy: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Output as a nested dictionary with JAX or NumPy arrays.

        Args:
            expand_to_batch: Whether to expand arrays to the batch size. Defaults to ``False``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``.
                Must be ``False`` if used within a jitted context, as NumPy arrays are not
                compatible with JAX transformations (jit, vmap, etc.).
            **kwargs: Arbitrary keyword arguments for the output dictionary

        Returns:
            Dictionary of the solution with JAX or NumPy arrays
        """

    @property
    def _split_solution(self) -> list[Float[Array, "#n_batch n_species"]]:
        """Log number of moles and log stability, split from the solution array in one pass."""
        return jnp.split(self.multi_attempt_solution.value, 2, axis=-1)

    @property
    def batch_size(self) -> int:
        """Batch size of the output"""
        return self.parameters.batch_size

    @property
    def log_number_moles(self) -> Float[Array, "#n_batch n_species"]:
        """Log number of moles for each species"""
        log_number_moles, _ = self._split_solution

        return log_number_moles

    @property
    def log_stability(self) -> Float[Array, "#n_batch n_species"]:
        """Log stability for each species"""
        _, log_stability = self._split_solution

        active_stability: ArrayLike = self.parameters.reaction_system.species.active_stability
        log_stability = jnp.where(active_stability, log_stability, -jnp.inf)

        return log_stability

    @property
    def solution(self) -> Float[Array, "#n_batch twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.multi_attempt_solution.value

    def _solution_array_to_dict(
        self, solution_array: Float[Array, "#n_batch twice_species"], suffix: str = ""
    ) -> dict[str, ArrayLike]:
        """Converts a solution array to a dictionary mapping.

        Args:
            solution_array: The solution array containing log number of moles and log stability
            suffix: Optional suffix to add to the keys. Defauls to an empty string.

        Returns:
            Dictionary mapping species names and their stability keys to arrays of values
        """
        out: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self.parameters.species.species_names):
            out[f"{species_}{suffix}"] = solution_array[..., nn]
            out[f"{species_}_stability{suffix}"] = solution_array[
                ..., nn + self.parameters.species.number_species
            ]

        return out

    def auto_initial_guess_to_dict(self) -> dict[str, ArrayLike]:
        """Gets the automatic initial guess for the solver.

        Returns:
            Dictionary mapping species names and their stability keys to arrays of values,
            plus global MAE and RMSE metrics for the log error for active solution quantities.
        """
        out: dict[str, ArrayLike] = self._solution_array_to_dict(self.solution, suffix="_solution")

        guess: Float[Array, "#n_batch twice_species"] = jnp.atleast_2d(
            generate_auto_initial_guess(self.parameters)
        )
        out.update(self._solution_array_to_dict(guess, suffix="_guess"))

        # Compute error metrics for active solution quantities
        guess_log_number_moles, guess_log_stability = jnp.split(guess, 2, axis=-1)
        diff_log_number_moles: Float[Array, "... n_species"] = (
            guess_log_number_moles - self.log_number_moles
        )
        diff_log_stability: Float[Array, "... n_species"] = (
            guess_log_stability - self.log_stability
        )

        # Mask inactive stabilities
        active_stability: Bool[Array, " n_species"] = self.parameters.species.active_stability
        diff_log_stability_masked: Float[Array, "... n_species"] = jnp.where(
            active_stability, diff_log_stability, jnp.nan
        )

        # Stack differences column-wise: [log_number_moles | log_stability]
        diff_log: Float[Array, "... twice_species"] = jnp.concatenate(
            [diff_log_number_moles, diff_log_stability_masked], axis=-1
        )
        diff_log = jnp.abs(diff_log)
        # jax.debug.print("diff_log = {out}", out=diff_log)

        # Compute MAE and RMSE per batch (ignore NaNs, which are inactive solution quantities)
        mae: Float[Array, "#n_batch"] = jnp.nanmean(diff_log, axis=-1)
        rmse: Float[Array, "#n_batch"] = jnp.sqrt(jnp.nanmean(diff_log**2, axis=-1))

        out.update(self._solution_array_to_dict(diff_log, suffix="_log_error"))
        out["log_error_mae"] = mae
        out["log_error_rmse"] = rmse

        return out

    def solution_to_dict(self) -> dict[str, ArrayLike]:
        """Returns a dictionary of solution arrays for each species.

        Returns:
            Dictionary mapping species names and their stability keys to arrays of values
        """
        return self._solution_array_to_dict(self.solution)

    @property
    def gas(self) -> GasPhaseOutput:
        """Gas phase output. By construction, the gas phase is at index 0 in the phase system."""
        return cast(GasPhaseOutput, self.get_phase_output_from_index(0))

    def get_phase_output_from_index(self, phase_index: int) -> PhaseOutput:
        """Gets the output for a phase from the phase index.

        Args:
            phase_index: The index of the phase to get the output for, where 0 corresponds to the
                gas phase by construction.

        Returns:
            The phase output for the specified phase index
        """
        phase_slice: slice = self.parameters.reaction_system.phase_system.phase_slice(phase_index)
        phase: BasePhase = self.parameters.reaction_system.phase_system.phases[phase_index]

        phase_output: PhaseOutput = phase.output(
            self.log_number_moles[..., phase_slice],
            self.log_stability[..., phase_slice],
            self._temperature,
            self._pressure,
        )

        return phase_output

    @property
    def _temperature(self) -> FloatArray:
        """Temperature in K"""
        return self.parameters.state.temperature

    @property
    def temperature(self) -> Float[Array, "#n_batch 1"]:
        """Temperature in K"""
        return jnp.atleast_1d(self._temperature)[:, None]

    @property
    def _pressure(self) -> FloatArray:
        """Pressure in bar"""
        return self.parameters.state.get_pressure(self.log_number_moles)

    @property
    def pressure(self) -> Float[Array, "#n_batch 1"]:
        """Pressure in bar"""
        return jnp.atleast_1d(self._pressure)[:, None]

    def phase_to_dict(self, phase_output: PhaseOutput[BasePhase]) -> dict[str, Any]:
        """Phase-level properties such as total mass, number of moles, molar mass, etc.

        Args:
            phase_output: The phase output to convert.

        Returns:
            A dictionary of phase-level properties
        """
        return {
            "background_mass": phase_output.background_mass,
            "background_number_moles": phase_output.background_number_moles,
            "background_molar_mass": phase_output.background_molar_mass,
            "mass": phase_output.phase_mass,
            "species_number_moles": phase_output.phase_species_number_moles,
            "elements_number_moles": phase_output.phase_element_number_moles,
            "molar_mass": phase_output.phase_molar_mass,
            "species_to_phase_mass_ratio": phase_output.species_to_phase_mass_ratio,
        }

    def solver_to_dict(self) -> dict[str, ArrayLike]:
        """Solver information such as success flags and number of iterations"""
        return self.multi_attempt_solution.asdict()

    def state_to_dict(self) -> dict[str, Any]:
        """Thermodynamic state of the system"""
        return self.parameters.state.asdict(self.solution)


class OutputNaturalDict(BaseOutputDict):
    """Represents the natural output of a model based on the arrays used internally.

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    @override
    def _phase_output_to_dict(self, phase_output: PhaseOutput[BasePhase]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        # Phase-level properties
        out["phase"] = self.phase_to_dict(phase_output)

        out["elements"] = {
            "names": phase_output.phase.species.unique_elements,
            "mass": phase_output.element_mass,
            "number_moles": phase_output.element_number_moles,
        }
        out["species"] = {
            "names": phase_output.phase.species_names,
            "activity": phase_output.species_activity,
            "mass": phase_output.species_mass,
            "mass_fraction": phase_output.species_mass_fraction,
            "number_moles": phase_output.species_number_moles,
            "mole_fraction": phase_output.species_mole_fraction,
            "include_in_phase_mass": phase_output.include_in_phase_mass,
        }

        return out

    @override
    def to_dict(
        self, *, expand_to_batch: bool = False, to_numpy: bool = False, **kwargs
    ) -> dict[str, Any]:
        del kwargs
        out: dict[str, Any] = {}

        # Save all phases, including gas, silicate melt, silicate solid, and condensates (if any)
        for phase_index, phase in enumerate(self.parameters.reaction_system.phase_system.phases):
            if not phase.is_empty:
                phase_output: PhaseOutput = self.get_phase_output_from_index(phase_index)
                out[phase.name] = self._phase_output_to_dict(phase_output)

        # Extra outputs for the gas phase
        if not self.gas.phase.is_empty:
            phase_name: str = self.gas.phase.name
            out[phase_name]["species"]["partial_pressure"] = self.gas.species_partial_pressure
            out[phase_name]["phase"]["volume"] = self.gas.volume
            out[phase_name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[phase_name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[phase_name]["phase"]["pressure"] = self.gas.pressure

        out["constraints"] = {
            "elements": {
                "number_moles": self.parameters.mass_constraints.abundance_mol(self.batch_size),
                "names": self.parameters.mass_constraints.species.unique_elements,
                "mass": self.parameters.mass_constraints.abundance_mass(self.batch_size),
            }
        }
        out["constraints"]["species"] = {
            "activity": jnp.exp(
                self.parameters.activity_constraints.log_activity(
                    jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
                )
            ),
            "names": self.parameters.activity_constraints.species.species_names,
        }

        out["solution"] = self.solution_to_dict()
        out["auto_initial_guess"] = self.auto_initial_guess_to_dict()
        out["solver"] = self.solver_to_dict()
        out["state"] = self.state_to_dict()

        # Order of operations matters here: expansion must be done before conversion to NumPy
        # arrays
        if expand_to_batch:
            out = expand_jax_arrays_in_pytree_to_batch(out, self.batch_size)

        if to_numpy:
            out = convert_jax_arrays_in_pytree_to_numpy(out)

        return out


class OutputNamedArraysDict(BaseOutputDict):
    """Output dictionary with arrays split and labeled by element and species names"""

    @staticmethod
    def _split_by_name_and_add(
        names: tuple[str, ...], inarray: Array, output: dict, keyname: str
    ) -> None:
        """Splits the species/element-level data by species/element and adds them to the output.

        Args:
            names: The species/elements corresponding to the columns of the input array
            inarray: The input array to split
            output: The output dictionary to which the split entries will be added
            keyname: The name of the property being split (e.g., "mass", "number_moles", etc.)
                to use in the output keys
        """
        split_data: list[Array] = jnp.split(inarray, max(len(names), 1), axis=-1)
        out_dict: dict = output.setdefault(keyname, {})

        for ii, name in enumerate(names):
            out_dict[name] = split_data[ii]

    def _split_by_elements_and_add(
        self, phase_output: PhaseOutput[BasePhase], inarray: Array, output: dict, keyname: str
    ) -> None:
        """Splits the element-level data by element and adds them to the output.

        Args:
            phase_output: The phase output object containing the element information
            inarray: The input array to split, with shape (... n_elements)
            output: The output dictionary to which the split entries will be added
            keyname: The name of the property being split (e.g., "mass", "number_moles", etc.)
                to use in the output keys
        """
        self._split_by_name_and_add(
            phase_output.phase.species.unique_elements, inarray, output, keyname
        )

    def _split_by_species_and_add(
        self, phase_output: PhaseOutput[BasePhase], inarray: Array, output: dict, keyname: str
    ) -> None:
        """Splits the species-level data by species' and adds them to the output.

        Args:
            phase_output: The phase output object containing the species information
            inarray: The input array to split, with shape (... n_species)
            output: The output dictionary to which the split entries will be added
            keyname: The name of the property being split (e.g., "mass", "activity", etc.) to
                use in the output keys
        """
        self._split_by_name_and_add(phase_output.phase.species_names, inarray, output, keyname)

    @override
    def _phase_output_to_dict(self, phase_output: PhaseOutput[BasePhase]) -> dict[str, Any]:

        out: dict[str, Any] = {}

        # Phase-level properties
        out["phase"] = self.phase_to_dict(phase_output)

        # Element-level properties, split by element
        elements_out: dict = out.setdefault("elements", {})
        self._split_by_elements_and_add(
            phase_output, phase_output.element_mass, elements_out, "mass"
        )
        self._split_by_elements_and_add(
            phase_output, phase_output.element_number_moles, elements_out, "number_moles"
        )

        # Species-level properties, split by species
        species_out: dict = out.setdefault("species", {})
        self._split_by_species_and_add(
            phase_output, phase_output.species_activity, species_out, "activity"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_mass, species_out, "mass"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_mass_fraction, species_out, "mass_fraction"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_number_moles, species_out, "number_moles"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_mole_fraction, species_out, "mole_fraction"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.include_in_phase_mass, species_out, "include_in_phase_mass"
        )

        return out

    @override
    def to_dict(
        self,
        *,
        expand_to_batch: bool = False,
        ravel: bool = False,
        to_numpy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Complete output as a nested dictionary with JAX or NumPy arrays, split by element and
        species names.

        Args:
            expand_to_batch: Whether to expand arrays to the batch size. Defaults to ``False``.
            ravel: Whether to ravel arrays to 1-D after expanding. Can be used when the expanded
                arrays are intended for conversion to DataFrames (which also requires
                ``to_numpy=False``). Defaults to ``False``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``. Must be
                ``False`` if used within a jitted context, as NumPy arrays are not compatible with
                JAX transformations (jit, vmap, etc.).
            **kwargs: Arbitrary keyword arguments for the output dictionary

        Returns:
            Dictionary of the solution with JAX or NumPy arrays, split by element and species names
        """
        del kwargs
        out: dict[str, Any] = {}

        # Save all phases, including gas, silicate melt, silicate solid, and condensates (if any)
        for phase_index, phase in enumerate(self.parameters.reaction_system.phase_system.phases):
            if not phase.is_empty:
                phase_output: PhaseOutput = self.get_phase_output_from_index(phase_index)
                out[phase.name] = self._phase_output_to_dict(phase_output)

        # Extra outputs for the gas phase
        if not self.gas.is_empty:
            self._split_by_species_and_add(
                self.gas,
                self.gas.species_partial_pressure,
                out[self.gas.phase.name]["species"],
                "partial_pressure",
            )
            out[self.gas.phase.name]["phase"]["volume"] = self.gas.volume
            out[self.gas.phase.name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[self.gas.phase.name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[self.gas.phase.name]["phase"]["pressure"] = self.gas.pressure

        out["constraints"] = {}
        elements_out: dict = out["constraints"].setdefault("elements", {})
        self._split_by_name_and_add(
            self.parameters.mass_constraints.species.unique_elements,
            self.parameters.mass_constraints.abundance_mass(self.batch_size),
            elements_out,
            "mass",
        )
        self._split_by_name_and_add(
            self.parameters.mass_constraints.species.unique_elements,
            self.parameters.mass_constraints.abundance_mol(self.batch_size),
            elements_out,
            "number_moles",
        )
        species_out: dict = out["constraints"].setdefault("species", {})

        # Evaluated activity constraints
        arrays: list[Array] = [
            jnp.asarray(
                constraint.log_activity(jnp.squeeze(self.temperature), jnp.squeeze(self.pressure))
            )
            for constraint in self.parameters.activity_constraints.ordered_constraints
        ]
        evaluated_activity_constraints: Float[Array, "... n_species"] = jnp.exp(
            stack_broadcast(arrays)
        )

        self._split_by_name_and_add(
            self.parameters.activity_constraints.species.species_names,
            evaluated_activity_constraints,
            species_out,
            "activity",
        )

        out["solution"] = self.solution_to_dict()
        out["auto_initial_guess"] = self.auto_initial_guess_to_dict()
        out["solver"] = self.solver_to_dict()
        out["state"] = self.state_to_dict()

        # Must vmap the residual evaluation to match what the solver did: parameters contains a
        # mix of scalar and batched leaves, so calling objective_function directly on the 2-D
        # solution gives incorrect results. vmap_axes_spec maps None/0 per leaf appropriately.
        # FIXME: This is breaking because all arrays including numpy are seen as batchable under
        # jit
        # objective_function_vmapped = eqx.filter_vmap(
        #    objective_function, in_axes=(0, vmap_axes_spec(self.parameters))
        # )
        # out["residual"] = objective_function_vmapped(self.solution, self.parameters)

        # Order of operations matters here: expansion must be done before ravel, and both before
        # conversion to NumPy arrays
        if expand_to_batch:
            out = expand_jax_arrays_in_pytree_to_batch(out, self.batch_size)

        if ravel:
            out = ravel_jax_arrays_in_pytree(out)

        if to_numpy:
            out = convert_jax_arrays_in_pytree_to_numpy(out)

        return out

    def compare(
        self,
        d1: dict,
        rtol: float,
        atol: float,
        log: bool = False,
    ) -> bool:
        """Compares two nested dictionaries of output.

        Args:
            d1: Target dictionary
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            log: Whether to compare the base-10 logarithm of the values. Defaults to ``False``.

        Returns:
            ``True`` if all values match within the specified tolerances, else ``False``
        """
        d2: dict[str, Any] = self.to_dict(expand_to_batch=True, ravel=True, to_numpy=True)

        return self._compare_recursive(d1, d2, rtol, atol, log)

    def _compare_recursive(
        self,
        d1: dict,
        d2: dict,
        rtol: float,
        atol: float,
        log: bool,
        path: tuple[str, ...] = (),
    ) -> bool:
        """Recursively compares two nested dictionaries.

        Args:
            d1: First dictionary
            d2: Second dictionary
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            log: Whether to compare the base-10 logarithm of the values
            path: Current key path (used for logging)

        Returns:
            ``True`` if all values match within the specified tolerances, else ``False``
        """
        all_match: bool = True

        for key in d1.keys():
            v1 = d1.get(key)
            v2 = d2.get(key)
            current_path = path + (key,)

            if isinstance(v1, dict) and isinstance(v2, dict):
                all_match = (
                    self._compare_recursive(v1, v2, rtol, atol, log, current_path) and all_match
                )
            else:
                if isinstance(v1, (np.ndarray, float, int)) and isinstance(
                    v2, (np.ndarray, float, int)
                ):
                    if log:
                        v1, v2 = np.log10(v1), np.log10(v2)
                    is_close: bool = np.allclose(v1, v2, rtol=rtol, atol=atol)
                    all_match = all_match and is_close
                    logger.info(
                        "Comparing %s: %s vs %s --> %s",
                        current_path,
                        v1,
                        v2,
                        "match" if is_close else "mismatch",
                    )

        return all_match


class OutputElementsSpeciesDict(BaseOutputDict):
    """Output dictionary grouped by element and species names"""

    @staticmethod
    def _split_array_by_names(names: tuple[str, ...], inarray: Array) -> list[Array]:
        """Splits the input array into a list of arrays corresponding to the input names.

        Args:
            names: The species/elements corresponding to the columns of the input array
            inarray: The input array to split

        Returns:
            A list of arrays corresponding to the input names
        """
        return jnp.split(inarray, max(len(names), 1), axis=-1)

    @override
    def _phase_output_to_dict(self, phase_output: PhaseOutput[BasePhase]) -> dict[str, Any]:

        out: dict[str, Any] = {}

        unique_elements: tuple[str, ...] = phase_output.phase.species.unique_elements
        element_mass: list[Array] = self._split_array_by_names(
            unique_elements, phase_output.element_mass
        )
        element_number_moles: list[Array] = self._split_array_by_names(
            unique_elements, phase_output.element_number_moles
        )

        for nn, element in enumerate(unique_elements):
            element_dict: dict[str, Any] = out.setdefault(element, {})
            phase_dict: dict[str, Any] = element_dict.setdefault(phase_output.phase.name, {})
            phase_dict["mass"] = element_mass[nn]
            phase_dict["number_moles"] = element_number_moles[nn]

        species_names: tuple[str, ...] = phase_output.phase.species_names
        species_activity: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_activity
        )
        species_mass: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_mass
        )
        species_mass_fraction: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_mass_fraction
        )
        species_number_moles: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_number_moles
        )
        species_mole_fraction: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_mole_fraction
        )
        include_in_mass_phase: list[Array] = self._split_array_by_names(
            species_names, phase_output.include_in_phase_mass
        )

        for nn, species in enumerate(species_names):
            species_dict: dict[str, Any] = out.setdefault(species, {})
            phase_dict: dict[str, Any] = species_dict.setdefault(phase_output.phase.name, {})
            phase_dict["activity"] = species_activity[nn]
            phase_dict["mass"] = species_mass[nn]
            phase_dict["mass_fraction"] = species_mass_fraction[nn]
            phase_dict["number_moles"] = species_number_moles[nn]
            phase_dict["mole_fraction"] = species_mole_fraction[nn]
            phase_dict["include_in_phase_mass"] = include_in_mass_phase[nn]

        return out

    @override
    def to_dict(
        self,
        *,
        expand_to_batch: bool = False,
        ravel: bool = False,
        to_numpy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        out: dict[str, Any] = {}

        # Save all phases, including gas, silicate melt, silicate solid, and condensates (if any)
        for phase_index, phase in enumerate(self.parameters.reaction_system.phase_system.phases):
            if not phase.is_empty:
                phase_output: PhaseOutput = self.get_phase_output_from_index(phase_index)
                out_ = self._phase_output_to_dict(phase_output)
                out = recursively_merge_dictionaries(out, out_)
                out[phase.name] = {}
                out[phase.name]["phase"] = self.phase_to_dict(phase_output)

        # Extra outputs for the gas phase
        if not self.gas.is_empty:
            # Add the partial pressure of each species in the gas phase to the output
            species_partial_pressure: list[Array] = self._split_array_by_names(
                self.gas.phase.species_names, self.gas.species_partial_pressure
            )
            for nn, species in enumerate(self.gas.phase.species_names):
                species_dict: dict[str, Any] = out.setdefault(species, {})
                phase_dict: dict[str, Any] = species_dict.setdefault(self.gas.phase.name, {})
                phase_dict["partial_pressure"] = species_partial_pressure[nn]

            out[self.gas.phase.name]["phase"]["volume"] = self.gas.volume
            out[self.gas.phase.name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[self.gas.phase.name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[self.gas.phase.name]["phase"]["pressure"] = self.gas.pressure

            # Metallicity
            gas_phase_dict: dict[str, Any] = out[self.gas.phase.name]["phase"]
            total_number_moles: ArrayLike = gas_phase_dict["elements_number_moles"]
            total_mass: ArrayLike = gas_phase_dict["mass"]

            heavy_elements: tuple[str, ...] = tuple(
                element
                for element in self.gas.phase.species.unique_elements
                if element not in ("H", "He")
            )
            if heavy_elements:
                z_by_moles: ArrayLike = jnp.sum(
                    jnp.stack(
                        [
                            out[element][self.gas.phase.name]["number_moles"]
                            for element in heavy_elements
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                z_by_mass: ArrayLike = jnp.sum(
                    jnp.stack(
                        [out[element][self.gas.phase.name]["mass"] for element in heavy_elements],
                        axis=0,
                    ),
                    axis=0,
                )
            else:
                z_by_moles = jnp.zeros_like(total_number_moles)
                z_by_mass = jnp.zeros_like(total_mass)

            z_by_moles = z_by_moles / total_number_moles
            z_by_mass = z_by_mass / total_mass
            out[self.gas.phase.name]["phase"]["metallicity_by_moles"] = z_by_moles
            out[self.gas.phase.name]["phase"]["metallicity_by_mass"] = z_by_mass

        # Mass constraints
        mass_constraints: MassConstraintSet = self.parameters.mass_constraints
        unique_elements: tuple[str, ...] = mass_constraints.species.unique_elements
        element_mass: list[Array] = self._split_array_by_names(
            unique_elements, mass_constraints.abundance_mass(self.batch_size)
        )
        element_number_moles: list[Array] = self._split_array_by_names(
            unique_elements, mass_constraints.abundance_mol(self.batch_size)
        )

        for nn, element in enumerate(unique_elements):
            element_dict: dict[str, Any] = out.setdefault(element, {})
            constraints_dict: dict[str, Any] = element_dict.setdefault("constraints", {})
            constraints_dict["mass"] = element_mass[nn]
            constraints_dict["number_moles"] = element_number_moles[nn]

        # Activity constraints
        activity_constraints: ActivityConstraintSet = self.parameters.activity_constraints
        unique_species: tuple[str, ...] = activity_constraints.species.species_names
        species_log_activity: list[Array] = self._split_array_by_names(
            unique_species,
            activity_constraints.log_activity(
                jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
            ),
        )

        for nn, species in enumerate(unique_species):
            species_dict: dict[str, Any] = out.setdefault(species, {})
            constraints_dict: dict[str, Any] = species_dict.setdefault("constraints", {})
            constraints_dict["activity"] = jnp.exp(species_log_activity[nn])

        out["solution"] = self.solution_to_dict()
        out["solver"] = self.solver_to_dict()
        out["state"] = self.state_to_dict()

        if expand_to_batch:
            out = expand_jax_arrays_in_pytree_to_batch(out, self.batch_size)

        if ravel:
            out = ravel_jax_arrays_in_pytree(out)

        if to_numpy:
            out = convert_jax_arrays_in_pytree_to_numpy(out)

        return out
