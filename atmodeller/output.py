# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Public output API for atmodeller results.

This module exposes the high-level output interface used to inspect, compare, and export simulation
results. It builds on base output infrastructure defined in :mod:`atmodeller.output_base` and
provides user-facing convenience methods for:

- Producing nested dictionary outputs in multiple formats
- Converting outputs to pandas DataFrames
- Exporting outputs to Excel and pickle
- Comparing outputs against reference data and quick logging/inspection

The :class:`Output` class is the main entry point for downstream workflows.

.. warning::
    Methods that convert to NumPy, compare values, or perform file export are intended for
    post-processing and are not compatible with JAX-compiled execution contexts (for example,
    inside ``jax.jit``).
"""

import dataclasses
import logging
import pickle
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import equinox as eqx
import numpy as np
import pandas as pd
from jaxtyping import Array, Float
from openpyxl.styles import PatternFill

from atmodeller import __version__
from atmodeller.jax_utils import NpArray
from atmodeller.solver_utils import CheckedSolution
from atmodeller.output_base import (
    OutputElementsSpeciesDict,
    OutputNamedArraysDict,
    OutputNaturalDict,
    flatten_dictionary,
)
from atmodeller.parameters import Parameters
from atmodeller.reactions import get_reaction_dictionary

logger: logging.Logger = logging.getLogger(__name__)

OUTPUT_UNITS: dict[str, str] = {
    "activity": "dimensionless (fugacity in bar for gas species)",
    "background_mass": "kg",
    "background_molar_mass": "kg/mol",
    "background_number_moles": "mol",
    "C_to_H": "mol/mol",
    "C_to_O": "mol/mol",
    "elements_number_moles": "mol",
    "include_in_phase_mass": "boolean",
    "log10dIW_1_bar": "dimensionless",
    "log10dIW_P": "dimensionless",
    "log_epsilon": "dex",
    "M_H": "dex",
    "mantle_mass": "kg",
    "mantle_melt_fraction": "kg/kg",
    "mantle_melt_mass": "kg",
    "mantle_solid_mass": "kg",
    "mass": "kg",
    "mass_fraction": "kg/kg",
    "metallic_core_mass": "kg",
    "metallicity_by_mass": "kg/kg",
    "metals_to_hydrogen": "mol/mol",
    "metals_to_hydrogen_by_mass": "kg/kg",
    "molar_mass": "kg/mol",
    "mole_fraction": "mol/mol",
    "num_steps": "integer",
    "number_moles": "mol",
    "optimistix_result": "text",
    "partial_pressure": "bar",
    "planet_mass": "kg",
    "pressure": "bar",
    "residual": "units of the objective function (see the documentation)",
    "result": "text",
    "silicate_mass": "kg",
    "silicate_melt_fraction": "kg/kg",
    "silicate_melt_mass": "kg",
    "silicate_solid_mass": "kg",
    "solution": "natural logarithm of the number of moles (mol) and of the stability",
    "solver_success": "boolean",
    "species_number_moles": "mol",
    "species_to_phase_mass_ratio": "kg/kg",
    "stable": "boolean",
    "success": "boolean",
    "successful_attempt": "integer",
    "surface_area": "m^2",
    "surface_gravity": "m/s^2",
    "surface_radius": "m",
    "temperature": "K",
    "volume": "m^3",
    "X_H": "dex",
}
"""Units of the output quantities, by quantity name, as in the documentation"""


class Output(eqx.Module):
    """Master output interface for model results.

    This class provides a unified entry point for all output formats, conversions, comparisons,
    and exports. It wraps the various output dictionary representations (natural, named arrays,
    grouped by element/species) and exposes methods for:

    - Outputting results in different formats (dict, DataFrame, Excel, pickle)
    - Comparing outputs for regression testing/validation
    - Quick inspection and logging

    .. warning::
        Some methods (such as ``to_dict(to_numpy=True)``, ``compare``, and export methods) are
        not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context), as they
        may use operations or objects that are not supported by JAX transformations.

    Args:
        parameters: Parameters
        checked_solution: Checked solution of the solve
        initial_guess: Initial guess the solver started from, i.e. log number of moles and log
            stability
        residual: Residual of every constraint at the solution, NaN for inactive constraints (see
            :func:`~atmodeller.engine.get_residual_by_constraint`). Defaults to ``None``, in which
            case the output has no ``residual`` section.
    """

    parameters: Parameters
    checked_solution: CheckedSolution
    initial_guess: Float[Array, "n_batch twice_species"]
    residual: Float[Array, "n_batch dim"] | None = None

    @property
    def solution(self) -> Float[Array, "n_batch twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.checked_solution.value

    def to_dict(
        self,
        output_format: Literal["natural", "named_arrays", "elements_species"] = "named_arrays",
        *,
        to_numpy: bool = False,
        expand_to_batch: bool = False,
        ravel: bool = False,
    ) -> dict[str, Any]:
        """Output as a nested dictionary with JAX or NumPy arrays.

        .. warning::
            ``to_numpy`` must be ``False`` if used within a jitted context, as NumPy arrays are not
            compatible with JAX transformations (jit, vmap, etc.).

        Args:
            output_format: The format of the output dictionary. Can be ``natural`` for the natural
                output format based on the arrays used internally, ``named_arrays`` for an
                alternative format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``.
                Must be ``False`` if used within a jitted context, as NumPy arrays are not
                compatible with JAX transformations (jit, vmap, etc.).
            expand_to_batch: Whether to broadcast every array to the batch size. Defaults to
                ``False``.
            ravel: Whether to ravel arrays to 1-D after expanding. Not available for the
                ``natural`` format. Defaults to ``False``.

        Returns:
            Dictionary of the solution with JAX or NumPy arrays in the specified format

        Raises:
            ValueError: If ``ravel`` is used with the ``natural`` format, or the format is invalid
        """
        if output_format == "natural":
            if ravel:
                raise ValueError("ravel is not available for the natural output format")
            return OutputNaturalDict(
                self.parameters, self.checked_solution, self.initial_guess, self.residual
            ).to_dict(to_numpy=to_numpy, expand_to_batch=expand_to_batch)
        elif output_format == "named_arrays":
            return OutputNamedArraysDict(
                self.parameters, self.checked_solution, self.initial_guess, self.residual
            ).to_dict(to_numpy=to_numpy, expand_to_batch=expand_to_batch, ravel=ravel)
        elif output_format == "elements_species":
            return OutputElementsSpeciesDict(
                self.parameters, self.checked_solution, self.initial_guess, self.residual
            ).to_dict(to_numpy=to_numpy, expand_to_batch=expand_to_batch, ravel=ravel)
        else:
            raise ValueError(f"Invalid output format: {output_format}")

    def compare(self, d1: dict, rtol: float, atol: float, log: bool = False) -> bool:
        """Compares a target dictionary to the model output.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            d1: Target dictionary
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            log: Whether to compare the base-10 logarithm of the values. Defaults to ``False``.

        Returns:
            ``True`` if all values match within the specified tolerances, else ``False``
        """
        return OutputNamedArraysDict(
            self.parameters, self.checked_solution, self.initial_guess, self.residual
        ).compare(d1, rtol, atol, log)

    def quick_look(
        self,
        output_format: Literal["natural", "named_arrays", "elements_species"] = "named_arrays",
        logger_: logging.Logger = logger,
    ) -> dict[str, Any]:
        """Quick look at the output.

        The output is logged at the ``INFO`` level of ``logger_``, so it is shown only if logging
        is set up, e.g. with :func:`atmodeller.console_logger`. It is also returned, so in a
        notebook the returned dictionary is displayed in any case.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            output_format: The format of the output dictionary. Can be ``natural`` for the natural
                output format based on the arrays used internally, ``named_arrays`` for an
                alternative format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.
            logger_: Logger to write the output to. Defaults to the ``atmodeller.output`` logger.

        Returns:
            The output dictionary with NumPy arrays, as from :meth:`to_dict` with
            ``to_numpy=True``
        """
        out: dict[str, Any] = self.to_dict(output_format=output_format, to_numpy=True)
        logger_.info("Quick look output:\n%s", pformat(out, sort_dicts=False))

        return out

    def _drop_unsuccessful_solves(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Drops unsuccessful solves.

        A solve is unsuccessful unless it passes both the objective-based check and the solver
        status, consistent with the rows highlighted by :meth:`to_excel`.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            dataframes: Dataframes from which to drop unsuccessful models

        Returns:
            Dictionary of dataframes without unsuccessful models
        """
        return {
            key: df.loc[np.asarray(self.checked_solution.success)]
            for key, df in dataframes.items()
        }

    def info(self) -> pd.DataFrame:
        """Gets a description of the run, which is the same for every model.

        The description has the *Atmodeller* version and the date (``run``), the solver settings
        (``solver``), the equation of each reaction whose residual is ``reaction_<index>``
        (``reactions``), the units of the output quantities (``units``) and the configuration of
        each species (``species``).

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Returns:
            Dataframe with columns ``section``, ``name`` and ``value``
        """

        def describe(value: Any) -> Any:
            """A readable value for a solver setting"""
            if isinstance(value, type) or callable(value):
                return getattr(value, "__name__", type(value).__name__)
            array: NpArray = np.asarray(value)
            if array.dtype != object:
                return array.item() if array.size == 1 else str(array.tolist())
            return type(value).__name__

        rows: list[tuple[str, str, Any]] = [
            ("run", "atmodeller_version", __version__),
            ("run", "date", datetime.now().isoformat(timespec="seconds")),
        ]
        solver_parameters: Any = self.parameters.solver_parameters
        rows += [
            ("solver", field.name, describe(getattr(solver_parameters, field.name)))
            for field in dataclasses.fields(solver_parameters)
        ]
        # The parameters of an output from a compiled solve hold JAX arrays
        reaction_system: Any = self.parameters.reaction_system
        reactions: dict[int, str] = get_reaction_dictionary(
            np.asarray(reaction_system.matrix), reaction_system.species.species_names
        )
        rows += [
            ("reactions", f"reaction_{index}", equation) for index, equation in reactions.items()
        ]
        rows += [("units", name, unit) for name, unit in OUTPUT_UNITS.items()]
        for name, species_ in zip(self.parameters.species.species_names, self.parameters.species):
            rows += [
                ("species", f"{name}.solve_for_stability", species_.solve_for_stability),
                ("species", f"{name}.include_in_phase_mass", species_.include_in_phase_mass),
            ]

        return pd.DataFrame(rows, columns=["section", "name", "value"])

    def to_dataframes(
        self,
        output_format: Literal["named_arrays", "elements_species"] = "named_arrays",
        drop_unsuccessful_solves: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Each top-level key becomes a DataFrame, with columns formed by joining nested keys with "."

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            output_format: The format of the output dictionary. Can be ``named_arrays`` for a
                format with named arrays, or ``elements_species`` for a format grouped by element
                and species names. Defaults to ``named_arrays``.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.

        Returns:
            Dictionary mapping top-level keys to pandas DataFrames
        """
        nested_dict: dict[str, Any] = self.to_dict(
            output_format=output_format, to_numpy=True, expand_to_batch=True, ravel=True
        )

        result: dict[str, pd.DataFrame] = {}

        for top_key, value in nested_dict.items():
            if isinstance(value, dict):
                flat: dict = flatten_dictionary(value)
                result[top_key] = pd.DataFrame(flat)
            else:
                result[top_key] = pd.DataFrame({top_key: value})
            result[top_key].index.name = "model"

        if drop_unsuccessful_solves:
            logger.info("Dropping unsuccessful solves from output")
            result = self._drop_unsuccessful_solves(result)

        return result

    def to_excel(
        self,
        file_prefix: Path | str = "atmodeller_out",
        output_format: Literal["named_arrays", "elements_species"] = "named_arrays",
        drop_unsuccessful_solves: bool = False,
        append_version: bool = True,
    ) -> None:
        """Writes the output to an Excel file.

        The first sheet, ``info``, describes the run (see :meth:`info`), and each other sheet has
        one row per model. Rows of models that were not solved are highlighted.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            file_prefix: Prefix of the output file. Accepts ``str`` or :class:`pathlib.Path`.
                Defaults to atmodeller_out.
            output_format: The format of the output dictionary. Can be ``named_arrays`` for a
                format with named arrays, or ``elements_species`` for a format grouped by element
                and species names. Defaults to ``named_arrays``.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
            append_version: Whether to append ``_v<package_version>`` to the output filename.
                Defaults to ``True``.
        """
        logger.info("Writing output to excel")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            output_format=output_format, drop_unsuccessful_solves=drop_unsuccessful_solves
        )
        output_path: Path = Path(file_prefix)
        base_name: str = output_path.name
        if base_name.endswith(".xlsx"):
            base_name = base_name[: -len(".xlsx")]
        if append_version:
            version_tag: str = f"_v{__version__}"
            if not base_name.endswith(version_tag):
                base_name = f"{base_name}{version_tag}"
        output_file: Path = output_path.with_name(f"{base_name}.xlsx")

        # Convenient to highlight rows where the solver failed to find a solution for follow-up
        # analysis. Define a fill colour for highlighting rows (e.g., yellow)
        highlight_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

        # Get the indices of the solves that fail the objective-based check or the solver status
        unsuccessful_indices: NpArray = np.where(~np.asarray(self.checked_solution.success))[0]

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            self.info().to_excel(writer, sheet_name="info", index=False)
            for column in writer.sheets["info"].columns:
                width: int = max(len(str(cell.value)) for cell in column) + 2
                writer.sheets["info"].column_dimensions[column[0].column_letter].width = width

            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)
                sheet = writer.sheets[df_name]

                # Freeze the header row and the model column, and fit the columns to the headers
                sheet.freeze_panes = "B2"
                for column in sheet.iter_cols(max_row=1):
                    sheet.column_dimensions[column[0].column_letter].width = max(
                        10, len(str(column[0].value)) + 2
                    )

                # Map batch indices to row positions, since unsuccessful solves may be dropped
                unsuccessful_rows: NpArray = np.flatnonzero(df.index.isin(unsuccessful_indices))

                # Apply highlighting to the rows where the solver failed to find a solution
                for idx in unsuccessful_rows:
                    # Highlight the entire row (starting from index 2 to skip header row)
                    for col in range(1, len(df.columns) + 2):
                        # row=idx+2 because Excel is 1-indexed and row 1 is the header
                        cell = sheet.cell(row=idx + 2, column=col)
                        cell.fill = highlight_fill

        logger.info("Output written to %s", output_file)

    def to_pickle(
        self,
        file_prefix: Path | str = "atmodeller_out",
        output_format: Literal["named_arrays", "elements_species"] = "named_arrays",
        drop_unsuccessful_solves: bool = False,
        append_version: bool = True,
    ) -> None:
        """Writes the output to a pickle file.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            file_prefix: Prefix of the output file. Accepts ``str`` or :class:`pathlib.Path`.
                Defaults to atmodeller_out.
            output_format: The format of the output dictionary. Can be ``named_arrays`` for a
                format with named arrays, or ``elements_species`` for a format grouped by element
                and species names. Defaults to ``named_arrays``.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
            append_version: Whether to append ``_v<package_version>`` to the output filename.
                Defaults to ``True``.
        """
        logger.info("Writing output to pickle")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            output_format=output_format, drop_unsuccessful_solves=drop_unsuccessful_solves
        )
        output_path: Path = Path(file_prefix)
        base_name: str = output_path.name
        if base_name.endswith(".pkl"):
            base_name = base_name[: -len(".pkl")]
        if append_version:
            version_tag: str = f"_v{__version__}"
            if not base_name.endswith(version_tag):
                base_name = f"{base_name}{version_tag}"
        output_file: Path = output_path.with_name(f"{base_name}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)

    def solver_stats_to_logger(self, logger_: logging.Logger = logger) -> None:
        """Logs solver statistics.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a :func:`jax.jit` context)

        Args:
            logger_: Logger to write the statistics to. Defaults to the ``atmodeller.output``
                logger.
        """
        return self.checked_solution.stats_to_logger(logger_)
