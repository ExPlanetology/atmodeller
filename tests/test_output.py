# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the public output API (:mod:`atmodeller.output`) and its base infrastructure
(:mod:`atmodeller.output_base`)."""

import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from jaxtyping import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import ActivityConstraintProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.output import Output
from atmodeller.output_base import flatten_dictionary, recursively_merge_dictionaries
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

# Gas species
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")

# Melt species
H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
)
CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "CO2", solubility=solubility_models["CO2_basalt_dixon95"], include_in_phase_mass=False
)


def _solve(
    gas_species: tuple[ChemicalSpecies, ...],
    mass_constraints: dict[str, ArrayLike],
    *,
    melt_species: tuple[SpeciesProtocol, ...] = (),
    activity_constraints: Optional[dict[str, ActivityConstraintProtocol]] = None,
) -> Output:
    """Builds and solves a model, shared by the fixtures below."""
    planet: BaseThermodynamicState = Planet.from_species(
        gas_species, silicate_melt_species=melt_species
    )
    parameters: Parameters = Parameters(
        planet, activity_constraints=activity_constraints, mass_constraints=mass_constraints
    )
    model: EquilibriumModel = EquilibriumModel(parameters)

    return model.solve_with_default()


@pytest.fixture(scope="module")
def h2o_output() -> Output:
    """Output for a simple H2O-H2O(melt) model.

    Used as the general-purpose fixture for exercising the ``Output`` API (to_dict formats,
    quick_look, dataframes, and file exports).
    """
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(2)

    return _solve((H2O_g,), {"H": h_kg}, melt_species=(H2O_d,))


@pytest.fixture(scope="module")
def batched_output() -> Output:
    """Output for a batch of two H budgets, following test_HO.py's batch-testing pattern.

    Used to exercise the array-broadcasting branch of
    :func:`~atmodeller.output_base.expand_jax_arrays_in_pytree_to_batch`, which is only triggered
    when scalar/shape-(1,) leaves must be broadcast up to a batch size greater than one.
    """
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(2)

    return _solve((H2O_g,), {"H": np.array([h_kg, 2 * h_kg])}, melt_species=(H2O_d,))


@pytest.fixture(scope="module")
def heavy_element_output() -> Output:
    """Output for a C-H-O model, following the C-H-O system in test_CHO.py.

    Carbon is present as a heavy element (i.e. not H or He), so this is used to exercise the
    metallicity-computed branch of :class:`~atmodeller.output_base.OutputElementsSpeciesDict`.
    """
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(1)
    c_kg: ArrayLike = h_kg  # ch_ratio = 1

    return _solve(
        (H2O_g, H2_g, O2_g, CO_g, CO2_g),
        {"C": c_kg, "H": h_kg},
        melt_species=(H2O_d, CO2_d),
        activity_constraints={"O2_g": IronWustiteBuffer()},
    )


@pytest.fixture(scope="module")
def h_only_output() -> Output:
    """Output for an H-only model (H2 and atomic H gas species), following test_HO.py's pattern.

    Since the only element present is H, there are no heavy elements (elements other than H or
    He), which exercises the ``else: zeros`` branch of the metallicity calculation in
    :class:`~atmodeller.output_base.OutputElementsSpeciesDict`. Two species (rather than a single
    species with no reactions) are used to avoid a degenerate solver edge case.
    """
    H_g: ChemicalSpecies = ChemicalSpecies.create_gas("H")
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(1)

    return _solve((H2_g, H_g), {"H": h_kg})


class TestToDict:
    """Tests for Output.to_dict()"""

    def test_natural(self, h2o_output: Output) -> None:
        out: dict[str, Any] = h2o_output.to_dict(output_format="natural", to_numpy=True)

        assert "gas" in out
        assert "solution" in out
        assert "solver" in out
        assert "state" in out
        assert "constraints" in out
        assert "H2O_g" in out["gas"]["species"]["names"]
        assert isinstance(out["gas"]["species"]["partial_pressure"], np.ndarray)
        assert out["gas"]["species"]["partial_pressure"].shape[-1] == len(
            out["gas"]["species"]["names"]
        )

    def test_named_arrays(self, h2o_output: Output) -> None:
        out: dict[str, Any] = h2o_output.to_dict(output_format="named_arrays", to_numpy=True)

        assert "gas" in out
        assert "solution" in out
        assert "solver" in out
        assert "state" in out
        assert "H2O_g" in out["gas"]["species"]["partial_pressure"]

    def test_elements_species(self, h2o_output: Output) -> None:
        out: dict[str, Any] = h2o_output.to_dict(output_format="elements_species", to_numpy=True)

        assert "H2O_g" in out
        assert "H" in out
        assert "O" in out
        assert "gas" in out
        assert "solution" in out
        assert "solver" in out
        assert "state" in out

    def test_default_format_is_named_arrays(self, h2o_output: Output) -> None:
        default_out: dict[str, Any] = h2o_output.to_dict(to_numpy=True)
        named_out: dict[str, Any] = h2o_output.to_dict(output_format="named_arrays", to_numpy=True)

        assert set(default_out.keys()) == set(named_out.keys())

    def test_elements_species_expand_and_ravel(self, batched_output: Output) -> None:
        """Exercises the expand_to_batch and ravel branches of OutputElementsSpeciesDict.to_dict,
        including array broadcasting (in expand_jax_arrays_in_pytree_to_batch) for leaves such as
        ``state.temperature`` that are not already shaped to the batch size (2)."""
        out: dict[str, Any] = batched_output.to_dict(
            output_format="elements_species", to_numpy=True, expand_to_batch=True, ravel=True
        )

        assert np.asarray(out["state"]["temperature"]).shape == (2,)
        assert np.asarray(out["gas"]["phase"]["mass"]).shape == (2,)

    def test_invalid_format_raises(self, h2o_output: Output) -> None:
        # The test session runs with beartype instrumentation enabled (see pyproject.toml's
        # ``beartype_packages``), which would otherwise intercept an invalid Literal value before
        # the method body executes. Call through ``__wrapped__`` (the plain, undecorated function
        # that beartype preserves) to exercise the method's own ``raise ValueError`` branch.
        to_dict = getattr(Output.to_dict, "__wrapped__", Output.to_dict)
        with pytest.raises(ValueError):
            to_dict(h2o_output, output_format="not_a_real_format")  # pyright: ignore


def test_solution_property(h2o_output: Output) -> None:
    """Tests the Output.solution property (log number of moles and log stability)."""
    solution = h2o_output.solution

    assert solution.shape[-1] == 2 * len(h2o_output.parameters.species.species_names)


def test_compare(h2o_output: Output) -> None:
    """Tests Output.compare(), following the pattern in test_HO.py's test_H2O."""
    target: dict[str, Any] = {
        "gas": {"species": {"partial_pressure": {"H2O_g": 1.0312913336898137}}}
    }

    assert h2o_output.compare(target, rtol=1.0e-6, atol=1.0e-6)

    mismatched_target: dict[str, Any] = {"gas": {"species": {"partial_pressure": {"H2O_g": -1.0}}}}

    assert not h2o_output.compare(mismatched_target, rtol=1.0e-6, atol=1.0e-6)

    # log=True compares the base-10 logarithm of the values
    assert h2o_output.compare(target, rtol=1.0e-6, atol=1.0e-6, log=True)


def test_quick_look(h2o_output: Output, caplog: pytest.LogCaptureFixture) -> None:
    """Tests that quick_look runs without raising and logs a pformat'd dict."""
    with caplog.at_level(logging.INFO, logger="atmodeller"):
        h2o_output.quick_look()

    assert "Quick look output" in caplog.text


class TestToDataframes:
    """Tests for Output.to_dataframes()"""

    def test_to_dataframes(self, h2o_output: Output) -> None:
        dataframes: dict[str, pd.DataFrame] = h2o_output.to_dataframes()

        assert isinstance(dataframes, dict)
        assert len(dataframes) > 0
        for df in dataframes.values():
            assert isinstance(df, pd.DataFrame)

        assert "gas" in dataframes
        assert any(col.startswith("species.partial_pressure") for col in dataframes["gas"].columns)

    def test_to_dataframes_drop_unsuccessful_solves(self, h2o_output: Output) -> None:
        dataframes: dict[str, pd.DataFrame] = h2o_output.to_dataframes(
            drop_unsuccessful_solves=True
        )

        assert isinstance(dataframes, dict)
        assert len(dataframes) > 0

        n_success: int = int(np.sum(np.asarray(h2o_output.multi_attempt_solution.solver_success)))
        for df in dataframes.values():
            assert len(df) == n_success

    def test_to_dataframes_elements_species(self, h2o_output: Output) -> None:
        dataframes: dict[str, pd.DataFrame] = h2o_output.to_dataframes(
            output_format="elements_species"
        )

        assert isinstance(dataframes, dict)
        assert "H2O_g" in dataframes
        assert isinstance(dataframes["H2O_g"], pd.DataFrame)


def _read_excel(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None, index_col=0)


def _read_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


# (export method, file extension). Both to_excel and to_pickle share the same
# file_prefix/append_version handling, so the behaviours below are parametrized over both rather
# than duplicated per format.
_EXPORT_FORMATS = [
    pytest.param(Output.to_excel, "xlsx", id="excel"),
    pytest.param(Output.to_pickle, "pkl", id="pickle"),
]
# Same as above, plus the function to read the written file back (only needed to check content).
_EXPORT_FORMATS_WITH_READER = [
    pytest.param(Output.to_excel, "xlsx", _read_excel, id="excel"),
    pytest.param(Output.to_pickle, "pkl", _read_pickle, id="pickle"),
]


class TestExports:
    """Tests for Output.to_excel() and Output.to_pickle()"""

    @pytest.mark.parametrize("export, extension, read_back", _EXPORT_FORMATS_WITH_READER)
    def test_default_prefix(
        self, h2o_output: Output, tmp_path: Path, export, extension: str, read_back
    ) -> None:
        prefix: Path = tmp_path / "atmodeller_out"
        export(h2o_output, file_prefix=prefix)

        expected: Path = tmp_path / f"atmodeller_out_v{__version__}.{extension}"
        assert expected.exists()

        # Sanity check that the written file can be read back and has expected content
        data = read_back(expected)
        assert "gas" in data

    @pytest.mark.parametrize("export, extension", _EXPORT_FORMATS)
    def test_no_version_suffix(
        self, h2o_output: Output, tmp_path: Path, export, extension: str
    ) -> None:
        prefix: Path = tmp_path / "no_version"
        export(h2o_output, file_prefix=prefix, append_version=False)

        expected: Path = tmp_path / f"no_version.{extension}"
        assert expected.exists()

    @pytest.mark.parametrize("export, extension", _EXPORT_FORMATS)
    def test_prefix_already_has_extension(
        self, h2o_output: Output, tmp_path: Path, export, extension: str
    ) -> None:
        """Tests that a file_prefix already ending in the export extension is not doubled up."""
        prefix: Path = tmp_path / f"already_suffixed.{extension}"
        export(h2o_output, file_prefix=prefix, append_version=False)

        expected: Path = tmp_path / f"already_suffixed.{extension}"
        assert expected.exists()
        assert not (tmp_path / f"already_suffixed.{extension}.{extension}").exists()

    def test_to_excel_relative_prefix(
        self, h2o_output: Output, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tests that a bare (non-absolute) file_prefix writes relative to the cwd."""
        monkeypatch.chdir(tmp_path)
        h2o_output.to_excel(file_prefix="relative_out", append_version=False)

        expected: Path = tmp_path / "relative_out.xlsx"
        assert expected.exists()


class TestElementsSpeciesMetallicity:
    """Tests for the metallicity branches of OutputElementsSpeciesDict, exercised via
    Output.to_dict(output_format="elements_species")."""

    def test_heavy_element_present(self, heavy_element_output: Output) -> None:
        out: dict[str, Any] = heavy_element_output.to_dict(
            output_format="elements_species", to_numpy=True
        )

        gas_phase: dict[str, Any] = out["gas"]["phase"]
        assert "metallicity_by_moles" in gas_phase
        assert "metallicity_by_mass" in gas_phase

        # Carbon is present as a heavy element, so metallicity should be non-zero.
        assert np.all(np.asarray(gas_phase["metallicity_by_moles"]) > 0)
        assert np.all(np.asarray(gas_phase["metallicity_by_mass"]) > 0)

    def test_no_heavy_elements(self, h_only_output: Output) -> None:
        out: dict[str, Any] = h_only_output.to_dict(
            output_format="elements_species", to_numpy=True
        )

        gas_phase: dict[str, Any] = out["gas"]["phase"]
        assert "metallicity_by_moles" in gas_phase
        assert "metallicity_by_mass" in gas_phase

        # No heavy elements (only H is present), so metallicity should be exactly zero.
        assert np.allclose(np.asarray(gas_phase["metallicity_by_moles"]), 0.0)
        assert np.allclose(np.asarray(gas_phase["metallicity_by_mass"]), 0.0)


class TestFlattenDictionary:
    """Tests for the flatten_dictionary helper function"""

    @pytest.mark.parametrize(
        "d, expected",
        [
            pytest.param({"a": 1, "b": 2}, {"a": 1, "b": 2}, id="flat_unchanged"),
            pytest.param(
                {"a": {"b": 1, "c": 2}, "d": 3}, {"a.b": 1, "a.c": 2, "d": 3}, id="nested"
            ),
            pytest.param({"a": {"b": {"c": {"d": 42}}}}, {"a.b.c.d": 42}, id="deeply_nested"),
            pytest.param({}, {}, id="empty"),
        ],
    )
    def test_flatten(self, d: dict[str, Any], expected: dict[str, Any]) -> None:
        assert flatten_dictionary(d) == expected

    def test_parent_key_prefix(self) -> None:
        d: dict[str, Any] = {"b": 1}
        assert flatten_dictionary(d, parent_key="a") == {"a.b": 1}


class TestRecursivelyMergeDictionaries:
    """Tests for the recursively_merge_dictionaries helper function"""

    @pytest.mark.parametrize(
        "d1, d2, expected",
        [
            pytest.param({"a": 1}, {"b": 2}, {"a": 1, "b": 2}, id="disjoint_keys"),
            pytest.param({"a": 1}, {"a": 2}, {"a": 2}, id="overwrite_non_dict_value"),
            pytest.param(
                {"a": {"x": 1, "y": 2}},
                {"a": {"y": 3, "z": 4}},
                {"a": {"x": 1, "y": 3, "z": 4}},
                id="recursive_merge_of_nested_dicts",
            ),
            pytest.param(
                {"a": {"x": 1}}, {"a": 5}, {"a": 5}, id="dict_overwritten_by_non_dict"
            ),
        ],
    )
    def test_merge(self, d1: dict[str, Any], d2: dict[str, Any], expected: dict[str, Any]) -> None:
        assert recursively_merge_dictionaries(d1, d2) == expected

    def test_original_dictionaries_not_mutated(self) -> None:
        d1: dict[str, Any] = {"a": {"x": 1}}
        d2: dict[str, Any] = {"a": {"y": 2}}
        merged = recursively_merge_dictionaries(d1, d2)

        assert merged == {"a": {"x": 1, "y": 2}}
        assert d1 == {"a": {"x": 1}}
        assert d2 == {"a": {"y": 2}}
