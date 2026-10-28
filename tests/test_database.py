# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermodynamic databases in :mod:`atmodeller.database`"""

import importlib.resources
from pathlib import Path
from typing import Self, cast

import equinox as eqx
import pytest

from atmodeller import override
from atmodeller.containers import ChemicalSpecies
from atmodeller.database import DataBase, GlennDataBase, get_default_database
from atmodeller.interfaces import species_name
from atmodeller.phases import PurePhase
from atmodeller.thermodata.core import DATA_DIRECTORY
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.glenn import THERMODYNAMIC_DATA_SOURCE, glenn_properties


def test_default_database() -> None:
    """The default database gives the same species as creating them without a database"""
    db: DataBase = get_default_database()

    assert eqx.tree_equal(
        ChemicalSpecies.create_gas("H2O", database=db), ChemicalSpecies.create_gas("H2O")
    )
    for component in ("graphite", "diamond"):
        assert eqx.tree_equal(
            ChemicalSpecies.from_component(component, database=db),
            ChemicalSpecies.from_component(component),
        )


def test_from_file(tmp_path: Path) -> None:
    """A database from a file matches the packaged data, and a file with missing columns raises"""
    packaged = DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)
    with importlib.resources.as_file(packaged) as path:  # type: ignore
        lines: list[str] = Path(path).read_text().splitlines(keepends=True)
    glenn_file: Path = tmp_path / "glenn.txt"
    glenn_file.write_text("".join(line for line in lines if not line.startswith("gas,H2O,")))
    db: DataBase = GlennDataBase.from_file(glenn_file)

    assert eqx.tree_equal(
        ChemicalSpecies.create_gas("CO2", database=db),
        ChemicalSpecies.create_gas("CO2", database=GlennDataBase()),
    )
    with pytest.raises(KeyError, match="gas is not a phase component of H2O"):
        ChemicalSpecies.create_gas("H2O", database=db)

    bad_file: Path = tmp_path / "bad.txt"
    bad_file.write_text("phase_component,hill_formula,T_min,T_max\n")
    with pytest.raises(ValueError, match="is missing the columns"):
        GlennDataBase.from_file(bad_file)


def test_set_phase_component() -> None:
    """Phase components are added or replaced in this database only, and a replaced phase
    component from the data file can still be stitched"""
    db: GlennDataBase = GlennDataBase()
    db.set_phase_component("SiO2", "test", diamond_gustafson)
    graphite = db.get_phase_component("C", "graphite").with_volume(GRAPHITE_VOLUME_GUSTAFSON)
    db.set_phase_component("C", "graphite", graphite)

    assert ChemicalSpecies.from_component("test", database=db).name == "O2Si_test"
    assert "test" not in db.components_of("SiO2")
    assert "test" not in GlennDataBase().phase_components()["O2Si"]
    assert db.components_of("C") == ("graphite",)
    assert eqx.tree_equal(db.stitch("C", ("graphite",)), graphite)
    with pytest.raises(ValueError, match="already given for O2Si"):
        db.set_phase_component("C", "test", diamond_gustafson)


def test_custom_database() -> None:
    """A custom database only needs to provide its data"""

    class MyDataBase(DataBase):
        @override
        @classmethod
        def default(cls) -> Self:
            return cls({"H2O": {"gas": glenn_properties["H2O"]["gas"]}})

    db: DataBase = MyDataBase.default()

    assert eqx.tree_equal(
        ChemicalSpecies.create_gas("H2O", database=db), ChemicalSpecies.create_gas("H2O")
    )
    with pytest.raises(KeyError, match="CO2 is not available"):
        ChemicalSpecies.create_gas("CO2", database=db)


def test_phase_components() -> None:
    """Condensed species from phase components match the phase components, and the rules hold"""
    sio2 = cast(ChemicalSpecies, PurePhase.from_formula("SiO2").species[0])
    carbon = cast(ChemicalSpecies, PurePhase.from_formula("C").species[0])

    db: DataBase = get_default_database()
    assert db.components_of("SiO2") == (
        "quartz-alpha",
        "quartz-beta",
        "cristobalite-beta",
        "liquid",
    )
    cho: dict[str, tuple[str, ...]] = db.phase_components(("C", "H", "O"))
    assert {"H2O", "CO2", "C"} <= set(cho) and "O2Si" not in cho
    assert (species_name("SO2"), species_name("SiO2", "l")) == ("O2S_g", "O2Si_l")
    assert (sio2.name, carbon.name) == ("O2Si_cd", "C_cd")
    # Quartz, then the liquid above the melting point of cristobalite (1996 K)
    for temperature, component in ((1000.0, "quartz-beta"), (2500.0, "liquid")):
        expected = ChemicalSpecies.from_component(component, formula="SiO2")
        assert sio2.standard_gibbs_over_RT(temperature, 1.0) == pytest.approx(
            expected.standard_gibbs_over_RT(temperature, 1.0)
        )
    # Graphite on its own keeps the volume, unlike graphite from the database without it
    assert carbon.standard_gibbs_over_RT(1000.0, 1.0e4) == pytest.approx(
        ChemicalSpecies.from_component("graphite").standard_gibbs_over_RT(1000.0, 1.0e4)
    )
    no_volume = cast(
        ChemicalSpecies, PurePhase.from_formula("C", database=GlennDataBase()).species[0]
    )
    assert carbon.standard_gibbs_over_RT(1000.0, 1.0e4) != pytest.approx(
        no_volume.standard_gibbs_over_RT(1000.0, 1.0e4)
    )

    assert PurePhase.from_component("cristobalite-beta").name == "O2Si_cristobalite-beta"
    assert PurePhase.from_component("liquid", formula="H2O").name == "H2O_l"
    assert PurePhase.from_component("diamond").name == "C_diamond"
    assert PurePhase(ChemicalSpecies.from_component("graphite")).name == "C_graphite"
    assert PurePhase.from_components(("quartz-alpha", "quartz-beta"), name="qz").name == "O2Si_qz"
    melt_sio2 = ChemicalSpecies.from_component(
        "liquid", formula="SiO2", include_in_phase_mass=False
    )
    assert (melt_sio2.name, melt_sio2.include_in_phase_mass) == ("O2Si_l", False)

    with pytest.raises(ValueError, match="hematite .* overlaps liquid"):
        PurePhase.from_formula("Fe2O3")
    with pytest.raises(ValueError, match="give the formula"):
        PurePhase.from_component("liquid")
    with pytest.raises(ValueError, match="not a condensed phase component"):
        PurePhase.from_components(("gas", "liquid"), name="x", formula="H2O")
    with pytest.raises(ValueError, match="from a data file"):
        PurePhase.from_components(("graphite", "diamond"), name="x")
    with pytest.raises(KeyError, match="ice is not a phase component of O2Si"):
        PurePhase.from_components(("quartz-alpha", "ice"), name="x")
