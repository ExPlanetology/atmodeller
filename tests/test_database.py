# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermodynamic databases in :mod:`atmodeller.database`"""

import importlib.resources
from pathlib import Path
from typing import cast

import equinox as eqx
import pytest

from atmodeller.containers import ChemicalSpecies
from atmodeller.database import DataBase
from atmodeller.interfaces import species_name
from atmodeller.phases import PurePhase
from atmodeller.thermodata.core import DATA_DIRECTORY
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.glenn import THERMODYNAMIC_DATA_SOURCE, glenn_properties


def test_default_database() -> None:
    """The default database gives the same species as creating them without a database"""
    db: DataBase = DataBase.default()

    assert eqx.tree_equal(
        ChemicalSpecies.create_gas("H2O", database=db), ChemicalSpecies.create_gas("H2O")
    )
    for end_member in ("graphite", "diamond"):
        assert eqx.tree_equal(
            ChemicalSpecies.from_end_member(end_member, database=db),
            ChemicalSpecies.from_end_member(end_member),
        )


def test_from_file(tmp_path: Path) -> None:
    """A database from a file matches the packaged data, and a file with missing columns raises"""
    packaged = DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)
    with importlib.resources.as_file(packaged) as path:  # type: ignore
        lines: list[str] = Path(path).read_text().splitlines(keepends=True)
    glenn_file: Path = tmp_path / "glenn.txt"
    glenn_file.write_text("".join(line for line in lines if not line.startswith("gas,H2O,")))
    db: DataBase = DataBase.from_glenn_file(glenn_file)

    assert eqx.tree_equal(
        ChemicalSpecies.create_gas("CO2", database=db),
        ChemicalSpecies.create_gas("CO2", database=DataBase(glenn_properties)),
    )
    with pytest.raises(KeyError, match="gas is not an end-member of H2O"):
        ChemicalSpecies.create_gas("H2O", database=db)

    bad_file: Path = tmp_path / "bad.txt"
    bad_file.write_text("end_member,hill_formula,T_min,T_max\n")
    with pytest.raises(ValueError, match="is missing the columns"):
        DataBase.from_glenn_file(bad_file)


def test_set_end_member() -> None:
    """End-members are added or replaced in this database only, and a replaced
    end-member that the database was created with can still be stitched"""
    db: DataBase = DataBase(glenn_properties)
    db.set_end_member("SiO2", "test", diamond_gustafson)
    graphite = db.get_end_member("C", "graphite").with_volume(GRAPHITE_VOLUME_GUSTAFSON)
    db.set_end_member("C", "graphite", graphite)

    assert ChemicalSpecies.from_end_member("test", database=db).name == "O2Si_test"
    assert "test" not in db.stitchable_end_members("SiO2")
    assert "test" not in DataBase(glenn_properties).end_members()["O2Si"]
    assert db.stitchable_end_members("C") == ("graphite",)
    assert eqx.tree_equal(db.stitch("C", ("graphite",)), graphite)
    with pytest.raises(ValueError, match="already given for O2Si"):
        db.set_end_member("C", "test", diamond_gustafson)


def test_custom_database() -> None:
    """A custom database only needs its data"""
    db: DataBase = DataBase({"H2O": {"gas": glenn_properties["H2O"]["gas"]}})

    assert eqx.tree_equal(
        ChemicalSpecies.create_gas("H2O", database=db), ChemicalSpecies.create_gas("H2O")
    )
    with pytest.raises(KeyError, match="CO2 is not available"):
        ChemicalSpecies.create_gas("CO2", database=db)


def test_end_members() -> None:
    """Condensed species from end-members match the end-members, and the rules hold"""
    sio2 = cast(ChemicalSpecies, PurePhase.from_formula("SiO2").species[0])
    sio2_s = cast(ChemicalSpecies, PurePhase.from_formula("SiO2", liquid=False).species[0])
    carbon = cast(ChemicalSpecies, PurePhase.from_formula("C").species[0])

    db: DataBase = DataBase.default()
    assert db.stitchable_end_members("SiO2") == (
        "quartz-alpha",
        "quartz-beta",
        "cristobalite-beta",
        "liquid",
    )
    cho: dict[str, tuple[str, ...]] = db.end_members(("C", "H", "O"))
    assert {"H2O", "CO2", "C"} <= set(cho) and "O2Si" not in cho
    assert (species_name("SO2"), species_name("SiO2", "l")) == ("O2S_g", "O2Si_l")
    assert (sio2.name, sio2_s.name, carbon.name) == ("O2Si_cd", "O2Si_s", "C_cd")
    # Without the liquid, the solids end at the melting point of cristobalite (1996 K)
    assert sio2_s.thermo.temperature_range() == (200.0, 1996.0)
    # Quartz, then the liquid above the melting point of cristobalite (1996 K)
    for temperature, end_member in ((1000.0, "quartz-beta"), (2500.0, "liquid")):
        expected = ChemicalSpecies.from_end_member(end_member, formula="SiO2")
        assert sio2.standard_gibbs_over_RT(temperature, 1.0) == pytest.approx(
            expected.standard_gibbs_over_RT(temperature, 1.0)
        )
    # Graphite on its own keeps the volume, unlike graphite from the database without it
    assert carbon.standard_gibbs_over_RT(1000.0, 1.0e4) == pytest.approx(
        ChemicalSpecies.from_end_member("graphite").standard_gibbs_over_RT(1000.0, 1.0e4)
    )
    glenn: DataBase = DataBase(glenn_properties)
    no_volume = cast(ChemicalSpecies, PurePhase.from_formula("C", database=glenn).species[0])
    assert carbon.standard_gibbs_over_RT(1000.0, 1.0e4) != pytest.approx(
        no_volume.standard_gibbs_over_RT(1000.0, 1.0e4)
    )

    assert PurePhase.from_end_member("cristobalite-beta").name == "O2Si_cristobalite-beta"
    assert PurePhase.from_end_member("liquid", formula="H2O").name == "H2O_l"
    assert PurePhase.from_end_member("diamond").name == "C_diamond"
    assert PurePhase(ChemicalSpecies.from_end_member("graphite")).name == "C_graphite"
    assert (
        PurePhase.from_end_members(("quartz-alpha", "quartz-beta"), suffix="qz").name == "O2Si_qz"
    )
    melt_sio2 = ChemicalSpecies.from_end_member(
        "liquid", formula="SiO2", include_in_phase_mass=False
    )
    assert (melt_sio2.name, melt_sio2.include_in_phase_mass) == ("O2Si_l", False)

    with pytest.raises(KeyError, match="Fe has no solid end-members"):
        PurePhase.from_formula("Fe", liquid=False)
    with pytest.raises(ValueError, match="hematite .* overlaps liquid"):
        PurePhase.from_formula("Fe2O3")
    with pytest.raises(ValueError, match="give the formula"):
        PurePhase.from_end_member("liquid")
    with pytest.raises(ValueError, match="not a condensed end-member"):
        PurePhase.from_end_members(("gas", "liquid"), suffix="x", formula="H2O")
    with pytest.raises(ValueError, match="that the database was created with"):
        PurePhase.from_end_members(("graphite", "diamond"), suffix="x")
    with pytest.raises(KeyError, match="ice is not an end-member of O2Si"):
        PurePhase.from_end_members(("quartz-alpha", "ice"), suffix="x")
