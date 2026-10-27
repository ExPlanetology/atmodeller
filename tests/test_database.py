# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermodynamic databases in :mod:`atmodeller.database`"""

import importlib.resources
from pathlib import Path
from typing import Self

import equinox as eqx
import pytest

from atmodeller import override
from atmodeller.containers import ChemicalSpecies
from atmodeller.database import DataBase, GlennDataBase, components_of, get_default_database
from atmodeller.phases import PurePhase
from atmodeller.thermodata.core import DATA_DIRECTORY, ThermodynamicProperties
from atmodeller.thermodata.gustafson import diamond_gustafson
from atmodeller.thermodata.janaf import THERMODYNAMIC_DATA_SOURCE, glenn_properties


def test_default_database() -> None:
    """The default database gives the same species as creating them directly"""
    db: DataBase = get_default_database()

    assert eqx.tree_equal(db.create_gas("H2O"), ChemicalSpecies.create_gas("H2O"))
    for formula, state in (("H2O", "l"), ("C", "s"), ("C", "diamond")):
        assert eqx.tree_equal(
            db.create_condensed(formula, state=state),
            ChemicalSpecies.create_condensed(formula, state=state),
        )


def test_from_file(tmp_path: Path) -> None:
    """A database from a file matches the packaged data, without the species removed from it"""
    packaged = DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)
    with importlib.resources.as_file(packaged) as path:  # type: ignore
        lines: list[str] = Path(path).read_text().splitlines(keepends=True)
    glenn_file: Path = tmp_path / "glenn.txt"
    glenn_file.write_text("".join(line for line in lines if not line.startswith("H2O,g,")))
    db: DataBase = GlennDataBase.from_file(glenn_file)

    assert eqx.tree_equal(db.create_gas("CO2"), GlennDataBase().create_gas("CO2"))
    with pytest.raises(KeyError, match="H2O_g not available"):
        db.create_gas("H2O")


def test_add_species() -> None:
    """Species are named from the Hill formula and replace existing ones, in this database only"""
    db: GlennDataBase = GlennDataBase()
    db.add_species("SiO2", "test", diamond_gustafson)
    db.add_species("C", "s", diamond_gustafson)

    assert db.create_condensed("SiO2", state="test").name == "O2Si_test"
    assert eqx.tree_equal(db.get_thermodynamic_properties("C_s"), diamond_gustafson)
    assert db.available_species().count("C_s") == 1
    assert "O2Si_test" not in glenn_properties
    assert "O2Si_test" not in GlennDataBase().available_species()


def test_custom_database() -> None:
    """A custom database only needs to provide its data"""

    class MyDataBase(DataBase):
        def __init__(self, coefficients: dict[str, ThermodynamicProperties]):
            super().__init__()
            self.coefficients = coefficients

        @override
        @classmethod
        def default(cls) -> Self:
            return cls({"H2O_g": glenn_properties["H2O_g"]})

        @override
        def set_thermodynamic_properties(
            self, name: str, properties: ThermodynamicProperties
        ) -> None:
            self.coefficients[name] = properties

        @override
        def available_species(self) -> tuple[str, ...]:
            return tuple(self.coefficients)

        @override
        def get_thermodynamic_properties(self, name: str) -> ThermodynamicProperties:
            return self.coefficients[name]

    db: DataBase = MyDataBase.default()

    assert eqx.tree_equal(db.create_gas("H2O"), ChemicalSpecies.create_gas("H2O"))
    with pytest.raises(KeyError, match=r"CO2_g not available. Available species are \('H2O_g',\)"):
        db.create_gas("CO2")


def test_phase_components() -> None:
    """Condensed species from phase components match the old species, and the rules hold"""
    db: DataBase = get_default_database()
    sio2: ChemicalSpecies = db.condensed_from_formula("SiO2")
    carbon: ChemicalSpecies = db.condensed_from_formula("C")

    assert components_of("SiO2") == ("quartz-alpha", "quartz-beta", "cristobalite-beta", "liquid")
    assert PurePhase.from_formula("SiO2").name == sio2.name == "O2Si_cd"
    # Solids, then the liquid above the melting point of cristobalite (1996 K)
    for temperature, state in ((1000.0, "s"), (2500.0, "l")):
        assert sio2.standard_gibbs_over_RT(temperature, 1.0) == pytest.approx(
            db.create_condensed("SiO2", state=state).standard_gibbs_over_RT(temperature, 1.0)
        )
    # Graphite on its own keeps the volume, unlike graphite from the database without it
    assert carbon.name == "C_cd"
    assert carbon.standard_gibbs_over_RT(1000.0, 1.0e4) == pytest.approx(
        db.create_condensed("C").standard_gibbs_over_RT(1000.0, 1.0e4)
    )
    no_volume: ChemicalSpecies = GlennDataBase().condensed_from_formula("C")
    assert carbon.standard_gibbs_over_RT(1000.0, 1.0e4) != pytest.approx(
        no_volume.standard_gibbs_over_RT(1000.0, 1.0e4)
    )

    assert PurePhase.from_component("cristobalite-beta").name == "O2Si_cristobalite-beta"
    assert PurePhase.from_component("liquid", formula="H2O").name == "H2O_l"
    assert PurePhase.from_component("diamond").name == "C_diamond"
    assert PurePhase.from_components(("quartz-alpha", "quartz-beta"), name="qz").name == "O2Si_qz"

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
