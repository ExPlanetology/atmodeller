# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catalogue information of the model registries"""

import re
from pathlib import Path

from atmodeller.eos import get_eos_catalogue
from atmodeller.interfaces import ModelInfo
from atmodeller.solubility import get_solubility_catalogue

BIBLIOGRAPHY: Path = Path(__file__).parents[1] / "docs" / "refs.bib"


def test_catalogue_complete() -> None:
    """Every model of the libraries has catalogue information (except ``NO_SOLUBILITY``, which is
    not listed), with a species and sources that are in the bibliography of the documentation"""
    bibliography: set[str] = set(re.findall(r"@\w+\{([^,]+),", BIBLIOGRAPHY.read_text()))
    catalogues = {"eos": get_eos_catalogue(), "solubility": get_solubility_catalogue()}

    for name, catalogue in catalogues.items():
        for key, entry in catalogue.items():
            if key == "NO_SOLUBILITY":
                assert entry.info == ModelInfo()
                continue
            assert entry.info.species, f"{name} {key}: no species"
            assert entry.info.sources, f"{name} {key}: no sources"
            missing: set[str] = set(entry.info.sources) - bibliography
            assert not missing, f"{name} {key}: {missing} not in the bibliography"
