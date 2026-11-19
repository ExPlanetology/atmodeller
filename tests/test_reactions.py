# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for atmodeller.reactions"""

import logging

import numpy as np
import pytest

from atmodeller import ChemicalSpecies, Planet, ReservoirSpecies, console_logger
from atmodeller.reactions import left_null_space
from atmodeller.solubility import get_solubility_models

logger: logging.Logger = console_logger()
logger.setLevel(logging.WARNING)


def test_left_null_space_rank_deficient_matrix() -> None:
    """A rank-deficient matrix gives nrows minus the rank combinations, each of which vanishes"""
    # A duplicate row and an all-zero row give rank 2, with 3 columns and 4 rows, and the column
    # without a pivot must not use up a row
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = left_null_space(matrix)

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result @ matrix, 0.0, atol=1e-12)
    # H2O_g and H2O_l: H and O in a fixed ratio, so one component and one reaction
    np.testing.assert_array_equal(left_null_space(np.array([[2.0, 1.0], [2.0, 1.0]])), [[-1, 1]])


def test_fO2_dependent_solubility_requires_O2() -> None:
    """A solubility law that depends on fO2 needs O2 as a gas species, checked when the model is
    built rather than at run time, where a check would also fire for a diverged model of a batch"""
    gas_species = tuple(ChemicalSpecies.create_gas(formula) for formula in ("H2", "H2O", "N2"))
    melt_species = (
        ReservoirSpecies.create_dissolved(
            "N2", solubility=get_solubility_models()["N2_basalt_libourel03"]
        ),
    )

    with pytest.raises(ValueError, match="O2 must be included as a gas species"):
        Planet.from_species(gas_species, silicate_melt_species=melt_species)
