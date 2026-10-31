# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for atmodeller.jax_utils"""

import logging

import numpy as np

from atmodeller import debug_logger
from atmodeller.jax_utils import left_null_space

logger: logging.Logger = debug_logger()
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
