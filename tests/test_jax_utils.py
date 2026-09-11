# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for atmodeller.jax_utils"""

import logging

import numpy as np

from atmodeller import debug_logger
from atmodeller.jax_utils import partial_rref

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)


def test_partial_rref_rank_deficient_matrix() -> None:
    """Tests partial_rref with a rank-deficient matrix to hit the no-pivot skip branch."""
    # This matrix has a duplicate row and an all-zero row, so it is rank deficient (rank 2, with 3
    # columns and 4 rows). This exercises both the forward "no pivot in this column" branch and the
    # backward "pivot is zero, skipping" branch.
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = partial_rref(matrix)

    assert result.shape == (1, 4)
    assert np.all(np.isfinite(result))
