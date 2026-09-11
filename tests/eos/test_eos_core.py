# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the core real gas EOS classes in :mod:`atmodeller.eos.core`"""

import numpy.testing as nptest
import pytest

from atmodeller.eos._aggregators import CombinedRealGas, CombinedRealGasFugacity
from atmodeller.eos._holland_powell import H2O_virial_compensation_holland91, dummy_critical_data
from atmodeller.eos.core import IdealGas, VirialCompensation
from atmodeller.sci_utils import ExperimentalCalibration

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""


def test_virial_compensation_below_P0() -> None:
    """Tests VirialCompensation.volume and volume_integral below P0 (no contribution).

    Uses the same coefficients as :obj:`~atmodeller.eos._holland_powell.H2O_virial_compensation_holland91`.
    """
    model: VirialCompensation = H2O_virial_compensation_holland91
    assert model.P0 == 2000.0

    temperature: float = 1000.0
    pressure: float = 1000.0  # Below P0

    volume = model.volume(temperature, pressure, dummy_critical_data)
    volume_integral = model.volume_integral(temperature, pressure, dummy_critical_data)

    nptest.assert_allclose(volume, 0.0, rtol=0, atol=ATOL)
    nptest.assert_allclose(volume_integral, 0.0, rtol=0, atol=ATOL)


def test_virial_compensation_above_P0() -> None:
    """Tests VirialCompensation.volume and volume_integral above P0 (non-zero contribution).

    This exercises the ``_a``, ``_b``, and ``_c`` helper methods together with the branch of
    :meth:`~atmodeller.eos.core.VirialCompensation._delta_pressure` where pressure exceeds P0.
    """
    model: VirialCompensation = H2O_virial_compensation_holland91

    temperature: float = 1000.0
    pressure: float = 5000.0  # Above P0

    volume = model.volume(temperature, pressure, dummy_critical_data)
    volume_integral = model.volume_integral(temperature, pressure, dummy_critical_data)

    expected_volume: float = -6.468509203936454e-07
    expected_volume_integral: float = -0.0012785783412872908

    nptest.assert_allclose(volume, expected_volume, rtol=RTOL, atol=ATOL)
    nptest.assert_allclose(volume_integral, expected_volume_integral, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("cls", [CombinedRealGas, CombinedRealGasFugacity])
def test_upper_pressure_bounds_malformed(cls: type) -> None:
    """Tests that a malformed calibrations tuple raises a ValueError.

    A non-last calibration entry that lacks ``pressure_max`` is not allowed because the upper
    pressure bound cannot be determined for it. ``CombinedRealGas`` and ``CombinedRealGasFugacity``
    each duplicate this ``_get_upper_pressure_bounds`` logic verbatim (they don't share a common
    base for it), so both are checked here rather than because they exercise different behaviour.
    """
    calibrations = (
        ExperimentalCalibration(pressure_min=1),
        ExperimentalCalibration(pressure_min=1, pressure_max=100),
    )

    with pytest.raises(ValueError):
        cls._get_upper_pressure_bounds(calibrations)


def test_combined_real_gas_fugacity_last_calibration_without_max() -> None:
    """Tests that a valid calibrations tuple whose last entry lacks pressure_max is accepted.

    This exercises the ``continue`` branch of
    :meth:`~atmodeller.eos._aggregators.CombinedRealGasFugacity._get_upper_pressure_bounds`, which
    is only reached when the (permitted) unbounded last calibration lacks a maximum pressure.
    """
    calibrations = (ExperimentalCalibration(pressure_min=1),)

    upper_pressure_bounds = CombinedRealGasFugacity._get_upper_pressure_bounds(calibrations)
    assert upper_pressure_bounds == ()

    model: CombinedRealGasFugacity = CombinedRealGasFugacity((IdealGas(),), calibrations)
    expected: float = 4.605170185988092  # log(100)

    nptest.assert_allclose(model.log_fugacity(1000.0, 100.0), expected, rtol=RTOL, atol=ATOL)
