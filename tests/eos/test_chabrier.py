# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`CD21`"""

from atmodeller.eos import RealGas
from atmodeller.eos._chabrier import (
    H2_3000K_chabrier21,
    H2_4000K_chabrier21,
    H2_chabrier21,
    _H2_3000K_chabrier21,
)
from atmodeller.sci_utils import unit_conversion


def test_H2_volume_100kbar(check_values) -> None:
    """Tests volume at 100 kbar"""
    expected: float = 9.005066169376918
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(3000, 100e3, H2_chabrier21, expected)


def test_H2_fugacity_coefficient_100kbar(check_values) -> None:
    """Tests fugacity coefficient at 100 kbar"""
    # Assumes 100 integration steps
    expected: float = 33.741562
    check_values.fugacity_coefficient(3000, 100e3, H2_chabrier21, expected)


def test_H2_volume_1000kbar(check_values) -> None:
    """Tests volume at 1000 kbar"""
    expected: float = 3.0100820540769166
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(5000, 1000e3, H2_chabrier21, expected)


def test_H2_fugacity_coefficient_1000kbar(check_values) -> None:
    """Tests fugacity coefficient at 1000 kbar"""
    # Assumes 100 integration steps
    expected: float = 482475.388584
    check_values.fugacity_coefficient(5000, 1000e3, H2_chabrier21, expected)


def test_broadcasting(check_values) -> None:
    """Tests methods with broadcasting"""
    model: RealGas = H2_chabrier21
    check_values.check_broadcasting(model)


def test_H2_3000K_ideal_branch(check_values) -> None:
    """Tests the ideal gas (extrapolated) branch below 1 bar for the 3000 K function fit

    This exercises :meth:`CombinedRealGasFugacity._get_index` and
    :meth:`CombinedRealGasFugacity.log_fugacity` for the first (ideal gas) EOS in the composite.
    """
    model: RealGas = H2_3000K_chabrier21
    # An ideal gas has a fugacity coefficient of unity by construction
    expected: float = 1.0
    check_values.fugacity_coefficient(3000, 0.5, model, expected)


def test_H2_3000K_fit_branch(check_values) -> None:
    """Tests the Chabrier function-fit branch above 1 bar for the 3000 K function fit"""
    model: RealGas = H2_3000K_chabrier21
    expected: float = 1.1909553080149688
    check_values.fugacity_coefficient(3000, 100, model, expected)
    expected_log_fugacity: float = 4.77992595090065
    check_values._check_property("log_fugacity", 3000, 100, model, expected_log_fugacity)


def test_H2_4000K_ideal_branch(check_values) -> None:
    """Tests the ideal gas (extrapolated) branch below 1 bar for the 4000 K function fit"""
    model: RealGas = H2_4000K_chabrier21
    expected: float = 1.0
    check_values.fugacity_coefficient(4000, 0.5, model, expected)


def test_H2_4000K_fit_branch(check_values) -> None:
    """Tests the Chabrier function-fit branch above 1 bar for the 4000 K function fit"""
    model: RealGas = H2_4000K_chabrier21
    expected: float = 4.239207102263808
    check_values.fugacity_coefficient(4000, 1000, model, expected)
    expected_log_fugacity: float = 8.352131526555477
    check_values._check_property("log_fugacity", 4000, 1000, model, expected_log_fugacity)


def test_ChabrierFunction_raw_log_fugacity(check_values) -> None:
    """Tests the raw ChabrierFunction class's log_fugacity override directly

    :class:`~atmodeller.eos._aggregators.CombinedRealGasFugacity` only ever calls
    ``log_fugacity_coefficient`` on its constituent EOS models (not ``log_fugacity``), so
    :meth:`~atmodeller.eos._chabrier.ChabrierFunction.log_fugacity` is otherwise never exercised.
    """
    model: RealGas = _H2_3000K_chabrier21
    expected: float = 4.77992595090065
    check_values._check_property("log_fugacity", 3000, 100, model, expected)
