# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`CD21`"""

from atmodeller.eos import RealGas
from atmodeller.eos._chabrier import H2_chabrier21
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
