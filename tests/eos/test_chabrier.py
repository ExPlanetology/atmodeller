#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Tests for the EOS models from :cite:t:`CD21`"""

from atmodeller.utilities import unit_conversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
MODEL_SUFFIX: str = "chabrier21"
"""Suffix of the :cite:t:`CD21` models"""


def test_Chabrier_H2_volume_100kbar(check_values) -> None:
    """Tests Chabrier volume at 100 kbar"""
    expected: float = 9.005066169376918
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        3000,
        100e3,
        check_values.get_eos_model("H2", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_Chabrier_H2_fugacity_coefficient_100kbar(check_values) -> None:
    """Tests Chabrier fugacity coefficient at 100 kbar"""
    # Assumes 1000 integration steps
    expected: float = 32.84516697626837
    check_values.fugacity_coefficient(
        3000,
        100e3,
        check_values.get_eos_model("H2", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_Chabrier_H2_volume_1000kbar(check_values) -> None:
    """Tests Chabrier volume at 1000 kbar"""
    expected: float = 3.0100820540769166
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        5000,
        1000e3,
        check_values.get_eos_model("H2", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_Chabrier_H2_fugacity_coefficient_1000kbar(check_values) -> None:
    """Tests Chabrier fugacity coefficient at 1000 kbar"""
    # Assumes 1000 integration steps
    expected: float = 451196.92186256585
    check_values.fugacity_coefficient(
        5000,
        1000e3,
        check_values.get_eos_model("H2", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )
