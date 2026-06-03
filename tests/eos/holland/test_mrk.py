# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the MRK EOS models from :cite:t:`HP91,HP98`

These are not intended to be used directly, but rather as a building block of the CORK models they
serve as convenient tests in the absence of a virial correction term.
"""

from atmodeller.eos import RealGas
from atmodeller.eos._holland_powell import (
    CO2_mrk_cs_holland91,
    CO2MrkHolland91,
    H2O_mrk_fluid_holland91,
    H2O_mrk_gas_holland91,
    H2OMrkHolland91,
)


def test_CO2_corresponding_states(check_values) -> None:
    """CO2 corresponding states"""
    model: RealGas = CO2_mrk_cs_holland91
    expected: float = 1.5831992703027848
    check_values.fugacity_coefficient(2000, 2e3, model, expected)


def test_CO2(check_values) -> None:
    """CO2"""
    model: RealGas = CO2MrkHolland91
    expected: float = 1.575457075165528
    check_values.fugacity_coefficient(2000, 2e3, model, expected)


def test_H2O_above_Tc(check_values) -> None:
    """H2O above Tc"""
    model: RealGas = H2O_mrk_fluid_holland91
    expected: float = 1.048278616058322
    check_values.fugacity_coefficient(2000, 1e3, model, expected)


def test_H2O_below_Tc_below_Psat(check_values) -> None:
    """H2O below Tc and below Psat"""
    # Psat = 0.118224 kbar at T = 600 K
    model: RealGas = H2O_mrk_gas_holland91
    expected: float = 0.7910907770688191
    check_values.fugacity_coefficient(600, 0.1e3, model, expected)


def test_H2O_below_Tc_above_Psat(check_values) -> None:
    """H2O below Tc and above Psat"""
    # Psat = 0.118224 kbar at T = 600 K
    expected: float = 0.13704706029361396
    check_values.fugacity_coefficient(600, 1e3, H2OMrkHolland91, expected)


def test_broadcasting_CO2(check_values) -> None:
    """Tests CO2 properties with broadcasting"""
    model: RealGas = CO2MrkHolland91
    check_values.check_broadcasting(model)


def test_broadcasting_H2O(check_values) -> None:
    """Tests H2O properties with broadcasting"""
    model: RealGas = H2OMrkHolland91
    check_values.check_broadcasting(model)
