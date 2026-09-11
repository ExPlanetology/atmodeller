# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the CORK EOS models from :cite:t:`HP91,HP98`"""

import pytest

from atmodeller.eos import RealGas
from atmodeller.eos._holland_powell import (
    CO2_cork_holland91,
    H2O_cork_holland91_bounded,
    H2O_cork_holland98_bounded,
)
from atmodeller.sci_utils import unit_conversion


def test_H2O_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 2a}`"""
    model: RealGas = H2O_cork_holland91_bounded
    expected: float = 47.502083040419844 * unit_conversion.cm3_to_m3
    check_values.volume(873, 1000, model, expected)


def test_CO2_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 7}`"""
    model: RealGas = check_values.get_eos_model("CO2", "cork_holland91")
    expected: float = 96.13326116472262 * unit_conversion.cm3_to_m3
    check_values.volume(873, 1000, model, expected)


# species, temperature (K), pressure (bar), expected volume (m3), citation figure. CO, CH4, and H2
# all share the "cork_cs_holland91" corresponding-states formulation, so this is one code path
# parametrized over each species' own reference points from the cited figure, not distinct
# branches of the implementation.
VOLUME_CASES = [
    pytest.param("CO", 1173, 1000, 131.475184896045, id="CO-1kbar-Figure8a"),
    pytest.param("CO", 973, 2000, 71.32153159834933, id="CO-2kbar-Figure8a"),
    pytest.param("CO", 1473, 4000, 62.22167162862537, id="CO-4kbar-Figure8a"),
    pytest.param("CH4", 1173, 1000, 131.6743085645421, id="CH4-1kbar-Figure8b"),
    pytest.param("CH4", 973, 2000, 72.14376119913776, id="CH4-2kbar-Figure8b"),
    pytest.param("CH4", 1473, 4000, 63.106094264549, id="CH4-4kbar-Figure8b"),
    pytest.param("H2", 773, 500, 149.1657987388235, id="H2-500bar-Figure8c"),
    pytest.param("H2", 773, 1800, 55.04174839002075, id="H2-1800bar-Figure8c"),
    pytest.param("H2", 773, 10000, 20.67497630046999, id="H2-10kbar-Figure8c"),
]


@pytest.mark.parametrize("species, temperature, pressure, expected_cm3", VOLUME_CASES)
def test_cork_cs_holland91_volume(
    check_values, species: str, temperature: float, pressure: float, expected_cm3: float
) -> None:
    """Tests volume against the reference value for each species/pressure point"""
    model: RealGas = check_values.get_eos_model(species, "cork_cs_holland91")
    expected: float = expected_cm3 * unit_conversion.cm3_to_m3
    check_values.volume(temperature, pressure, model, expected)


# species, expected fugacity coefficient at T=2000 K, P=10 kbar. All species share the
# "cork_cs_holland91" formulation, so this is one code path parametrized over each species'
# reference value, not distinct branches of the implementation.
FUGACITY_COEFFICIENT_CASES = [
    pytest.param("H2", 4.67146087585007, id="H2"),
    pytest.param("CO", 7.735168014913625, id="CO"),
    pytest.param("CH4", 8.01145999484921, id="CH4"),
    pytest.param("CO2", 7.118598073639082, id="simple_CO2"),
]


@pytest.mark.parametrize("species, expected", FUGACITY_COEFFICIENT_CASES)
def test_cork_cs_holland91_fugacity_coefficient(check_values, species: str, expected: float) -> None:
    """Tests fugacity coefficient against the reference value for each species"""
    model: RealGas = check_values.get_eos_model(species, "cork_cs_holland91")
    check_values.fugacity_coefficient(2000, 10e3, model, expected)


def test_CO2_at_P0(check_values) -> None:
    """CO2 below P0 so virial contribution excluded"""
    model: RealGas = check_values.get_eos_model("CO2", "cork_holland98")
    expected: float = 1.57505991404597
    check_values.fugacity_coefficient(2000, 2e3, model, expected)


def test_CO2_above_P0(check_values) -> None:
    """CO2 above P0 so virial contribution included"""
    model: RealGas = check_values.get_eos_model("CO2", "cork_holland98")
    expected: float = 7.142958711915495
    check_values.fugacity_coefficient(2000, 10e3, model, expected)


def test_H2O_above_Tc_above_P0(check_values) -> None:
    """H2O above Tc and above P0"""
    model: RealGas = H2O_cork_holland98_bounded
    expected: float = 1.344344209075713
    check_values.fugacity_coefficient(2000, 5e3, model, expected)


def test_H2O_below_Tc_above_P0(check_values) -> None:
    """H2O below Tc and above P0"""
    model: RealGas = H2O_cork_holland98_bounded
    expected: float = 0.39156756638038
    check_values.fugacity_coefficient(600, 10e3, model, expected)


def test_broadcasting(check_values) -> None:
    """Tests methods with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2O", "cork_holland98")
    check_values.check_broadcasting(model)


def test_CO2_raw_cork_log_fugacity(check_values) -> None:
    """Tests the raw (unbounded) CORK class's log_fugacity called directly

    This exercises :meth:`~atmodeller.eos.core.CORK.log_fugacity`, which is inherited from
    :class:`~atmodeller.eos.core.RealGas` and delegates to :meth:`CORK.volume_integral`, without
    going via :class:`~atmodeller.eos._aggregators.CombinedRealGas`.
    """
    model: RealGas = CO2_cork_holland91
    expected: float = 7.171588313368521
    check_values._check_property("log_fugacity", 873, 1000, model, expected)
