# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the CORK EOS models from :cite:t:`HP91,HP98`"""

from atmodeller.eos import RealGas
from atmodeller.eos._holland_powell import H2O_cork_holland91_bounded, H2O_cork_holland98_bounded
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


def test_CO_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8a}`"""
    model: RealGas = check_values.get_eos_model("CO", "cork_cs_holland91")
    expected: float = 131.475184896045 * unit_conversion.cm3_to_m3
    check_values.volume(1173, 1000, model, expected)


def test_CO_volume_2kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8a}`"""
    model: RealGas = check_values.get_eos_model("CO", "cork_cs_holland91")
    expected: float = 71.32153159834933 * unit_conversion.cm3_to_m3
    check_values.volume(973, 2000, model, expected)


def test_CO_volume_4kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8a}`"""
    model: RealGas = check_values.get_eos_model("CO", "cork_cs_holland91")
    expected: float = 62.22167162862537 * unit_conversion.cm3_to_m3
    check_values.volume(1473, 4000, model, expected)


def test_CH4_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8b}`"""
    model: RealGas = check_values.get_eos_model("CH4", "cork_cs_holland91")
    expected: float = 131.6743085645421 * unit_conversion.cm3_to_m3
    check_values.volume(1173, 1000, model, expected)


def test_CH4_volume_2kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8b}`"""
    model: RealGas = check_values.get_eos_model("CH4", "cork_cs_holland91")
    expected: float = 72.14376119913776 * unit_conversion.cm3_to_m3
    check_values.volume(973, 2000, model, expected)


def test_CH4_volume_4kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8b}`"""
    model: RealGas = check_values.get_eos_model("CH4", "cork_cs_holland91")
    expected: float = 63.106094264549 * unit_conversion.cm3_to_m3
    check_values.volume(1473, 4000, model, expected)


def test_H2_volume_500bar(check_values) -> None:
    """:cite:t:`HP91{Figure 8c}`"""
    model: RealGas = check_values.get_eos_model("H2", "cork_cs_holland91")
    expected: float = 149.1657987388235 * unit_conversion.cm3_to_m3
    check_values.volume(773, 500, model, expected)


def test_H2_volume_1800bar(check_values) -> None:
    """:cite:t:`HP91{Figure 8c}`"""
    model: RealGas = check_values.get_eos_model("H2", "cork_cs_holland91")
    expected: float = 55.04174839002075 * unit_conversion.cm3_to_m3
    check_values.volume(773, 1800, model, expected)


def test_H2_volume_10kb(check_values) -> None:
    """:cite:t:`HP91{Figure 8c}`"""
    model: RealGas = check_values.get_eos_model("H2", "cork_cs_holland91")
    expected: float = 20.67497630046999 * unit_conversion.cm3_to_m3
    check_values.volume(773, 10000, model, expected)


def test_H2(check_values) -> None:
    """H2"""
    model: RealGas = check_values.get_eos_model("H2", "cork_cs_holland91")
    expected: float = 4.67146087585007
    check_values.fugacity_coefficient(2000, 10e3, model, expected)


def test_CO(check_values) -> None:
    """CO"""
    model: RealGas = check_values.get_eos_model("CO", "cork_cs_holland91")
    expected: float = 7.735168014913625
    check_values.fugacity_coefficient(2000, 10e3, model, expected)


def test_CH4(check_values) -> None:
    """CH4"""
    model: RealGas = check_values.get_eos_model("CH4", "cork_cs_holland91")
    expected: float = 8.01145999484921
    check_values.fugacity_coefficient(2000, 10e3, model, expected)


def test_simple_CO2(check_values) -> None:
    """Simple CO2"""
    model: RealGas = check_values.get_eos_model("CO2", "cork_cs_holland91")
    expected: float = 7.118598073639082
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
