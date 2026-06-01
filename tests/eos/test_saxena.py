# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`SF87,SF87a,SF88,SS92`"""

from atmodeller.eos import RealGas
from atmodeller.eos._saxena import H2_SF87
from atmodeller.sci_utils import unit_conversion


def test_Ar(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGas = check_values.get_eos_model("Ar", "cs_saxena87")
    expected: float = 7.41624600755374
    check_values.compressibility(2510, 100e3, model, expected)


def test_CH4(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGas = check_values.get_eos_model("CH4", "cs_shi92")
    expected: float = 17.77499804453072
    check_values.compressibility(1912, 159e3, model, expected)


def test_CO2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGas = check_values.get_eos_model("CO2", "cs_shi92")
    expected: float = 33.886349109271734
    check_values.compressibility(1167, 184e3, model, expected)


def test_H2_SF87(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGas = H2_SF87
    expected: float = 4.975497264839999
    check_values.compressibility(1222, 41.66e3, model, expected)


def test_N2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGas = check_values.get_eos_model("N2", "cs_saxena87")
    expected: float = 10.293087737779091
    check_values.compressibility(1573, 75e3, model, expected)


def test_O2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGas = check_values.get_eos_model("O2", "cs_shi92")
    expected: float = 12.409268281002012
    check_values.compressibility(1823, 133e3, model, expected)


def test_H2_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGas = check_values.get_eos_model("H2", "shi92")
    expected: float = 7279.356114821697 * unit_conversion.cm3_to_m3
    check_values.volume(873, 10, model, expected)


def test_H2_medium_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGas = check_values.get_eos_model("H2", "shi92")
    expected: float = 164.38851468757488 * unit_conversion.cm3_to_m3
    check_values.volume(873, 500, model, expected)


def test_H2_high_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGas = check_values.get_eos_model("H2", "shi92")
    expected: float = 41.97871061892679 * unit_conversion.cm3_to_m3
    check_values.volume(1473, 4000, model, expected)


def test_H2_high_pressure2_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGas = check_values.get_eos_model("H2", "shi92")
    expected: float = 20.806595067793276 * unit_conversion.cm3_to_m3
    check_values.volume(1073, 10000, model, expected)


def test_H2_high_pressure3_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGas = check_values.get_eos_model("H2", "shi92")
    expected: float = 71.50153474005484 * unit_conversion.cm3_to_m3
    check_values.volume(673, 1000, model, expected)


def test_H2S_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 3}`"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    expected: float = 272.7266232763035 * unit_conversion.cm3_to_m3
    check_values.volume(673, 200, model, expected)


def test_H2S_medium_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 3}`"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    expected: float = 116.55537998390933 * unit_conversion.cm3_to_m3
    check_values.volume(1873, 2000, model, expected)


def test_SO2_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 2}`"""
    model: RealGas = check_values.get_eos_model("SO2", "shi92")
    expected: float = 8308.036738813245 * unit_conversion.cm3_to_m3
    check_values.volume(1073, 10, model, expected)


def test_SO2_high_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 2}`"""
    model: RealGas = check_values.get_eos_model("SO2", "shi92")
    expected: float = 70.86864302460566 * unit_conversion.cm3_to_m3
    check_values.volume(1873, 4000, model, expected)


def test_lower_extrapolation(check_values) -> None:
    """Tests the lower bound extrapolation"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    temperature: float = 1073
    pressure: float = 0.01
    expected: float = 0.01
    check_values.fugacity(temperature, pressure, model, expected)


def test_upper_extrapolation(check_values) -> None:
    """Tests the upper bound extrapolation"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    temperature: float = 1073
    pressure: float = 10e3  # Maximum pressure of the highest pressure EOS in this composite EOS
    pressure_above_max: float = 20e3  # To test pressure above the maximum
    # Expected compressibility factor at 3000 K and (the maximum calibrated pressure) 10000 bar
    expected: float = 4.472116811082408
    check_values.compressibility(temperature, pressure, model, expected)
    # The upper bound extrapolation should maintain the same compressibility for higher pressures
    # for dz/dp if reinstated at some point
    # expected: float = 8.098866090720415
    check_values.compressibility(temperature, pressure_above_max, model, expected)


def test_volume_integral_standard_pressure(check_values) -> None:
    """Tests the volume integral"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    temperature: float = 1873
    pressure: float = 1.0
    # At the standard pressure the volume integral is zero by construction
    expected: float = 0.0
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_integral_index0(check_values) -> None:
    """Tests the volume integral for the first EOS"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    # Within ranges of experimental data, otherwise the extrapolated behaviour might be unphysical
    temperature: float = 1073
    pressure: float = 1000.0
    expected: float = 0.619184870412741
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_integral_index1(check_values) -> None:
    """Tests the volume integral for the first and second EOS"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    temperature: float = 1073
    pressure: float = 9999
    expected: float = 1.088570827561088
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_integral_index2(check_values) -> None:
    """Tests the volume integral in the extrapolated region above the maximum pressure"""
    model: RealGas = check_values.get_eos_model("H2S", "shi92")
    temperature: float = 1073
    pressure: float = 20000
    # for dz/dp if reinstated at some point
    # expected: float = 1.4644445135917623
    expected: float = 1.365159989472265
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_with_broadcasting(check_values) -> None:
    """Tests volume with broadcasting"""
    model: RealGas = check_values.get_eos_model("CO2", "cs_shi92")
    check_values.check_broadcasting("volume", model)


def test_fugacity_with_broadcasting(check_values) -> None:
    """Tests volume with broadcasting"""
    model: RealGas = check_values.get_eos_model("CO2", "cs_shi92")
    check_values.check_broadcasting("fugacity", model)
