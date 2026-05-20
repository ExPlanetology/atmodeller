# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`ZD09`"""

from typing import cast

import jax.numpy as jnp
import numpy as np
import numpy.testing as nptest
from scipy.constants import kilo

from atmodeller.eos import get_eos_models
from atmodeller.eos._zhang_duan import ZhangDuanMixture
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import NpArray
from atmodeller.sci_utils import GAS_CONSTANT_BAR, unit_conversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
MODEL_SUFFIX: str = "zhang09"
"""Suffix of the :cite:t:`ZD09` models"""


def test_H2O_vol_low_TP(check_values) -> None:
    """Tests H2O volume at 1203.15 K and 950 MPa :cite:p:`ZD09{Table 6}`"""
    expected: float = 22.20343433408026
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1203.15,
        9500,  # 950 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_H2O_vol_high_TP(check_values) -> None:
    """Tests H2O volume at 1873.15 K and 2500 MPa :cite:p:`ZD09{Table 6}`"""
    expected: float = 19.41089977577485
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1873.15,
        25000,  # 2500 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_H2O_vol_high_TP2(check_values) -> None:
    """Tests H2O volume at 1373.15 K and 3500 MPa :cite:p:`ZD09{Table 6}`"""
    expected: float = 16.02290245403692
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1373.15,
        35000,  # 3500 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


# def test_volume_with_broadcasting(check_values) -> None:
#     """Tests volume with broadcasting"""
#     model: RealGas = check_values.get_eos_model("H2O", MODEL_SUFFIX)
#     check_values.check_broadcasting("volume", model)


# def test_fugacity_with_broadcasting(check_values) -> None:
#     """Tests volume with broadcasting"""
#     model: RealGas = check_values.get_eos_model("H2O", MODEL_SUFFIX)
#     check_values.check_broadcasting("fugacity", model)


def test_H2O_CO2_vol_mix_low_H2O() -> None:
    """Experimental molar volume in the H2O-CO2 system

    Tests H2O-CO2 mixture at 1573.15 K and 14500 MPa :cite:p:`ZD09{Figure 1}` using data from
    :cite:t:`Frost1997`.
    """
    species: tuple[str, ...] = ("H2O", "CO2")
    model = ZhangDuanMixture(species)

    mole_fractions = jnp.array([0.62, 0.38])

    expected: float = 28.19426
    expected *= unit_conversion.cm3_to_m3
    volume = model.volume(1573.15, 14500, mole_fractions)

    nptest.assert_allclose(volume, expected, rtol=RTOL, atol=ATOL)


def test_H2O_CO2_vol_mix_high_H2O() -> None:
    """Experimental molar volume in the H2O-CO2 system

    Tests H2O-CO2 mixture at 1573.15 K and 14500 MPa :cite:p:`ZD09{Figure 1}` using data from
    :cite:t:`Frost1997`.
    """
    species: tuple[str, ...] = ("H2O", "CO2")
    model = ZhangDuanMixture(species)

    mole_fractions = jnp.array([0.875, 0.125])

    expected: float = 23.93116
    expected *= unit_conversion.cm3_to_m3
    volume = model.volume(1573.15, 14500, mole_fractions)

    nptest.assert_allclose(volume, expected, rtol=RTOL, atol=ATOL)


def test_RTlnfCO2() -> None:
    """Comparison of RTlnf for CO2

    Tests based on the data for MgCO3 --> MgO + CO2 :cite:p:`ZD09{Table 8}`.
    """
    # MPa to bar
    unit_scale = 10

    pressure = np.array([500, 700, 1000, 1630, 1920, 2200]) * unit_scale
    temperature = np.array([1298, 1278, 1373, 1600, 1700, 1800])

    model_CO2: RealGas = cast(RealGas, get_eos_models()["CO2_zhang09"])
    RTlnfCO2 = model_CO2.volume_integral_J(temperature, pressure) / kilo  # convert J->kJ

    expected: NpArray = np.array([106.5, 114.7, 135.1, 178.1, 197.1, 215.6])

    # rtol chosen to account for rounding in the expected values, which are only given to 1
    # decimal place in the paper. The expected values are likely rounded, but the model is not,
    # so we need to allow for some tolerance in the comparison.
    nptest.assert_allclose(RTlnfCO2, expected, rtol=3.3e-4, atol=ATOL)


# TODO: template test for RTlnf of a species in a mixture?
def test_H2O_CO2_RTlnf_CO2_working() -> None:
    """Comparison of RTlnf for CO2

    Tests based on :cite:p:`ZD09{Table 8}`.
    """
    species: tuple[str, ...] = ("H2O", "CO2")
    model = ZhangDuanMixture(species)

    pressure = np.array([500, 700, 1000, 1630, 1920, 2200]) * 10
    temperature = np.array([1298, 1278, 1373, 1600, 1700, 1800])

    mole_fractions = jnp.array([0.875, 0.125])
    # mole_fractions = jnp.array([1.0, 0.0])

    epsilon = model.get_species_epsilon(mole_fractions)
    print("epsilon:", epsilon)

    # Pure models
    model_H2O = get_eos_models()["H2O_zhang09_unbounded"]
    log_fugacity = model_H2O.log_fugacity(temperature, pressure)
    print("Pure log_fugacity (H2O):", log_fugacity)

    RTlnfH2O = GAS_CONSTANT_BAR * temperature * log_fugacity
    print("Pure RTlnf (H2O):", RTlnfH2O)

    model_CO2 = get_eos_models()["CO2_zhang09"]
    volume = model_CO2.volume(temperature, pressure)  # type: ignore
    print("Pure volume (CO2):", volume)
    log_fugacity_CO2 = model_CO2.log_fugacity(temperature, pressure)
    print("Pure log_fugacity (CO2):", log_fugacity_CO2)

    RTlnfCO2 = GAS_CONSTANT_BAR * temperature * log_fugacity_CO2
    print("Pure RTlnf (CO2):", RTlnfCO2)

    # Mixture models
    model_H2O = ZhangDuanMixture(species, species_to_output="H2O")
    log_fugacity = model_H2O.log_fugacity(temperature, pressure, mole_fractions)
    print("Mixture log_fugacity (H2O):", log_fugacity)

    model_CO2 = ZhangDuanMixture(species, species_to_output="CO2")
    log_fugacity_CO2 = model_CO2.log_fugacity(temperature, pressure, mole_fractions)
    print("Mixturelog_fugacity (CO2):", log_fugacity_CO2)

    # expected: float = 32.45717
    # expected *= unit_conversion.cm3_to_m3
    # volume = model.volume(temperature, pressure, mole_fractions)

    # nptest.assert_allclose(volume, expected, rtol=RTOL, atol=ATOL)
