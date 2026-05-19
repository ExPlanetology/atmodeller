# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`ZD09`"""

import jax.numpy as jnp
import numpy.testing as nptest

from atmodeller.eos._zhang_duan import ZhangDuanMixture
from atmodeller.sci_utils import unit_conversion

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
