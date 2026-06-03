# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the Viral Quadratic EOS models :cite:p:`WLL18`"""

from atmodeller.eos import RealGas
from atmodeller.eos._wang import H4Si_wang18_bounded


def test_H4Si_fugacity_coefficient_300K_1bar(check_values) -> None:
    """H4Si fugacity coefficients"""
    model: RealGas = H4Si_wang18_bounded
    expected: float = 1
    check_values.fugacity_coefficient(300, 1, model, expected)


def test_H4Si_fugacity_coefficient_3000K_50e3bar(check_values) -> None:
    """H4Si fugacity coefficients"""
    model: RealGas = H4Si_wang18_bounded
    expected: float = 28.758474
    check_values.fugacity_coefficient(3000, 50e3, model, expected)


def test_H4Si_volume_1230K_515e3bar(check_values) -> None:
    """H4Si volumes :cite:p:`WLL18{Table 4}`"""
    model: RealGas = H4Si_wang18_bounded
    expected: float = 1.616524434600913e-05  # published value 1.611e-05 (+/- 0.005e-05) m3/mol
    check_values.volume(1230, 515e3, model, expected)  # 1230 (+/- 30) K, 515e3 (+/- 3e3) bar


def test_H4Si_volume_2540K_924e3bar(check_values) -> None:
    """H4Si volumes :cite:p:`WLL18{Table 4}`"""
    model: RealGas = H4Si_wang18_bounded
    expected: float = 1.41539090679e-05  # published value 1.401e-05 (+/- 0.007e-05) m3/mol
    check_values.volume(2540, 924e3, model, expected)  # 2540 (+/- 60) K, 924e3 (+/- 10e3) bar


def test_broadcasting(check_values) -> None:
    """Tests methods with broadcasting"""
    model: RealGas = H4Si_wang18_bounded
    check_values.check_broadcasting(model)
