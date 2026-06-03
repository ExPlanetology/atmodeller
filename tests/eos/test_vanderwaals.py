# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`Lide2005`"""

from atmodeller.eos import RealGas

MODEL_SUFFIX: str = "vdw_lide05"
"""Suffix of the :cite:t:`Lide2005` models"""


def test_H2_volume_100bar(check_values) -> None:
    """Tests H2 volume at 300 K and 100 bar"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.volume(300, 100, model, 2.85325715e-4)


def test_H2_volume_1kbar(check_values) -> None:
    """Tests H2 volume at 1000 K and 1000 bar"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.volume(1000, 1000, model, 1.12342095e-4)


def test_broadcasting(check_values) -> None:
    """Tests methods with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.check_broadcasting(model)
