# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`HWZ58`"""

import pytest

from atmodeller.eos import RealGas
from atmodeller.sci_utils import unit_conversion

# Probably due to rounding of the model parameters in the paper, some compressibilities in the
# table in the paper don't quite match exactly with what we compute. Hence relax the tolerance.
RTOL: float = 1.0e-4
"""Relative tolerance"""
ATOL: float = 1.0e-4
"""Absolute tolerance"""

MODEL_SUFFIX: str = "beattie_holley58"
"""Suffix of the :cite:t:`HWZ58` models"""

# species, temperature (K), pressure (atm), expected compressibility factor
# All species share the same Beattie-Bridgeman formulation (HolleyModel), so this is a single code
# path exercised with each species' own reference value from the cited table (see the id, which
# names the table), not distinct branches of the implementation.
COMPRESSIBILITY_FACTOR_CASES = [
    pytest.param("H2", 300, 100, 1.06217, id="H2-low-TableII"),
    pytest.param("H2", 1000, 1000, 1.26294, id="H2-high-TableII"),
    pytest.param("N2", 300, 100, 1.00464, id="N2-low-TableIII"),
    pytest.param("N2", 1000, 1000, 1.36551, id="N2-high-TableIII"),
    pytest.param("O2", 300, 100, 0.95454, id="O2-low-TableIV"),
    pytest.param("O2", 1000, 1000, 1.28897, id="O2-high-TableIV"),
    pytest.param("CO2", 400, 100, 0.81853, id="CO2-low-TableV"),
    pytest.param("CO2", 1000, 1000, 1.07058, id="CO2-high-TableV"),
    pytest.param("NH3", 400, 100, 0.56165, id="NH3-low-TableVI"),
    pytest.param("NH3", 1000, 500, 0.93714, id="NH3-high-TableVI"),
    pytest.param("CH4", 300, 100, 0.85583, id="CH4-low-TableVII"),
    pytest.param("CH4", 1000, 1000, 1.36201, id="CH4-high-TableVII"),
    pytest.param("He", 300, 100, 1.05148, id="He-low-TableVIII"),
    pytest.param("He", 1000, 1000, 1.14766, id="He-high-TableVIII"),
]


@pytest.mark.parametrize("species, temperature, pressure_atm, expected", COMPRESSIBILITY_FACTOR_CASES)
def test_compressibility_factor(
    check_values, species: str, temperature: float, pressure_atm: float, expected: float
) -> None:
    """Tests compressibility factor against the reference value for each species/table entry"""
    model: RealGas = check_values.get_eos_model(species, MODEL_SUFFIX)
    pressure: float = pressure_atm * unit_conversion.atmosphere_to_bar
    check_values.compressibility_factor(temperature, pressure, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_high_fugacity(check_values) -> None:
    """Tests that a fugacity can be calculated"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.fugacity(1000, pressure, model, 1301.672235770893, rtol=RTOL, atol=ATOL)


def test_broadcasting(check_values) -> None:
    """Tests methods with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.check_broadcasting(model)
