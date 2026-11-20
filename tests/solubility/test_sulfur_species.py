# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the solubility laws of sulfur"""

import inspect
import logging

import pytest

from atmodeller import console_logger
from atmodeller.solubility import get_solubility_models
from atmodeller.solubility.core import Solubility

logger: logging.Logger = console_logger()
logger.setLevel(logging.WARNING)

TEST_PRESSURE: float = 2  # bar

solubility_models: dict[str, Solubility] = get_solubility_models()


def test_S2_sulfate_andesite_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

    Andesite at 1673 K, log fO2 = -1.10 and log fS2 = -13.16 (:cite:t:`BW22{Tables 2
    and 3}`), where the measured S is 37 ppm. The model is pinned, and within 50% of the
    measurement.
    Across Table 3 the model is 0.94-1.36 times the measured S from 1623 K up, but 2.6-3 times
    at 1523 and 1573 K.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfate_andesite_boulliung23"]
    check_values.concentration(
        function_name,
        solubility_model,
        37,
        10**-13.16,
        1673,
        TEST_PRESSURE,
        10**-1.10,
        rtol=0.5,
        pinned=50.35001192113412,
    )


def test_S2_sulfide_andesite_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in andesite :cite:p:`BW23`

    Andesite at 1473 K, log fO2 = -9.4 and log fS2 = -1.05 (:cite:t:`BW23{Table 2}`),
    where the measured S is 376 ppm. The model is pinned, and 0.59 times the measurement.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfide_andesite_boulliung23"]
    check_values.concentration(
        function_name,
        solubility_model,
        376,
        10**-1.05,
        1473,
        TEST_PRESSURE,
        10**-9.4,
        rtol=0.45,
        pinned=220.59185727726214,
    )


def test_S2_sulfate_basalt_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in basalt :cite:p:`BW22,BW23corr`

    Icelandic basalt (NIB) at 1573 K, log fO2 = -1.10 and log fS2 = -14.6 (:cite:t:`BW22{Tables 2
    and 3}`), where the measured S is 450 ppm. The model is pinned, and within 30% of the
    measurement.
    Across Table 3 the model is 0.82-1.27 times the measured S.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfate_basalt_boulliung23"]
    check_values.concentration(
        function_name,
        solubility_model,
        450,
        10**-14.6,
        1573,
        TEST_PRESSURE,
        10**-1.10,
        rtol=0.3,
        pinned=454.3062304741978,
    )


def test_S2_sulfide_basalt_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in basalt :cite:p:`BW23`

    Basalt at 1473 K, log fO2 = -9.4 and log fS2 = -1.05 (:cite:t:`BW23{Table 2}`),
    where the measured S is 760 ppm. The model is pinned, and 1.14 times the measurement.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfide_basalt_boulliung23"]
    check_values.concentration(
        function_name,
        solubility_model,
        760,
        10**-1.05,
        1473,
        TEST_PRESSURE,
        10**-9.4,
        rtol=0.2,
        pinned=866.6630990827014,
    )


def test_S2_sulfate_trachybasalt_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

    T-Basalt at 1473 K, log fO2 = -1.10 and log fS2 = -16.22 (:cite:t:`BW22{Tables 2
    and 3}`), where the measured S is 1550 ppm. The model is pinned, and within 50% of the
    measurement.
    Across Table 3 the model is 1.35-2.8 times the measured S, about 2 times at most temperatures:
    the law uses the starting composition (Table 1), whereas the glasses lost Na2O and K2O.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfate_trachybasalt_boulliung23"]
    check_values.concentration(
        function_name,
        solubility_model,
        1550,
        10**-16.22,
        1473,
        TEST_PRESSURE,
        10**-1.10,
        rtol=0.5,
        pinned=2086.430552541709,
    )


def test_S2_sulfide_trachybasalt_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

    Trachybasalt at 1473 K, log fO2 = -9.4 and log fS2 = -1.05 (:cite:t:`BW23{Table 2}`),
    where the measured S is 1157 ppm. The model is pinned, and 1.03 times the measurement.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfide_trachybasalt_boulliung23"]
    check_values.concentration(
        function_name,
        solubility_model,
        1157,
        10**-1.05,
        1473,
        TEST_PRESSURE,
        10**-9.4,
        rtol=0.1,
        pinned=1190.7802861599728,
    )


@pytest.mark.parametrize(
    "melt, expected",
    [
        ("andesite", 43.462822237086414),
        ("basalt", 149.7046707623272),
        ("trachybasalt", 188.4785666630943),
    ],
)
def test_S2_boulliung(melt: str, expected: float) -> None:
    """Tests S by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`

    At 1473 K, log fO2 = -7 and log fS2 = -1.05, where sulfide and sulfate are comparable, the
    law is pinned and is the sum of the sulfide and sulfate laws.
    """

    def concentration(key: str) -> float:
        return float(
            solubility_models[key].concentration(
                10**-1.05, temperature=1473, pressure=TEST_PRESSURE, fO2=10**-7
            )
        )

    total: float = concentration(f"S2_{melt}_boulliung23")
    sulfide: float = concentration(f"S2_sulfide_{melt}_boulliung23")
    sulfate: float = concentration(f"S2_sulfate_{melt}_boulliung23")
    assert total == pytest.approx(expected, rel=1e-8)
    assert total == pytest.approx(sulfide + sulfate, rel=1e-12)
    assert 0.2 < sulfate / total < 0.8
