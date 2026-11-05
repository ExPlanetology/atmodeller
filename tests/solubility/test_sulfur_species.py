# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests solubility laws for other species"""

import inspect
import logging

import numpy as np
from atmodeller.sci_utils import unit_conversion
from jaxtyping import ArrayLike

from atmodeller import console_logger
from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.solubility import get_solubility_models
from atmodeller.solubility.core import Solubility
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = console_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

LOG10_SHIFT: ArrayLike = 0
IW: RedoxBufferProtocol = IronWustiteBuffer(LOG10_SHIFT)

# Test a non-unity fugacity so the exponent is relevant for a power law solubility.
TEST_FUGACITY: ArrayLike = 2
TEST_TEMPERATURE: ArrayLike = 2000
TEST_PRESSURE: ArrayLike = 2  # bar
# Several models are calibrated in the low GPa range, so use this instead
TEST_PRESSURE_GPA: ArrayLike = 2 * unit_conversion.GPa_to_bar  # GPa
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))
TEST_FO2_GPA: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE_GPA))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_PRESSURE_GPA = %e bar", TEST_PRESSURE_GPA)
logger.info("TEST_FO2 = %e bar", TEST_FO2)
logger.info("TEST_FO2_GPA = %e bar", TEST_FO2_GPA)

solubility_models: dict[str, Solubility] = get_solubility_models()


def test_S2_sulfate_andesite_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfate_andesite_boulliung23"]
    # target_concentration: ArrayLike = 0.0002601378182149385
    target_concentration: ArrayLike = 95
    test_fugacity_S2_sulfate_boulliung_andesite: ArrayLike = 4.16869e-16
    test_fo2_sulfate_boulliung_andesite: ArrayLike = 0.079432823
    test_temperature_sulfate_boulliung_andesite: ArrayLike = 1523
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_S2_sulfate_boulliung_andesite,
        test_temperature_sulfate_boulliung_andesite,
        TEST_PRESSURE,
        test_fo2_sulfate_boulliung_andesite,
    )


def test_S2_sulfide_andesite_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in andesite :cite:p:`BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfide_andesite_boulliung23"]
    # target_concentration: ArrayLike = 2765.534194086474
    target_concentration: ArrayLike = 376
    test_fugacity_S2_sulfide_boulliung_andesite: ArrayLike = 0.089125094
    test_fo2_sulfide_boulliung_andesite: ArrayLike = 3.98107e-10
    test_temperature_sulfide_boulliung_andesite: ArrayLike = 1473
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_S2_sulfide_boulliung_andesite,
        test_temperature_sulfide_boulliung_andesite,
        TEST_PRESSURE,
        test_fo2_sulfide_boulliung_andesite,
    )


def test_S2_andesite_boulliung(check_values) -> None:
    """Tests S in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_andesite_boulliung23"]
    target_concentration: ArrayLike = 2765.5344542242924

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfate_basalt_boulliung(check_values) -> None:
    """Tests S in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfate_basalt_boulliung23"]
    # target_concentration: ArrayLike = 0.00020528360011091678
    target_concentration: ArrayLike = 210
    test_fugacity_S2_sulfate_boulliung_basalt: ArrayLike = 1.38038e-14
    test_fo2_sulfate_boulliung_basalt: ArrayLike = 0.079432823
    test_temperature_sulfate_boulliung_basalt: ArrayLike = 1623
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_S2_sulfate_boulliung_basalt,
        test_temperature_sulfate_boulliung_basalt,
        TEST_PRESSURE,
        test_fo2_sulfate_boulliung_basalt,
    )


def test_S2_sulfide_basalt_boulliung(check_values) -> None:
    """Tests S in basalt as sulfide (S^2-) :cite:p:`BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfide_basalt_boulliung23"]
    # target_concentration: ArrayLike = 7576.212242182479
    target_concentration: ArrayLike = 760
    test_fugacity_S2_sulfide_boulliung_basalt: ArrayLike = 0.089125094
    test_fo2_sulfide_boulliung_basalt: ArrayLike = 3.98107e-10
    test_temperature_sulfide_boulliung_basalt: ArrayLike = 1473
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_S2_sulfide_boulliung_basalt,
        test_temperature_sulfide_boulliung_basalt,
        TEST_PRESSURE,
        test_fo2_sulfide_boulliung_basalt,
    )


def test_S2_basalt_boulliung(check_values) -> None:
    """Tests S in basalt due to sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_basalt_boulliung23"]
    target_concentration: ArrayLike = 7576.212447466079

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfate_trachybasalt_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfate_trachybasalt_boulliung23"]
    # target_concentration: ArrayLike = 0.0007002721319248917
    target_concentration: ArrayLike = 230
    test_fugacity_S2_sulfate_boulliung_trachybasalt: ArrayLike = 2.51189e-15
    test_fo2_sulfate_boulliung_trachybasalt: ArrayLike = 0.079432823
    test_temperature_sulfate_boulliung_trachybasalt: ArrayLike = 1573
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_S2_sulfate_boulliung_trachybasalt,
        test_temperature_sulfate_boulliung_trachybasalt,
        TEST_PRESSURE,
        test_fo2_sulfate_boulliung_trachybasalt,
    )


def test_S2_sulfide_trachybasalt_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in trachybasalt :cite:p:`BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_sulfide_trachybasalt_boulliung23"]
    # target_concentration: ArrayLike = 9573.602305681254
    target_concentration: ArrayLike = 1157
    test_fugacity_S2_sulfide_boulliung_trachybasalt: ArrayLike = 0.089125094
    test_fo2_sulfide_boulliung_trachybasalt: ArrayLike = 3.98107e-10
    test_temperature_sulfide_boulliung_trachybasalt: ArrayLike = 1473
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_S2_sulfide_boulliung_trachybasalt,
        test_temperature_sulfide_boulliung_trachybasalt,
        TEST_PRESSURE,
        test_fo2_sulfide_boulliung_trachybasalt,
    )


def test_S2_trachybasalt_boulliung(check_values) -> None:
    """Tests S in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["S2_trachybasalt_boulliung23"]
    target_concentration: ArrayLike = 9573.603005953386

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )
