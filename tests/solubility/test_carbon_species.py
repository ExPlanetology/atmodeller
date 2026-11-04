# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests solubility laws for carbon species"""

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

RTOL: float = 1e-8
"""Relative tolerance"""
ATOL: float = 1e-8
"""Absolute tolerance"""

LOG10_SHIFT: ArrayLike = 0
IW: RedoxBufferProtocol = IronWustiteBuffer(LOG10_SHIFT)

# Test a non-unity fugacity so the exponent is relevant for a power law solubility.
TEST_FUGACITY: ArrayLike = 2  # bar
TEST_TEMPERATURE: ArrayLike = 2000  # K
TEST_PRESSURE: ArrayLike = 500  # bar, motivated by Dixon experimental range
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))
# Several models are calibrated in the low GPa range, so use this instead for testing
TEST_PRESSURE_GPA: ArrayLike = 2 * unit_conversion.GPa_to_bar  # GPa
TEST_FO2_GPA: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE_GPA))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_PRESSURE_GPA = %e bar", TEST_PRESSURE_GPA)
logger.info("TEST_FO2 = %e bar", TEST_FO2)
logger.info("TEST_FO2_GPA = %e bar", TEST_FO2_GPA)

solubility_models: dict[str, Solubility] = get_solubility_models()


def test_CH4_basalt_ardia(check_values) -> None:
    """Tests CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["CH4_basalt_ardia13"]
    # target_concentration: ArrayLike = 0.0005831884445042942
    target_concentration: ArrayLike = 69.39
    test_fugacity_CH4_ardia_basalt: ArrayLike = 19360
    test_pressureGPa_CH4_ardia_basalt: ArrayLike = 7000
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_CH4_ardia_basalt,
        TEST_TEMPERATURE,
        test_pressureGPa_CH4_ardia_basalt,
        TEST_FO2_GPA,
    )


def test_CO_basalt_armstrong(check_values) -> None:
    """Tests volatiles in mafic melts under reduced conditions :cite:p:`AHS15`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["CO_basalt_armstrong15"]
    # target_concentration: ArrayLike = 0.027396953726422667
    target_concentration: ArrayLike = 17.3
    test_fugacity_CO_armstrong_basalt: ArrayLike = 1000
    test_pressure_CO_armstrong_basalt: ArrayLike = 1.20e4
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_CO_armstrong_basalt,
        TEST_TEMPERATURE,
        test_pressure_CO_armstrong_basalt,
        TEST_FO2_GPA,
    )


def test_CO_basalt_yoshioka(check_values) -> None:
    """Tests carbon in silicate melts :cite:p:`YNN19`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["CO_basalt_yoshioka19"]
    # target_concentration: ArrayLike = 0.1098560543306116
    target_concentration: ArrayLike = 59
    test_fugacity_CO_yoshioka_basalt: ArrayLike = 5248
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_CO_yoshioka_basalt,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_CO_rhyolite_yoshioka(check_values) -> None:
    """Tests carbon in silicate melts :cite:p:`YNN19`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["CO_rhyolite_yoshioka19"]
    # target_concentration: ArrayLike = 1.19271202468211
    target_concentration: ArrayLike = 161
    test_fugacity_CO_yoshioka_rhyolite: ArrayLike = 31623
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_CO_yoshioka_rhyolite,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_CO2_basalt_dixon(check_values) -> None:
    """Tests CO2 in MORB liquids :cite:p:`DSH95`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["CO2_basalt_dixon95"]
    # target_concentration: ArrayLike = 0.8527333099685608
    target_concentration: ArrayLike = 11
    test_fugacity_CO2_dixon_basalt: ArrayLike = 25
    test_temperature_CO2_dixon_basalt: ArrayLike = 1473.15
    test_pressure_CO2_dixon_basalt: ArrayLike = 25
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_CO2_dixon_basalt,
        test_temperature_CO2_dixon_basalt,
        test_pressure_CO2_dixon_basalt,
        TEST_FO2,
    )
