# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests solubility laws for other species"""

import inspect
import logging

import jax.numpy as jnp
import numpy as np
import pytest
from atmodeller.sci_utils import unit_conversion
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.solubility import get_solubility_models
from atmodeller.solubility.core import NoSolubility, Solubility
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
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


def test_no_solubility() -> None:
    """Tests that NoSolubility().concentration(...) always returns zero."""
    no_solubility: NoSolubility = NoSolubility()

    concentration = no_solubility.concentration(
        TEST_FUGACITY, temperature=TEST_TEMPERATURE, pressure=TEST_PRESSURE, fO2=TEST_FO2
    )

    assert concentration == pytest.approx(0.0)
    assert concentration == jnp.array(0.0)


def test_Cl2_ano_dio_for_thomas(check_values) -> None:
    """Tests Cl in silicate melts :cite:p:`TW21`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["Cl2_ano_dio_for_thomas21"]
    # target_concentration: ArrayLike = 1987252.8978466734
    target_concentration: ArrayLike = 1800
    test_fugacity_Cl2_ano_dio: ArrayLike = 1.66e-6
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_Cl2_ano_dio,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_Cl2_basalt_thomas(check_values) -> None:
    """Tests Cl in silicate melts :cite:p:`TW21`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["Cl2_basalt_thomas21"]
    # target_concentration: ArrayLike = 1111006.1746003036
    target_concentration: ArrayLike = 16000
    test_fugacity_Cl2_basalt: ArrayLike = 4.24e-4
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_Cl2_basalt,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_He_basalt(check_values) -> None:
    """He in tholeittic basalt melt :cite:p:`JWB86`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["He_basalt_jambon86"]
    # target_concentration: ArrayLike = 0.20013
    target_concentration: ArrayLike = 0.1
    test_fugacity_He_basalt: ArrayLike = 1
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_He_basalt,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_N2_basalt_bernadou(check_values) -> None:
    """Tests N2 in basaltic silicate melt :cite:p:`BGF21`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["N2_basalt_bernadou21"]
    # target_concentration: ArrayLike = 1.8621737654355521
    target_concentration: ArrayLike = 20.86
    test_fugacity_N2_basalt_bernadou: ArrayLike = 741
    test_totalP_N2_basalt_bernadou: ArrayLike = 800
    test_temperature_N2_basalt_bernadou: ArrayLike = 1473.15
    test_fO2_N2_basalt_bernadou: ArrayLike = 38.9045145
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_N2_basalt_bernadou,
        test_temperature_N2_basalt_bernadou,
        test_totalP_N2_basalt_bernadou,
        test_fO2_N2_basalt_bernadou,
    )


def test_N2_basalt_dasgupta(check_values) -> None:
    """Tests N2 in silicate melts :cite:p:`DFP22`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["N2_basalt_dasgupta22"]
    # target_concentration: ArrayLike = 2.280293304957063
    target_concentration: ArrayLike = 1000
    test_fugacity_N2_basalt_dasgupta: ArrayLike = 1550
    test_totalP_N2_basalt_dasgupta: ArrayLike = 1708.7
    test_temperature_N2_basalt_dasgupta: ArrayLike = 1773.15
    test_fO2_N2_basalt_dasgupta: ArrayLike = 1.8e-13
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_N2_basalt_dasgupta,
        test_temperature_N2_basalt_dasgupta,
        test_totalP_N2_basalt_dasgupta,
        test_fO2_N2_basalt_dasgupta,
    )


def test_N2_basalt_libourel(check_values) -> None:
    """Tests N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["N2_basalt_libourel03"]
    # target_concentration: ArrayLike = 0.12236470428806222
    target_concentration: ArrayLike = 446.7
    test_fugacity_N2_basalt_libourel: ArrayLike = 0.20
    test_temperature_N2_basalt_libourel: ArrayLike = 1698.15
    test_fO2_N2_basalt_libourel: ArrayLike = 6.31e-17
    test_totalP_N2_basalt_libourel: ArrayLike = 1
    # target_concentration: ArrayLike = 1000

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_N2_basalt_libourel,
        test_temperature_N2_basalt_libourel,
        test_totalP_N2_basalt_libourel,
        test_fO2_N2_basalt_libourel,
    )
