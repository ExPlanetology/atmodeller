#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Tests solubility laws for hydrogen species"""

import inspect
import logging

import numpy as np
from jax.typing import ArrayLike

from atmodeller import debug_logger
from atmodeller.interfaces import RedoxBufferProtocol, SolubilityProtocol
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

# RTOL: float = 1.0e-8
RTOL: float = 0.05
"""Relative tolerance"""
# ATOL: float = 1.0e-8
ATOL: float = 0.05
"""Absolute tolerance"""

LOG10_SHIFT: ArrayLike = 0
IW: RedoxBufferProtocol = IronWustiteBuffer(LOG10_SHIFT)

# Test a non-unity fugacity so the exponent is relevant for a power law solubility.
TEST_FUGACITY: ArrayLike = 2  # bar
TEST_TEMPERATURE: ArrayLike = 2000  # K
TEST_PRESSURE: ArrayLike = 10  # bar
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_FO2 = %e bar", TEST_FO2)

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()


def test_H2_andesite_hirschmann(check_values) -> None:
    """Tests H2 in synthetic andesite :cite:p:`HWA12`.

    Reference Parameters (fH2, H2 Conc) from Table 2 Values for Andesite, Experiment 901
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2_andesite_hirschmann12"]
    target_concentration: ArrayLike = 9000
    test_fugacity_H2_hirschmann_andesite: ArrayLike = 72269
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2_hirschmann_andesite,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2_basalt_hirschmann(check_values) -> None:
    """Tests H2 in synthetic basalt :cite:p:`HWA12`.

    Reference Parameters (fH2, H2 Conc) from Table 2 Values for Basalt, Average of Experiments A697
    and A711.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2_basalt_hirschmann12"]
    target_concentration: ArrayLike = 2200
    test_fugacity_H2_hirschmann_basalt: ArrayLike = 19058
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2_hirschmann_basalt,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2_silicic_melts_gaillard(check_values) -> None:
    """Tests Fe-H redox exchange in silicate glasses :cite:p:`GSM03`.

    Reference Parameters (fH2 and H2 Conc) from Table 4, No. 29
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2_silicic_melts_gaillard03"]
    target_concentration: ArrayLike = 2.1
    test_fugacity_H2_gaillard: ArrayLike = 8
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2_gaillard,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_ano_dio_newcombe(check_values) -> None:
    """Tests H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`.

    Reference Parameters (fH2O and H2O Conc) from Table 2, Experiment AD26
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2O_ano_dio_newcombe17"]
    target_concentration: ArrayLike = 229
    test_fugacity_H2O_newcombe_andio: ArrayLike = 0.1
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2O_newcombe_andio,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_basalt_dixon(check_values) -> None:
    """Tests H2O in MORB liquids :cite:p:`DSH95`.

    Reference Parameters (fH2O and H2O Conc) from Table 5, fH2O=25 bar
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2O_basalt_dixon95"]
    target_concentration: ArrayLike = 5200
    test_fugacity_H2O_dixon_basalt: ArrayLike = 25
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2O_dixon_basalt,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_basalt_mitchell(check_values) -> None:
    """Tests H2O in basaltic melt :cite:p:`MGO17`.

    Reference Parameters (fH2O and H2O Conc) from Figure 7, Maroon Point from 'This Study'.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2O_basalt_mitchell17"]
    target_concentration: ArrayLike = 202247
    test_fugacity_H2O_mitchell_basalt: ArrayLike = 20600.63
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2O_mitchell_basalt,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_lunar_glass_newcombe(check_values) -> None:
    """Tests H2O in lunar basalt :cite:p:`NBB17`.

    Reference Parameters (fH2O and H2O Conc) from Table 2, Average of Experiments LG2 and LG4
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2O_lunar_glass_newcombe17"]
    target_concentration: ArrayLike = 353.5
    test_fugacity_H2O_newcombe_lunarglass: ArrayLike = 0.27
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2O_newcombe_lunarglass,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_peridotite_sossi(check_values) -> None:
    """Tests H2O in peridotite liquids :cite:p:`STB23`.

    Reference Parameters (fH2O and H2O Conc) from Table 1, Sample Per-7, using epsilon_3550 of 5.1.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["H2O_peridotite_sossi23"]
    target_concentration: ArrayLike = 42.9
    test_fugacity_H2O_sossi_peridotite: ArrayLike = 0.0038
    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        test_fugacity_H2O_sossi_peridotite,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )
