# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests redox buffers"""

import logging

from atmodeller.sci_utils import unit_conversion
from jaxtyping import ArrayLike
from pytest import approx

from atmodeller import console_logger
from atmodeller.thermodata._redox_buffers import (
    IronWustiteBufferHirschmann,
    IronWustiteBufferHirschmann08,
    IronWustiteBufferHirschmann21,
    RedoxBuffer,
)

logger: logging.Logger = console_logger()
logger.setLevel(logging.WARNING)

LOG10_SHIFT: ArrayLike = 0
"""Log10 shift"""
RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

TEST_LOW_TEMPERATURE: ArrayLike = 1000
TEST_LOW_PRESSURE: ArrayLike = 1  # bar
TEST_MEDIUM_TEMPERATURE: ArrayLike = 2000
TEST_MEDIUM_PRESSURE: ArrayLike = 5 * unit_conversion.GPa_to_bar
TEST_HIGH_TEMPERATURE: ArrayLike = 3000
TEST_HIGH_PRESSURE: ArrayLike = 50 * unit_conversion.GPa_to_bar

# The switch temperature is contained in the temperature_min attribute of the calibration class
# within the high temperature IW buffer. The value is 1000 K, which these constants bridge.
TEST_BELOW_SWITCH_TEMPERATURE: ArrayLike = 900
TEST_ABOVE_SWITCH_TEMPERATURE: ArrayLike = 1100


def test_IW_Hirschmann08() -> None:
    """Tests the low temperature iron-wustite buffer :cite:p:`OP93,HGD08`"""
    buffer: RedoxBuffer = IronWustiteBufferHirschmann08(LOG10_SHIFT, None)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(
        TEST_BELOW_SWITCH_TEMPERATURE, TEST_LOW_PRESSURE
    )
    logger.info("log10_fugacity (Very low T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-23.93938230619323, RTOL, ATOL)


def test_IW_Hirschmann21() -> None:
    """Tests the high temperature iron-wustite buffer :cite:p:`H21`

    The values below agree with the calculator provided in Table S7 of the Excel spreadsheet
    provided in the supplementary materials of the online publication.
    """
    buffer: RedoxBuffer = IronWustiteBufferHirschmann21(LOG10_SHIFT, None)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_LOW_TEMPERATURE, TEST_LOW_PRESSURE)
    logger.info("log10_fugacity (Low T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-20.816597461900887, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_MEDIUM_TEMPERATURE, TEST_LOW_PRESSURE)
    logger.info("log10_fugacity (Med T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-7.053620769567622, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_HIGH_TEMPERATURE, TEST_LOW_PRESSURE)
    logger.info("log10_fugacity (High T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-2.562340960147525, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_LOW_TEMPERATURE, TEST_MEDIUM_PRESSURE)
    logger.info("log10_fugacity (Low T, med P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-17.97502528402518, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(
        TEST_MEDIUM_TEMPERATURE, TEST_MEDIUM_PRESSURE
    )
    logger.info("log10_fugacity (Med T, med P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-5.619533227830953, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_HIGH_TEMPERATURE, TEST_MEDIUM_PRESSURE)
    logger.info("log10_fugacity (High T, med P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-1.5951793921085295, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_LOW_TEMPERATURE, TEST_HIGH_PRESSURE)
    logger.info("log10_fugacity (Low T, high P) = %s", log10_fugacity)

    assert log10_fugacity == approx(4.232976104931987, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_MEDIUM_TEMPERATURE, TEST_HIGH_PRESSURE)
    logger.info("log10_fugacity (Med T, high P) = %s", log10_fugacity)

    assert log10_fugacity == approx(5.534585436736885, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_HIGH_TEMPERATURE, TEST_HIGH_PRESSURE)
    logger.info("log10_fugacity (High T, high P) = %s", log10_fugacity)

    assert log10_fugacity == approx(5.887511178821558, RTOL, ATOL)


def test_IW_Hirschmann() -> None:
    """Tests the switch between the high and low temperature iron-wustite buffer."""

    low_temperature_buffer: RedoxBuffer = IronWustiteBufferHirschmann08(LOG10_SHIFT, None)
    high_temperature_buffer: RedoxBuffer = IronWustiteBufferHirschmann21(LOG10_SHIFT, None)
    composite_buffer: RedoxBuffer = IronWustiteBufferHirschmann(LOG10_SHIFT, None)

    # Evaluate the buffers either side of the switch temperature.
    log10_fugacity_low_temperature: ArrayLike = low_temperature_buffer.log10_fugacity(
        TEST_BELOW_SWITCH_TEMPERATURE, TEST_LOW_PRESSURE
    )
    log10_fugacity_composite: ArrayLike = composite_buffer.log10_fugacity(
        TEST_BELOW_SWITCH_TEMPERATURE, TEST_LOW_PRESSURE
    )

    assert log10_fugacity_composite == approx(float(log10_fugacity_low_temperature), RTOL, ATOL)

    log10_fugacity_high_temperature: ArrayLike = high_temperature_buffer.log10_fugacity(
        TEST_ABOVE_SWITCH_TEMPERATURE, TEST_LOW_PRESSURE
    )
    log10_fugacity_composite: ArrayLike = composite_buffer.log10_fugacity(
        TEST_ABOVE_SWITCH_TEMPERATURE, TEST_LOW_PRESSURE
    )

    assert log10_fugacity_composite == approx(float(log10_fugacity_high_temperature), RTOL, ATOL)


def test_IW_Hirschmann08_fixed_P() -> None:
    """Tests the low temperature iron-wustite buffer with constant pressure

    When the user sets an evaluation pressure the total pressure should not enter the calculation
    of the fugacity.
    """
    # Define an evaluation pressure at 1 bar
    buffer: RedoxBuffer = IronWustiteBufferHirschmann08(LOG10_SHIFT, 1)

    log10_fugacity_low_pressure: ArrayLike = buffer.log10_fugacity(
        TEST_BELOW_SWITCH_TEMPERATURE, TEST_LOW_PRESSURE
    )
    logger.info("log10_fugacity_low_pressure = %s", log10_fugacity_low_pressure)

    log10_fugacity_high_pressure: ArrayLike = buffer.log10_fugacity(
        TEST_BELOW_SWITCH_TEMPERATURE, TEST_HIGH_PRESSURE
    )
    logger.info("log10_fugacity_high_pressure = %s", log10_fugacity_high_pressure)

    # The evaluation should be at 1 bar pressure and the input pressure ignored
    assert log10_fugacity_low_pressure == approx(-23.93938230619323, RTOL, ATOL)
    assert log10_fugacity_high_pressure == approx(-23.93938230619323, RTOL, ATOL)
