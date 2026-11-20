# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests solubility laws for other species"""

import inspect
import logging

import jax.numpy as jnp
import numpy as np
import pytest
from atmodeller.sci_utils import unit_conversion
from jaxtyping import ArrayLike

from atmodeller import console_logger
from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.solubility import get_solubility_models
from atmodeller.solubility.core import NoSolubility, Solubility
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
        rtol=0.01,
        pinned=1810.4735530794146,
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
        rtol=0.02,
        pinned=16176.494077518775,
    )


@pytest.mark.parametrize(
    "species, molar_mass, pinned, partial_pressure, measured, rtol",
    [
        ("He", 4.0026, 0.100065, 95300e-6, 51.0, 0.2),
        ("Ne", 20.1797, 0.22521986607142858, 9830e-6, 22.55, 0.12),
        ("Ar", 39.948, 0.10522017857142857, 907000e-6, 5.6, 0.15),
        ("Kr", 83.798, 0.11222946428571429, 1000e-6, 3.5, 0.23),
        ("Xe", 131.293, 0.09964200892857143, 547e-6, 1.75, 0.23),
    ],
)
def test_noble_gas_basalt_jambon(
    species: str,
    molar_mass: float,
    pinned: float,
    partial_pressure: float,
    measured: float,
    rtol: float,
) -> None:
    """Tests the noble gases in tholeiitic basalt melt :cite:p:`JWB86`

    The concentration at 1 bar is pinned, and is proportional to the fugacity (Henry's law). It
    is compared with the measured solubility (1e-5 cm3 STP g-1 bar-1), the mean of the two runs at
    1400 C with their noble gas mixtures ARM and ARM (1:9) (:cite:t:`JWB86{Table 5}`), at the
    partial pressure in ARM (:cite:t:`JWB86{Table 2}`). The law is 0.86-1.11 times the
    measurement, within the 1-sigma scatter about their regression (12-23%).
    """
    model: Solubility = solubility_models[f"{species}_basalt_jambon86"]

    def concentration(fugacity: float) -> float:
        return float(
            model.concentration(
                fugacity, temperature=TEST_TEMPERATURE, pressure=TEST_PRESSURE, fO2=TEST_FO2
            )
        )

    assert concentration(1) == pytest.approx(pinned, rel=1e-8)
    assert concentration(partial_pressure) == pytest.approx(partial_pressure * pinned, rel=1e-12)
    # cm3 STP to mol (22414 cm3 mol-1), then mass fraction to ppmw
    measured_ppmw: float = (
        measured * 1e-5 * partial_pressure / 22414 * molar_mass * unit_conversion.fraction_to_ppm
    )
    assert concentration(partial_pressure) == pytest.approx(measured_ppmw, rel=rtol)


def test_N2_basalt_bernadou(check_values) -> None:
    """Tests N2 in basaltic silicate melt :cite:p:`BGF21` at their oxidised experiment AR1FeN

    1200 C, 800 bar, fN2 = 741 bar and IW+1.59 (:cite:t:`BGF21{Tables 3-4}`; fO2 from the IW
    buffer of Atmodeller), where N2 dominates. The model is pinned, and within 30% of the measured
    20.86 ppm, the threshold :cite:t:`BGF21` used for the uncertainty of their fit. The N3- term at
    reduced conditions does not reproduce their data (see the docstring of the law).
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: Solubility = solubility_models["N2_basalt_bernadou21"]
    fugacity: ArrayLike = 741.0
    temperature: ArrayLike = 1473.15
    pressure: ArrayLike = 800.0
    fO2: ArrayLike = 10**-10.365073101913051
    check_values.concentration(
        function_name,
        solubility_model,
        20.86,
        fugacity,
        temperature,
        pressure,
        fO2,
        rtol=0.3,
        pinned=25.578390596094554,
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
        rtol=0.01,
        pinned=1002.1074478063852,
    )


@pytest.mark.parametrize(
    "fugacity, log10_fO2, pinned, measured, rtol",
    [
        (0.20, -16.2, 377.1406984833259, 446.675, 0.2),
        (0.78, -0.7, 0.04765800176612579, 0.0486, 0.05),
    ],
)
def test_N2_basalt_libourel(
    check_values, fugacity: float, log10_fO2: float, pinned: float, measured: float, rtol: float
) -> None:
    """Tests N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`

    Two runs at 1425 C (:cite:t:`LMH03{Tables 1 and 2}`), with the measured N the mean of their
    replicate analyses: CM-1#8 at IW-6.8 (pN2 = 0.20 atm, 447 ppm), where N3- dominates, and
    CM-1#17 in air (pN2 = 0.78 atm, 0.049 ppm), where N2 dominates. The model is pinned, and
    0.84 and 0.98 times the measurements.
    """

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    check_values.concentration(
        function_name,
        solubility_models["N2_basalt_libourel03"],
        measured,
        fugacity,
        1698.15,
        1,
        10**log10_fO2,
        rtol=rtol,
        pinned=pinned,
    )
