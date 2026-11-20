# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tests"""

import logging

import pytest
from jaxtyping import ArrayLike
from pytest import approx

from atmodeller.interfaces import SolubilityProtocol

logger: logging.Logger = logging.getLogger("atmodeller.tests.solubility")

PINNED_RTOL: float = 1.0e-8
"""Relative tolerance of a pinned value, which detects any change of a law"""


class CheckValues:
    """Helper to check and confirm values"""

    @classmethod
    def concentration(
        cls,
        function_name: str,
        solubility_model: SolubilityProtocol,
        expected_concentration: ArrayLike,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
        *,
        rtol: float,
        pinned: float | None = None,
    ) -> None:
        """Checks the concentration (ppmw) against a reference value and a pinned value

        Args:
            function_name: Name of the test, for logging
            solubility_model: Solubility law
            expected_concentration: Reference concentration from the source, a measurement or a
                published value of the law
            fugacity: Fugacity in bar
            temperature: Temperature in K
            pressure: Pressure in bar
            fO2: Oxygen fugacity in bar
            rtol: Relative tolerance of the reference concentration, stated for each test from
                how well the law reproduces it
            pinned: Concentration the law gives, checked to :data:`PINNED_RTOL`. Defaults to
                None, when the reference concentration is itself exact.
        """
        concentration: ArrayLike = solubility_model.concentration(
            fugacity, temperature=temperature, pressure=pressure, fO2=fO2
        )
        logger.debug("%s, concentration = %s ppmw", function_name, concentration)

        if pinned is not None:
            assert concentration == approx(pinned, rel=PINNED_RTOL, abs=0)
        assert concentration == approx(expected_concentration, rel=rtol, abs=0)


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
