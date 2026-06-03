# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for tests"""

import logging
from collections.abc import Callable

import numpy as np
import numpy.testing as nptest
import pytest
from jaxtyping import Array, ArrayLike

from atmodeller import debug_logger
from atmodeller.eos import RealGas, get_eos_models
from atmodeller.eos.core import RealGasBase
from atmodeller.jax_utils import as_j64

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

# Tolerances to compare the test results with target output.
RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""


class CheckValues:
    """Helper class with methods to check and confirm values"""

    def __init__(self) -> None:
        self._eos_models: dict[str, RealGasBase] = get_eos_models()

    @classmethod
    def _check_property(
        cls,
        property_name: str,
        temperature: ArrayLike,
        pressure: ArrayLike,
        eos: RealGas,
        expected: ArrayLike,
        *,
        rtol=RTOL,
        atol=ATOL,
    ) -> None:
        """Generalized method to check a property (e.g., compressibility, fugacity, etc.)

        Args:
            property_name: Name of the property to check
            temperature: Temperature in K
            pressure: Pressure in bar
            eos: EOS model
            expected: The expected value
            rtol: Relative tolerance. Defaults to RTOL.
            atol: Absolute tolerance. Defaults to ATOL.
        """
        # Dynamically get the method from the eos model based on property_name
        method: Callable = getattr(eos, property_name)
        # Call the method with the provided temperature and pressure avoiding recompilation
        temperature = as_j64(temperature)
        pressure = as_j64(pressure)
        result: ArrayLike = method(temperature, pressure)

        # Compare the result with the expected value
        nptest.assert_allclose(result, expected, rtol, atol)

    @classmethod
    def compressibility_factor(cls, *args, **kwargs) -> None:
        """Checks the compressibility factor."""
        cls._check_property("compressibility_factor", *args, **kwargs)

    @classmethod
    def fugacity(cls, *args, **kwargs) -> None:
        """Checks the fugacity."""
        cls._check_property("fugacity", *args, **kwargs)

    @classmethod
    def fugacity_coefficient(cls, *args, **kwargs) -> None:
        """Checks the fugacity coefficient."""
        cls._check_property("fugacity_coefficient", *args, **kwargs)

    @classmethod
    def volume(cls, *args, **kwargs) -> None:
        """Checks the volume."""
        cls._check_property("volume", *args, **kwargs)

    @classmethod
    def volume_integral(cls, *args, **kwargs) -> None:
        """Checks the volume integral."""
        cls._check_property("volume_integral", *args, **kwargs)

    @classmethod
    def check_broadcasting_for_property(cls, property_name: str, eos: RealGas) -> None:
        """Checks that the EOS model handles broadcasting correctly.

        Args:
            property_name: Name of the property to check
            eos: EOS model
        """
        # Dynamically get the method from the eos model based on property_name
        method: Callable = getattr(eos, property_name)

        # Since the shapes of the arrays are always changing here there's no point in converting to
        # jax arrays in order to avoid recompilation because recompilation will occur anyway due to
        # the changing array shapes.

        # Tests pressure broadcasting
        temperature = 2000
        pressure = np.array([1, 10, 100])
        result: Array = method(temperature, pressure)
        logger.debug(f"Result of {property_name} with pressure broadcasting: {result}")
        assert result.shape == (3,)

        # Tests temperature broadcasting
        temperature = np.array([1500, 2000])
        pressure = 100
        result = method(temperature, pressure)
        logger.debug(f"Result of {property_name} with temperature broadcasting: {result}")
        assert result.shape == (2,)

        # Tests both temperature and pressure broadcasting with equal length arrays
        temperature = np.array([1500, 2000])
        pressure = np.array([0.5, 100])
        results = method(temperature, pressure)
        logger.debug(
            f"Result of {property_name} with both temperature and pressure broadcasting: {results}"
        )
        assert results.shape == (2,)

        # Tests both temperature and pressure broadcasting
        temperature = np.array([1500, 2000])[:, None]
        pressure = np.array([1, 10, 100])[None, :]
        result = method(temperature, pressure)
        logger.debug(
            f"Result of {property_name} with temperature and pressure broadcasting to form a grid: {result}"
        )
        assert result.shape == (2, 3)

        # Tests both temperature and pressure broadcasting with switched axes
        temperature = np.array([1500, 2000])[None, :]
        pressure = np.array([1, 10, 100])[:, None]
        result = method(temperature, pressure)
        logger.debug(
            f"Result of {property_name} with temperature and pressure broadcasting to form a grid with switched axes: {result}"
        )
        assert result.shape == (3, 2)

    def check_broadcasting(self, eos: RealGas) -> None:
        """Checks that the EOS model handles broadcasting correctly for all properties.

        Args:
            eos: EOS model
        """
        self.check_broadcasting_for_property("compressibility_factor", eos)
        self.check_broadcasting_for_property("log_fugacity", eos)
        self.check_broadcasting_for_property("log_fugacity_coefficient", eos)
        self.check_broadcasting_for_property("volume", eos)
        self.check_broadcasting_for_property("volume_integral", eos)

        assert True

    def get_eos_model(self, species_name: str, suffix: str) -> RealGasBase:
        """Gets a model for a species.

        Args:
            species_name: Species name
            suffix: Model suffix

        Returns:
            EOS model
        """
        return self._eos_models[f"{species_name}_{suffix}"]


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
