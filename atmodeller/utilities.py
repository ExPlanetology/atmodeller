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
"""General utilities

This module is designed to have minimal dependencies on the core Atmodeller package, as its
functionality is broadly applicable across different parts of the codebase. Keeping this module
lightweight also helps avoid circular imports.
"""

import logging
from typing import Optional

import equinox as eqx
import numpy as np
from jaxmod.constants import OCEAN_MASS_H2
from jaxtyping import ArrayLike

from atmodeller.type_aliases import Scalar

logger: logging.Logger = logging.getLogger(__name__)


class ExperimentalCalibration(eqx.Module):
    r"""Experimental calibration

    Args:
        temperature_min: Minimum calibrated temperature. Defaults to ``None``.
        temperature_max: Maximum calibrated temperature. Defaults to ``None``.
        pressure_min: Minimum calibrated pressure. Defaults to ``None``.
        pressure_max: Maximum calibrated pressure. Defaults to ``None``.
        log10_fO2_min: Minimum calibrated :math:`\log_{10} f\rm{O}_2`. Defaults to ``None``.
        log10_fO2_max: Maximum calibrated :math:`\log_{10} f\rm{O}_2`. Defaults to ``None``.
    """

    temperature_min: Optional[float]
    """Minimum calibrated temperature"""
    temperature_max: Optional[float]
    """Maximum calibrated temperature"""
    pressure_min: Optional[float]
    """Minimum calibrated pressure"""
    pressure_max: Optional[float]
    """Maximum calibrated pressure"""
    log10_fO2_min: Optional[float]
    r"""Minimum calibrated :math:`\log_{10} f\rm{O}_2`"""
    log10_fO2_max: Optional[float]
    r"""Maximum calibrated :math:`\log_{10} f\rm{O}_2`"""

    def __init__(
        self,
        temperature_min: Optional[Scalar] = None,
        temperature_max: Optional[Scalar] = None,
        pressure_min: Optional[Scalar] = None,
        pressure_max: Optional[Scalar] = None,
        log10_fO2_min: Optional[Scalar] = None,
        log10_fO2_max: Optional[Scalar] = None,
    ):
        self.temperature_min = float(temperature_min) if temperature_min is not None else None
        self.temperature_max = float(temperature_max) if temperature_max is not None else None
        self.pressure_min = float(pressure_min) if pressure_min is not None else None
        self.pressure_max = float(pressure_max) if pressure_max is not None else None
        self.log10_fO2_min = float(log10_fO2_min) if log10_fO2_min is not None else None
        self.log10_fO2_max = float(log10_fO2_max) if log10_fO2_max is not None else None


def bulk_silicate_earth_abundances() -> dict[str, dict[str, float]]:
    """Bulk silicate Earth element masses in kg

    Hydrogen, carbon, and nitrogen from :cite:t:`SKG21`, sulfur from :cite:t:`H16`, and chlorine
    from :cite:t:`KHK17`

    Returns:
        A dictionary of Earth BSE element masses in kg
    """
    earth_bse: dict[str, dict[str, float]] = {
        "H": {"min": 1.852e20, "max": 1.894e21},
        "C": {"min": 1.767e20, "max": 3.072e21},
        "S": {"min": 8.416e20, "max": 1.052e21},
        "N": {"min": 3.493e18, "max": 1.052e19},
        "Cl": {"min": 7.574e19, "max": 1.431e20},
    }

    for _, values in earth_bse.items():
        values["mean"] = np.mean((values["min"], values["max"]))  # type: ignore

    return earth_bse


def earth_oceans_to_hydrogen_mass(number_of_earth_oceans: ArrayLike = 1) -> ArrayLike:
    """Converts Earth oceans to hydrogen mass.

    Args:
        number_of_earth_oceans: Number of Earth oceans. Defaults to ``1`` kg.

    Returns:
        Hydrogen mass in kg
    """
    h_kg: ArrayLike = number_of_earth_oceans * OCEAN_MASS_H2

    return h_kg
