# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone helper functions and constants for scientific modeling

This module provides physical, chemical, and mathematical constants, unit conversions, and utility
functions for scientific calculations, particularly in planetary and atmospheric modeling. It is
designed to be standalone and does not depend on other modules within the atmodeller package.
"""

from typing import Optional, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike
from molmass import Formula
from scipy import constants
from scipy.constants import atmosphere, bar, kilo, mega

# Module should be standalone, so avoid importing from the jax_utils module
Scalar: TypeAlias = int | float
"""Scalar"""


class UnitConversion(eqx.Module):
    """Unit conversions"""

    # Pressure
    atmosphere_to_bar: float = atmosphere / bar
    bar_to_Pa: float = 1.0e5
    bar_to_MPa: float = 1.0e-1
    bar_to_GPa: float = 1.0e-4
    Pa_to_bar: float = 1.0e-5
    MPa_to_bar: float = 1.0e1
    GPa_to_bar: float = 1.0e4

    # Concentration / fraction
    fraction_to_ppm: float = mega
    ppm_to_fraction: float = 1 / mega
    ppm_to_percent: float = 100 / mega
    percent_to_ppm: float = 1.0e4

    # Mass / volume
    g_to_kg: float = 1 / kilo
    cm3_to_m3: float = 1.0e-6
    m3_to_cm3: float = 1.0e6
    litre_to_m3: float = 1.0e-3

    # Energy / work
    m3_bar_to_J: float = 1.0e5
    J_to_m3_bar: float = 1.0e-5


# Single instance for convenient access
unit_conversion: UnitConversion = UnitConversion()


class PlanetParameters(eqx.Module):
    """Planet parameters"""

    mass: float
    """Mass of the planet (kg)"""
    radius: float
    """Radius of the planet (m)"""
    core_mass_fraction: float
    """Core mass fraction (dimensionless)"""

    @property
    def bulk_density(self) -> float:
        r"""Calculates bulk density from mass and radius

        .. math::
                \rho = \frac{M}{\frac{4}{3}\pi R^3}

        where :math:`M` is the mass of the planet and :math:`R` is the radius of the planet.

        Returns:
            Bulk density of the planet (kgm\ :sup:`-3`)
        """
        return self.mass / self.volume

    @property
    def core_mass(self) -> float:
        """Calculates core mass from mass and core mass fraction

        .. math::
                M_{core} = f_{core} M

        where :math:`f_{core}` is the core mass fraction and :math:`M` is the mass of the planet.

        Returns:
            Mass of the core (kg)
        """
        return self.core_mass_fraction * self.mass

    @property
    def mantle_mass(self) -> float:
        """Calculates mantle mass from mass and core mass fraction

        .. math::
                M_{mantle} = (1 - f_{core}) M

        where :math:`f_{core}` is the core mass fraction and :math:`M` is the mass of the planet.

        Returns:
            Mass of the mantle (kg)
        """
        return (1 - self.core_mass_fraction) * self.mass

    @property
    def surface_gravity(self) -> float:
        """Calculates surface gravity from mass and radius

        .. math::
                g = \\frac{GM}{R^2}

        where :math:`G` is the gravitational constant, :math:`M` is the mass of the planet, and
        :math:`R` is the radius of the planet.

        Returns:
            Surface gravity of the planet (ms\\ :sup:`-2`)
        """
        return constants.gravitational_constant * self.mass / self.radius**2

    @property
    def volume(self) -> float:
        """Calculates volume from radius

        .. math::
                V = \\frac{4}{3}\\pi R^3

        where :math:`R` is the radius of the planet.

        Returns:
            Volume of the planet (m\\ :sup:`3`)
        """
        return (4 / 3) * np.pi * self.radius**3


class EarthParameters(PlanetParameters):
    """Earth parameters"""

    ocean_moles: float
    """Moles of H\\ :sub:`2` or H\\ :sub:`2`\\ O in present-day Earth's ocean (mol)"""
    ocean_mass_H2: float
    """Mass of H\\ :sub:`2` in present-day Earth's ocean (kg)"""
    ocean_mass_H2O: float
    """Mass of H\\ :sub:`2`\\ O in present-day Earth's ocean (kg)"""

    def __init__(self):
        self.mass = 5.972e24
        self.radius = 6371000.0
        self.core_mass_fraction = 0.295334691460966
        self.ocean_moles = 7.68894973907177e22
        self.ocean_mass_H2 = self.ocean_moles * Formula("H2").mass * unit_conversion.g_to_kg
        self.ocean_mass_H2O = self.ocean_moles * Formula("H2O").mass * unit_conversion.g_to_kg

    def oceans_to_hydrogen_mass(self, number_of_earth_oceans: ArrayLike = 1) -> ArrayLike:
        """Converts number of Earth oceans to hydrogen mass.

        Args:
            number_of_earth_oceans: Number of Earth oceans. Defaults to ``1``.

        Returns:
            Hydrogen mass in kg corresponding to the given number of Earth oceans
        """
        return number_of_earth_oceans * self.ocean_mass_H2


earth: EarthParameters = EarthParameters()
"""Earth parameters"""
trappist1e: PlanetParameters = PlanetParameters(
    mass=0.6919722531398028 * earth.mass,
    radius=0.9199497724062157 * earth.radius,
    core_mass_fraction=earth.core_mass_fraction,
)
"""Parameters for TRAPPIST-1e based on :cite:t:`ADG21` and used by :cite:t:`BTH25`"""

GAS_CONSTANT: float = constants.gas_constant
r"""Gas constant (:math:`\mathrm{J}\mathrm{K}^{-1}\mathrm{mol}^{-1}`)"""

GAS_CONSTANT_BAR: float = GAS_CONSTANT * unit_conversion.Pa_to_bar
r"""Gas constant (:math:`\mathrm{m}^3\mathrm{bar}^{-1}\mathrm{K}^{-1}\mathrm{mol}^{-1}`)"""

SIO2_MOLAR_MASS: float = Formula("SiO2").mass * unit_conversion.g_to_kg
"""Molar mass of SiO\\ :sub:`2` (kg mol\\ :sup:`-1`)"""

FE_MOLAR_MASS: float = Formula("Fe").mass * unit_conversion.g_to_kg
"""Molar mass of Fe (kg mol\\ :sup:`-1`), used as the default background molar mass for a metal 
phase"""


class ExperimentalCalibration(eqx.Module):
    r"""Experimental calibration

    Args:
        temperature_min: Minimum calibrated temperature (K). Defaults to ``None``.
        temperature_max: Maximum calibrated temperature (K). Defaults to ``None``.
        pressure_min: Minimum calibrated pressure (bar). Defaults to ``None``.
        pressure_max: Maximum calibrated pressure (bar). Defaults to ``None``.
        log10_fO2_min: Minimum calibrated :math:`\log_{10}f\rm{O}_2` (bar). Defaults to ``None``.
        log10_fO2_max: Maximum calibrated :math:`\log_{10}f\rm{O}_2` (bar). Defaults to ``None``.
    """

    temperature_min: Optional[float]
    """Minimum calibrated temperature (K)"""
    temperature_max: Optional[float]
    """Maximum calibrated temperature (K)"""
    pressure_min: Optional[float]
    """Minimum calibrated pressure (bar)"""
    pressure_max: Optional[float]
    """Maximum calibrated pressure (bar)"""
    log10_fO2_min: Optional[float]
    r"""Minimum calibrated :math:`\log_{10} f\rm{O}_2` (bar)"""
    log10_fO2_max: Optional[float]
    r"""Maximum calibrated :math:`\log_{10} f\rm{O}_2` (bar)"""

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
    from :cite:t:`KHK17`.

    Returns:
        Dictionary mapping element symbols to dicts with 'min', 'max', and 'mean' masses (kg)
    """
    earth_bse: dict[str, dict[str, float]] = {
        "H": {"min": 1.852e20, "max": 1.894e21},
        "C": {"min": 1.767e20, "max": 3.072e21},
        "S": {"min": 8.416e20, "max": 1.052e21},
        "N": {"min": 3.493e18, "max": 1.052e19},
        "Cl": {"min": 7.574e19, "max": 1.431e20},
    }

    for _, values in earth_bse.items():
        values["mean"] = np.mean((values["min"], values["max"]))  # pyright: ignore

    return earth_bse
