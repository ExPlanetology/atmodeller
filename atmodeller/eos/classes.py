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
"""Concrete classes for real gas equations of state"""

import importlib.resources
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import optimistix as optx
import pandas as pd
from jax import Array, jit
from jax.scipy.interpolate import RegularGridInterpolator
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller import PRESSURE_REFERENCE
from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos import DATA_DIRECTORY
from atmodeller.eos.core import RealGas
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@register_pytree_node_class
class BeattieBridgeman(RealGas):
    r"""Beattie-Bridgeman equation :cite:p:`HWZ58{Equation 1}`.

    .. math::

        PV^2 = RT\left(1-\frac{c}{VT^3}\right)\left(V+B_0-\frac{bB_0}{V}\right)
        - A_0\left(1-\frac{a}{V}\right)

    Args:
        A0: A0 empirical constant determined experimentally
        a: a empirical constant determined experimentally
        B0: B0 empirical constant determined experimentally
        b: b empirical constant determined experimentally
        c: c empirical constant determined experimentally
        calibration: Experimental calibration. Defaults to empty.

    Attributes:
        A0: A0 empirical constant determined experimentally
        a: a empirical constant determined experimentally
        B0: B0 empirical constant determined experimentally
        b: b empirical constant determined experimentally
        c: c empirical constant determined experimentally
        calibration: Experimental calibration. Defaults to empty.
    """

    def __init__(
        self,
        A0: float,
        a: float,
        B0: float,
        b: float,
        c: float,
    ):
        self._A0: float = A0
        self._a: float = a
        self._B0: float = B0
        self._b: float = b
        self._c: float = c

    @jit
    def _objective_function(
        self, compressibility_factor: ArrayLike, kwargs: dict[str, ArrayLike]
    ) -> Array:
        r"""Objective function to solve for the compressibility factor :cite:p:`HWZ58{Equation 2}`

        .. math::

            PV^4 - RTV^3 - \left(RTB_0 - \frac{Rc}{T^2}-A_0\right)V^2
            +\left(RTbB_0+\frac{RcB_0}{T^2}-aA_0\right)V - \frac{RcbB_0}{T^2}=0

        Args:
            compressibility_factor: Compressibility factor
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]
        volume: Array = compressibility_factor * self.ideal_volume(temperature, pressure)

        coeff0: Array = (
            1 / jnp.square(temperature) * -GAS_CONSTANT_BAR * self._c * self._b * self._B0
        )
        coeff1: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self._c * self._B0
            + GAS_CONSTANT_BAR * temperature * self._b * self._B0
            - self._a * self._A0
        )
        coeff2: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self._c
            - GAS_CONSTANT_BAR * temperature * self._B0
            + self._A0
        )
        coeff3: ArrayLike = -GAS_CONSTANT_BAR * temperature

        residual: Array = (
            coeff0
            + coeff1 * volume
            + coeff2 * jnp.power(volume, 2)
            + coeff3 * jnp.power(volume, 3)
            + pressure * jnp.power(volume, 4)
        )

        return residual

    @override
    @jit
    def log_fugacity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> Array:
        """Log fugacity :cite:p:`HWZ58{Equation 11}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        log_fugacity: Array = (
            jnp.log(GAS_CONSTANT_BAR * temperature / volume)
            + (
                self._B0
                - self._c / jnp.power(temperature, 3)
                - self._A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 2
            / volume
            - (
                self._b * self._B0
                + self._c * self._B0 / jnp.power(temperature, 3)
                - self._a * self._A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 3
            / (2 * jnp.square(volume))
            + (self._c * self._b * self._B0 / jnp.power(temperature, 3))
            * 4
            / (3 * jnp.power(volume, 3))
        )

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Volume

        :cite:t:`HWZ58` doesn't say which root to take, but one real root is very small and the
        maximum real root gives a volume that agrees with the tabulated compressibility factor
        for all species.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        # Based on the tabulated data, most compressibility factors are around unity
        initial_compressibility_factor: float = 1.0
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver = optx.Newton(rtol=1.0e-8, atol=1.0e-8)
        sol = optx.root_find(
            self._objective_function,
            solver,
            initial_compressibility_factor,
            args=kwargs,
        )
        volume: ArrayLike = sol.value * self.ideal_volume(temperature, pressure)
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        return GAS_CONSTANT_BAR * temperature * self.log_fugacity(temperature, pressure)

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "A0": self._A0,
            "a": self._a,
            "B0": self._B0,
            "b": self._b,
            "c": self._c,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class Chabrier(RealGas):
    r"""Chabrier EOS from :cite:t:`CD21`

    This uses rho-T-P tables to lookup density (rho).

    Args:
        log10_density_func: Spline lookup for density from :cite:t:`CD21` T-P-rho tables
        He_fraction: He fraction
        H2_molar_mass_g_mol: Molar mass of H2
        He_molar_mass_g_mol: Molar mass of He
        integration_steps: Number of integration steps. Defaults to 1000.
    """

    CHABRIER_DIRECTORY: Path = Path("chabrier")
    """Directory of the Chabrier data within :obj:`~atmodeller.eos.data`"""
    He_fraction_map: ImmutableMap = ImmutableMap(
        {
            "TABLE_H_TP_v1": 0,
            "TABLE_HE_TP_v1": 1,
            "TABLEEOS_2021_TP_Y0275_v1": 0.275,
            "TABLEEOS_2021_TP_Y0292_v1": 0.292,
            "TABLEEOS_2021_TP_Y0297_v1": 0.297,
        }
    )
    """Mole fraction of He in the gas mixture, the other component being H2.
    
    Dictionary keys should correspond to the name of the Chabrier file.
    """

    def __init__(
        self,
        log10_density_func: RegularGridInterpolator,
        He_fraction: float,
        H2_molar_mass_g_mol: float,
        He_molar_mass_g_mol: float,
        integration_steps: int = 1000,
    ):
        self._log10_density_func: RegularGridInterpolator = log10_density_func
        self._He_fraction: float = He_fraction
        self._H2_molar_mass_g_mol: float = H2_molar_mass_g_mol
        self._He_molar_mass_g_mol: float = He_molar_mass_g_mol
        self._integration_steps: int = integration_steps

<<<<<<< HEAD
    @classmethod
    def create(cls, filename: Path, integration_steps: int = 1000) -> Self:
        """Creates a Chabrier instance

        Args:
            filename: Filename of the density-T-P data
            integration_steps: Number of integration steps. Defaults to 1000.

        Returns:
            Instance
        """
        log10_density_func: RegularGridInterpolator = cls._get_interpolator(filename)
        He_fraction: float = cls.He_fraction_map[filename.name]
        H2_molar_mass_g_mol: float = Formula("H2").mass
        He_molar_mass_g_mol: float = Formula("He").mass

        return cls(
            log10_density_func,
            He_fraction,
            H2_molar_mass_g_mol,
            He_molar_mass_g_mol,
            integration_steps,
        )

    @jit
    def _convert_to_molar_density(self, log10_density_gcc: ArrayLike) -> Array:
        r"""Converts density to molar density

        Convert units: g/cm3 to mol/cm3 to mol/m3 for H2 (1e6 cm3 = 1 m3; 1 mol H2 = 2.016 g H2)

        Args:
            log10_density_gcc: Log10 density in g/cc

        Returns:
            Molar density in :math:`\mathrm{mol}\mathrm{m}^{-3}`
        """
        molar_density: Array = jnp.power(10, log10_density_gcc) / unit_conversion.cm3_to_m3
        composition_factor: float = (
            self._He_molar_mass_g_mol * self._He_fraction
            + self._H2_molar_mass_g_mol * (1 - self._He_fraction)
        )
        molar_density = molar_density / composition_factor

        return molar_density

    @classmethod
    def _get_interpolator(cls, filename: Path) -> RegularGridInterpolator:
        """Gets spline lookup for density from :cite:t:`CD21` T-P-rho tables.

        The data tables have a slightly different organisation of the header line. But in all cases
        the first three columns contain the required data: log10 T [K], log10 P [GPa], and
        log10 rho [g/cc].

        Args:
            filename: Filename of the density-T-P data

        Returns:
            Interpolator
        """
        # Define column names for the first three columns
        T_name: str = "log T [K]"
        P_name: str = "log P [GPa]"
        rho_name: str = "log rho [g/cc]"
        column_names: list[str] = [T_name, P_name, rho_name]

        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(str(cls.CHABRIER_DIRECTORY.joinpath(filename)))
        )
        with data as datapath:
            df: pd.DataFrame = pd.read_csv(
                datapath,  # type: ignore
                sep=r"\s+",
                comment="#",
                usecols=[0, 1, 2],  # type: ignore
                names=column_names,
                skiprows=2,
            )
        pivot_table: pd.DataFrame = df.pivot(index=T_name, columns=P_name, values=rho_name)
        log_T: Array = jnp.array(pivot_table.index.to_numpy())
        log_P: Array = jnp.array(pivot_table.columns.to_numpy())
        log_rho: Array = jnp.array(pivot_table.to_numpy())

        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (log_T, log_P), log_rho, method="linear"
        )

        return interpolator

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        # jax.debug.print("temperature_in = {out}", out=temperature)
        # jax.debug.print("pressure_in = {out}", out=pressure)
        # Pressure range to integrate over
        pressures: Array = jnp.logspace(
            jnp.log10(PRESSURE_REFERENCE), jnp.log10(pressure), num=self._integration_steps
        )
        # jax.debug.print("pressures = {out}", out=pressures)
        volumes: Array = self.volume(temperature, pressures)
        # jax.debug.print("volumes = {out}", out=volumes)

        # Optimized trapezoidal rule (avoids jax.scipy.integrate.trapezoid overhead)
        dP: Array = jnp.diff(pressures)
        avg_volumes: Array = (volumes[:-1] + volumes[1:]) * 0.5
        volume_integral: Array = jnp.sum(avg_volumes * dP)  # Equivalent to trapezoid integration
        # jax.debug.print("volume_integral = {out}", out=volume_integral)
        log_fugacity: Array = volume_integral / (GAS_CONSTANT_BAR * temperature)

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log10_density_gcc: Array = self._log10_density_func(
            (jnp.log10(temperature), jnp.log10(unit_conversion.bar_to_GPa * pressure))
        )
        # jax.debug.print("log10_density_gcc = {out}", out=log10_density_gcc)
        molar_density: Array = self._convert_to_molar_density(log10_density_gcc)
        volume: Array = jnp.reciprocal(molar_density)
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.log_fugacity(temperature, pressure)
        volume_integral: Array = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "log10_density_func": self._log10_density_func,
        children = (self._log10_density_func,)  # Store the interpolator as a child
        aux_data = {
            "filename": self._filename,
            "He_fraction": self._He_fraction,
            "H2_molar_mass_g_mol": self._H2_molar_mass_g_mol,
            "He_molar_mass_g_mol": self._He_molar_mass_g_mol,
            "integration_steps": self._integration_steps,
        }

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        obj = cls.__new__(cls)  # Avoids calling __init__
        obj._log10_density_func = children[0]  # Restore the interpolator
        obj._filename = aux_data["filename"]
        obj._He_fraction = aux_data["He_fraction"]
        obj._H2_molar_mass_g_mol = aux_data["H2_molar_mass_g_mol"]
        obj._He_molar_mass_g_mol = aux_data["He_molar_mass_g_mol"]
        obj._integration_steps = aux_data["integration_steps"]

        return obj
