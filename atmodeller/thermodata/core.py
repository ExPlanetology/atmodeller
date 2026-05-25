# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions for thermochemical and critical data"""

import importlib.resources
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from atmodeller.constants import TEMPERATURE_REFERENCE
from atmodeller.jax_utils import FloatArray, as_j64, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory"""
THERMODYNAMIC_DATA_SOURCE: Path = Path("nasa_glenn_coefficients.txt")
"""Source of the thermodynamic data"""
CRITICAL_DATA_SOURCE: Path = Path("critical_data.txt")
"""Source of the critical data"""


class ActivityCoefficient(eqx.Module):
    """Activity coefficient of a stable condensate.

    Args:
        gamma: Activity coefficient. Defaults to 1 (ideal).
    """

    gamma: Array = eqx.field(converter=as_j64, default=1)
    """Activity coefficient"""

    def active(self) -> Bool[Array, "..."]:
        """Active activity constraint

        Condensate activity is imposed in the reaction network and therefore is never part of an
        active constraint in the residual.

        Returns:
            Always ``False`` because it does not require solution.
        """
        return jnp.full_like(self.gamma, False, dtype=jnp.bool_)

    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: FloatArray | None = None
    ) -> FloatArray:
        """Log of the activity coefficient (dimensionless).

        This is the primary access point for calling the EOS within the main engine so must adhere
        to the expected interface for activity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions. Defaults to ``None`` if unused.

        Returns:
            Log activity coefficient
        """
        del temperature
        del pressure
        del mole_fractions

        return jnp.log(self.gamma)


class ThermodynamicCoefficients(eqx.Module):
    """NASA Glenn coefficients for the thermodynamic properties of an individual species

    Coefficients are available at https://ntrs.nasa.gov/citations/20020085330

    Args:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in K in the range
        T_max: Maximum temperature(s) in K in the range
    """

    b1: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Enthalpy constant(s) of integration"""
    b2: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Entropy constant(s) of integration"""
    cp_coeffs: tuple[tuple[float, ...], ...] = eqx.field(converter=to_native_floats)
    """Heat capacity coefficients"""
    T_min: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Minimum temperature(s) in K in the range"""
    T_max: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Maximum temperature(s) in K in the range"""

    def _get_index(self, temperature: ArrayLike) -> Integer[Array, "..."]:
        """Gets the index of the temperature range for the given temperature

        This assumes the temperature is within one of the ranges and will produce unexpected output
        if the temperature is outside the ranges.

        Args:
            temperature: Temperature in K

        Returns:
            Index of the temperature range
        """
        temperature = as_j64(temperature)
        T_max: Array = as_j64(self.T_max)
        T_min: Array = as_j64(self.T_min)

        # Reshape T_min/T_max to (N, 1, 1, ...) to broadcast against any temperature shape,
        # giving bool_mask shape (N, *temperature.shape) and index shape temperature.shape.
        n_extra = jnp.ndim(temperature)
        T_min_b = T_min.reshape((-1,) + (1,) * n_extra)
        T_max_b = T_max.reshape((-1,) + (1,) * n_extra)

        # Reshape for broadcasting
        bool_mask: Bool[Array, "N ..."] = (T_min_b <= temperature) & (temperature <= T_max_b)
        index: Integer[Array, "..."] = jnp.argmax(bool_mask, axis=0)

        return index

    def _cp_over_R(
        self, cp_coefficients: Float[Array, "... 7"], temperature: ArrayLike
    ) -> FloatArray:
        """Heat capacity relative to :const:`~atmodeller.constants.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            temperature: Temperature in K

        Returns:
            Heat capacity relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        """
        temperature = as_j64(temperature)
        temperature_terms: Float[Array, "... 7"] = jnp.stack(
            [
                jnp.power(temperature, -2),
                jnp.power(temperature, -1),
                jnp.ones_like(temperature),
                temperature,
                jnp.power(temperature, 2),
                jnp.power(temperature, 3),
                jnp.power(temperature, 4),
            ],
            axis=-1,
        )

        return jnp.sum(cp_coefficients * temperature_terms, axis=-1)

    def _S_over_R(
        self, cp_coefficients: Float[Array, "... 7"], b2: ArrayLike, temperature: ArrayLike
    ) -> FloatArray:
        """Entropy relative to :const:`~atmodeller.constants.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            b2: Entropy integration constant
            temperature: Temperature in K

        Returns:
            Entropy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        """
        temperature = as_j64(temperature)
        temperature_terms: Float[Array, "... 7"] = jnp.stack(
            [
                -jnp.power(temperature, -2) / 2,
                -jnp.power(temperature, -1),
                jnp.log(temperature),
                temperature,
                jnp.power(temperature, 2) / 2,
                jnp.power(temperature, 3) / 3,
                jnp.power(temperature, 4) / 4,
            ],
            axis=-1,
        )

        return jnp.sum(cp_coefficients * temperature_terms, axis=-1) + b2

    def _H_over_RT(
        self, cp_coefficients: Float[Array, "... 7"], b1: ArrayLike, temperature: ArrayLike
    ) -> FloatArray:
        r"""Enthalpy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            temperature: Temperature in K

        Returns:
            Enthalpy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        temperature = as_j64(temperature)
        temperature_terms: Float[Array, "... 7"] = jnp.stack(
            [
                -jnp.power(temperature, -2),
                jnp.log(temperature) / temperature,
                jnp.ones_like(temperature),
                temperature / 2,
                jnp.power(temperature, 2) / 3,
                jnp.power(temperature, 3) / 4,
                jnp.power(temperature, 4) / 5,
            ],
            axis=-1,
        )

        enthalpy: FloatArray = (
            jnp.sum(cp_coefficients * temperature_terms, axis=-1) + b1 / temperature
        )

        return enthalpy

    def _G_over_RT(
        self,
        cp_coefficients: Float[Array, "... 7"],
        b1: ArrayLike,
        b2: ArrayLike,
        temperature: ArrayLike,
    ) -> FloatArray:
        r"""Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            b2: Entropy integration constant
            temperature: Temperature in K

        Returns:
            Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        enthalpy: FloatArray = self._H_over_RT(cp_coefficients, b1, temperature)
        # jax.debug.print("enthalpy = {out}", out=enthalpy)
        entropy: FloatArray = self._S_over_R(cp_coefficients, b2, temperature)
        # jax.debug.print("entropy = {out}", out=entropy)

        # No temperature multiplication is correct since the return is Gibbs energy relative to RT
        gibbs: FloatArray = enthalpy - entropy

        return gibbs

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets Gibbs energy to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        index: Integer[Array, "..."] = self._get_index(temperature)
        # jax.debug.print(
        #     "temperature.shape = {t}, index.shape = {i}",
        #     t=jnp.shape(temperature),
        #     i=jnp.shape(index),
        # )
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Float[Array, "... 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        # jax.debug.print("cp_coeffs_for_index.shape = {out}", out=jnp.shape(cp_coeffs_for_index))
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index)
        b1_for_index: FloatArray = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        b2_for_index: FloatArray = jnp.take(jnp.array(self.b2), index)
        # jax.debug.print("b2_for_index = {out}", out=b2_for_index)
        gibbs_for_index: FloatArray = self._G_over_RT(
            cp_coeffs_for_index, b1_for_index, b2_for_index, temperature
        )
        # jax.debug.print("gibbs_for_index.shape = {out}", out=jnp.shape(gibbs_for_index))

        return gibbs_for_index

    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "... 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index.shape)
        cp: FloatArray = self._cp_over_R(cp_coeffs_for_index, temperature) * GAS_CONSTANT

        return cp

    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "... 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        b1_for_index: FloatArray = jnp.take(jnp.array(self.b1), index)
        enthalpy: FloatArray = (
            self._H_over_RT(cp_coeffs_for_index, b1_for_index, temperature)
            * GAS_CONSTANT
            * temperature
        )

        return enthalpy

    def reference_enthalpy(self) -> Float[Array, ""]:
        r"""Gets reference enthalpy.

        This is :math:`H^{\circ}(T_r)` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Reference enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        index: Integer[Array, ""] = self._get_index(TEMPERATURE_REFERENCE)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Float[Array, "7"] = jnp.take(jnp.array(self.cp_coeffs), index, axis=0)
        b1_for_index: Float[Array, ""] = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        reference_enthalpy: Float[Array, ""] = (
            self._H_over_RT(cp_coeffs_for_index, b1_for_index, TEMPERATURE_REFERENCE)
            * GAS_CONSTANT
            * TEMPERATURE_REFERENCE
        )

        return reference_enthalpy

    def enthalpy_function(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy function/increment.

        This is :math:`H-H^{\circ}(T_r)` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy increment in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return self.enthalpy(temperature) - self.reference_enthalpy()

    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy

        This is :math:`S^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "... 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        b2_for_index: FloatArray = jnp.take(jnp.array(self.b2), index)
        entropy: FloatArray = (
            self._S_over_R(cp_coeffs_for_index, b2_for_index, temperature) * GAS_CONSTANT
        )

        return entropy

    def gibbs_function(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets Gibbs energy function.

        This is :math:`-[G^\circ-H^{\circ}(T_r)]/T` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy function in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        gibbs: FloatArray = self.get_gibbs_over_RT(temperature) * GAS_CONSTANT * temperature
        gibbs_function: FloatArray = -(gibbs - self.reference_enthalpy()) / temperature

        return gibbs_function


@dataclass
class ThermodynamicDataSource:
    """Thermodynamic data source for all species"""

    data: pd.DataFrame
    """Thermodynamic data for all species"""

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            self.data = pd.read_csv(datapath, comment="#")

    @property
    def formula_column(self) -> str:
        """Name of the column that refers to the hill formula"""
        return "hill_formula"

    @property
    def state_column(self) -> str:
        """Name of the column that refers to the state of aggregation"""
        return "state"

    def available_species(self) -> tuple[str, ...]:
        """Available species

        Returns:
            Available species
        """
        df: pd.DataFrame = cast(
            pd.DataFrame, self.data[[self.formula_column, self.state_column]].drop_duplicates()
        )
        available_species: tuple[str, ...] = tuple(
            f"{getattr(row, self.formula_column)}_{getattr(row, self.state_column)}"
            for row in df.itertuples(index=False)
        )

        return available_species

    def create_dictionary(self) -> dict[str, ThermodynamicCoefficients]:
        """Dictionary of thermodynamic coefficients for all species

        Returns:
            Dictionary of thermodynamic coefficients for all species
        """
        unique_combinations: pd.DataFrame = cast(
            pd.DataFrame, self.data[[self.formula_column, self.state_column]].drop_duplicates()
        )
        coefficient_dict: dict[str, ThermodynamicCoefficients] = {}

        for row in unique_combinations.itertuples(index=False):
            hill_formula: str = str(getattr(row, self.formula_column))
            state: str = str(getattr(row, self.state_column))
            name: str = f"{hill_formula}_{state}"

            # Find all data across all temperature ranges
            df: pd.DataFrame = cast(
                pd.DataFrame,
                self.data[
                    (self.data[self.formula_column] == hill_formula)
                    & (self.data[self.state_column] == state)
                ],
            )
            cp_coeffs: pd.DataFrame | pd.Series = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]]
            coefficient_dict[name] = ThermodynamicCoefficients(
                df["b1"], df["b2"], cp_coeffs, df["T_min"], df["T_max"]
            )

        return coefficient_dict


class CriticalData(eqx.Module):
    """Critical temperature and pressure of a gas species

    Args:
        temperature: Critical temperature in K
        pressure: Critical pressure in bar
    """

    temperature: float = eqx.field(converter=float, default=1)
    """Critical temperature in K"""
    pressure: float = eqx.field(converter=float, default=1)
    """Critical pressure in bar"""


@dataclass
class CriticalDataSource:
    """Critical data source for all species"""

    data: pd.DataFrame
    """Critical data for all species"""

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(CRITICAL_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            self.data = pd.read_csv(datapath, comment="#")

    @property
    def name_column(self) -> str:
        """Name of the column that refers to the hill formula and an optional suffix"""
        return "name"

    @property
    def critical_temperature_column(self) -> str:
        """Name of the column that refers to the critical temperature in K"""
        return "Tc"

    @property
    def critical_pressure_column(self) -> str:
        """Name of the column that refers to the critical pressure"""
        return "Pc"

    def create_dictionary(self) -> dict[str, CriticalData]:
        """Dictionary of critical data for all species

        Returns:
            Dictionary of critical data for all species
        """
        critical_dict: dict[str, CriticalData] = {}

        for row in self.data.itertuples(index=False):
            name: str = str(getattr(row, self.name_column))
            critical_dict[name] = CriticalData(
                temperature=float(getattr(row, self.critical_temperature_column)),
                pressure=float(getattr(row, self.critical_pressure_column)),
            )

        return critical_dict


# Create dictionaries of instantiated data (JAX-compliant Pytrees) that we can use for lookup.
# It should also be net faster to create these data once and then access (potentially many times).
# These are also set to private to avoid sphinx (autodoc) from printing long strings.
thermodynamic_data_source: ThermodynamicDataSource = ThermodynamicDataSource()
"""Thermodynamic data source

:meta private:
"""
thermodynamic_coefficients_dictionary: dict[str, ThermodynamicCoefficients] = (
    thermodynamic_data_source.create_dictionary()
)
"""Thermodynamic coefficients dictionary

:meta private:
"""
critical_data_source: CriticalDataSource = CriticalDataSource()
"""Critical data source

:meta private:
"""
critical_data_dictionary: dict[str, CriticalData] = critical_data_source.create_dictionary()
"""Critical data dictionary

:meta private:
"""
