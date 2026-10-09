# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JANAF-consistent thermodynamic data from the NASA Glenn polynomials of :cite:t:`MZG02`

The NASA Glenn formulation expresses the heat capacity of each species as a 7-term polynomial
over one or more temperature ranges, together with enthalpy and entropy constants of integration
(``b1`` and ``b2``) for each range. These are split into three models:

    1. :class:`JanafHeatCapacity`, the heat capacity polynomial,
    2. :class:`JanafEnthalpy`, the analytical integral of the heat capacity with ``b1``, and
    3. :class:`JanafEntropy`, the analytical integral of the heat capacity over temperature with
       ``b2``.

Coefficients are available at https://ntrs.nasa.gov/citations/20020085330
"""

import importlib.resources
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy.typing as npt
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from atmodeller import override
from atmodeller.jax_utils import FloatArray, as_j64, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT
from atmodeller.thermodata.core import (
    DATA_DIRECTORY,
    Enthalpy,
    Entropy,
    HeatCapacity,
    ThermodynamicCoefficients,
)

THERMODYNAMIC_DATA_SOURCE: Path = Path("nasa_glenn_coefficients.txt")
"""Source of the thermodynamic data"""


class JanafHeatCapacity(HeatCapacity):
    """Heat capacity from the NASA Glenn polynomials of :cite:t:`MZG02`

    Args:
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in K in the range
        T_max: Maximum temperature(s) in K in the range
    """

    cp_coeffs: tuple[tuple[float, ...], ...] = eqx.field(converter=to_native_floats)
    """Heat capacity coefficients"""
    T_min: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Minimum temperature(s) in K in the range"""
    T_max: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Maximum temperature(s) in K in the range"""

    def get_index(self, temperature: ArrayLike) -> Integer[Array, "..."]:
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

        # Append a range axis: (..., 1) against (N,) gives a mask of shape (..., N)
        bool_mask: Bool[Array, "... N"] = (T_min <= temperature[..., None]) & (
            temperature[..., None] <= T_max
        )
        index: Integer[Array, "..."] = jnp.argmax(bool_mask, axis=-1)

        return index

    def get_cp_coeffs(self, index: Integer[Array, "..."]) -> Float[Array, "... 7"]:
        """Gets the heat capacity coefficients for the temperature range index

        Args:
            index: Index of the temperature range

        Returns:
            Heat capacity coefficients
        """
        return jnp.take(jnp.array(self.cp_coeffs), index, axis=0)

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

    @override
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self.get_index(temperature)

        return self._cp_over_R(self.get_cp_coeffs(index), temperature) * GAS_CONSTANT

    @override
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the heat capacity model is valid.

        Returns:
            Minimum and maximum temperature in K
        """
        return min(self.T_min), max(self.T_max)


class JanafEnthalpy(Enthalpy):
    """Enthalpy from the NASA Glenn polynomials of :cite:t:`MZG02`

    Args:
        heat_capacity_model: JANAF heat capacity model
        b1: Enthalpy constant(s) of integration
    """

    heat_capacity_model: JanafHeatCapacity
    """JANAF heat capacity model"""
    b1: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Enthalpy constant(s) of integration"""

    def _H_over_RT(
        self, cp_coefficients: Float[Array, "... 7"], b1: ArrayLike, temperature: ArrayLike
    ) -> FloatArray:
        r"""Enthalpy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients
            b1: Enthalpy constant of integration
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

        return jnp.sum(cp_coefficients * temperature_terms, axis=-1) + b1 / temperature

    @override
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)
        index: Integer[Array, "..."] = self.heat_capacity_model.get_index(temperature)
        b1_for_index: FloatArray = jnp.take(jnp.array(self.b1), index)

        return (
            self._H_over_RT(
                self.heat_capacity_model.get_cp_coeffs(index), b1_for_index, temperature
            )
            * GAS_CONSTANT
            * temperature
        )


class JanafEntropy(Entropy):
    """Entropy from the NASA Glenn polynomials of :cite:t:`MZG02`

    Args:
        heat_capacity_model: JANAF heat capacity model
        b2: Entropy constant(s) of integration
    """

    heat_capacity_model: JanafHeatCapacity
    """JANAF heat capacity model"""
    b2: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Entropy constant(s) of integration"""

    def _S_over_R(
        self, cp_coefficients: Float[Array, "... 7"], b2: ArrayLike, temperature: ArrayLike
    ) -> FloatArray:
        """Entropy relative to :const:`~atmodeller.constants.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            b2: Entropy constant of integration
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

    @override
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        This is :math:`S^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self.heat_capacity_model.get_index(temperature)
        b2_for_index: FloatArray = jnp.take(jnp.array(self.b2), index)

        return (
            self._S_over_R(
                self.heat_capacity_model.get_cp_coeffs(index), b2_for_index, temperature
            )
            * GAS_CONSTANT
        )


def nasa_glenn_thermodynamic_coefficients(
    b1: npt.ArrayLike,
    b2: npt.ArrayLike,
    cp_coeffs: npt.ArrayLike,
    T_min: npt.ArrayLike,
    T_max: npt.ArrayLike,
) -> ThermodynamicCoefficients:
    """Creates thermodynamic coefficients from the NASA Glenn coefficients

    Args:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in K in the range
        T_max: Maximum temperature(s) in K in the range

    Returns:
        Thermodynamic coefficients
    """
    heat_capacity_model: JanafHeatCapacity = JanafHeatCapacity(cp_coeffs, T_min, T_max)

    return ThermodynamicCoefficients(
        heat_capacity_model,
        JanafEnthalpy(heat_capacity_model, b1),
        JanafEntropy(heat_capacity_model, b2),
    )


def read_glenn_coefficients(
    path: str | Path | None = None,
) -> dict[str, ThermodynamicCoefficients]:
    """Reads NASA Glenn coefficients and creates thermodynamic coefficients for all species

    Args:
        path: Path to a file of NASA Glenn coefficients in the same format as the packaged
            :const:`THERMODYNAMIC_DATA_SOURCE`. Defaults to ``None``, which uses the packaged
            file.

    Returns:
        Thermodynamic coefficients for all species, keyed by Hill formula and state, e.g.
        ``H2O_g``
    """
    if path is not None:
        data: pd.DataFrame = pd.read_csv(path, comment="#")
    else:
        packaged: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)  # type: ignore
        )
        with packaged as datapath:
            data = pd.read_csv(datapath, comment="#")

    unique_combinations: pd.DataFrame = cast(
        pd.DataFrame, data[["hill_formula", "state"]].drop_duplicates()
    )
    coefficient_dict: dict[str, ThermodynamicCoefficients] = {}

    for row in unique_combinations.itertuples(index=False):
        hill_formula: str = str(row.hill_formula)
        state: str = str(row.state)
        name: str = f"{hill_formula}_{state}"

        # Find all data across all temperature ranges
        df: pd.DataFrame = cast(
            pd.DataFrame, data[(data["hill_formula"] == hill_formula) & (data["state"] == state)]
        )
        cp_coeffs: pd.DataFrame | pd.Series = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]]
        coefficient_dict[name] = nasa_glenn_thermodynamic_coefficients(
            df["b1"], df["b2"], cp_coeffs, df["T_min"], df["T_max"]
        )

    return coefficient_dict


# Create the default data once, since they are accessed (potentially many times) when creating
# species. This is set to private to avoid sphinx (autodoc) from printing long strings.
glenn_coefficients: dict[str, ThermodynamicCoefficients] = read_glenn_coefficients()
"""Thermodynamic coefficients for all species from the packaged NASA Glenn coefficients

:meta private:
"""
