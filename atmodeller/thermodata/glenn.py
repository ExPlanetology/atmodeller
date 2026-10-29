# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""JANAF-consistent thermodynamic data from the NASA Glenn polynomials of :cite:t:`MZG02`

The NASA Glenn formulation expresses the heat capacity of each species as a 7-term polynomial
over one or more temperature ranges, together with enthalpy and entropy constants of integration
(``b1`` and ``b2``) for each range. :class:`NasaGlennThermodynamicProperties` evaluates the heat
capacity polynomial and its analytical integrals for the enthalpy and entropy.

Coefficients are available at https://ntrs.nasa.gov/citations/20020085330
"""

from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Self, cast

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from atmodeller import override
from atmodeller.jax_utils import FloatArray, as_j64, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT
from atmodeller.thermodata.core import DATA_DIRECTORY, ThermodynamicProperties

THERMODYNAMIC_DATA_SOURCE: Path = Path("nasa_glenn_coefficients.txt")
"""Source of the thermodynamic data"""

# Temperature tolerance (K) for matching the end of one temperature range to the start of the next
_RANGE_TOLERANCE: float = 1e-6


class NasaGlennThermodynamicProperties(ThermodynamicProperties):
    """Thermodynamic properties from the NASA Glenn polynomials of :cite:t:`MZG02`

    Args:
        cp_coeffs: Heat capacity coefficients for each temperature range
        b1: Enthalpy constant of integration for each temperature range
        b2: Entropy constant of integration for each temperature range
        T_min: Minimum temperature of each range (K)
        T_max: Maximum temperature of each range (K)
        volume_model: Volume model. Defaults to ``None``, which ignores the pressure dependence.
    """

    cp_coeffs: tuple[tuple[float, ...], ...] = eqx.field(converter=to_native_floats)
    """Heat capacity coefficients"""
    b1: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Enthalpy constant(s) of integration"""
    b2: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Entropy constant(s) of integration"""
    T_min: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Minimum temperature(s) of the range (K)"""
    T_max: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """Maximum temperature(s) of the range (K)"""

    def get_index(self, temperature: ArrayLike) -> Integer[Array, "..."]:
        """Gets the index of the temperature range for the given temperature

        Outside the ranges, the nearest range is used, so the data are extrapolated from the
        lowest range below it and from the highest range above it.

        Args:
            temperature: Temperature (K)

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
        # Below the lowest range the mask is all False, so argmax gives the lowest range (index 0)
        index: Integer[Array, "..."] = jnp.argmax(bool_mask, axis=-1)
        # Above the highest range, use the highest range rather than the lowest
        index = jnp.where(temperature > jnp.max(T_max), jnp.argmax(T_max), index)

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
        """Heat capacity relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            temperature: Temperature (K)

        Returns:
            Heat capacity relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`
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

    def _H_over_RT(
        self, cp_coefficients: Float[Array, "... 7"], b1: ArrayLike, temperature: ArrayLike
    ) -> FloatArray:
        r"""Enthalpy relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients
            b1: Enthalpy constant of integration
            temperature: Temperature (K)

        Returns:
            Enthalpy relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`
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

    def _S_over_R(
        self, cp_coefficients: Float[Array, "... 7"], b2: ArrayLike, temperature: ArrayLike
    ) -> FloatArray:
        """Entropy relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`

        Args:
            cp_coefficients: Heat capacity coefficients
            b2: Entropy constant of integration
            temperature: Temperature (K)

        Returns:
            Entropy relative to :data:`~atmodeller.sci_utils.GAS_CONSTANT`
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
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature (K)

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self.get_index(temperature)

        return self._cp_over_R(self.get_cp_coeffs(index), temperature) * GAS_CONSTANT

    @override
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature (K)

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)
        index: Integer[Array, "..."] = self.get_index(temperature)
        b1_for_index: FloatArray = jnp.take(jnp.array(self.b1), index)

        return (
            self._H_over_RT(self.get_cp_coeffs(index), b1_for_index, temperature)
            * GAS_CONSTANT
            * temperature
        )

    @override
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        This is :math:`S^\circ` in the JANAF tables.

        Args:
            temperature: Temperature (K)

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, "..."] = self.get_index(temperature)
        b2_for_index: FloatArray = jnp.take(jnp.array(self.b2), index)

        return self._S_over_R(self.get_cp_coeffs(index), b2_for_index, temperature) * GAS_CONSTANT

    @override
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the thermodynamic data are valid.

        Returns:
            Minimum and maximum temperature (K)
        """
        return min(self.T_min), max(self.T_max)

    @classmethod
    def stitch(cls, end_members: Mapping[str, Self]) -> Self:
        """Stitches the end-members of one formula together in order of temperature

        The NASA Glenn data are at 1 bar, and the temperature ranges of the end-members of
        a formula (e.g. quartz-alpha, quartz-beta, cristobalite-beta and liquid) are its stable
        sequence at 1 bar, so stitching them gives the stable end-member at each
        temperature. A single end-member is returned as it is, with any volume model.
        Several end-members must have no volume model, since the pressure dependence would
        move the transitions between them.

        Args:
            end_members: Thermodynamic properties of each end-member, keyed by its name

        Returns:
            Thermodynamic properties of the end-members stitched together

        Raises:
            ValueError: If several end-members are stitched and any has a volume model, or if
                their temperature ranges overlap or leave a gap
        """
        if len(end_members) == 1:
            return next(iter(end_members.values()))

        with_volume: list[str] = [
            name for name, properties in end_members.items() if properties.volume_model
        ]
        if with_volume:
            raise ValueError(
                f"End-members with a volume model cannot be stitched together: {with_volume}"
            )

        ordered: list[tuple[str, NasaGlennThermodynamicProperties]] = sorted(
            end_members.items(), key=lambda item: min(item[1].T_min)
        )
        for (name, properties), (next_name, next_properties) in zip(ordered, ordered[1:]):
            end: float = max(properties.T_max)
            start: float = min(next_properties.T_min)
            if start < end - _RANGE_TOLERANCE:
                raise ValueError(f"{name} (to {end} K) overlaps {next_name} (from {start} K)")
            if start > end + _RANGE_TOLERANCE:
                raise ValueError(
                    f"There is a gap between {name} (to {end} K) and {next_name} (from {start} K)"
                )

        return cls(
            tuple(coeffs for _, properties in ordered for coeffs in properties.cp_coeffs),
            tuple(value for _, properties in ordered for value in properties.b1),
            tuple(value for _, properties in ordered for value in properties.b2),
            tuple(value for _, properties in ordered for value in properties.T_min),
            tuple(value for _, properties in ordered for value in properties.T_max),
        )


def read_glenn_coefficients(
    path: str | Path | StringIO,
) -> dict[str, dict[str, NasaGlennThermodynamicProperties]]:
    """Reads NASA Glenn coefficients and creates thermodynamic properties for each end-member

    Each row of the file gives the coefficients of one temperature range of an end-member
    (``gas``, ``liquid`` or a mineral) of a formula, and the rows of an end-member together
    give its thermodynamic properties. End-members are stitched together on demand with
    :meth:`NasaGlennThermodynamicProperties.stitch`.

    Args:
        path: Path to a file of NASA Glenn coefficients in the same format as the packaged
            :data:`THERMODYNAMIC_DATA_SOURCE`, or its contents in a :class:`~io.StringIO`

    Returns:
        Thermodynamic properties of each end-member, keyed by Hill formula and then
        end-member, e.g. ``["O2Si"]["quartz-alpha"]`` or ``["H2O"]["gas"]``

    Raises:
        ValueError: If the file does not have the columns of the packaged file
    """
    data: pd.DataFrame = pd.read_csv(path, comment="#")
    columns: tuple[str, ...] = (
        "end_member",
        "hill_formula",
        "T_min",
        "T_max",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "b1",
        "b2",
    )
    missing: list[str] = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing the columns {missing}")

    end_members: dict[str, dict[str, NasaGlennThermodynamicProperties]] = {}

    for key, df in data.groupby(["hill_formula", "end_member"], sort=False):
        hill_formula, end_member = (str(value) for value in cast(tuple[str, str], key))
        end_members.setdefault(hill_formula, {})[end_member] = NasaGlennThermodynamicProperties(
            df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]],
            df["b1"],
            df["b2"],
            df["T_min"],
            df["T_max"],
        )

    return end_members


# Create the default data once, since they are accessed (potentially many times) when creating
# species. The value is hidden from sphinx (autodoc), which would otherwise print it in full.
glenn_properties: dict[str, dict[str, NasaGlennThermodynamicProperties]] = read_glenn_coefficients(
    StringIO(DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE).read_text())
)
"""Thermodynamic properties of each end-member from the packaged NASA Glenn coefficients,
keyed by Hill formula and then end-member, e.g. ``glenn_properties["H2O"]["gas"]``

:meta hide-value:
"""
