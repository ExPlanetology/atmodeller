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

import importlib.resources
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy.typing as npt
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from atmodeller import override
from atmodeller.constants import GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT
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


def nasa_glenn_thermodynamic_properties(
    b1: npt.ArrayLike,
    b2: npt.ArrayLike,
    cp_coeffs: npt.ArrayLike,
    T_min: npt.ArrayLike,
    T_max: npt.ArrayLike,
) -> ThermodynamicProperties:
    """Creates thermodynamic properties from the NASA Glenn coefficients

    Args:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) of the range (K)
        T_max: Maximum temperature(s) of the range (K)

    Returns:
        Thermodynamic properties
    """
    return NasaGlennThermodynamicProperties(cp_coeffs, b1, b2, T_min, T_max)


def stitch_glenn_properties(
    phase_components: Mapping[str, NasaGlennThermodynamicProperties],
) -> NasaGlennThermodynamicProperties:
    """Stitches the phase components of one formula together in order of temperature

    The NASA Glenn data are at 1 bar, and the temperature ranges of the phase components of a
    formula (e.g. quartz-alpha, quartz-beta, cristobalite-beta and liquid) are its stable sequence
    at 1 bar, so stitching them gives the stable phase component at each temperature. A single
    phase component is returned as it is, with any volume model. Several phase components must
    have no volume model, since the pressure dependence would move the transitions between them.

    Args:
        phase_components: Thermodynamic properties of each phase component, keyed by its name

    Returns:
        Thermodynamic properties of the phase components stitched together

    Raises:
        ValueError: If several phase components are stitched and any has a volume model, or if
            their temperature ranges overlap or leave a gap
    """
    if len(phase_components) == 1:
        return next(iter(phase_components.values()))

    with_volume: list[str] = [
        name for name, properties in phase_components.items() if properties.volume_model
    ]
    if with_volume:
        raise ValueError(
            f"Phase components with a volume model cannot be stitched together: {with_volume}"
        )

    ordered: list[tuple[str, NasaGlennThermodynamicProperties]] = sorted(
        phase_components.items(), key=lambda item: min(item[1].T_min)
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

    return NasaGlennThermodynamicProperties(
        tuple(coeffs for _, properties in ordered for coeffs in properties.cp_coeffs),
        tuple(value for _, properties in ordered for value in properties.b1),
        tuple(value for _, properties in ordered for value in properties.b2),
        tuple(value for _, properties in ordered for value in properties.T_min),
        tuple(value for _, properties in ordered for value in properties.T_max),
    )


_GLENN_COLUMNS: tuple[str, ...] = (
    "phase_component",
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
"""Columns of a file of NASA Glenn coefficients"""


def _read_glenn_data(path: str | Path | None) -> pd.DataFrame:
    """Reads a file of NASA Glenn coefficients, or the packaged file if ``path`` is ``None``"""
    if path is not None:
        data: pd.DataFrame = pd.read_csv(path, comment="#")
    else:
        packaged: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)  # type: ignore
        )
        with packaged as datapath:
            data = pd.read_csv(datapath, comment="#")

    if "state" in data.columns:
        raise ValueError(
            f"{path} has a state column, which is no longer used. Drop it and move "
            "phase_component to the first column, e.g. with pandas: "
            'data = pd.read_csv(path, comment="#"); '
            'data.drop(columns="state")[["phase_component", "hill_formula", *data.columns[3:]]]'
            ".to_csv(new_path, index=False)"
        )
    if tuple(data.columns) != _GLENN_COLUMNS:
        raise ValueError(f"{path} must have the columns {_GLENN_COLUMNS}")

    return data


def _from_rows(df: pd.DataFrame) -> NasaGlennThermodynamicProperties:
    """Thermodynamic properties from all the temperature ranges in the rows"""
    cp_coeffs: pd.DataFrame | pd.Series = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]]
    return NasaGlennThermodynamicProperties(
        cp_coeffs, df["b1"], df["b2"], df["T_min"], df["T_max"]
    )


def read_glenn_coefficients(
    path: str | Path | None = None,
) -> dict[str, dict[str, NasaGlennThermodynamicProperties]]:
    """Reads NASA Glenn coefficients and creates thermodynamic properties for each phase component

    Each row of the file gives the coefficients of one temperature range of a phase component
    (``gas``, ``liquid`` or a mineral) of a formula, and the rows of a phase component together
    give its thermodynamic properties. Phase components are stitched together on demand with
    :func:`stitch_glenn_properties`.

    Args:
        path: Path to a file of NASA Glenn coefficients in the same format as the packaged
            :data:`THERMODYNAMIC_DATA_SOURCE`. Defaults to ``None``, which uses the packaged
            file.

    Returns:
        Thermodynamic properties of each phase component, keyed by Hill formula and then phase
        component, e.g. ``["O2Si"]["quartz-alpha"]`` or ``["H2O"]["gas"]``

    Raises:
        ValueError: If the file is not in the format of the packaged file, or the same mineral is
            given for more than one formula
    """
    data: pd.DataFrame = _read_glenn_data(path)
    phase_components: dict[str, dict[str, NasaGlennThermodynamicProperties]] = {}
    formula_of_component: dict[str, str] = {}

    for key, df in data.groupby(["hill_formula", "phase_component"], sort=False):
        hill_formula, component = (str(value) for value in cast(tuple[str, str], key))
        if component not in (GAS_PHASE_COMPONENT, LIQUID_PHASE_COMPONENT):
            if formula_of_component.setdefault(component, hill_formula) != hill_formula:
                raise ValueError(
                    f"Phase component {component} is given for "
                    f"{formula_of_component[component]} and {hill_formula}"
                )
        phase_components.setdefault(hill_formula, {})[component] = _from_rows(df)

    return phase_components


# Create the default data once, since they are accessed (potentially many times) when creating
# species. The value is hidden from sphinx (autodoc), which would otherwise print it in full.
glenn_properties: dict[str, dict[str, NasaGlennThermodynamicProperties]] = (
    read_glenn_coefficients()
)
"""Thermodynamic properties of each phase component from the packaged NASA Glenn coefficients,
keyed by Hill formula and then phase component, e.g. ``glenn_properties["H2O"]["gas"]``

:meta hide-value:
"""
