# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions for thermochemical and critical data"""

import importlib.resources
from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float

from atmodeller.constants import TEMPERATURE_REFERENCE
from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.sci_utils import GAS_CONSTANT

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory"""
CRITICAL_DATA_SOURCE: Path = Path("critical_data.txt")
"""Source of the critical data"""


class ActivityCoefficient(eqx.Module):
    """Activity coefficient of a stable condensate.

    Args:
        gamma: Activity coefficient. Defaults to 1 (ideal).
    """

    gamma: Array = eqx.field(converter=as_j64, default=1)
    """Activity coefficient"""

    def active(self) -> Bool[Array, "..."]:  # pragma: no cover
        """Active activity constraint

        Condensate activity is imposed in the reaction network and therefore is never part of an
        active constraint in the residual. Not part of :class:`~atmodeller.interfaces.
        ActivityProtocol` and not currently called anywhere.

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
        shape: tuple[int, ...] = jnp.broadcast_shapes(
            jnp.shape(self.gamma), jnp.shape(temperature), jnp.shape(pressure)
        )
        if mole_fractions is not None:
            # mole_fractions has shape (..., n_species); drop the trailing species axis so the
            # broadcast shape matches the batch dimensions other activity models derive from it.
            shape = jnp.broadcast_shapes(shape, jnp.shape(mole_fractions)[:-1])

        return jnp.broadcast_to(jnp.log(self.gamma), shape)


class HeatCapacity(eqx.Module):
    r"""Heat capacity model."""

    @abstractmethod
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity at constant pressure.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        raise NotImplementedError

    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the heat capacity model is valid.

        Returns:
            Minimum and maximum temperature in K
        """
        raise NotImplementedError


class Enthalpy(eqx.Module):
    r"""Enthalpy model."""

    @abstractmethod
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        raise NotImplementedError


class Entropy(eqx.Module):
    r"""Entropy model."""

    @abstractmethod
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        raise NotImplementedError


class ThermodynamicProperties(eqx.Module):
    r"""Thermodynamic properties of an individual species

    The standard state is 1 bar and the reference temperature is
    :const:`~atmodeller.constants.TEMPERATURE_REFERENCE`.

    Args:
        heat_capacity_model: Heat capacity model
        enthalpy_model: Enthalpy model
        entropy_model: Entropy model
    """

    heat_capacity_model: HeatCapacity
    """Heat capacity model"""
    enthalpy_model: Enthalpy
    """Enthalpy model"""
    entropy_model: Entropy
    """Entropy model"""

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets Gibbs energy to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        temperature = as_j64(temperature)

        return (
            self.enthalpy(temperature) / (GAS_CONSTANT * temperature)
            - self.entropy(temperature) / GAS_CONSTANT
        )

    def cp(self, temperature: ArrayLike) -> FloatArray:  # pragma: no cover
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return self.heat_capacity_model.cp(temperature)

    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the thermodynamic data are valid.

        Returns:
            Minimum and maximum temperature in K
        """
        return self.heat_capacity_model.temperature_range()

    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return self.enthalpy_model.enthalpy(temperature)

    def reference_enthalpy(self) -> Float[Array, ""]:  # pragma: no cover
        r"""Gets reference enthalpy.

        This is :math:`H^{\circ}(T_r)` in the JANAF tables.

        Returns:
            Reference enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return self.enthalpy(TEMPERATURE_REFERENCE)

    def enthalpy_function(self, temperature: ArrayLike) -> FloatArray:  # pragma: no cover
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
        return self.entropy_model.entropy(temperature)

    def gibbs_function(self, temperature: ArrayLike) -> FloatArray:  # pragma: no cover
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


# Create a dictionary of instantiated data (JAX-compliant Pytrees) that we can use for lookup.
# It should also be net faster to create these data once and then access (potentially many times).
# These are also set to private to avoid sphinx (autodoc) from printing long strings.
critical_data_source: CriticalDataSource = CriticalDataSource()
"""Critical data source

:meta private:
"""
critical_data_dictionary: dict[str, CriticalData] = critical_data_source.create_dictionary()
"""Critical data dictionary

:meta private:
"""
