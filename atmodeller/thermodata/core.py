# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions for thermochemical and critical data"""

import importlib.resources
from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Self, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float

from atmodeller import override
from atmodeller.constants import TEMPERATURE_REFERENCE
from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.sci_utils import GAS_CONSTANT

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory"""
CRITICAL_DATA_SOURCE: Path = Path("critical_data.txt")
"""Source of the critical data"""

# Gauss-Legendre quadrature for integrating heat capacities from the reference temperature. Over
# 200-6000 K, 64 nodes integrate the NASA Glenn heat capacity of graphite to within 0.021 J/mol
# (enthalpy) and 2.5e-5 J/K/mol (entropy) of the analytical result, with the error arising from the
# discontinuities in the derivative of the heat capacity between temperature ranges.
_QUADRATURE_ORDER: int = 64
_quadrature_nodes, _quadrature_weights = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
_QUADRATURE_NODES: Array = jnp.asarray(_quadrature_nodes)
_QUADRATURE_WEIGHTS: Array = jnp.asarray(_quadrature_weights)


def _integrate_from_reference(
    integrand: Callable[[Array], Array], temperature: ArrayLike
) -> FloatArray:
    """Integrates from the reference temperature to the temperature with Gauss-Legendre quadrature

    Args:
        integrand: Function of temperature to integrate
        temperature: Temperature in K, which can be less than the reference temperature

    Returns:
        Integral from :const:`~atmodeller.constants.TEMPERATURE_REFERENCE` to ``temperature``
    """
    temperature = as_j64(temperature)
    # Append a node axis: (..., 1) against (N,) gives quadrature temperatures of shape (..., N)
    half_width: Array = (temperature[..., None] - TEMPERATURE_REFERENCE) / 2
    quadrature_temperature: Array = TEMPERATURE_REFERENCE + half_width * (_QUADRATURE_NODES + 1)

    return jnp.sum(half_width * _QUADRATURE_WEIGHTS * integrand(quadrature_temperature), axis=-1)


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


class RelativeHeatCapacity(HeatCapacity):
    r"""Heat capacity of a phase relative to a base phase.

    .. math::

        C_p(T) = C_p^{\mathrm{base}}(T) + \left[C_p^{\mathrm{phase}}(T)
            - C_p^{\mathrm{reference}}(T)\right]

    The heat capacity of a well-characterised base phase is combined with the difference in heat
    capacity between two models fitted in the same way, so that systematic errors common to those
    two models largely cancel. For example, diamond can be described by the NASA Glenn heat
    capacity of graphite plus the difference between the heat capacities of diamond and graphite
    from :cite:t:`Vassiliev2021`. This keeps graphite, the reference state of carbon, consistent
    with the other carbon-bearing species.

    Args:
        base: Heat capacity of the base phase
        phase: Heat capacity of the phase from the same model as ``reference``
        reference: Heat capacity of the base phase from the same model as ``phase``
    """

    base: HeatCapacity
    """Heat capacity of the base phase"""
    phase: HeatCapacity
    """Heat capacity of the phase from the same model as reference"""
    reference: HeatCapacity
    """Heat capacity of the base phase from the same model as phase"""

    @override
    def cp(self, temperature: ArrayLike) -> FloatArray:  # pragma: no cover
        r"""Gets heat capacity at constant pressure.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return (
            self.base.cp(temperature) + self.phase.cp(temperature) - self.reference.cp(temperature)
        )

    @override
    def temperature_range(self) -> tuple[float, float]:  # pragma: no cover
        """Gets the temperature range over which the heat capacity model is valid.

        This is the range of the base phase.

        Returns:
            Minimum and maximum temperature in K
        """
        return self.base.temperature_range()


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


class IntegratedEnthalpy(Enthalpy):
    r"""Enthalpy from integrating a heat capacity model.

    .. math::

        H(T) = H^\circ(T_r) + \int_{T_r}^T C_p\, dT

    where :math:`T_r` is :const:`~atmodeller.constants.TEMPERATURE_REFERENCE`.

    Args:
        heat_capacity_model: Heat capacity model
        enthalpy_reference: Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}` at the reference
            temperature
    """

    heat_capacity_model: HeatCapacity
    """Heat capacity model"""
    enthalpy_reference: float = eqx.field(converter=float)
    """Enthalpy in J/mol at the reference temperature"""

    @override
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return self.enthalpy_reference + _integrate_from_reference(
            self.heat_capacity_model.cp, temperature
        )


class IntegratedEntropy(Entropy):
    r"""Entropy from integrating a heat capacity model.

    .. math::

        S(T) = S^\circ(T_r) + \int_{T_r}^T \frac{C_p}{T}\, dT

    where :math:`T_r` is :const:`~atmodeller.constants.TEMPERATURE_REFERENCE`.

    Args:
        heat_capacity_model: Heat capacity model
        entropy_reference: Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}` at
            the reference temperature
    """

    heat_capacity_model: HeatCapacity
    """Heat capacity model"""
    entropy_reference: float = eqx.field(converter=float)
    """Entropy in J/K/mol at the reference temperature"""

    @override
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return self.entropy_reference + _integrate_from_reference(
            lambda t: self.heat_capacity_model.cp(t) / t, temperature
        )


class MurnaghanEOS(eqx.Module):
    r"""Volume of a solid with the equation of state of :cite:t:`HP98`.

    The volume at 1 bar follows from the temperature-dependent thermal expansion
    :math:`\alpha_T = a^\circ(1 - 10/\sqrt{T})` and its pressure dependence from the Murnaghan
    equation of state, with a bulk modulus that decreases linearly with temperature.

    Args:
        V0: Volume at 298 K and 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        alpha0: Thermal expansion parameter :math:`a^\circ` in :math:`\mathrm{K}^{-1}`
        K0: Bulk modulus at 298 K in bar
        dKdP: Pressure derivative of the bulk modulus. Defaults to 4.
    """

    V0: float
    alpha0: float
    K0: float
    dKdP: float = 4.0

    @property
    def dKdT(self) -> float:
        """Temperature derivative of the bulk modulus in bar/K"""
        return -1.0 * self.K0 * 1.5e-4

    def thermal_expansion(self, temperature: ArrayLike) -> Array:
        r"""Gets the thermal expansion.

        Args:
            temperature: Temperature in K

        Returns:
            Thermal expansion in :math:`\mathrm{K}^{-1}`
        """
        return self.alpha0 * (1 - 10 / jnp.sqrt(temperature))

    def bulk_modulus(self, temperature: ArrayLike) -> ArrayLike:
        """Gets the bulk modulus.

        Args:
            temperature: Temperature in K

        Returns:
            Bulk modulus in bar
        """
        return self.K0 + self.dKdT * (temperature - 298)

    def volume_1bar(self, temperature: ArrayLike) -> Array:
        r"""Gets the volume at 1 bar.

        This integrates the thermal expansion exactly. See :meth:`volume_1bar_linear` for the
        linearised form used by :cite:t:`HP98`.

        Args:
            temperature: Temperature in K

        Returns:
            Volume at 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        """
        volume: Array = self.V0 * jnp.exp(
            self.alpha0 * (temperature - 298)
            - 2 * 10.0 * self.alpha0 * (temperature**0.5 - 298**0.5)
        )

        return volume

    def volume_1bar_linear(self, temperature: ArrayLike) -> Array:
        r"""Gets the volume at 1 bar using the linearised form of :cite:t:`HP98`.

        .. math::

            V_{1,T} = V_{1,298}\left[1 + a^\circ(T - 298) - 20a^\circ(\sqrt{T} - \sqrt{298})\right]

        This is the exact expression used by :cite:t:`HP98` to derive their data set. It is the
        first-order expansion of :meth:`volume_1bar`, which integrates the thermal expansion
        exactly; the two differ by about 0.3% for diamond at 6000 K.

        Args:
            temperature: Temperature in K

        Returns:
            Volume at 1 bar in :math:`\mathrm{J}\ \mathrm{bar}^{-1}`
        """
        return cast(
            Array,
            self.V0
            * (
                1
                + self.alpha0 * (temperature - 298)
                - 2 * 10.0 * self.alpha0 * (jnp.sqrt(temperature) - jnp.sqrt(298))
            ),
        )

    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Gets the integral of volume with respect to pressure from 1 bar.

        .. math::

            \int_1^P V\, dP = \frac{V_{1,T}\, K_T}{K' - 1}
                \left[\left(1 + \frac{K'(P - 1)}{K_T}\right)^{1 - 1/K'} - 1\right]

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Integral of volume with respect to pressure in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        bulk_modulus: ArrayLike = self.bulk_modulus(temperature)
        integral: Array = (
            self.volume_1bar(temperature)
            * bulk_modulus
            / (self.dKdP - 1)
            * ((1 + self.dKdP * (pressure - 1.0) / bulk_modulus) ** (1.0 - 1.0 / self.dKdP) - 1)
        )

        return integral


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
    pv_model: MurnaghanEOS | None = None
    """Pressure-volume model. Defaults to ``None`` if unused."""

    @classmethod
    def from_reference_values(
        cls,
        heat_capacity_model: HeatCapacity,
        enthalpy_reference: float,
        entropy_reference: float,
    ) -> Self:
        r"""Creates thermodynamic properties by integrating a heat capacity model.

        Args:
            heat_capacity_model: Heat capacity model
            enthalpy_reference: Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}` at the
                reference temperature
            entropy_reference: Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1}
                \mathrm{mol}^{-1}` at the reference temperature

        Returns:
            Thermodynamic properties
        """
        return cls(
            heat_capacity_model,
            IntegratedEnthalpy(heat_capacity_model, enthalpy_reference),
            IntegratedEntropy(heat_capacity_model, entropy_reference),
        )

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
