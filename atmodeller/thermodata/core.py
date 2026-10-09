# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core classes and functions for thermochemical and critical data"""

import importlib.resources
from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float

from atmodeller import override
from atmodeller.constants import STANDARD_PRESSURE, TEMPERATURE_REFERENCE
from atmodeller.interfaces import HeatCapacityProtocol, VolumeProtocol
from atmodeller.jax_utils import FloatArray, as_j64, elementwise_derivative
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


class ThermodynamicProperties(eqx.Module):
    r"""Thermodynamic properties of an individual species

    Subclasses provide the heat capacity, enthalpy and entropy of a data source, from which this
    class gets the Gibbs energy. The standard state is 1 bar and the reference temperature is
    :const:`~atmodeller.constants.TEMPERATURE_REFERENCE`. An optional volume model adds the
    pressure dependence of a condensed phase to the Gibbs energy. Gases do not need one, since
    their pressure dependence enters through the fugacity.
    """

    volume_model: VolumeProtocol | None = eqx.field(default=None, kw_only=True)
    """Volume model, or ``None`` to ignore the pressure dependence. Keyword only."""

    @abstractmethod
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        raise NotImplementedError

    @abstractmethod
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the thermodynamic data are valid.

        Returns:
            Minimum and maximum temperature in K
        """
        raise NotImplementedError

    @abstractmethod
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        raise NotImplementedError

    @abstractmethod
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy

        This is :math:`S^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        raise NotImplementedError

    def gibbs_energy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets the Gibbs energy at 1 bar.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)

        return self.enthalpy(temperature) - temperature * self.entropy(temperature)

    def with_volume(self, volume_model: VolumeProtocol) -> Self:
        """Gets a copy of the thermodynamic properties with a volume model.

        Args:
            volume_model: Volume model, which replaces any existing one

        Returns:
            Thermodynamic properties with the volume model
        """
        return replace(self, volume_model=volume_model)

    def get_gibbs_over_RT(
        self, temperature: ArrayLike, pressure: ArrayLike = STANDARD_PRESSURE
    ) -> FloatArray:
        r"""Gets Gibbs energy to :const:`~atmodeller.constants.GAS_CONSTANT`
        :math:`\times T`

        Without a volume model this is the standard-state Gibbs energy at 1 bar for any pressure.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar. Defaults to 1 bar.

        Returns:
            Gibbs energy relative to :const:`~atmodeller.constants.GAS_CONSTANT`
            :math:`\times T`
        """
        temperature = as_j64(temperature)
        pressure = as_j64(pressure)

        gibbs_over_RT: FloatArray = self.gibbs_energy(temperature) / (GAS_CONSTANT * temperature)
        if self.volume_model is not None:
            gibbs_over_RT = gibbs_over_RT + self.volume_model.volume_integral(
                temperature, pressure
            ) / (GAS_CONSTANT * temperature)

        # All species must return the same shape when evaluated together in the reaction network
        return jnp.broadcast_to(
            gibbs_over_RT, jnp.broadcast_shapes(temperature.shape, pressure.shape)
        )

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


class IntegratedThermodynamicProperties(ThermodynamicProperties):
    r"""Thermodynamic properties from integrating a heat capacity model

    .. math::

        H(T) = H^\circ(T_r) + \int_{T_r}^T C_p\, dT, \qquad
        S(T) = S^\circ(T_r) + \int_{T_r}^T \frac{C_p}{T}\, dT

    where :math:`T_r` is :const:`~atmodeller.constants.TEMPERATURE_REFERENCE`. The integrals use
    Gauss-Legendre quadrature.

    Args:
        heat_capacity_model: Heat capacity model
        enthalpy_reference: Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}` at the reference
            temperature
        entropy_reference: Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}` at
            the reference temperature
        volume_model: Volume model. Defaults to ``None``, which ignores the pressure dependence.
    """

    heat_capacity_model: HeatCapacityProtocol
    """Heat capacity model"""
    enthalpy_reference: float = eqx.field(converter=float)
    """Enthalpy in J/mol at the reference temperature"""
    entropy_reference: float = eqx.field(converter=float)
    """Entropy in J/K/mol at the reference temperature"""

    @override
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return self.heat_capacity_model.cp(temperature)

    @override
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the thermodynamic data are valid.

        Returns:
            Minimum and maximum temperature in K
        """
        return self.heat_capacity_model.temperature_range()

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


class GibbsThermodynamicProperties(ThermodynamicProperties):
    r"""Thermodynamic properties from a Gibbs energy function at 1 bar

    Subclasses provide the Gibbs energy :math:`G(T)`, from which the entropy, enthalpy and heat
    capacity follow by automatic differentiation

    .. math::

        S = -\frac{\partial G}{\partial T}, \qquad H = G + TS, \qquad
        C_p = -T\frac{\partial^2 G}{\partial T^2}
    """

    @override
    @abstractmethod
    def gibbs_energy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets the Gibbs energy at 1 bar.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        raise NotImplementedError

    @override
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return -elementwise_derivative(self.gibbs_energy, as_j64(temperature))

    @override
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)

        return self.gibbs_energy(temperature) + temperature * self.entropy(temperature)

    @override
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        temperature = as_j64(temperature)

        return -temperature * elementwise_derivative(
            lambda t: elementwise_derivative(self.gibbs_energy, t), temperature
        )


class RelativeThermodynamicProperties(ThermodynamicProperties):
    r"""Thermodynamic properties of a phase relative to a base phase.

    .. math::

        X(T) = X^{\mathrm{base}}(T) + \left[X^{\mathrm{phase}}(T)
            - X^{\mathrm{reference}}(T)\right]

    for the Gibbs energy, enthalpy, entropy and heat capacity. A well-characterised base phase is
    combined with the difference between two phases from the same assessment, so that, for
    example, diamond has the Gibbs energy of graphite from the NASA Glenn coefficients plus the
    Gibbs energy of diamond relative to graphite from :cite:t:`Gustafson1986`. Only the volume
    model of this class is used, not those of the base, phase or reference.

    Args:
        base: Thermodynamic properties of the base phase
        phase: Thermodynamic properties of the phase from the same assessment as ``reference``
        reference: Thermodynamic properties of the base phase from the same assessment as
            ``phase``
        volume_model: Volume model. Defaults to ``None``, which ignores the pressure dependence.
    """

    base: ThermodynamicProperties
    """Thermodynamic properties of the base phase"""
    phase: ThermodynamicProperties
    """Thermodynamic properties of the phase from the same assessment as reference"""
    reference: ThermodynamicProperties
    """Thermodynamic properties of the base phase from the same assessment as phase"""

    @override
    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return (
            self.base.cp(temperature)
            + self.phase.cp(temperature)
            - self.reference.cp(temperature)
        )

    @override
    def enthalpy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets enthalpy.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return (
            self.base.enthalpy(temperature)
            + self.phase.enthalpy(temperature)
            - self.reference.enthalpy(temperature)
        )

    @override
    def entropy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets entropy.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        return (
            self.base.entropy(temperature)
            + self.phase.entropy(temperature)
            - self.reference.entropy(temperature)
        )

    @override
    def gibbs_energy(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets the Gibbs energy at 1 bar.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        return (
            self.base.gibbs_energy(temperature)
            + self.phase.gibbs_energy(temperature)
            - self.reference.gibbs_energy(temperature)
        )

    @override
    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the thermodynamic data are valid.

        This is the range of the base phase.

        Returns:
            Minimum and maximum temperature in K
        """
        return self.base.temperature_range()


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
