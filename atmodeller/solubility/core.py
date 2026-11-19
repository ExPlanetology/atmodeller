# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Core classes and functions for solubility laws"""

from abc import abstractmethod
from typing import ClassVar, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from atmodeller import override
from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.jax_utils import FloatArray, power_law
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer


class Solubility(eqx.Module):
    """Solubility interface"""

    requires_fO2: ClassVar[bool] = False
    """Whether the law depends on the oxygen fugacity, in which case a model that uses it must
    include O2 as a gas species. This is checked when the model is built."""

    @abstractmethod
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
        fO2: Optional[ArrayLike] = None,
    ) -> FloatArray:
        """Concentration in ppmw

        Args:
            fugacity: Fugacity (bar)
            temperature: Temperature (K). Defaults to ``None`` for not used.
            pressure: Pressure (bar). Defaults to ``None`` for not used.
            fO2: Oxygen fugacity (bar). Defaults to ``None`` for not used.

        Returns:
            Concentration in ppmw
        """

    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> FloatArray:
        """Wrapper to pass concentration arguments by position.

        This is also required for gradient calculations, which require all arguments to be passed
        by position.

        Args:
            fugacity: Fugacity (bar)
            temperature: Temperature (K)
            pressure: Pressure (bar)
            fO2: Oxygen fugacity (bar)

        Returns:
            Concentration in ppmw
        """
        return self.concentration(fugacity, temperature=temperature, pressure=pressure, fO2=fO2)


class NoSolubility(Solubility):
    """No solubility"""

    @override
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
        fO2: Optional[ArrayLike] = None,
    ) -> FloatArray:
        del fugacity
        del temperature
        del pressure
        del fO2

        return jnp.array(0.0)  # Must be a JAX array for type compatibility


class SolubilityPowerLaw(Solubility):
    """A solubility power law

    Args:
        constant: Constant
        exponent: Exponent
    """

    constant: float = eqx.field(converter=float)
    """Constant"""
    exponent: float = eqx.field(converter=float)
    """Exponent"""

    @override
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
        fO2: Optional[ArrayLike] = None,
    ) -> FloatArray:
        del temperature, pressure, fO2

        return power_law(fugacity, self.constant, self.exponent)


class SolubilityPowerLawLog10(Solubility):
    """A solubility power law with log10 coefficients

    Args:
        log10_constant: Log10 constant
        log10_exponent: Log10 exponent

    """

    log10_constant: float = eqx.field(converter=float)
    """Log10 constant"""
    log10_exponent: float = eqx.field(converter=float)
    """Log10 exponent"""

    @override
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
        fO2: Optional[ArrayLike] = None,
    ) -> FloatArray:
        del temperature, pressure, fO2

        return jnp.power(10, (self.log10_constant + self.log10_exponent * jnp.log10(fugacity)))


def fO2_temperature_correction(
    fO2: ArrayLike,
    *,
    temperature: ArrayLike,
    pressure: ArrayLike,
    reference_temperature: ArrayLike,
) -> FloatArray:
    r"""Applies a temperature correction to :math:`f\rm{O}_2`.

    Some experimentally derived solubility laws operate on absolute :math:`\log_{10} f\rm{O}_2`,
    which depends on temperature and pressure. A temperature correction has to be applied to
    maintain the same :math:`\log_{10} f\rm{O}_2` shift at arbitrary temperature.

    Args:
        fO2: Absolute oxygen fugacity at ``temperature`` (bar)
        temperature: Temperature (K)
        pressure: Absolute pressure (bar). The iron-wustite buffer used for the correction is
            evaluated at 1 bar by default, so this has no effect on the result.
        reference_temperature: Reference temperature, which is usually the temperature at which the
            experiment was performed.

    Returns:
        Oxygen fugacity (bar) at ``reference_temperature`` with the same :math:`\log_{10}
        f\rm{O}_2` shift from the iron-wustite buffer as ``fO2`` has at ``temperature``
    """
    iron_wustite_buffer: RedoxBufferProtocol = IronWustiteBuffer()
    logiw_fugacity_at_current_temp: ArrayLike = iron_wustite_buffer.log10_fugacity(
        temperature, pressure
    )
    fo2_shift: FloatArray = jnp.log10(fO2) - logiw_fugacity_at_current_temp

    logiw_fugacity_at_reference_temp: ArrayLike = iron_wustite_buffer.log10_fugacity(
        reference_temperature, pressure
    )
    adjusted_fo2: FloatArray = jnp.power(10, logiw_fugacity_at_reference_temp + fo2_shift)

    return adjusted_fo2
