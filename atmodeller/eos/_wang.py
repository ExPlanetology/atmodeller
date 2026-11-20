# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Real gas Virial EOS

The papers state a volume integration from :math:`P_0` to :math:`P`, where :math:`f(P_0=1)=1`.
Hence for bounded EOS a minimum pressure of 1 bar is assumed.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from atmodeller import override
from atmodeller.interfaces import CatalogueEntry, ModelInfo
from atmodeller.constants import STANDARD_PRESSURE
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import FloatArray, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration


class VirialQuadratic(RealGas):
    """A real gas EOS that implements the virial formulation

    Args:
        a_coefficients: :math:`a` coefficients
        b_coefficients: :math:`b` coefficients
        c_coefficients: :math:`c` coefficients
    """

    a_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`a` coefficients"""
    b_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`b` coefficients"""
    c_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """:math:`c` coefficients"""

    def _get_compressibility_coefficient(
        self, temperature: ArrayLike, coefficients: tuple[float, ...]
    ) -> FloatArray:
        """General form of the coefficients for the compressibility calculation

        Args:
            temperature: Temperature (K)
            coefficients: Tuple of the coefficients :math:`a`, :math:`b`, :math:`c`

        Returns
            The relevant coefficient
        """
        coefficient: FloatArray = (
            jnp.asarray(coefficients[0])
            + coefficients[1] * temperature
            + coefficients[2] * jnp.square(temperature)
        )

        return coefficient

    def a(self, temperature: ArrayLike) -> FloatArray:
        """:math:`a` parameter

        Args:
            temperature: Temperature (K)

        Returns:
            a parameter
        """
        return self._get_compressibility_coefficient(temperature, self.a_coefficients)

    def b(self, temperature: ArrayLike) -> FloatArray:
        """:math:`b` parameter

        Args:
            temperature: Temperature (K)

        Returns:
            b parameter
        """
        return self._get_compressibility_coefficient(temperature, self.b_coefficients)

    def c(self, temperature: ArrayLike) -> FloatArray:
        """:math:`c` parameter

        Args:
            temperature: Temperature (K)

        Returns:
            c parameter
        """
        return self._get_compressibility_coefficient(temperature, self.c_coefficients)

    @override
    def compressibility_factor(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Compressibility factor

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            The compressibility factor (dimensionless)
        """
        del mole_fractions

        Z: FloatArray = (
            self.a(temperature)
            + self.b(temperature) * pressure
            + self.c(temperature) * jnp.square(pressure)
        )

        return Z

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        log_fugacity: FloatArray = self.volume_integral(temperature, pressure, mole_fractions) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        Z: FloatArray = self.compressibility_factor(temperature, pressure, mole_fractions)
        volume_ideal: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        volume: FloatArray = Z * volume_ideal

        return volume

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Volume integral

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Volume integral (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        del mole_fractions

        volume_integral: FloatArray = (
            (
                self.a(temperature) * jnp.log(pressure / STANDARD_PRESSURE)
                + self.b(temperature) * (pressure - STANDARD_PRESSURE)
                + (1.0 / 2) * self.c(temperature) * (jnp.square(pressure) - STANDARD_PRESSURE**2)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )

        return volume_integral


experimental_calibration_wang18: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=1200, temperature_max=4100, pressure_min=1, pressure_max=1387e3
)
"""Calibration range of the :cite:t:`WLL18` shock data fitted by :cite:t:`Hakim2026`"""

# Quadratic virial EOS for H4Si fitted by Hakim et al. (2026, Table 2) to the shock data of Wang
# et al. (2018); the coefficients are converted from GPa to bar

H4Si_wang18: RealGas = VirialQuadratic(
    a_coefficients=(1.0, 0, 0),
    b_coefficients=(3.8552e-4, -1.822e-7, 2.54e-11),
    c_coefficients=(-1.942e-10, 1.088e-13, -1.62e-17),
)
"""H4Si quadratic virial fitted by :cite:t:`Hakim2026{Table 2}` to the data of :cite:t:`WLL18`"""

H4Si_wang18_bounded: RealGas = CombinedRealGas.create(
    [H4Si_wang18], [experimental_calibration_wang18]
)
"""H4Si quadratic virial bounded :cite:p:`Hakim2026,WLL18`"""


def get_wang_eos_catalogue() -> dict[str, CatalogueEntry[RealGas]]:
    """Gets a dictionary of virial EOS

    Returns:
        Dictionary of EOS models with their catalogue information
    """
    eos_models: dict[str, CatalogueEntry[RealGas]] = {}
    eos_models["H4Si_wang18"] = CatalogueEntry(
        H4Si_wang18_bounded, ModelInfo("H4Si", sources=("Hakim2026", "WLL18"))
    )

    return eos_models
