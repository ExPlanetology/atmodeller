# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Real gas EOS from :cite:t:`CD21`"""

import importlib.resources
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import ArrayLike, Float
from molmass import Formula

from atmodeller import override
from atmodeller.constants import STANDARD_PRESSURE
from atmodeller.eos import DATA_DIRECTORY
from atmodeller.eos._aggregators import CombinedRealGas, CombinedRealGasFugacity
from atmodeller.eos.core import IdealGas, RealGas, RealGasBase
from atmodeller.jax_utils import FloatArray, as_j64, to_native_floats
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration, unit_conversion

MIN_LOG_FUGACITY_COEFFICIENT: float = -15.0
"""Minimum log fugacity coefficient to avoid numerical issues when solving with the Chabrier EOS"""
MAX_LOG_FUGACITY_COEFFICIENT: float = 15.0
"""Maximum log fugacity coefficient to avoid numerical issues when solving with the Chabrier EOS"""


class Chabrier(RealGas):
    r"""Chabrier EOS from :cite:t:`CD21`

    This uses rho-T-P tables to lookup density (rho), from which the log fugacity is integrated
    from 1 bar.

    .. note::
        The tabulated densities are not reliable in two ranges, which the bounded versions do not
        exclude:

        - At very low densities, roughly :math:`P/T \lesssim 10^{-7}` bar/K (below about
          :math:`10^{-3}` bar at :math:`10^4` K and :math:`10^{-5}` bar at 300 K), the densities
          are erroneous; for He the compressibility factor ranges from 0.05 to 5 instead of 1.
        - At high pressures and low temperatures the densities jump to unphysical values, with
          compressibility factors of up to about :math:`10^8` and log fugacities of up to about
          :math:`10^{10}`: for :math:`\mathrm{H}_2` between about 460 and 780 K above about
          :math:`10^5` bar, and for the :math:`\mathrm{H}_2`-He mixtures between about 270 and
          380 K above about :math:`2.5 \times 10^6` bar. He is unaffected.

    Args:
        log10_density_func: Spline lookup for density from :cite:t:`CD21` T-P-rho tables
        He_fraction: He mass fraction (Y)
        H2_molar_mass_g_mol: Molar mass of :math:`\mathrm{H}_2`
        He_molar_mass_g_mol: Molar mass of He
        integration_steps: Number of pressure nodes for Simpson integration (odd)
    """

    CHABRIER_DIRECTORY: ClassVar[Path] = Path("chabrier")
    """Directory of the Chabrier data within :obj:`~atmodeller.eos.data`"""
    log10_density_func: Callable
    """Spline lookup for density from :cite:t:`CD21` T-P-rho tables"""
    He_fraction: float = eqx.field(converter=float)
    """He mass fraction (Y)"""
    H2_molar_mass_g_mol: float = eqx.field(converter=float)
    r"""Molar mass of :math:`\mathrm{H}_2`"""
    He_molar_mass_g_mol: float = eqx.field(converter=float)
    """Molar mass of He"""
    integration_steps: int
    """Number of pressure nodes for Simpson integration (odd)"""

    @classmethod
    def create(cls, filename: Path, integration_steps: int = 101) -> RealGas:
        r"""Creates a Chabrier instance.

        Args:
            filename: Filename of the density-T-P data
            integration_steps: Number of pressure nodes for Simpson integration, which must be
                odd (an even value is increased by one). Defaults to ``101``, which computes the
                log fugacity to within 1.2e-3 for T from 500 to 10000 K and pressure to 10 GPa,
                and to within 3e-3 for T from 1000 to 10000 K and pressure to 100 GPa, relative to
                a converged integration. The error does not decrease monotonically with the number
                of nodes below about 150 because the tabulated density is linearly interpolated.

        Returns:
            An instance
        """
        log10_density_func: Callable = cls._get_interpolator(filename)
        He_fraction: float = cls.get_He_fraction_map()[filename.name]
        H2_molar_mass_g_mol: float = Formula("H2").mass
        He_molar_mass_g_mol: float = Formula("He").mass

        return cls(
            log10_density_func,
            He_fraction,
            H2_molar_mass_g_mol,
            He_molar_mass_g_mol,
            integration_steps,
        )

    @classmethod
    def _get_interpolator(cls, filename: Path) -> Callable:
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
                datapath,
                sep=r"\s+",
                comment="#",
                usecols=[0, 1, 2],  # type: ignore
                names=column_names,
                skiprows=2,
            )
        pivot_table: pd.DataFrame = df.pivot(index=T_name, columns=P_name, values=rho_name)
        log_T: FloatArray = jnp.array(pivot_table.index.to_numpy())
        log_P: FloatArray = jnp.array(pivot_table.columns.to_numpy())
        log_rho: FloatArray = jnp.array(pivot_table.to_numpy())

        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (log_T, log_P), log_rho, method="linear"
        )

        def interpolator_hashable_function_wrapper(x) -> FloatArray:
            """Converts interpolator to a hashable function"""
            return interpolator(x)

        return interpolator_hashable_function_wrapper

    def _convert_to_molar_density(self, log10_density_gcc: ArrayLike) -> FloatArray:
        r"""Converts density to molar density

        Args:
            log10_density_gcc: Log10 density (:math:`\mathrm{g}\ \mathrm{cm}^{-3}`)

        Returns:
            Molar density (:math:`\mathrm{mol}\ \mathrm{m}^{-3}`)
        """
        molar_density: FloatArray = jnp.power(10, log10_density_gcc) / unit_conversion.cm3_to_m3
        # He_fraction is a mass fraction, so the mean molar mass is the mass-weighted harmonic mean
        mean_molar_mass_g_mol: float = 1.0 / (
            self.He_fraction / self.He_molar_mass_g_mol
            + (1 - self.He_fraction) / self.H2_molar_mass_g_mol
        )
        molar_density = molar_density / mean_molar_mass_g_mol

        return molar_density

    @staticmethod
    def get_He_fraction_map() -> dict[str, float]:
        """Mass fraction of He (Y) in the gas mixture, the other component being hydrogen.

        Y is the helium mass fraction as defined by :cite:t:`CD21`, not a mole fraction.
        Dictionary keys should correspond to the name of the Chabrier file.
        """
        He_fraction_map: dict[str, float] = {
            "TABLE_H_TP_v1": 0.0,
            "TABLE_HE_TP_v1": 1.0,
            "TABLEEOS_2021_TP_Y0275_v1": 0.275,
            "TABLEEOS_2021_TP_Y0292_v1": 0.292,
            "TABLEEOS_2021_TP_Y0297_v1": 0.297,
        }

        return He_fraction_map

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Log fugacity

        Computed as :math:`\ln f = \ln P + \int_{P_0}^{P} (Z - 1)\, d\ln P` with
        :math:`P_0 = 1` bar, using Simpson's rule on nodes evenly spaced in :math:`\ln P`. Writing
        the integrand as :math:`Z - 1` treats the ideal-gas part exactly. An obvious speedup to
        implement is to precompute the fugacity (integral), either by calculating it during
        initialisation or by storing it in a lookup table that is read in.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log fugacity (bar)
        """
        del mole_fractions

        # pressure = eqx.error_if(
        #    jnp.asarray(pressure), jnp.asarray(pressure) <= 0, "pressure must be positive"
        # )
        temperature = as_j64(temperature)
        log10_pressure: FloatArray = jnp.log10(pressure)
        temperature, log10_pressure = jnp.broadcast_arrays(temperature, log10_pressure)

        # Pressure nodes evenly spaced in ln P from the standard pressure to the pressure. Simpson's
        # rule requires an odd number of nodes, so an even value is increased by one.
        num: int = self.integration_steps + 1 - self.integration_steps % 2
        pressures: FloatArray = jnp.logspace(jnp.log10(STANDARD_PRESSURE), log10_pressure, num=num)
        dlnP: FloatArray = (
            (log10_pressure - jnp.log10(STANDARD_PRESSURE)) * jnp.log(10) / (num - 1)
        )

        volumes: FloatArray = self.volume(temperature, pressures)
        integrand: FloatArray = pressures * volumes / (GAS_CONSTANT_BAR * temperature) - 1.0

        # Simpson's rule weights 1, 4, 2, 4, ..., 2, 4, 1 along the node axis
        weights: FloatArray = jnp.ones(num).at[1:-1:2].set(4.0).at[2:-1:2].set(2.0)
        weights = weights.reshape((num,) + (1,) * (integrand.ndim - 1))
        integral: FloatArray = dlnP / 3.0 * jnp.sum(weights * integrand, axis=0)

        log_fugacity: FloatArray = log10_pressure * jnp.log(10) + integral

        return log_fugacity

    @override
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions

        log10_density_gcc: FloatArray = self.log10_density_func(
            (jnp.log10(temperature), jnp.log10(unit_conversion.bar_to_GPa * pressure))
        )
        # jax.debug.print("log10_density_gcc = {out}", out=log10_density_gcc)
        molar_density: FloatArray = self._convert_to_molar_density(log10_density_gcc)
        volume: FloatArray = jnp.reciprocal(molar_density)
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    def volume_integral(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions

        log_fugacity: FloatArray = self.log_fugacity(temperature, pressure)
        volume_integral: FloatArray = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral


class ChabrierFunction(RealGasBase):
    r"""Chabrier EOS from :cite:t:`CD21` using a function fit.

    This provides a faster alternative to the full Chabrier EOS, which integrates the tabulated
    volume. The fits at 3000 K and 4000 K are fitted to the log fugacity coefficient of
    :data:`H2_chabrier21` from 1 bar to :math:`10^6` bar (100 GPa), and reproduce it to within
    about 0.15 over that range. This is well within the range of typical atmospheres considered
    for planetary applications. Above :math:`10^6` bar the fits diverge from the table.

    Args:
        coeffs: Coefficients for the log fugacity coefficient function
    """

    coeffs: tuple[float, ...] = eqx.field(converter=to_native_floats)

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        return self.log_fugacity_coefficient(temperature, pressure, mole_fractions) + jnp.log(
            pressure
        )

    @override
    def log_fugacity_coefficient(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        """Log fugacity coefficient

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless). Ignored by
                default for pure-phase EOSs; may be used by overriding subclasses. Defaults to
                ``None``.

        Returns:
            Log fugacity coefficient, which is dimensionless
        """
        del mole_fractions

        _, logP = jnp.broadcast_arrays(jnp.log(temperature), jnp.log(pressure))
        c: Float[FloatArray, " coeffs"] = jnp.array(self.coeffs)

        # The chosen functional form is empirical, but by construction recovers a unity fugacity
        # coefficient at 1 bar. The c[0]*exp(c[1]*logP) term captures the overall exponential
        # behaviour, while the c[2] + c[3]*logP terms provide additional flexibility to fit the data
        # more accurately before the exponential term dominates at higher pressures.
        # The terms are calculated and summed term-by-term to ensure broadcasting works correctly.
        log_fugacity_coefficient: FloatArray = logP * (
            c[0] * jnp.exp(c[1] * logP) + c[2] + c[3] * logP
        )

        return log_fugacity_coefficient


_H2_3000K_chabrier21: RealGasBase = ChabrierFunction(
    coeffs=((0.000796975857387194, 0.537830199470527, 0.102923173815428, -0.0162512662920581))
)
"""Function-fit version at 3000 K for H2 Chabrier EOS between 1 bar and 1e6 bar :cite:p:`CD21`"""

ideal_gas: RealGasBase = IdealGas()
H2_3000K_chabrier21: RealGasBase = CombinedRealGasFugacity(
    (ideal_gas, _H2_3000K_chabrier21),
    (
        ExperimentalCalibration(pressure_max=1),
        ExperimentalCalibration(
            temperature_min=3000, temperature_max=3000, pressure_min=1, pressure_max=8e5
        ),
    ),
    min_log_fugacity_coefficient=MIN_LOG_FUGACITY_COEFFICIENT,
    max_log_fugacity_coefficient=MAX_LOG_FUGACITY_COEFFICIENT,
)
"""H2 Chabrier EOS at 3000 K between 1 bar and 8e5 bar (80 GPa) :cite:p:`CD21`

The fit is accurate to 1e6 bar, but its log fugacity coefficient reaches the maximum of
:data:`MAX_LOG_FUGACITY_COEFFICIENT` at about 8.3e5 bar, above which it is capped.
"""

_H2_4000K_chabrier21: RealGasBase = ChabrierFunction(
    coeffs=((0.00179360295945961, 0.467251813716831, 0.49726658673974, -0.0484259966686226))
)
"""Function-fit version at 4000 K for H2 Chabrier EOS between 1 bar and 1e6 bar :cite:p:`CD21`"""

H2_4000K_chabrier21: RealGasBase = CombinedRealGasFugacity(
    (ideal_gas, _H2_4000K_chabrier21),
    (
        ExperimentalCalibration(pressure_max=1),
        ExperimentalCalibration(
            temperature_min=4000, temperature_max=4000, pressure_min=1, pressure_max=1e6
        ),
    ),
    min_log_fugacity_coefficient=MIN_LOG_FUGACITY_COEFFICIENT,
    max_log_fugacity_coefficient=MAX_LOG_FUGACITY_COEFFICIENT,
)
"""H2 Chabrier EOS at 4000 K between 1 bar and 1e6 bar (100 GPa) :cite:p:`CD21`"""

calibration_chabrier21: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=100, temperature_max=1.0e8, pressure_max=1.0e17
)
"""Calibration for :cite:t:`CD21`"""
H2_chabrier21: RealGas = Chabrier.create(Path("TABLE_H_TP_v1"))
r""":math:`\mathrm{H}_2` :cite:p:`CD21`"""
H2_chabrier21_bounded: RealGas = CombinedRealGas.create([H2_chabrier21], [calibration_chabrier21])
r""":math:`\mathrm{H}_2` bounded :cite:p:`CD21`"""
He_chabrier21: RealGas = Chabrier.create(Path("TABLE_HE_TP_v1"))
"""He :cite:p:`CD21`"""
He_chabrier21_bounded: RealGas = CombinedRealGas.create([He_chabrier21], [calibration_chabrier21])
"""He bounded :cite:p:`CD21`"""
H2_He_Y0275_chabrier21: RealGas = Chabrier.create(Path("TABLEEOS_2021_TP_Y0275_v1"))
"""H2HeY0275 :cite:p:`CD21`"""
H2_He_Y0275_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_He_Y0275_chabrier21], [calibration_chabrier21]
)
"""H2HeY0275 bounded :cite:p:`CD21`"""
H2_He_Y0292_chabrier21: RealGas = Chabrier.create(Path("TABLEEOS_2021_TP_Y0292_v1"))
"""H2HeY0292 :cite:p:`CD21`"""
H2_He_Y0292_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_He_Y0292_chabrier21], [calibration_chabrier21]
)
"""H2HeY0292 bounded :cite:p:`CD21`"""
H2_He_Y0297_chabrier21: RealGas = Chabrier.create(Path("TABLEEOS_2021_TP_Y0297_v1"))
"""H2HeY0297 :cite:p:`CD21`"""
H2_He_Y0297_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_He_Y0297_chabrier21], [calibration_chabrier21]
)
"""H2HeY0297 bounded :cite:p:`CD21`"""


def get_chabrier_eos_models() -> dict[str, RealGasBase]:
    """Gets a dictionary of EOS models

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGasBase] = {}
    eos_models["H2_chabrier21"] = H2_chabrier21_bounded
    eos_models["H2_3000K_chabrier21"] = H2_3000K_chabrier21
    eos_models["H2_4000K_chabrier21"] = H2_4000K_chabrier21
    eos_models["H2_He_Y0275_chabrier21"] = H2_He_Y0275_chabrier21_bounded
    eos_models["H2_He_Y0292_chabrier21"] = H2_He_Y0292_chabrier21_bounded
    eos_models["H2_He_Y0297_chabrier21"] = H2_He_Y0297_chabrier21_bounded
    eos_models["He_chabrier21"] = He_chabrier21_bounded

    return eos_models
