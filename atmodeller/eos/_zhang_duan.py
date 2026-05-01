# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOS from :cite:t:`ZD09`"""

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW, VOLUME_EPSILON
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import OptxSolver, as_j64, safe_exp
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration, unit_conversion


class ZhangDuan(RealGas):
    r"""Real gas EOS :cite:p:`ZD09`

    Args:
        epsilon: Lenard-Jones parameter (epsilon/kB) in K
        sigma: Lenard-Jones parameter in :math:`10^{-10}` m
    """

    coefficients: ClassVar[tuple[float, ...]] = (
        2.95177298930e-2,
        -6.33756452413e3,
        -2.75265428882e5,
        1.29128089283e-3,
        -1.45797416153e2,
        7.65938947237e4,
        2.58661493537e-6,
        0.52126532146,
        -1.39839523753e2,
        -2.36335007175e-8,
        5.35026383543e-3,
        -0.27110649951,
        2.50387836486e4,
        0.73226726041,
        1.54833359970e-2,
    )
    """Coefficients"""

    epsilon: float = eqx.field(converter=float)
    """Lenard-Jones parameter (epsilon/kB) in K"""
    sigma: float = eqx.field(converter=float)
    r"""Lenard-Jones parameter in :math:`10^{-10}` m"""

    @eqx.filter_jit
    def _Pm(self, pressure: ArrayLike) -> Array:
        """Scaled pressure

        Args:
            pressure: Pressure in bar

        Returns:
            Scaled pressure
        """
        pressure_MPa: ArrayLike = pressure * unit_conversion.bar_to_MPa
        scaled_pressure: Array = 3.0636 * jnp.power(self.sigma, 3) * pressure_MPa / self.epsilon
        # jax.debug.print("scaled_pressure = {out}", out=scaled_pressure)

        return scaled_pressure

    @eqx.filter_jit
    def _Tm(self, temperature: ArrayLike) -> ArrayLike:
        """Scaled temperature

        Args:
            temperature: Temperature in K

        Returns:
            Scaled temperature
        """
        scaled_temperature: ArrayLike = 154 * temperature / self.epsilon
        # jax.debug.print("scaled_temperature = {out}", out=scaled_temperature)

        return scaled_temperature

    @eqx.filter_jit
    def _Vm(self, volume: ArrayLike) -> Array:
        r"""Scaled volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`

        Returns:
            Scaled volume
        """
        volume_cm3: ArrayLike = volume * unit_conversion.m3_to_cm3
        sigma_term: Array = jnp.power(self.sigma / 3.691, 3)
        scaled_volume: Array = volume_cm3 / 1000 / sigma_term  # type: ignore
        # jax.debug.print("scaled_volume = {out}", out=scaled_volume)

        return scaled_volume

    @eqx.filter_jit
    def _get_S1_parameter(self, Tm: ArrayLike, coefficients: tuple[float, ...]) -> Array:
        """Gets the parameter (coefficient) for the S1 term for polynomials involving Tm terms

        Args:
            Tm: Scaled temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return (
            coefficients[0] + coefficients[1] / jnp.square(Tm) + coefficients[2] / jnp.power(Tm, 3)
        )

    @eqx.filter_jit
    def _get_S2_parameter(self, Tm: ArrayLike, coefficients: tuple[float, ...]) -> Array:
        """Gets the parameter (coefficient) for the S2 term for polynomials involving Tm terms

        Args:
            Tm: Scaled temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return 2 * coefficients[1] / jnp.square(Tm) + 3 * coefficients[2] / jnp.power(Tm, 3)

    @eqx.filter_jit
    def _S1(self, Tm: ArrayLike, Vm: ArrayLike) -> Array:
        """S1 term :cite:p:`ZD09{Equation 15}`

        Args:
            Tm: Scaled temperature
            Vm: Scaled volume

        Returns:
            S1 term
        """
        b: Array = self._get_S1_parameter(Tm, self.coefficients[0:3])
        c: Array = self._get_S1_parameter(Tm, self.coefficients[3:6])
        d: Array = self._get_S1_parameter(Tm, self.coefficients[6:9])
        e: Array = self._get_S1_parameter(Tm, self.coefficients[9:12])
        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]

        S1: Array = (
            b / Vm
            + c / (2 * jnp.square(Vm))
            + d / (4 * jnp.power(Vm, 4))
            + e / (5 * jnp.power(Vm, 5))
        ) + (
            a13
            / (2 * a15 * jnp.power(Tm, 3))
            * (a14 + 1 - (a14 + 1 + a15 / jnp.square(Vm)) * safe_exp(-a15 / jnp.square(Vm)))
        )
        # jax.debug.print("S1 = {out}", out=S1)

        return S1

    @eqx.filter_jit
    def _S2(self, Tm: ArrayLike, Vm: ArrayLike) -> Array:
        """S2 term :cite:p:`ZD09{Equation 16}`

        Args:
            Tm: Scaled temperature
            Vm: Scaled volume

        Returns:
            S2 term
        """
        b: Array = self._get_S2_parameter(Tm, self.coefficients[0:3])
        c: Array = self._get_S2_parameter(Tm, self.coefficients[3:6])
        d: Array = self._get_S2_parameter(Tm, self.coefficients[6:9])
        e: Array = self._get_S2_parameter(Tm, self.coefficients[9:12])
        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]

        S2: Array = (
            b / Vm
            + c / (2 * jnp.square(Vm))
            + d / (4 * jnp.power(Vm, 4))
            + e / (5 * jnp.power(Vm, 5))
        ) + (
            3
            * a13
            / (2 * a15 * jnp.power(Tm, 3))
            * (a14 + 1 - (a14 + 1 + a15 / jnp.square(Vm)) * safe_exp(-a15 / jnp.square(Vm)))
        )
        # jax.debug.print("S2 = {out}", out=S2)

        return S2

    @eqx.filter_jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume :cite:p:`ZD09{Equation 8}`.

        Note that the left-hand side of :cite:t:`ZD09{Equation 8}` is the compressibility factor
        so should be expressed in terms of P, V, R, and T, and not the scaled equivalents.

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]
        # jax.debug.print("temperature = {temperature}", temperature=temperature)
        # jax.debug.print("pressure = {pressure}", pressure=pressure)

        Tm: ArrayLike = self._Tm(temperature)
        # jax.debug.print("Tm = {Tm}", Tm=Tm)
        Vm: ArrayLike = self._Vm(volume)
        # jax.debug.print("Vm = {Vm}", Vm=Vm)
        # Zhang and Duan (2009) use scaled quantities, according to the paper, but this is
        # presumably a typo and in fact the actual P and T should be used. This agrees with the
        # Perple_X implementation and the reported volumes in Table 6.
        ptr: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        # jax.debug.print("ptr = {ptr}", ptr=ptr)

        b: ArrayLike = self._get_S1_parameter(Tm, self.coefficients[0:3])
        # jax.debug.print("b = {b}", b=b)
        c: ArrayLike = self._get_S1_parameter(Tm, self.coefficients[3:6])
        # jax.debug.print("c = {c}", c=c)
        d: ArrayLike = self._get_S1_parameter(Tm, self.coefficients[6:9])
        # jax.debug.print("d = {d}", d=d)
        e: ArrayLike = self._get_S1_parameter(Tm, self.coefficients[9:12])
        # jax.debug.print("e = {e}", e=e)

        term1: Array = (
            as_j64(1)
            + b / as_j64(Vm)
            + c / jnp.power(Vm, 2)
            + d / jnp.power(Vm, 4)
            + e / jnp.power(Vm, 5)
        )
        # jax.debug.print("term1 = {term1}", term1=term1)

        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]
        term2: Array = (
            a13
            / jnp.power(Tm, 3)
            / jnp.power(Vm, 2)
            * (a14 + a15 / jnp.square(Vm))
            * safe_exp(-a15 / jnp.square(Vm))
        )
        # jax.debug.print("term2 = {term2}", term2=term2)

        residual: Array = term1 + term2 - ptr
        # jax.debug.print("residual = {residual}", residual=residual)

        return residual

    @eqx.filter_jit
    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Initial guess volume to ensure convergence to the correct root

        .. math::
            V = \frac{RT}{P} \left(1 + \frac{b}{V_m}\right)

        This is the ideal gas law with a correction for the b term based on truncating the full
        equation and substituting in the ideal gas volume. The 1/Vm**2 is not included because
        convergence did not improve with it, in fact, it got worse. Also, the initial volume is
        required to be positive, so the initial compressibility parameter is limited to a minimum
        of 0.1, which is guided by the behaviour of water to some extent.

        This initial guess will eventually fail for large pressures, leaving room for an improved
        approach.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Initial volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        safe_volume: ArrayLike = ideal_volume + VOLUME_EPSILON
        Tm: ArrayLike = self._Tm(temperature)
        Vm: Array = self._Vm(safe_volume)
        b: Array = self._get_S1_parameter(Tm, self.coefficients[0:3])

        compressibility_factor: Array = 1 + b / Vm
        # TODO: This is ad-hoc, but works for now.
        minimum_compressibility_factor: float = 0.1
        compressibility_factor = jnp.where(
            compressibility_factor < minimum_compressibility_factor,
            minimum_compressibility_factor,
            compressibility_factor,
        )
        initial_volume: Array = compressibility_factor * ideal_volume
        # jax.debug.print("initial_volume = {out}", out=initial_volume)

        return initial_volume

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        initial: Array = self.initial_volume(temperature, pressure)
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(self._objective_function, solver, initial, args=kwargs, throw=THROW)
        volume: ArrayLike = sol.value
        # jax.debug.print("volume = {out}", out=volume)
        # jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

        # For comparing the initial and final volumes to refine the choice of the initial volume
        # jax.debug.print("initial_volume = {out}", out=initial)
        # jax.debug.print("final_volume = {out}", out=volume)
        # relative_volume_error: Array = (initial - volume) / volume
        # jax.debug.print("Relative volume error = {out}", out=relative_volume_error)

        return volume

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity :cite:p:`ZD09{Equation 14}`

        This is for a pure species and does not include the terms to enable end member mixing.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        Vm: Array = self._Vm(volume)
        Tm: ArrayLike = self._Tm(temperature)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: Array = -jnp.log(Z) + self._S1(Tm, Vm) + Z - 1
        log_fugacity: Array = log_fugacity_coefficient + jnp.log(pressure)
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        return log_fugacity

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        log_fugacity: Array = self.log_fugacity(temperature, pressure)
        volume_integral: Array = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral


CH4_zhang09: RealGas = ZhangDuan(154, 3.691)
"""CH4 unbounded :cite:p:`ZD09`"""
CH4_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=273,
    temperature_max=2573,
    pressure_min=0.1 * unit_conversion.MPa_to_bar,
    pressure_max=10000 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for CH4 :cite:p:`ZD09{Table 5}`"""
CH4_zhang09_bounded: RealGas = CombinedRealGas.create(
    [CH4_zhang09], [CH4_experimental_calibration]
)
"""CH4 bounded to data range :cite:p:`ZD09{Table 5}`"""

H2O_zhang09: RealGas = ZhangDuan(510, 2.88)
"""H2O unbounded :cite:p:`ZD09`"""
H2O_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=673,
    temperature_max=2573,
    pressure_min=0.1 * unit_conversion.MPa_to_bar,
    pressure_max=10000 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for H2O :cite:p:`ZD09{Table 5}`"""
H2O_zhang09_bounded: RealGas = CombinedRealGas.create(
    [H2O_zhang09], [H2O_experimental_calibration]
)
"""H2O bounded to data range :cite:p:`ZD09{Table 5}`"""

CO2_zhang09: RealGas = ZhangDuan(235, 3.79)
"""CO2 unbounded :cite:p:`ZD09`"""
CO2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=473,
    temperature_max=2573,
    pressure_min=0.1 * unit_conversion.MPa_to_bar,
    pressure_max=10000 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for CO2 :cite:p:`ZD09{Table 5}`"""
CO2_zhang09_bounded: RealGas = CombinedRealGas.create(
    [CO2_zhang09], [CO2_experimental_calibration]
)

# Tested boundedness (not the same as physical correctness) for 500<T<10000 K and 0<P<10 GPa
H2_zhang09: RealGas = ZhangDuan(31.2, 2.93)
"""H2 unbounded :cite:p:`ZD09`"""
H2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=250,
    temperature_max=423,
    pressure_min=2 * unit_conversion.MPa_to_bar,
    pressure_max=700 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for H2 :cite:p:`ZD09{Table 5}`"""
H2_zhang09_bounded: RealGas = CombinedRealGas.create([H2_zhang09], [H2_experimental_calibration])
"""H2 bounded to data range :cite:p:`ZD09{Table 5}`"""

CO_zhang09: RealGas = ZhangDuan(105.6, 3.66)
"""CO unbounded :cite:p:`ZD09`"""
CO_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=300,
    temperature_max=573.2,
    pressure_min=10 * unit_conversion.MPa_to_bar,
    pressure_max=1020.6 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for CO :cite:p:`ZD09{Table 5}`"""
CO_zhang09_bounded: RealGas = CombinedRealGas.create([CO_zhang09], [CO_experimental_calibration])
"""CO bounded to data range :cite:p:`ZD09{Table 5}`"""

O2_zhang09: RealGas = ZhangDuan(124.5, 3.36)
"""O2 unbounded :cite:p:`ZD09`"""
O2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=300,
    temperature_max=1000,
    pressure_min=7 * unit_conversion.MPa_to_bar,
    pressure_max=1013.2 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for O2 :cite:p:`ZD09{Table 5}`"""
O2_zhang09_bounded: RealGas = CombinedRealGas.create([O2_zhang09], [O2_experimental_calibration])
"""O2 bounded to data range :cite:p:`ZD09{Table 5}`"""

C2H6_zhang09: RealGas = ZhangDuan(246.1, 4.35)
"""C2H6 unbounded :cite:p:`ZD09`"""
C2H6_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=373,
    temperature_max=673,
    pressure_min=30 * unit_conversion.MPa_to_bar,
    pressure_max=900 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for C2H6 :cite:p:`ZD09{Table 5}`"""
C2H6_zhang09_bounded: RealGas = CombinedRealGas.create(
    [C2H6_zhang09], [C2H6_experimental_calibration]
)
"""C2H6 bounded to data range :cite:p:`ZD09{Table 5}`"""


def get_zhang_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of Zhang and Duan EOS models.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["CH4_zhang09"] = CH4_zhang09_bounded
    # eos_models["CH4_zhang09_unbounded"] = CH4_zhang09
    eos_models["H2O_zhang09"] = H2O_zhang09_bounded
    # eos_models["H2O_zhang09_unbounded"] = H2O_zhang09
    eos_models["CO2_zhang09"] = CO2_zhang09_bounded
    # eos_models["CO2_zhang09_unbounded"] = CO2_zhang09
    eos_models["H2_zhang09"] = H2_zhang09_bounded
    # eos_models["H2_zhang09_unbounded"] = H2_zhang09
    eos_models["CO_zhang09"] = CO_zhang09_bounded
    # eos_models["CO_zhang09_unbounded"] = CO2_zhang09
    eos_models["O2_zhang09"] = O2_zhang09_bounded
    # eos_models["O2_zhang09_unbounded"] = O2_zhang09
    eos_models["C2H6_zhang09"] = C2H6_zhang09_bounded
    # eos_models["C2H6_zhang09_unbounded"] = C2H6_zhang09

    return eos_models
