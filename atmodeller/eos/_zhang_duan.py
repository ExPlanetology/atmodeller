# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOS from :cite:t:`ZD09`"""

from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from atmodeller import override
from atmodeller.constants import STANDARD_FUGACITY
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW, VOLUME_EPSILON
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import FloatArray, NpFloat, OptxSolver, as_j64, safe_exp
from atmodeller.sci_utils import GAS_CONSTANT_BAR, ExperimentalCalibration, unit_conversion

epsilon_species: dict[str, float] = {
    "CH4": 154.0,
    "H2O": 510.0,
    "CO2": 235.0,
    "H2": 31.2,
    "CO": 105.6,
    "O2": 124.5,
    "C2H6": 246.1,
}
"""Epsilon values for each species (K) :cite:p:`ZD09{Table 4}`. Ensure these use Hill notation."""
sigma_species: dict[str, float] = {
    "CH4": 3.691,
    "H2O": 2.88,
    "CO2": 3.79,
    "H2": 2.93,
    "CO": 3.66,
    "O2": 3.36,
    "C2H6": 4.35,
}
r"""sigma values for each species (10\ :sup:`-10` m) :cite:p:`ZD09{Table 4}`. Ensure these use Hill
notation."""

REFERENCE_EPSILON: float = epsilon_species["CH4"]
"""Reference epsilon"""
REFERENCE_SIGMA: float = sigma_species["CH4"]
"""Reference sigma"""


class ZhangDuanBase(RealGas):
    """A real gas EOS from :cite:t:`ZD09`"""

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

    @abstractmethod
    def get_epsilon(self, mole_fractions: Optional[ArrayLike] = None) -> ArrayLike:
        """Gets epsilon.

        Args:
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Epsilon (K)
        """

    @abstractmethod
    def get_sigma(self, mole_fractions: Optional[ArrayLike] = None) -> ArrayLike:
        """Gets sigma.

        Args:
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Sigma (:math:`10^{-10}` m)
        """

    def reduced_pressure(
        self, pressure: ArrayLike, mole_fractions: Optional[ArrayLike] = None
    ) -> FloatArray:
        """Reduced pressure

        Args:
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Reduced pressure (dimensionless)
        """
        sigma: ArrayLike = self.get_sigma(mole_fractions)
        epsilon: ArrayLike = self.get_epsilon(mole_fractions)
        pressure_MPa: ArrayLike = pressure * unit_conversion.bar_to_MPa
        # Zhang and Duan (2009) give 3.0636, but we can calculate the exact value as follows:
        factor: float = epsilon_species["CH4"] / sigma_species["CH4"] ** 3
        reduced_pressure: FloatArray = factor * jnp.power(sigma, 3) * pressure_MPa / epsilon
        # jax.debug.print("reduced_pressure = {out}", out=reduced_pressure)

        return reduced_pressure

    def reduced_temperature(
        self, temperature: ArrayLike, mole_fractions: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """Reduced temperature

        Args:
            temperature: Temperature (K)
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Reduced temperature (dimensionless)
        """
        epsilon: ArrayLike = self.get_epsilon(mole_fractions)
        reduced_temperature: ArrayLike = REFERENCE_EPSILON * temperature / epsilon
        # jax.debug.print("reduced_temperature = {out}", out=reduced_temperature)

        return reduced_temperature

    def reduced_volume(
        self, volume: ArrayLike, mole_fractions: Optional[ArrayLike] = None
    ) -> FloatArray:
        r"""Reduced volume

        Args:
            volume: Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Reduced volume (dimensionless)
        """
        sigma: ArrayLike = self.get_sigma(mole_fractions)
        volume_cm3: ArrayLike = volume * unit_conversion.m3_to_cm3
        sigma_term: FloatArray = jnp.power(sigma / REFERENCE_SIGMA, 3)
        reduced_volume: FloatArray = volume_cm3 / 1000 / sigma_term  # type: ignore
        # jax.debug.print("reduced_volume = {out}", out=reduced_volume)

        return reduced_volume

    def _get_S1_parameter(self, Tr: ArrayLike, coefficients: tuple[float, ...]) -> FloatArray:
        """Gets the parameter (coefficient) for the S1 term for polynomials involving Tr terms.

        Args:
            Tr: Reduced temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return (
            coefficients[0] + coefficients[1] / jnp.square(Tr) + coefficients[2] / jnp.power(Tr, 3)
        )

    def get_S1(self, Tr: ArrayLike, Vr: ArrayLike) -> FloatArray:
        """Computes the S1 term :cite:p:`ZD09{Equation 15}`.

        Args:
            Tr: Reduced temperature
            Vr: Reduced volume

        Returns:
            S1 term
        """
        b: FloatArray = self._get_S1_parameter(Tr, self.coefficients[0:3])
        c: FloatArray = self._get_S1_parameter(Tr, self.coefficients[3:6])
        d: FloatArray = self._get_S1_parameter(Tr, self.coefficients[6:9])
        e: FloatArray = self._get_S1_parameter(Tr, self.coefficients[9:12])
        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]

        S1_term: FloatArray = (
            b / Vr
            + c / (2 * jnp.square(Vr))
            + d / (4 * jnp.power(Vr, 4))
            + e / (5 * jnp.power(Vr, 5))
        ) + (
            a13
            / (2 * a15 * jnp.power(Tr, 3))
            * (a14 + 1 - (a14 + 1 + a15 / jnp.square(Vr)) * safe_exp(-a15 / jnp.square(Vr)))
        )
        # jax.debug.print("S1_term = {out}", out=S1_term)

        return S1_term

    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> FloatArray:
        r"""Objective function to solve for the volume :cite:p:`ZD09{Equation 8}`.

        Note that the left-hand side of :cite:t:`ZD09{Equation 8}` is the compressibility factor,
        which can be expressed in terms of P, V, R, and T. If the scaled equivalents are used
        instead, care should be taken to ensure that the 1000 scaling factor is accounted for.

        Args:
            volume: Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
            kwargs: Dictionary with other required and optional parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]
        mole_fractions: Optional[ArrayLike] = kwargs.get("mole_fractions", None)
        # jax.debug.print("temperature = {temperature}", temperature=temperature)
        # jax.debug.print("pressure = {pressure}", pressure=pressure)
        # jax.debug.print("mole_fractions = {mole_fractions}", mole_fractions=mole_fractions)

        Tm: ArrayLike = self.reduced_temperature(temperature, mole_fractions)
        # jax.debug.print("Tm = {Tm}", Tm=Tm)
        Vm: FloatArray = self.reduced_volume(volume, mole_fractions)
        # jax.debug.print("Vm = {Vm}", Vm=Vm)
        ptr: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        # jax.debug.print("ptr = {ptr}", ptr=ptr)

        b: FloatArray = self._get_S1_parameter(Tm, self.coefficients[0:3])
        # jax.debug.print("b = {b}", b=b)
        c: FloatArray = self._get_S1_parameter(Tm, self.coefficients[3:6])
        # jax.debug.print("c = {c}", c=c)
        d: FloatArray = self._get_S1_parameter(Tm, self.coefficients[6:9])
        # jax.debug.print("d = {d}", d=d)
        e: FloatArray = self._get_S1_parameter(Tm, self.coefficients[9:12])
        # jax.debug.print("e = {e}", e=e)

        term1: FloatArray = (
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
        term2: FloatArray = (
            a13
            / jnp.power(Tm, 3)
            / jnp.power(Vm, 2)
            * (a14 + a15 / jnp.square(Vm))
            * safe_exp(-a15 / jnp.square(Vm))
        )
        # jax.debug.print("term2 = {term2}", term2=term2)

        residual: FloatArray = term1 + term2 - ptr
        # jax.debug.print("residual = {residual}", residual=residual)

        return residual

    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        r"""Initial guess volume is the ideal gas volume plus a small epsilon

        The factor of 10 biases the solver towards the largest (gas phase) root, otherwise an
        incorrect root may be found. But this does not guarantee that the correct root will always
        be found.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Initial volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        safe_volume: FloatArray = as_j64(ideal_volume + VOLUME_EPSILON)
        # jax.debug.print("initial_volume = {out}", out=safe_volume)

        return safe_volume * 10

    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Optional[ArrayLike] = None,
    ) -> FloatArray:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
        """
        initial: FloatArray = self.initial_volume(temperature, pressure)
        kwargs: dict[str, Optional[ArrayLike]] = {
            "temperature": temperature,
            "pressure": pressure,
            "mole_fractions": mole_fractions,
        }

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(self._objective_function, solver, initial, args=kwargs, throw=THROW)
        volume: FloatArray = sol.value
        # jax.debug.print("volume = {out}", out=volume)
        # jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

        # For comparing the initial and final volumes to refine the choice of the initial volume
        # jax.debug.print("initial_volume = {out}", out=initial)
        # jax.debug.print("final_volume = {out}", out=volume)
        # relative_volume_error: Array = (initial - volume) / volume
        # jax.debug.print("Relative volume error = {out}", out=relative_volume_error)

        return volume

    def volume_integral(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Optional[ArrayLike] = None,
    ) -> FloatArray:
        r"""Volume integral

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Ignored for pure fluids. Defaults to ``None``.

        Returns:
            Volume integral (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        log_fugacity: FloatArray = self.log_fugacity(temperature, pressure)
        volume_integral: FloatArray = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral


class ZhangDuanPureFluid(ZhangDuanBase):
    """Real gas EOS for a pure fluid :cite:p:`ZD09`

    Args:
        epsilon: Lenard-Jones parameter (epsilon/kB) in K
        sigma: Lenard-Jones parameter (:math:`10^{-10}` m)
    """

    epsilon: float = eqx.field(converter=float)
    """Lenard-Jones parameter (epsilon/kB) in K"""
    sigma: float = eqx.field(converter=float)
    """Lenard-Jones parameter (:math:`10^{-10}` m)"""

    @override
    def get_epsilon(self, mole_fractions: Optional[ArrayLike] = None) -> ArrayLike:
        del mole_fractions
        return as_j64(self.epsilon)

    @override
    def get_sigma(self, mole_fractions: Optional[ArrayLike] = None) -> ArrayLike:
        del mole_fractions
        return as_j64(self.sigma)

    @override
    def log_fugacity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Optional[ArrayLike] = None,
    ) -> FloatArray:
        """Log fugacity :cite:p:`ZD09{Equation 14}`

        This is for a pure fluid and does not include the terms to enable end member mixing.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Ignored for pure fluids. Defaults to ``None``.

        Returns:
            Log fugacity in bar
        """
        volume: FloatArray = self.volume(temperature, pressure)
        Vm: FloatArray = self.reduced_volume(volume)
        Tm: ArrayLike = self.reduced_temperature(temperature)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: FloatArray = -jnp.log(Z) + self.get_S1(Tm, Vm) + Z - 1
        log_fugacity: FloatArray = log_fugacity_coefficient + jnp.log(pressure)
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        return log_fugacity


class ZhangDuanMixture(ZhangDuanBase):
    """Zhang and Duan EOS for mixtures :cite:p:`ZD09`"""

    species: tuple[str, ...]
    k1: NpFloat  # (n_species, n_species)
    k2: NpFloat  # (n_species, n_species)
    epsilon_matrix: NpFloat  # (n_species, n_species)
    sigma_matrix: NpFloat  # (n_species, n_species)
    species_index: int
    _grad_fn: Callable

    @override
    def __init__(self, species: tuple[str, ...], species_to_output: str | None = None):
        """Initializes the mixture model.

        Args:
            species: Tuple of species names
            species_to_output: Name of the species for which to output species-specific quantities,
                such as fugacity. Defaults to ``None``.
        """
        self.species = species
        self.k1 = self.get_k1_mixing_matrix(species)
        self.k2 = self.get_k2_mixing_matrix(species)
        self.epsilon_matrix = self.get_epsilon_berthelot_rule(species)
        self.sigma_matrix = self.get_sigma_lorentz_rule(species)
        # TODO: Hacky
        self.species_index = (
            species.index(species_to_output) if species_to_output is not None else 0
        )
        # print("species_index:", self.species_index)
        self._grad_fn = eqx.filter_grad(self.wrap_log_fugacity_bulk)

    @override
    def get_epsilon(self, mole_fractions: Float[Array, "... n_species"]) -> FloatArray:
        """Gets epsilon for a mixture using the Berthelot mixing rule.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Epsilon for the mixture
        """
        return self.binary_mixing_rule(mole_fractions, self.k1, self.epsilon_matrix)

    @override
    def get_sigma(self, mole_fractions: Float[Array, "... n_species"]) -> FloatArray:
        """Gets sigma for a mixture using the Lorentz mixing rule.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Sigma for the mixture
        """
        return self.binary_mixing_rule(mole_fractions, self.k2, self.sigma_matrix)

    def get_species_epsilon(
        self, mole_fractions: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Gets the epsilon values for each species in the mixture using the Berthelot mixing rule.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Epsilon values for each species in the mixture
        """
        epsilon: Float[Array, "... n_species"] = jnp.einsum(
            "...j,ij,ij->...i", mole_fractions, self.k1, self.epsilon_matrix
        )

        return epsilon

    def get_species_sigma(
        self, mole_fractions: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Gets the sigma values for each species in the mixture using the Lorentz mixing rule.

        There is a typo in :cite:t:`ZD09` that gives the coefficient as k1 rather than k2.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Sigma values for each species in the mixture
        """
        sigma: Float[Array, "... n_species"] = jnp.einsum(
            "...j,ij,ij->...i", mole_fractions, self.k2, self.sigma_matrix
        )

        return sigma

    def _get_S2_parameter(self, Tm: ArrayLike, coefficients: tuple[float, ...]) -> FloatArray:
        """Gets the parameter (coefficient) for the S2 term for polynomials involving Tm terms

        Args:
            Tm: Scaled temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return 2 * coefficients[1] / jnp.square(Tm) + 3 * coefficients[2] / jnp.power(Tm, 3)

    def _S2(self, Tm: ArrayLike, Vm: ArrayLike) -> FloatArray:
        """S2 term :cite:p:`ZD09{Equation 16}`

        Args:
            Tm: Scaled temperature
            Vm: Scaled volume

        Returns:
            S2 term
        """
        b: FloatArray = self._get_S2_parameter(Tm, self.coefficients[0:3])
        c: FloatArray = self._get_S2_parameter(Tm, self.coefficients[3:6])
        d: FloatArray = self._get_S2_parameter(Tm, self.coefficients[6:9])
        e: FloatArray = self._get_S2_parameter(Tm, self.coefficients[9:12])
        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]

        S2: FloatArray = (
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

    # def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> FloatArray:
    #     r"""Objective function to solve for the volume :cite:p:`ZD09{Equation 8}`.

    #     Note that the left-hand side of :cite:t:`ZD09{Equation 8}` is the compressibility factor
    #     so should be expressed in terms of P, V, R, and T, and not the scaled equivalents.

    #     Args:
    #         volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
    #         kwargs: Dictionary with other required parameters

    #     Returns:
    #         Residual of the objective function
    #     """
    #     temperature: ArrayLike = kwargs["temperature"]
    #     pressure: ArrayLike = kwargs["pressure"]
    #     mole_fractions: Float[Array, "... n_species"] = kwargs["mole_fractions"]  # type: ignore
    #     # jax.debug.print("temperature = {temperature}", temperature=temperature)
    #     # jax.debug.print("pressure = {pressure}", pressure=pressure)

    #     Tm: ArrayLike = self._Tm(temperature, mole_fractions)
    #     # jax.debug.print("Tm = {Tm}", Tm=Tm)
    #     Vm: FloatArray = self._Vm(volume, mole_fractions)
    #     # jax.debug.print("Vm = {Vm}", Vm=Vm)
    #     ptr: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
    #     # jax.debug.print("ptr = {ptr}", ptr=ptr)

    #     b: FloatArray = self._get_S1_parameter(Tm, self.coefficients[0:3])
    #     # jax.debug.print("b = {b}", b=b)
    #     c: FloatArray = self._get_S1_parameter(Tm, self.coefficients[3:6])
    #     # jax.debug.print("c = {c}", c=c)
    #     d: FloatArray = self._get_S1_parameter(Tm, self.coefficients[6:9])
    #     # jax.debug.print("d = {d}", d=d)
    #     e: FloatArray = self._get_S1_parameter(Tm, self.coefficients[9:12])
    #     # jax.debug.print("e = {e}", e=e)

    #     term1: FloatArray = (
    #         as_j64(1)
    #         + b / as_j64(Vm)
    #         + c / jnp.power(Vm, 2)
    #         + d / jnp.power(Vm, 4)
    #         + e / jnp.power(Vm, 5)
    #     )
    #     # jax.debug.print("term1 = {term1}", term1=term1)

    #     a13: float = self.coefficients[12]
    #     a14: float = self.coefficients[13]
    #     a15: float = self.coefficients[14]
    #     term2: FloatArray = (
    #         a13
    #         / jnp.power(Tm, 3)
    #         / jnp.power(Vm, 2)
    #         * (a14 + a15 / jnp.square(Vm))
    #         * safe_exp(-a15 / jnp.square(Vm))
    #     )
    #     # jax.debug.print("term2 = {term2}", term2=term2)

    #     residual: FloatArray = term1 + term2 - ptr
    #     # jax.debug.print("residual = {residual}", residual=residual)

    #     return residual

    # @override
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    # def volume(
    #     self,
    #     temperature: ArrayLike,
    #     pressure: ArrayLike,
    #     mole_fractions: Float[Array, "... n_species"],
    # ) -> Array:
    #     r"""Computes the volume numerically.

    #     Args:
    #         temperature: Temperature (K)
    #         pressure: Pressure (bar)
    #         mole_fractions: Mole fractions of species in the gas phase

    #     Returns:
    #         Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
    #     """
    #     initial: FloatArray = self.initial_volume(temperature, pressure)
    #     kwargs: dict[str, ArrayLike] = {
    #         "temperature": temperature,
    #         "pressure": pressure,
    #         "mole_fractions": mole_fractions,
    #     }

    #     solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
    #     sol = optx.root_find(self._objective_function, solver, initial, args=kwargs, throw=THROW)
    #     volume: FloatArray = sol.value
    #     jax.debug.print("volume = {out}", out=volume)
    #     jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

    #     # For comparing the initial and final volumes to refine the choice of the initial volume
    #     jax.debug.print("initial_volume = {out}", out=initial)
    #     jax.debug.print("final_volume = {out}", out=volume)
    #     relative_volume_error: FloatArray = (initial - volume) / volume
    #     jax.debug.print("Relative volume error = {out}", out=relative_volume_error)

    #     return volume

    @override
    def log_fugacity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        """Log fugacity computed using autodiff for the volume.

        This is a useful check to ensure that the numerical solver for the volume is working correctly,
        as the log fugacity should be consistent regardless of how the volume is computed.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Gradient of log fugacity with respect to mole fractions
        """
        # This works, roughly
        # grad_fn = jax.grad(self.log_fugacity_bulk, argnums=2)
        # Try using equinox instead
        # grad_fn = eqx.filter_grad(self.wrap_log_fugacity_bulk)
        log_fugacity_coeff = self.log_fugacity_bulk(temperature, pressure, mole_fractions)
        # jax.debug.print("log_fugacity_coeff_bulk = {out}", out=log_fugacity_coeff)

        # For case without equinox
        # log_fugacity_coeff = (
        #     log_fugacity_coeff + grad_fn(temperature, pressure, mole_fractions)[self.species_index]
        # )
        # # jax.debug.print("grad_fn = {out}", out=self._grad_fn(temperature, pressure, mole_fractions))
        # # jax.debug.print("mole_fractions = {out}", out=mole_fractions)
        # log_fugacity_coeff = log_fugacity_coeff - jnp.sum(
        #     grad_fn(temperature, pressure, mole_fractions) * mole_fractions
        # )

        # With equinox, switch argument order
        # Use vmap over the batch dimension when mole_fractions is batched (ndim > 1),
        # since eqx.filter_grad requires a scalar-output function.
        if mole_fractions.ndim == 1:
            grads = self._grad_fn(mole_fractions, temperature, pressure)
        else:
            grads = jax.vmap(self._grad_fn, in_axes=(0, None, None))(
                mole_fractions, temperature, pressure
            )
        log_fugacity_coeff = log_fugacity_coeff + grads[..., self.species_index]
        log_fugacity_coeff = log_fugacity_coeff - jnp.sum(grads * mole_fractions, axis=-1)

        return log_fugacity_coeff + jnp.log(pressure)

    def log_fugacity_bulk(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        """Log fugacity computed using a bulk method that does not require solving for the volume.

        This is a useful check to ensure that the numerical solver for the volume is working correctly,
        as the log fugacity should be consistent regardless of how the volume is computed.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase
        """
        volume: FloatArray = self.volume(temperature, pressure, mole_fractions)
        Vm: FloatArray = self.reduced_volume(volume, mole_fractions)
        Tm: ArrayLike = self.reduced_temperature(temperature, mole_fractions)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: FloatArray = -jnp.log(Z) + self.get_S1(Tm, Vm) + Z - 1

        return log_fugacity_coefficient

    def wrap_log_fugacity_bulk(
        self,
        mole_fractions: Float[Array, "... n_species"],
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> FloatArray:
        """Wrapper for log fugacity bulk method to ensure it is not traced by JAX.

        This is necessary to avoid issues with the numerical solver for the volume when using
        autodiff, as the solver may not be compatible with JAX's tracing mechanism.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Log fugacity coefficient without the pressure term
        """
        return self.log_fugacity_bulk(temperature, pressure, mole_fractions)

    @override
    def log_fugacity_original(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        """Log fugacity :cite:p:`ZD09{Equation 14}`

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Log fugacity in bar
        """
        volume: FloatArray = self.volume(temperature, pressure, mole_fractions)
        Vm: FloatArray = self.reduced_volume(volume, mole_fractions)
        Tm: ArrayLike = self.reduced_temperature(temperature, mole_fractions)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: FloatArray = -jnp.log(Z) + self.get_S1(Tm, Vm) + Z - 1
        # jax.debug.print("species_index = {out}", out=self.species_index)
        # jax.debug.print(
        #    "log_fugacity_coefficient (before mixing) = {out}", out=log_fugacity_coefficient
        # )

        # Extra terms for mixture
        correction1 = self.get_species_epsilon(mole_fractions)[
            ..., self.species_index
        ] / self.get_epsilon(mole_fractions)
        # jax.debug.print("correction1 = {out}", out=correction1)
        log_fugacity_coefficient = log_fugacity_coefficient - 2 * self._S2(Tm, Vm) * (
            1 - correction1
        )
        correction2 = self.get_species_sigma(mole_fractions)[
            ..., self.species_index
        ] / self.get_sigma(mole_fractions)
        log_fugacity_coefficient = log_fugacity_coefficient + 6 * (1 - Z) * (1 - correction2)
        # jax.debug.print("correction2 = {out}", out=correction2)

        # jax.debug.print(
        #    "log_fugacity_coefficient (after mixing) = {out}", out=log_fugacity_coefficient
        # )

        # TODO: Check if this is partial fugacity?
        # TODO: Note multiplication by mole fraction
        # mole_fraction = mole_fractions[self.species_index]
        log_fugacity: FloatArray = (
            log_fugacity_coefficient
            + jnp.log(
                pressure
            )  # FIXME: The mixing is done outside this function + jnp.log(mole_fraction)
        )
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        # jax.debug.print("log_fugacity = {out}", out=log_fugacity)

        return log_fugacity

    @eqx.filter_jit
    def log_activity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        """Log activity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions. Defaults to ``None`` if unused.

        Returns:
            Log activity, which is dimensionless
        """
        # jax.debug.print("mole_fractions = {out}", out=mole_fractions)
        # log_fugacity = self.log_fugacity(temperature, pressure, mole_fractions)
        # jax.debug.print("log_fugacity = {out}", out=log_fugacity)
        return (
            self.log_fugacity(temperature, pressure, mole_fractions) - jnp.log(STANDARD_FUGACITY)
            # FIXME
            # - jnp.log(mole_fractions)[self.species_index]
        )

    @staticmethod
    def get_epsilon_berthelot_rule(
        species: tuple[str, ...],
    ) -> NpFloat:
        """Gets the epsilon matrix for a given set of species using the Berthelot rule.

        This is a simple geometric mean, which is the standard mixing rule for parameters like the
        Lennard-Jones energy parameter (epsilon).

        Uses the mean of known values as a fallback for missing species, which is a simple approach
        that allows the mixing rules to be applied even when some species are missing.

        Args:
            species: Tuple of species names

        Returns:
            Epsilon matrix
        """
        mean_epsilon = np.mean(np.array(list(epsilon_species.values())))
        epsilon: NpFloat = np.array([epsilon_species.get(sp, mean_epsilon) for sp in species])
        epsilon_matrix: NpFloat = np.sqrt(np.outer(epsilon, epsilon))

        return epsilon_matrix

    @staticmethod
    def get_sigma_lorentz_rule(species: tuple[str, ...]) -> NpFloat:
        """Gets the sigma matrix for a given set of species using the Lorentz mixing rule.

        This is a simple arithmetic mean, which is the standard mixing rule for parameters like the
        Lennard-Jones diameter (sigma).

        Uses the mean of known values as a fallback for missing species, which is a simple approach
        that allows the mixing rules to be applied even when some species are missing.

        Args:
            species: Tuple of species names

        Returns:
            sigma matrix
        """
        mean_sigma = np.mean(np.array(list(sigma_species.values())))
        sigma: NpFloat = np.array([sigma_species.get(sp, mean_sigma) for sp in species])
        sigma_matrix: NpFloat = 0.5 * (sigma[:, None] + sigma[None, :])

        return sigma_matrix

    @staticmethod
    def get_k1_mixing_matrix(species: tuple[str, ...]) -> NpFloat:
        """Gets the k1 matrix for a given set of species.

        Args:
            species: Tuple of species names

        Returns:
            k1 matrix
        """
        num_species: int = len(species)
        k1_matrix: NpFloat = np.ones((num_species, num_species))

        # Values from Zhang and Duan (2009) for CO2-H2O
        if "CO2" in species and "H2O" in species:
            k1_matrix[species.index("CO2"), species.index("H2O")] = 0.85
            k1_matrix[species.index("H2O"), species.index("CO2")] = 0.85

        # Values from Zhang and Duan (2009) for CH4-H2O
        if "CH4" in species and "H2O" in species:
            k1_matrix[species.index("CH4"), species.index("H2O")] = 0.8
            k1_matrix[species.index("H2O"), species.index("CH4")] = 0.8

        return k1_matrix

    @staticmethod
    def get_k2_mixing_matrix(species: tuple[str, ...]) -> NpFloat:
        """Gets the k2 matrix for a given set of species.

        Args:
            species: Tuple of species names

        Returns:
            k2 matrix
        """
        num_species: int = len(species)
        k2_matrix: NpFloat = np.ones((num_species, num_species))

        # Values from Zhang and Duan (2009) for CO2-H2O
        if "CO2" in species and "H2O" in species:
            k2_matrix[species.index("CO2"), species.index("H2O")] = 1.02
            k2_matrix[species.index("H2O"), species.index("CO2")] = 1.02

        # Values from Zhang and Duan (2009) for CH4-H2O
        if "CH4" in species and "H2O" in species:
            k2_matrix[species.index("CH4"), species.index("H2O")] = 1.0
            k2_matrix[species.index("H2O"), species.index("CH4")] = 1.0

        return k2_matrix

    @staticmethod
    def binary_mixing_rule(
        mole_fraction: Float[Array, "... n_species"], kn: ArrayLike, arg: ArrayLike
    ) -> Float[Array, "..."]:
        """Binary mixing rule

        Args:
            mole_fraction: Mole fraction of species
            kn: Binary interaction parameter
            arg: Argument to mix (e.g., epsilon or sigma)

        Returns:
            Mixed argument
        """
        return jnp.sum(
            jnp.einsum("...i,...j->...ij", mole_fraction, mole_fraction) * kn * arg, axis=(-2, -1)
        )


CH4_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CH4"], sigma_species["CH4"])
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

H2O_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["H2O"], sigma_species["H2O"])
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

CO2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CO2"], sigma_species["CO2"])
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
H2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["H2"], sigma_species["H2"])
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

CO_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CO"], sigma_species["CO"])
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

O2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["O2"], sigma_species["O2"])
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

C2H6_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["C2H6"], sigma_species["C2H6"])
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
