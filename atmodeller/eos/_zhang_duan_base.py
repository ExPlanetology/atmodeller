# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base module for real gas EOS from :cite:t:`ZD09`"""

from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from atmodeller import override
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW
from atmodeller.eos.core import RealGas, safe_ideal_initial_volume
from atmodeller.jax_utils import FloatArray, NpFloat, OptxSolver, as_j64, safe_exp
from atmodeller.sci_utils import GAS_CONSTANT_BAR, unit_conversion

epsilon_species: dict[str, float] = {
    # From Zhang and Duan (2009) Table 4. Ensure these use Hill notation.
    "CH4": 154.0,
    "H2O": 510.0,
    "CO2": 235.0,
    "H2": 31.2,
    "CO": 105.6,
    "O2": 124.5,
    "C2H6": 246.1,
    # From Poling et al. (2000), Appendix B. Ensure these use Hill notation.
    "Ar": 93.3,
    "He": 10.22,
    "Kr": 178.9,
    "Ne": 32.8,
    "Xe": 231.0,
    # "CH4": 148.6,
    # "CO": 91.7,
    "COS": 336.0,
    # "CO2": 195.2,
    # "C2H6": 215.7,
    "Cl2": 316.0,
    "F2": 112.6,
    "CHN": 569.1,
    "ClH": 344.7,
    # "H2": 59.7,
    # "H2O": 809.1,
    "H2S": 301.1,
    "H3N": 558.3,
    "NO": 116.7,
    "N2": 71.4,
    "N2O": 232.4,
    # "O2": 106.7,
    "O2S": 335.4,
}
"""Epsilon values for each species (K) :cite:p:`ZD09{Table 4}` and :cite:p:`PPO00{Appendix B}`.
Ensure these use Hill notation and prefer the values from :cite:t:`ZD09` where available."""
sigma_species: dict[str, float] = {
    # From Zhang and Duan (2009) Table 4. Ensure these use Hill notation.
    "CH4": 3.691,
    "H2O": 2.88,
    "CO2": 3.79,
    "H2": 2.93,
    "CO": 3.66,
    "O2": 3.36,
    "C2H6": 4.35,
    # From Poling et al. (2000), Appendix B. Ensure these use Hill notation.
    "Ar": 3.542,
    "He": 2.551,
    "Kr": 3.655,
    "Ne": 2.820,
    "Xe": 4.047,
    # "CH4": 3.758,
    # "CO": 3.690,
    "COS": 4.130,
    # "CO2": 3.941,
    # "C2H6": 4.443,
    "Cl2": 4.217,
    "F2": 3.357,
    "CHN": 3.630,
    "ClH": 3.339,
    # "H2": 2.827,
    # "H2O": 2.641,
    "H2S": 3.623,
    "H3N": 2.900,
    "NO": 3.492,
    "N2": 3.798,
    "N2O": 3.828,
    # "O2": 3.467,
    "O2S": 4.112,
}
r"""Sigma values for each species (10\ :sup:`-10` m) :cite:p:`ZD09{Table 4}` and 
:cite:p:`PPO00{Appendix B}`. Ensure these use Hill notation and prefer the values from 
:cite:t:`ZD09` where available."""

REFERENCE_EPSILON: float = epsilon_species["CH4"]
"""Reference epsilon"""
REFERENCE_SIGMA: float = sigma_species["CH4"]
"""Reference sigma"""
K1_ZHANG_DUAN: dict[str, float] = {"CO2-H2O": 0.85, "CH4-H2O": 0.8}
"""Default binary interaction parameters for k1 :cite:p:`ZD09`"""
K2_ZHANG_DUAN: dict[str, float] = {"CO2-H2O": 1.02, "CH4-H2O": 1.0}
"""Default binary interaction parameters for k2 :cite:p:`ZD09`"""


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
    def get_epsilon(self, mole_fractions: ArrayLike | None = None) -> ArrayLike:
        """Gets epsilon.

        Args:
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Epsilon (K)
        """

    @abstractmethod
    def get_sigma(self, mole_fractions: ArrayLike | None = None) -> ArrayLike:
        """Gets sigma.

        Args:
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Sigma (:math:`10^{-10}` m)
        """

    def reduced_pressure(
        self, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
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
        self, temperature: ArrayLike, mole_fractions: ArrayLike | None = None
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
        self, volume: ArrayLike, mole_fractions: ArrayLike | None = None
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
            Tr: Reduced temperature (dimensionless)
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
            Tr: Reduced temperature (dimensionless)
            Vr: Reduced volume (dimensionless)

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

    def _objective_function(
        self, volume: ArrayLike, kwargs: dict[str, ArrayLike | None]
    ) -> FloatArray:
        r"""Objective function to solve for the volume :cite:p:`ZD09{Equation 8}`.

        Note that the left-hand side of :cite:t:`ZD09{Equation 8}` is the compressibility factor,
        which can be expressed in terms of `P`, `V`, `R`, and `T`. If the scaled equivalents are
        used instead, care should be taken to ensure that the 1000 scaling factor is accounted for.

        Args:
            volume: Volume (:math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`)
            kwargs: Dictionary with other required and optional parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = cast(ArrayLike, kwargs["temperature"])
        pressure: ArrayLike = cast(ArrayLike, kwargs["pressure"])
        mole_fractions: ArrayLike | None = kwargs.get("mole_fractions", None)
        # jax.debug.print("temperature = {temperature}", temperature=temperature)
        # jax.debug.print("pressure = {pressure}", pressure=pressure)
        # jax.debug.print("mole_fractions = {mole_fractions}", mole_fractions=mole_fractions)

        Tr: ArrayLike = self.reduced_temperature(temperature, mole_fractions)
        # jax.debug.print("Tr = {Tr}", Tr=Tr)
        Vm: FloatArray = self.reduced_volume(volume, mole_fractions)
        # jax.debug.print("Vm = {Vm}", Vm=Vm)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        # jax.debug.print("ptr = {ptr}", ptr=ptr)

        b: FloatArray = self._get_S1_parameter(Tr, self.coefficients[0:3])
        # jax.debug.print("b = {b}", b=b)
        c: FloatArray = self._get_S1_parameter(Tr, self.coefficients[3:6])
        # jax.debug.print("c = {c}", c=c)
        d: FloatArray = self._get_S1_parameter(Tr, self.coefficients[6:9])
        # jax.debug.print("d = {d}", d=d)
        e: FloatArray = self._get_S1_parameter(Tr, self.coefficients[9:12])
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
            / jnp.power(Tr, 3)
            / jnp.power(Vm, 2)
            * (a14 + a15 / jnp.square(Vm))
            * safe_exp(-a15 / jnp.square(Vm))
        )
        # jax.debug.print("term2 = {term2}", term2=term2)

        residual: FloatArray = term1 + term2 - Z
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
        return safe_ideal_initial_volume(temperature, pressure) * 10

    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
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
        kwargs: dict[str, ArrayLike | None] = {
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
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Pressure-integrated volume contribution

        Computes the quantity

        .. math::
            RT \ln f,

        where :math:`f` is the fugacity returned by :meth:`log_fugacity`.

        For bulk fluid fugacities, this corresponds to the pressure integral involving the molar
        volume. For species fugacities in mixtures, the corresponding integral involves the partial
        molar volume.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Pressure-integrated volume contribution
            (:math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`)
        """
        log_fugacity: FloatArray = self.log_fugacity(temperature, pressure, mole_fractions)
        volume_integral: FloatArray = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral

    def log_mixture_fugacity_coefficient(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Log fugacity coefficient of the fluid mixture :cite:p:`ZD09{Equation 14}`

        Computes the fugacity coefficient associated with the bulk fluid described by the Zhang and
        Duan EOS.

        .. math::
            \ln \phi = -\ln Z + S_1 + Z - 1

        where :math:`\phi` is the mixture fugacity coefficient and :math:`Z` is the compressibility
        factor. For a pure fluid, this reduces to the standard pure-fluid fugacity coefficient.

        This quantity does not include the compositional derivative terms required to obtain
        species (partial) fugacity coefficients in mixtures.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase. Defaults to ``None`` for
                pure fluids.

        Returns:
            Logarithm of the mixture fugacity coefficient (dimensionless)
        """
        volume: FloatArray = self.volume(temperature, pressure, mole_fractions)
        Vm: FloatArray = self.reduced_volume(volume, mole_fractions)
        Tr: ArrayLike = self.reduced_temperature(temperature, mole_fractions)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: FloatArray = -jnp.log(Z) + self.get_S1(Tr, Vm) + Z - 1
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        return log_fugacity_coefficient

    def log_mixture_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        r"""Logarithm of the mixture fugacity

        Computes

        .. math::
            \ln f = \ln \phi + \ln P

        where :math:`f` is the mixture fugacity, :math:`\phi` is the mixture fugacity coefficient,
        and :math:`P` is the pressure.

        For pure fluids, this reduces to the standard definition of fugacity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase. Defaults
                to ``None`` for pure fluids.

        Returns:
            Logarithm of the mixture fugacity
        """
        return self.log_mixture_fugacity_coefficient(
            temperature, pressure, mole_fractions
        ) + jnp.log(pressure)


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
    def get_epsilon(self, mole_fractions: ArrayLike | None = None) -> ArrayLike:
        del mole_fractions
        return as_j64(self.epsilon)

    @override
    def get_sigma(self, mole_fractions: ArrayLike | None = None) -> ArrayLike:
        del mole_fractions
        return as_j64(self.sigma)

    @override
    def log_fugacity_coefficient(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions
        return self.log_mixture_fugacity_coefficient(temperature, pressure)

    @override
    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: ArrayLike | None = None
    ) -> FloatArray:
        del mole_fractions
        return self.log_mixture_fugacity(temperature, pressure)


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
    def __init__(
        self,
        species: tuple[str, ...],
        species_to_output: str,
        k1: dict[str, float] = K1_ZHANG_DUAN,
        k2: dict[str, float] = K2_ZHANG_DUAN,
    ):
        """Initializes the mixture model.

        Args:
            species: Tuple of species names
            species_to_output: Name of the species for which to output species-specific quantities,
                such as partial fugacity
            k1: Binary interaction parameter matrix for epsilon (n_species, n_species). Defaults to
                :attr:`K1_ZHANG_DUAN` to use the Zhang and Duan (2009) coefficients.
            k2: Binary interaction parameter matrix for sigma (n_species, n_species). Defaults to
                :attr:`K2_ZHANG_DUAN` to use the Zhang and Duan (2009) coefficients.
        """
        self.species = species
        self.k1 = self.build_mixing_matrix(species, k1)
        self.k2 = self.build_mixing_matrix(species, k2)
        self.epsilon_matrix = self.get_epsilon_berthelot_rule(species)
        self.sigma_matrix = self.get_sigma_lorentz_rule(species)
        self.species_index = species.index(species_to_output)
        # print("species_index:", self.species_index)
        self._grad_fn = eqx.filter_grad(self._wrap_log_mixture_fugacity_coefficient)

    def _wrap_log_mixture_fugacity_coefficient(
        self,
        mole_fractions: Float[Array, "... n_species"],
        temperature: ArrayLike,
        pressure: ArrayLike,
    ):
        """Wrapper to switch argument order to allow for differentiation with respect to mole
        fractions using eqx.filter_grad"""
        return self.log_mixture_fugacity_coefficient(temperature, pressure, mole_fractions)

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

        There is a typo in :cite:t:`ZD09` and the coefficient should be k2.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Sigma values for each species in the mixture
        """
        sigma: Float[Array, "... n_species"] = jnp.einsum(
            "...j,ij,ij->...i", mole_fractions, self.k2, self.sigma_matrix
        )

        return sigma

    def _get_S2_parameter(self, Tr: ArrayLike, coefficients: tuple[float, ...]) -> FloatArray:
        """Gets the parameter (coefficient) for the S2 term for polynomials involving Tr terms

        Args:
            Tr: Scaled temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return 2 * coefficients[1] / jnp.square(Tr) + 3 * coefficients[2] / jnp.power(Tr, 3)

    def _S2(self, Tr: ArrayLike, Vm: ArrayLike) -> FloatArray:
        """S2 term :cite:p:`ZD09{Equation 16}`

        Args:
            Tr: Scaled temperature (dimensionless)
            Vm: Scaled volume (dimensionless)

        Returns:
            S2 term
        """
        b: FloatArray = self._get_S2_parameter(Tr, self.coefficients[0:3])
        c: FloatArray = self._get_S2_parameter(Tr, self.coefficients[3:6])
        d: FloatArray = self._get_S2_parameter(Tr, self.coefficients[6:9])
        e: FloatArray = self._get_S2_parameter(Tr, self.coefficients[9:12])
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
            / (2 * a15 * jnp.power(Tr, 3))
            * (a14 + 1 - (a14 + 1 + a15 / jnp.square(Vm)) * safe_exp(-a15 / jnp.square(Vm)))
        )
        # jax.debug.print("S2 = {out}", out=S2)

        return S2

    def log_partial_fugacity_coefficient_autodiff(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        r"""Log partial fugacity coefficient obtained using autodiff

        .. math::
            \ln \phi_i = \left(\frac{\partial(n \ln \phi)}{\partial n_i}\right)_{T,P,n_{j \neq i}}.

        However, it is desirable to express this in terms of the mole fractions rather than the
        number of moles, since the mole fractions are the natural variables for the mixing rules.

        Differentiating, applying the quotient rule, and simplifying gives:

        .. math::
            \ln \phi_i = \ln \phi + \frac{\partial \ln \phi}{\partial x_i} - \sum_k x_k \frac{\partial \ln \phi}{\partial x_k}.

        This method is not used in the final implementation, but it serves as a useful check on the
        analytical expression for the species fugacity coefficient since the two should agree.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Log species fugacity coefficient (dimensionless)
        """
        log_fugacity_coeff_mix = self.log_mixture_fugacity_coefficient(
            temperature, pressure, mole_fractions
        )

        # Use vmap over the batch dimension when mole_fractions is batched (ndim > 1),
        # since eqx.filter_grad requires a scalar-output function.
        if mole_fractions.ndim == 2:
            grads = eqx.filter_vmap(self._grad_fn, in_axes=(0, None, None))(
                mole_fractions, temperature, pressure
            )
        else:
            grads = self._grad_fn(mole_fractions, temperature, pressure)

        log_fugacity_coeff_i = log_fugacity_coeff_mix + grads[..., self.species_index]
        log_fugacity_coeff_i = log_fugacity_coeff_i - jnp.sum(grads * mole_fractions, axis=-1)

        return log_fugacity_coeff_i

    def log_partial_fugacity_coefficient(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        """Log partial fugacity coefficient using analytical expression :cite:p:`ZD09{Equation 14}`

        Note the sign error in the compositional correction term in :cite:t:`ZD09{Equation 14}`.
        The correct sign before S2 should be negative.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Log partial fugacity coefficient (dimensionless)
        """
        log_fugacity_coeff_mix: FloatArray = self.log_mixture_fugacity_coefficient(
            temperature, pressure, mole_fractions
        )
        volume: FloatArray = self.volume(temperature, pressure, mole_fractions)
        Vm: FloatArray = self.reduced_volume(volume, mole_fractions)
        Tr: ArrayLike = self.reduced_temperature(temperature, mole_fractions)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)

        # Compositional correction terms
        correction1 = self.get_species_epsilon(mole_fractions)[
            ..., self.species_index
        ] / self.get_epsilon(mole_fractions)
        # jax.debug.print("correction1 = {out}", out=correction1)
        log_fugacity_coeff_i = log_fugacity_coeff_mix - 2 * self._S2(Tr, Vm) * (1 - correction1)
        correction2 = self.get_species_sigma(mole_fractions)[
            ..., self.species_index
        ] / self.get_sigma(mole_fractions)
        # jax.debug.print("correction2 = {out}", out=correction2)
        log_fugacity_coeff_i = log_fugacity_coeff_i + 6 * (1 - Z) * (1 - correction2)
        # jax.debug.print("log_fugacity_coeff_i = {out}", out=log_fugacity_coeff_i)

        return log_fugacity_coeff_i

    @override
    def log_fugacity_coefficient(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        return self.log_partial_fugacity_coefficient(temperature, pressure, mole_fractions)

    @override
    def log_fugacity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, "... n_species"],
    ) -> FloatArray:
        r"""Logarithm of the species fugacity, excluding the mole fraction contribution

        Computes

        .. math::
            \ln \phi_i + \ln P,

        which is the species fugacity without the :math:`\ln x_i` term. The full species fugacity
        is :math:`\ln f_i = \ln \phi_i + \ln x_i + \ln P`, where the :math:`\ln x_i` contribution
        is accounted for externally.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Logarithm of the species fugacity excluding the mole fraction term (dimensionless)
        """
        return self.log_fugacity_coefficient(temperature, pressure, mole_fractions) + jnp.log(
            pressure
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
    def build_mixing_matrix(species: tuple[str, ...], parameter_dict: dict[str, float]) -> NpFloat:
        """Builds a mixing matrix for a given set of species and parameter dictionary.

        This is a helper function to construct the k1 and k2 matrices from the provided parameter
        dictionary. The parameter dictionary should have keys in the format "species1-species2" and
        values corresponding to the interaction parameter for that pair of species.

        Args:
            species: Tuple of species names
            parameter_dict: Dictionary with keys in the format "species1-species2" and values
                corresponding to the interaction parameter for that pair of species

        Returns:
            Mixing matrix
        """
        num_species: int = len(species)
        mixing_matrix: NpFloat = np.ones((num_species, num_species))

        for i in range(num_species):
            for j in range(i + 1, num_species):
                key: str = f"{species[i]}-{species[j]}"
                rev_key: str = f"{species[j]}-{species[i]}"
                value: float | None = parameter_dict.get(key) or parameter_dict.get(rev_key)
                if value is not None:
                    mixing_matrix[i, j] = value
                    mixing_matrix[j, i] = value
                else:
                    mixing_matrix[i, j] = 1.0  # Default value if not specified
                    mixing_matrix[j, i] = 1.0  # Default value if not specified

        return mixing_matrix

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
