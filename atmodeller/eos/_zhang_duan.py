# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOS from :cite:t:`ZD09`"""

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from atmodeller import override
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW, VOLUME_EPSILON
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import FloatArray, NpFloat, OptxSolver, as_j64, safe_exp
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

    def get_epsilon(self) -> Float[Array, ""]:
        """Gets epsilon.

        Returns:
            Epsilon
        """
        return as_j64(self.epsilon)

    def get_sigma(self) -> Float[Array, ""]:
        """Gets sigma.

        Returns:
            Sigma
        """
        return as_j64(self.sigma)

    def _Pm(self, pressure: ArrayLike) -> Float[Array, ""]:
        """Scaled pressure

        Args:
            pressure: Pressure in bar

        Returns:
            Scaled pressure
        """
        sigma: Float[Array, ""] = self.get_sigma()
        epsilon: Float[Array, ""] = self.get_epsilon()
        pressure_MPa: ArrayLike = pressure * unit_conversion.bar_to_MPa
        scaled_pressure: FloatArray = 3.0636 * jnp.power(sigma, 3) * pressure_MPa / epsilon
        # jax.debug.print("scaled_pressure = {out}", out=scaled_pressure)

        return scaled_pressure

    def _Tm(self, temperature: ArrayLike) -> ArrayLike:
        """Scaled temperature

        Args:
            temperature: Temperature in K

        Returns:
            Scaled temperature
        """
        epsilon: Float[Array, ""] = self.get_epsilon()
        scaled_temperature: ArrayLike = 154 * temperature / epsilon
        # jax.debug.print("scaled_temperature = {out}", out=scaled_temperature)

        return scaled_temperature

    def _Vm(self, volume: ArrayLike) -> FloatArray:
        r"""Scaled volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`

        Returns:
            Scaled volume
        """
        sigma: Float[Array, ""] = self.get_sigma()
        volume_cm3: ArrayLike = volume * unit_conversion.m3_to_cm3
        sigma_term: FloatArray = jnp.power(sigma / 3.691, 3)
        scaled_volume: FloatArray = volume_cm3 / 1000 / sigma_term  # type: ignore
        # jax.debug.print("scaled_volume = {out}", out=scaled_volume)

        return scaled_volume

    def _get_S1_parameter(self, Tm: ArrayLike, coefficients: tuple[float, ...]) -> FloatArray:
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

    def _get_S2_parameter(self, Tm: ArrayLike, coefficients: tuple[float, ...]) -> FloatArray:
        """Gets the parameter (coefficient) for the S2 term for polynomials involving Tm terms

        Args:
            Tm: Scaled temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return 2 * coefficients[1] / jnp.square(Tm) + 3 * coefficients[2] / jnp.power(Tm, 3)

    def _S1(self, Tm: ArrayLike, Vm: ArrayLike) -> FloatArray:
        """S1 term :cite:p:`ZD09{Equation 15}`

        Args:
            Tm: Scaled temperature
            Vm: Scaled volume

        Returns:
            S1 term
        """
        b: FloatArray = self._get_S1_parameter(Tm, self.coefficients[0:3])
        c: FloatArray = self._get_S1_parameter(Tm, self.coefficients[3:6])
        d: FloatArray = self._get_S1_parameter(Tm, self.coefficients[6:9])
        e: FloatArray = self._get_S1_parameter(Tm, self.coefficients[9:12])
        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]

        S1: FloatArray = (
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

    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> FloatArray:
        r"""Objective function to solve for the volume :cite:p:`ZD09{Equation 8}`.

        Note that the left-hand side of :cite:t:`ZD09{Equation 8}` is the compressibility factor,
        which can be expressed in terms of P, V, R, and T. If the scaled equivalents are used
        instead, care should be taken to ensure that the 1000 scaling factor is accounted for.

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
        Vm: FloatArray = self._Vm(volume)
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

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Initial volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        safe_volume: FloatArray = as_j64(ideal_volume + VOLUME_EPSILON)
        # jax.debug.print("initial_volume = {out}", out=safe_volume)

        return safe_volume

    @override
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        initial: FloatArray = self.initial_volume(temperature, pressure)
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

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

    @override
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Log fugacity :cite:p:`ZD09{Equation 14}`

        This is for a pure species and does not include the terms to enable end member mixing.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        volume: FloatArray = self.volume(temperature, pressure)
        Vm: FloatArray = self._Vm(volume)
        Tm: ArrayLike = self._Tm(temperature)
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: FloatArray = -jnp.log(Z) + self._S1(Tm, Vm) + Z - 1
        log_fugacity: FloatArray = log_fugacity_coefficient + jnp.log(pressure)
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        return log_fugacity

    @override
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        r"""Volume integral

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        log_fugacity: FloatArray = self.log_fugacity(temperature, pressure)
        volume_integral: FloatArray = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral


class ZhangDuanMixture(ZhangDuan):
    """Zhang and Duan EOS for mixtures :cite:p:`ZD09`"""

    species: tuple[str, ...]
    k1: NpFloat  # (n_species, n_species)
    k2: NpFloat  # (n_species, n_species)
    epsilon_matrix: NpFloat  # (n_species, n_species)
    sigma_matrix: NpFloat  # (n_species, n_species)

    @override
    def __init__(self, species: tuple[str, ...]):
        """Initializes the mixture model.

        Args:
            species: Tuple of species names
        """
        self.species = species
        self.k1 = self.get_k1_mixing_matrix(species)
        self.k2 = self.get_k2_mixing_matrix(species)
        self.epsilon_matrix = self.get_epsilon_berthelot_rule(species)
        self.sigma_matrix = self.get_sigma_lorentz_rule(species)
        # FIXME: Hacky, but this is required to avoid errors in the base class methods that expect
        # these attributes to be defined. The actual values will be computed using the mixing
        # rules, so these are just placeholders.
        self.epsilon = 0
        self.sigma = 0

    @override
    def get_epsilon(self, mole_fractions: Float[Array, " n_species"]) -> Float[Array, ""]:
        """Gets epsilon for a mixture using the Berthelot mixing rule.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Epsilon for the mixture
        """
        return self.binary_mixing_rule(mole_fractions, self.k1, self.epsilon_matrix)

    @override
    def get_sigma(self, mole_fractions: Float[Array, " n_species"]) -> Float[Array, ""]:
        """Gets sigma for a mixture using the Lorentz mixing rule.

        Args:
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Sigma for the mixture
        """
        return self.binary_mixing_rule(mole_fractions, self.k2, self.sigma_matrix)

    def _Pm(self, pressure: ArrayLike, mole_fractions: Float[Array, " n_species"]) -> FloatArray:
        """Scaled pressure

        Args:
            pressure: Pressure in bar
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Scaled pressure
        """
        sigma: FloatArray = self.get_sigma(mole_fractions)
        epsilon: FloatArray = self.get_epsilon(mole_fractions)
        pressure_MPa: ArrayLike = pressure * unit_conversion.bar_to_MPa
        scaled_pressure: FloatArray = 3.0636 * jnp.power(sigma, 3) * pressure_MPa / epsilon
        # jax.debug.print("scaled_pressure = {out}", out=scaled_pressure)

        return scaled_pressure

    def _Tm(self, temperature: ArrayLike, mole_fractions: Float[Array, " n_species"]) -> ArrayLike:
        """Scaled temperature

        Args:
            temperature: Temperature in K
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Scaled temperature
        """
        epsilon: Float[Array, ""] = self.get_epsilon(mole_fractions)
        scaled_temperature: ArrayLike = 154 * temperature / epsilon
        # jax.debug.print("scaled_temperature = {out}", out=scaled_temperature)

        return scaled_temperature

    def _Vm(self, volume: ArrayLike, mole_fractions: Float[Array, " n_species"]) -> FloatArray:
        r"""Scaled volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Scaled volume
        """
        sigma: Float[Array, ""] = self.get_sigma(mole_fractions)
        volume_cm3: ArrayLike = volume * unit_conversion.m3_to_cm3
        sigma_term: FloatArray = jnp.power(sigma / 3.691, 3)
        scaled_volume: FloatArray = volume_cm3 / 1000 / sigma_term  # type: ignore
        # jax.debug.print("scaled_volume = {out}", out=scaled_volume)

        return scaled_volume

    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> FloatArray:
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
        mole_fractions: Float[Array, " n_species"] = kwargs["mole_fractions"]  # type: ignore
        # jax.debug.print("temperature = {temperature}", temperature=temperature)
        # jax.debug.print("pressure = {pressure}", pressure=pressure)

        Tm: ArrayLike = self._Tm(temperature, mole_fractions)
        # jax.debug.print("Tm = {Tm}", Tm=Tm)
        Vm: FloatArray = self._Vm(volume, mole_fractions)
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

    @override
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: Float[Array, " n_species"],
    ) -> Array:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            mole_fractions: Mole fractions of species in the gas phase

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        initial: FloatArray = self.initial_volume(temperature, pressure)
        kwargs: dict[str, ArrayLike] = {
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
        # relative_volume_error: FloatArray = (initial - volume) / volume
        # jax.debug.print("Relative volume error = {out}", out=relative_volume_error)

        return volume

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
        mole_fraction: Float[Array, " n_species"], kn: ArrayLike, arg: ArrayLike
    ) -> Float[Array, ""]:
        """Binary mixing rule

        Args:
            mole_fraction: Mole fraction of species
            kn: Binary interaction parameter
            arg: Argument to mix (e.g., epsilon or sigma)

        Returns:
            Mixed argument
        """
        return jnp.sum(jnp.outer(mole_fraction, mole_fraction) * kn * arg)


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
"""sigma values for each species (10\\ :sup:`-10` m) :cite:p:`ZD09{Table 4}`. Ensure these use Hill
notation."""

CH4_zhang09: RealGas = ZhangDuan(epsilon_species["CH4"], sigma_species["CH4"])
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

H2O_zhang09: RealGas = ZhangDuan(epsilon_species["H2O"], sigma_species["H2O"])
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

CO2_zhang09: RealGas = ZhangDuan(epsilon_species["CO2"], sigma_species["CO2"])
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
H2_zhang09: RealGas = ZhangDuan(epsilon_species["H2"], sigma_species["H2"])
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

CO_zhang09: RealGas = ZhangDuan(epsilon_species["CO"], sigma_species["CO"])
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

O2_zhang09: RealGas = ZhangDuan(epsilon_species["O2"], sigma_species["O2"])
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

C2H6_zhang09: RealGas = ZhangDuan(epsilon_species["C2H6"], sigma_species["C2H6"])
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
