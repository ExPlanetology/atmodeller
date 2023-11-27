"""Interfaces for real gas equations of state.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import RealGasABC
from atmodeller.utilities import debug_decorator

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ModifiedRedlichKwongABC(RealGasABC):
    """A Modified Redlich Kwong (MRK) EOS

    For example, Equation 3, Holland and Powell (1991):
        P = RT/(V-b) - a/(V(V+b)T**0.5)

    where:
        P is pressure
        T is temperature
        V is the molar volume
        R is the gas constant
        a is the Redlich-Kwong function, which is a function of T
        b is the Redlich-Kwong constant b

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter
        b0: Coefficient to compute the Redlich-Kwong constant b

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter
        b0: Coefficient to compute the Redlich-Kwong constant b
    """

    a_coefficients: tuple[float, ...]
    b0: float

    @abstractmethod
    def a(self, temperature: float) -> float:
        """MRK a parameter computed from self.a_coefficients

        Args:
            temperature: Temperature in kelvin

        Returns:
            MRK a parameter
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self) -> float:
        """MRK b parameter, which is is independent of temperature, computed from self.b0."""
        raise NotImplementedError


@dataclass(kw_only=True)
class MRKExplicitABC(ModifiedRedlichKwongABC):
    """A Modified Redlich Kwong (MRK) EOS with explicit equations for the volume and its integral.

    See base class.
    """

    def a(self, temperature: float) -> float:
        """Parameter a in Equation 9, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin

        Returns:
            Parameter a in J^2 bar^(-1) K^(1/2) mol^(-2)
        """
        a: float = (
            self.a_coefficients[0] * self.critical_temperature ** (5.0 / 2)
            + self.a_coefficients[1] * self.critical_temperature ** (3.0 / 2) * temperature
            + self.a_coefficients[2] * self.critical_temperature ** (1.0 / 2) * temperature**2
        )
        a /= self.critical_pressure

        return a

    @property
    def b(self) -> float:
        """Parameter b in Equation 9, Holland and Powell (1991)

        Returns:
            Parameter b in J bar^(-1) mol^(-1)
        """
        b: float = self.b0 * self.critical_temperature / self.critical_pressure

        return b

    @debug_decorator(logger)
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume-explicit equation. Equation 7, Holland and Powell (1991).

        Without complications of critical phenomena the MRK equation can be simplified using the
        approximation:

            V ~ RT/P + b

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            MRK volume in m^3/mol
        """
        volume: float = (
            GAS_CONSTANT * temperature / pressure
            + self.b
            - self.a(temperature)
            * GAS_CONSTANT
            * np.sqrt(temperature)
            / (GAS_CONSTANT * temperature + self.b * pressure)
            / (GAS_CONSTANT * temperature + 2.0 * self.b * pressure)
        )

        return volume

    @debug_decorator(logger)
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume-explicit integral (VdP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        volume_integral: float = (
            GAS_CONSTANT * temperature * np.log(pressure)
            + self.b * pressure
            + self.a(temperature)
            / self.b
            / np.sqrt(temperature)
            * (
                np.log(GAS_CONSTANT * temperature + self.b * pressure)
                - np.log(GAS_CONSTANT * temperature + 2.0 * self.b * pressure)
            )
        )

        return volume_integral


@dataclass(kw_only=True)
class MRKImplicitABC(ModifiedRedlichKwongABC):
    """A Modified Redlich Kwong (MRK) EOS in an implicit form

    See base class.
    """

    Ta: float = 0

    def a(self, temperature: float) -> float:
        """MRK a parameter for Equation 6, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin

        Returns:
            MRK a parameter
        """

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * self.delta_temperature_for_a(temperature)
            + self.a_coefficients[2] * self.delta_temperature_for_a(temperature) ** 2
            + self.a_coefficients[3] * self.delta_temperature_for_a(temperature) ** 3
        )

        return a

    @abstractmethod
    def delta_temperature_for_a(self, temperature: float) -> float:
        """Temperature difference for the calculation of the a parameter"""
        ...

    @property
    def b(self) -> float:
        """This class is not used for corresponding states which means b0 is the b coefficient."""
        return self.b0

    def A_factor(self, temperature: float, pressure: float) -> float:
        """A factor in Appendix A of Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            A factor, which is non-dimensional
        """
        del pressure
        A: float = self.a(temperature) / (self.b * GAS_CONSTANT * temperature**1.5)

        return A

    def B_factor(self, temperature: float, pressure: float) -> float:
        """B factor in Appendix A of Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            B factor, which is non-dimensional
        """
        B: float = self.b * pressure / (GAS_CONSTANT * temperature)

        return B

    @debug_decorator(logger)
    def volume_integral(
        self,
        temperature: float,
        pressure: float,
    ) -> float:
        """Volume integral. Equation A.2., Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        z: float = self.compressibility_parameter(temperature, pressure)
        A: float = self.A_factor(temperature, pressure)
        B: float = self.B_factor(temperature, pressure)
        # The base class requires a specification of the volume_integral, but the equations in
        # Holland and Powell (1991) are in terms of the fugacity coefficient.
        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)
        ln_fugacity: float = np.log(pressure) + ln_fugacity_coefficient
        volume_integral: float = GAS_CONSTANT * temperature * ln_fugacity

        return volume_integral

    def volume_roots(self, temperature: float, pressure: float) -> np.ndarray:
        """Real and (potentially) physically meaningful volume solutions of the MRK equation

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume solutions of the MRK equation in m^3/mol
        """
        coefficients: list[float] = []
        coefficients.append(-self.a(temperature) * self.b / np.sqrt(temperature))
        coefficients.append(
            -self.b * GAS_CONSTANT * temperature
            - self.b**2 * pressure
            + self.a(temperature) / np.sqrt(temperature)
        )
        coefficients.append(-GAS_CONSTANT * temperature)
        coefficients.append(pressure)

        polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="VMRK")
        logger.debug("MRK equation = %s", polynomial)
        volume_roots: np.ndarray = polynomial.roots()
        # Numerical solution could result in a small imaginery component, even though the real
        # root is purely real.
        real_roots: np.ndarray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
        # Physically meaningful volumes must be positive.
        positive_roots: np.ndarray = real_roots[real_roots > 0]
        # In general, several roots could be returned, and subclasses will need to determine which
        # is the correct volume to use.
        logger.debug("VMRK = %s", positive_roots)

        return positive_roots


@dataclass(kw_only=True)
class MRKCriticalBehaviour(RealGasABC):
    """A MRK model that accommodates critical behaviour

    Args:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation
        Tc: Critical temperature

    Attributes:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation
        Tc: Critical temperature
    """

    mrk_fluid: MRKImplicitABC
    mrk_gas: MRKImplicitABC
    mrk_liquid: MRKImplicitABC
    Ta: float
    Tc: float

    @abstractmethod
    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin

        Returns:
            Saturation curve pressure in bar
        """
        ...

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3/mol
        """
        Psat: float = self.Psat(temperature)

        if temperature <= self.Ta and pressure <= Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Gas phase")
            volume = self.mrk_gas.volume(temperature, pressure)

        elif temperature <= self.Ta and pressure > Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Liquid phase")
            volume = self.mrk_liquid.volume(temperature, pressure)

        else:
            logger.debug("Fluid phase")
            volume = self.mrk_fluid.volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral. Appendix A, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            volume integral in J mol^(-1)
        """
        Psat: float = self.Psat(temperature)

        if temperature <= self.Ta and pressure <= Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Gas phase")
            volume_integral = self.mrk_gas.volume_integral(temperature, pressure)

        elif temperature <= self.Ta and pressure > Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Performing pressure integration")
            volume_integral = self.mrk_gas.volume_integral(temperature, Psat)
            volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
            volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)

        else:
            logger.debug("Fluid phase")
            volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class VirialCompensation(RealGasABC):
    """A compensation term for the increasing deviation of the MRK volumes with pressure

    General form of the equation from Holland and Powell (1998), and also see Holland and Powell
    (1991) Equations 4 and 9:

        V_virial = a(P-P0) + b(P-P0)**0.5 + c(P-P0)**0.25

    This form also works for the virial compensation term from Holland and Powell (1991), in which
    case c=0. critical_pressure and critical_temperature are required for gases which are known to
    obey approximately the principle of corresponding states.

    Although this looks similar to an EOS, it's important to remember that for Holland and Powell
    it only calculates an additional perturbation to the volume and the volume integral of an MRK
    EOS, and hence it does not return a meaningful volume or volume integral by itself.

    Args:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be scaled (internally) by critical parameters for corresponding states
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Defaults to zero, which is
            appropriate for the corresponding states case.
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)

    Attributes:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally (internally) scaled by Tc and Pc in the case of corresponding
            states.
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
    """

    a_coefficients: tuple[float, float]
    b_coefficients: tuple[float, float]
    c_coefficients: tuple[float, float]
    P0: float

    @debug_decorator(logger)
    def a(self, temperature: float) -> float:
        """a parameter in Holland and Powell (1998)

        Args:
            temperature: Temperature in kelvin

        Returns:
            a parameter in J bar^(-2) mol^(-1)
        """
        a: float = (
            self.a_coefficients[0] * self.critical_temperature
            + self.a_coefficients[1] * temperature
        )
        a /= self.critical_pressure**2

        return a

    @debug_decorator(logger)
    def b(self, temperature: float) -> float:
        """b parameter in Holland and Powell (1998)

        Args:
            temperature: Temperature in kelvin

        Returns:
            b parameter in J bar^(-3/2) mol^(-1)
        """
        b: float = (
            self.b_coefficients[0] * self.critical_temperature
            + self.b_coefficients[1] * temperature
        )
        b /= self.critical_pressure ** (3 / 2)

        return b

    @debug_decorator(logger)
    def c(self, temperature: float) -> float:
        """c parameter in Holland and Powell (1998)

        Note the scalings by self.Tc and self.Pc to accommodate corresponding states.

        Args:
            temperature: Temperature in kelvin

        Returns:
            c parameter in J bar^(-5/4) mol^(-1)
        """
        c: float = (
            self.c_coefficients[0] * self.critical_temperature
            + self.c_coefficients[1] * temperature
        )
        c /= self.critical_pressure ** (5 / 4)

        return c

    @debug_decorator(logger)
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume contribution

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume contribution
        """
        volume: float = (
            self.a(temperature) * (pressure - self.P0)
            + self.b(temperature) * (pressure - self.P0) ** 0.5
            + self.c(temperature) * (pressure - self.P0) ** 0.25
        )

        return volume

    @debug_decorator(logger)
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP) contribution

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral contribution
        """
        volume_integral: float = (
            self.a(temperature) / 2.0 * (pressure - self.P0) ** 2
            + 2.0 / 3.0 * self.b(temperature) * (pressure - self.P0) ** (3.0 / 2.0)
            + 4.0 / 5.0 * self.c(temperature) * (pressure - self.P0) ** (5.0 / 4.0)
        )

        return volume_integral


@dataclass(kw_only=True)
class CORK(RealGasABC):
    """A Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991)

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)
        P0: Pressure in bar at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data
        mrk: MRK model for computing the MRK contribution
        a_virial: a coefficients for the virial compensation. Defaults to zero coefficients
        b_virial: b coefficients for the virial compensation. Defaults to zero coefficients
        c_virial: c coefficients for the virial compensation. Defaults to zero coefficients

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data
        mrk: MRK model for computing the MRK contribution
        a_virial: a coefficients for the virial compensation
        b_virial: b coefficients for the virial compensation
        c_virial: c coefficients for the virial compensation
        virial: A VirialCompensation instance
    """

    P0: float
    mrk: RealGasABC
    a_virial: tuple[float, float] = (0, 0)
    b_virial: tuple[float, float] = (0, 0)
    c_virial: tuple[float, float] = (0, 0)
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            P0=self.P0,
            critical_temperature=self.critical_temperature,
            critical_pressure=self.critical_pressure,
        )

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume including virial compensation. Equation 7a, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume including the virial compensation in m^3 mol^(-1)
        """
        volume: float = self.mrk.volume(temperature, pressure)

        if pressure > self.P0:
            volume += self.virial.volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral including virial compensation. Equation 8, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral including the virial compensation in J mol^(-1)
        """
        volume_integral: float = self.mrk.volume_integral(temperature, pressure)

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CombinedEOSModel(RealGasABC):
    """Combines multiple EOS models for different pressure ranges into a single model.

    Args:
        models: EOS models ordered by increasing pressure from lowest to highest
        upper_pressure_bounds: Upper pressure bound in bar relevant to the EOS by position

    Attributes:
        models: EOS models ordered by increasing pressure from lowest to highest
        upper_pressure_bounds: Upper pressure bound in bar relevant to the EOS by position
    """

    models: tuple[RealGasABC, ...]
    upper_pressure_bounds: tuple[float, ...]

    def _get_index(self, pressure: float) -> int:
        """Gets the index of the appropriate EOS model using the pressure

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """
        for index, pressure_high in enumerate(self.upper_pressure_bounds):
            if pressure < pressure_high:
                return index
        # If the pressure is higher than all specified pressure ranges, use the last model.
        return len(self.models) - 1

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        index: int = self._get_index(pressure)
        volume: float = self.models[index].volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        index: int = self._get_index(pressure)
        volume: float = self.models[index].volume_integral(temperature, pressure)

        return volume


@dataclass(frozen=True)
class critical_data:
    """Critical temperature and pressure of a gas species.

    Args:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure in bar

    Attributes:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure in bar
    """

    temperature: float
    pressure: float


# Critical temperature and pressure data for a corresponding states model, based on Table 2 in
# Shi and Saxena (1992) with some additions. For simplicity, we just compile one set of critical
# data, even though they vary a little between studies which could result in subtle differences.

# Holland and Powell use slightly different critical data in their 1991 paper, which makes
# insignificant differences in most cases, but their values are give in comments for completeness.
# However, for H2 their critical values are significantly different and are therefore retained as
# a separate entry.
critical_data_dictionary: dict[str, critical_data] = {
    "H2O": critical_data(647.25, 221.1925),
    "CO2": critical_data(304.15, 73.8659),  # 304.2, 73.8 from Holland and Powell (1991)
    "CH4": critical_data(191.05, 46.4069),  # 190.6, 46 from Holland and Powell (1991)
    "CO": critical_data(133.15, 34.9571),  # 132.9, 35 from Holland and Powell (1991)
    "O2": critical_data(154.75, 50.7638),
    "H2": critical_data(33.25, 12.9696),
    # Holland and Powell (1991) require different critical parameters
    "H2_Holland": critical_data(41.2, 21.1),
    # Holland and Powell (2011) state that the critical constants for S2 are taken from:
    # Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
    # McGraw-Hill, New York.
    # In the fifth edition of this book S2 is not given (only S is), so instead the critical
    # constants for S2 are taken from:
    # Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
    # Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.
    # http://www.minsocam.org/ammin/AM77/AM77_1038.pdf
    "S2": critical_data(208.15, 72.954),
    "SO2": critical_data(430.95, 78.7295),
    "COS": critical_data(377.55, 65.8612),
    # Appendix A.19 in:
    # Poling, Prausnitz, and O'Connell, 2001. The Properties of Gases and Liquids, 5th edition.
    # McGraw-Hill, New York. DOI: 10.1036/0070116822.
    "H2S": critical_data(373.55, 90.0779),  # 373.4, 0.08963
    "N2": critical_data(126.2, 33.9),  # Saxena and Fei (1987)
    "Ar": critical_data(151, 48.6),  # Saxena and Fei (1987)
}
