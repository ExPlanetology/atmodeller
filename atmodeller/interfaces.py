"""Interfaces.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, Union

import numpy as np
import pandas as pd
from molmass import Formula
from thermochem import janaf

from atmodeller import DATA_ROOT_PATH, GAS_CONSTANT
from atmodeller.utilities import UnitConversion, debug_decorator

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet


class GetValueABC(ABC):
    """An object with a get_value method."""

    @abstractmethod
    def get_value(self, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            **kwargs: Keyword arguments only

        Returns:
            An evaluation based on the provided arguments
        """
        ...

    def get_log10_value(self, **kwargs) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            **kwargs: Keyword arguments only

        Returns:
            An evaluation of the log10 value based on the provided arguments
        """
        return np.log10(self.get_value(**kwargs))


@dataclass(kw_only=True)
class RealGasABC(GetValueABC):
    """A real gas equation of state (EOS)

    This base class requires a specification for the volume and volume integral. Then the
    fugacity and related quantities can be computed using the standard relation:

    RTlnf = integral(VdP)

    If critical_temperature and critical_pressure are set to their default value of unity, then
    these quantities are effectively not used, and the model coefficients should be in terms of
    the real temperature and pressure. But for corresponding state models, which are formulated in
    terms of a reduced temperature and a reduced pressure, the critical_temperature and
    critical_pressure must be set to appropriate values for the species under consideration.

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
    """

    critical_temperature: float = 1  # Default of one is equivalent to not used
    critical_pressure: float = 1  # Default of one is equivalent to not used
    standard_state_pressure: float = field(init=False, default=1)  # 1 bar

    @debug_decorator(logger)
    def scaled_pressure(self, pressure: float) -> float:
        """Scaled pressure, i.e. a reduced pressure when critical pressure is not unity.

        Args:
            pressure: Pressure in bar

        Returns:
            The scaled (reduced) pressure, which is dimensionless
        """
        scaled_pressure: float = pressure / self.critical_pressure

        return scaled_pressure

    @debug_decorator(logger)
    def scaled_temperature(self, temperature: float) -> float:
        """Scaled temperature, i.e. a reduced temperature when critical temperature is not unity.

        Args:
            temperature: Temperature in kelvin

        Returns:
            The scaled (reduced) temperature, which is dimensionless.
        """
        scaled_temperature: float = temperature / self.critical_temperature

        return scaled_temperature

    @debug_decorator(logger)
    def compressibility_parameter(self, temperature: float, pressure: float, **kwargs) -> float:
        """Compressibility parameter at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar
            **kwargs: Catches unused keyword arguments. Used for overrides in subclasses.

        Returns:
            The compressibility parameter, Z, which is dimensionless.
        """
        del kwargs
        volume: float = self.volume(temperature, pressure)
        volume_ideal: float = self.ideal_volume(temperature, pressure)
        Z: float = volume / volume_ideal

        return Z

    @debug_decorator(logger)
    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Fugacity coefficient evaluated at temperature and pressure, which is dimensionaless.
        """
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)

        return fugacity_coefficient

    @debug_decorator(logger)
    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity.

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Natural log of the fugacity
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT * temperature
        )

        return ln_fugacity

    @debug_decorator(logger)
    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in the same units as the input pressure.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))

        return fugacity

    @debug_decorator(logger)
    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            fugacity coefficient, which is non-dimensional.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    @debug_decorator(logger)
    def ideal_volume(self, temperature: float, pressure: float) -> float:
        """Ideal volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            ideal volume in m^3 mol^(-1)
        """
        volume_ideal: float = GAS_CONSTANT * temperature / pressure

        return volume_ideal

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP).

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        ...


@dataclass(kw_only=True)
class IdealGas(RealGasABC):
    """An ideal gas, PV=RT"""

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        return self.ideal_volume(temperature, pressure)

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        volume_integral: float = GAS_CONSTANT * temperature * np.log(pressure)

        return volume_integral


@dataclass(kw_only=True, frozen=True)
class ConstraintABC(GetValueABC):
    """A constraint to apply to an interior-atmosphere system.

    Args:
        name: The name of the constraint, which should be one of: 'fugacity', 'pressure', or
            'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.

    Attributes:
        name: The name of the constraint.
        species: The species to constrain.
    """

    name: str
    species: str


@dataclass(kw_only=True, frozen=True)
class ConstantConstraint(ConstraintABC):
    """A constraint of a constant value.

    Args:
        name: The name of the constraint, which should be one of: 'fugacity', 'pressure', or
            'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.
        value: The constant value, which is usually in kg for masses and bar for pressures or
            fugacities.

    Attributes:
        name: The name of the constraint.
        species: The species to constrain.
        value: The constant value.
    """

    value: float

    def get_value(self, **kwargs) -> float:
        """Returns the constant value. See base class."""
        del kwargs
        return self.value


@dataclass(kw_only=True, frozen=True)
class IdealityConstant(ConstantConstraint):
    """A constant activity.

    The constructor must accept no arguments to enable it to be used as a default factory when the
    user does not specify an activity model for a solid species. Therefore, the name and species
    arguments are set to empty strings because they are not used.

    Args:
        value: The constant value. Defaults to 1 (i.e. ideal behaviour).

    Attributes:
        value: The constant value.
    """

    name: str = field(init=False, default="")
    species: str = field(init=False, default="")
    value: float = 1.0


# Solubility limiter applied universally
MAXIMUM_PPMW: float = UnitConversion.weight_percent_to_ppmw(10)  # 10% by weight.


def limit_solubility(bound: float = MAXIMUM_PPMW) -> Callable:
    """A decorator to limit the solubility in ppmw.

    Args:
        bound: The maximum limit of the solubility in ppmw. Defaults to MAXIMUM_PPMW.

    Returns:
        The decorator.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Solubility, *args, **kwargs):
            result: float = func(self, *args, **kwargs)
            if result > bound:
                msg: str = "%s solubility (%d ppmw) will be limited to %d ppmw" % (
                    self.__class__.__name__,
                    result,
                    bound,
                )
                logger.warning(msg)

            return np.clip(result, 0, bound)  # Limit the result between 0 and 'bound'

        return wrapper

    return decorator


class Solubility(GetValueABC):
    """A solubility law for a species."""

    def power_law(self, fugacity: float, constant: float, exponent: float) -> float:
        """Computes solubility from a power law.

        Args:
            fugacity: Fugacity of the species in bar.
            constant: Constant for the power law.
            exponent: Exponent for the power law.

        Returns:
            Dissolved volatile concentration in the melt in ppmw.
        """
        return constant * fugacity**exponent

    @abstractmethod
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Dissolved volatile concentration in the melt in ppmw.

        Args:
            fugacity: Fugacity of the species in bar.
            temperature: Temperature in kelvin.
            log10_fugacities_dict: Log10 fugacities of all species in the system.

        Returns:
            Dissolved volatile concentration in the melt in ppmw.
        """
        raise NotImplementedError

    @limit_solubility()  # Note this limiter is always applied.
    def get_value(
        self,
        *,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Dissolved volatile concentration in the melt in ppmw.

        See self._solubility.
        """
        solubility: float = self._solubility(
            fugacity, temperature, log10_fugacities_dict, pressure
        )
        logger.debug(
            "%s, f = %f, T = %f, ppmw = %f",
            self.__class__.__name__,
            fugacity,
            temperature,
            solubility,
        )
        return solubility


class NoSolubility(Solubility):
    """No solubility."""

    def _solubility(self, *args, **kwargs) -> float:
        """See base class."""
        del args
        del kwargs
        return 0.0


class ThermodynamicDataBase(ABC):
    """Thermodynamic data base class."""

    _DATA_SOURCE: str
    _ENTHALPY_REFERENCE_TEMPERATURE: float  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @abstractmethod
    def __init__(self, species: ChemicalComponent):
        self.species: ChemicalComponent = species

    @property
    def DATA_SOURCE(self) -> str:
        """Identifies the source of the data."""
        return self._DATA_SOURCE

    @property
    def ENTHALPY_REFERENCE_TEMPERATURE(self) -> float:
        """Enthalpy reference temperature in kelvin."""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def STANDARD_STATE_PRESSURE(self) -> float:
        """Standard state pressure in bar."""
        return self._STANDARD_STATE_PRESSURE

    @abstractmethod
    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Computes the standard Gibbs free energy of formation in units of J/mol.

        Args:
            temperature: Temperature in kelvin.
            pressure: Total pressure in bar.

        Returns:
            The standard Gibbs free energy of formation in J/mol.
        """
        ...


@dataclass(kw_only=True)
class GasSpeciesOutput:
    """Output for a gas species."""

    mass_in_atmosphere: float  # kg
    mass_in_solid: float  # kg
    mass_in_melt: float  # kg
    moles_in_atmosphere: float  # moles
    moles_in_melt: float  # moles
    moles_in_solid: float  # moles
    ppmw_in_solid: float  # ppm by weight
    ppmw_in_melt: float  # ppm by weight
    fugacity: float  # bar
    fugacity_coefficient: float  # dimensionless
    pressure_in_atmosphere: float  # bar
    volume_mixing_ratio: float  # dimensionless
    mass_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the gas species or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: GasSpecies, element: Optional[str] = None, **kwargs) -> float:
        """Wrapper to return the mass of either the gas species or one of its elements.

        Args:
            element: Returns the mass of this element. Defaults to None to return the species mass.
            **kwargs: Catches keyword arguments to forward to func.

        Returns:
            Mass of either the gas species or one of its elements.
        """
        mass: float = func(self, **kwargs)
        if element is not None:
            try:
                mass *= (
                    UnitConversion.g_to_kg(self.formula.composition()[element].mass)
                    / self.molar_mass
                )
            except KeyError:  # Element not in formula so mass is zero.
                mass = 0

        return mass

    return mass_wrapper


class ThermodynamicDataJANAF(ThermodynamicDataBase):
    """Thermodynamic data from the JANAF tables.

    Args:
        species: Chemical component.
    """

    _DATA_SOURCE: str = "JANAF"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    def __init__(self, species: ChemicalComponent):
        """Init.

        Quoting from the JANAF documentation: In the JANAF tables, we have generally chosen the
        ideal diatomic gas for the reference state of permanent gases such as O2, N2, Cl2 etc.
        """

        super().__init__(species)
        self.data: janaf.JanafPhase = self._get_phase_data()

    def _get_phase_data(self) -> janaf.JanafPhase:
        """Gets the relevant phase data for the species.

        Returns:
            A JanafPhase instance.
        """

        db: janaf.Janafdb = janaf.Janafdb()

        def get_phase_data(phase) -> Union[janaf.JanafPhase, None]:
            try:
                phase_data: janaf.JanafPhase = db.getphasedata(
                    formula=self.species.modified_hill_formula, phase=phase
                )
            except ValueError:
                return None
            return phase_data

        if isinstance(self.species, GasSpecies):
            if self.species.is_homonuclear_diatomic:
                phase_data_ref: Union[janaf.JanafPhase, None] = get_phase_data("ref")
                phase_data_g: Union[janaf.JanafPhase, None] = get_phase_data("g")
                if phase_data_ref is None and phase_data_g is None:
                    msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                        self.species.name_in_thermodynamic_data,
                        self.species.modified_hill_formula,
                        self.DATA_SOURCE,
                    )
                    logger.warning(msg)
                    raise KeyError(msg)
                phase = phase_data_ref or phase_data_g
            else:
                phase = get_phase_data("g")
                if phase is None:
                    msg: str = "Thermodynamic data for %s (%s) is not available in %s" % (
                        self.species.name_in_thermodynamic_data,
                        self.species.modified_hill_formula,
                        self.DATA_SOURCE,
                    )
                    logger.warning(msg)
                    raise KeyError(msg)
        else:
            phase = get_phase_data("ref")

        assert phase is not None
        logger.debug(
            "Thermodynamic data for %s (%s) = %s",
            self.species.name_in_thermodynamic_data,
            self.species.modified_hill_formula,
            phase,
        )

        return phase

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Computes the standard Gibbs free energy of formation in J/mol.

        Args:
            temperature: Temperature in kelvin.
            pressure: Total pressure in bar.

        Returns:
            The standard Gibbs free energy of formation in J/mol.
        """
        del pressure
        gibbs: float = self.data.DeltaG(temperature)

        logger.debug(
            "Species = %s, standard Gibbs energy of formation = %f",
            self.species.name_in_thermodynamic_data,
            gibbs,
        )

        return gibbs


class ThermodynamicDataHollandAndPowell(ThermodynamicDataBase):
    """Thermodynamic data from Holland and Powell (1998).

    https://ui.adsabs.harvard.edu/abs/1998JMetG..16..309H

    The book 'Equilibrium thermodynamics in petrology: an introduction' by R. Powell also has
    a useful appendix A with equations.

    See the comments in the data file that is parsed by __init__

    Args:
        species: Chemical component.

    Raises:
        KeyError: Thermodynamic data is not available for the species.
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar
    dKdP: float = 4.0  # Derivative of bulk modulus (K) with respect to pressure.
    dKdT_factor: float = -1.5e-4  # Factor for computing the temperature-dependence of K.

    def __init__(self, species: ChemicalComponent):
        super().__init__(species)
        self.data: pd.Series = self._get_phase_data()

    def _get_phase_data(self) -> pd.Series:
        """Gets the relevant phase data for the species.

        Returns:
            The phase data.
        """

        data_path: Path = DATA_ROOT_PATH / Path("Mindata161127.csv")  # type: ignore
        data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        data["name of phase component"] = data["name of phase component"].str.strip()
        data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        data.drop(columns="Abbreviation", inplace=True)
        data.set_index("name of phase component", inplace=True)
        data = data.loc[:, :"Vmax"]
        data = data.astype(float)

        try:
            return data.loc[self.species.name_in_thermodynamic_data]
        except KeyError as exc:
            msg: str = "Thermodynamic data for %s (%s) is not available in %s" % (
                self.species.name_in_thermodynamic_data,
                self.species.hill_formula,
                self.DATA_SOURCE,
            )
            logger.error(msg)
            raise KeyError from exc

    def get_enthalpy(self, temperature: float) -> float:
        """Calculates the enthalpy at temperature.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Enthalpy in J.
        """
        H = self.data["Hf"]  # J
        a = self.data["a"]  # J/K           Coeff for calc heat capacity.
        b = self.data["b"]  # J/K^2         Coeff for calc heat capacity.
        c = self.data["c"]  # J K           Coeff for calc heat capacity.
        d = self.data["d"]  # J K^(-1/2)    Coeff for calc heat capacity.

        integral_H: float = (
            H
            + a * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
            + b / 2 * (temperature**2 - self.ENTHALPY_REFERENCE_TEMPERATURE**2)
            - c * (1 / temperature - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE)
            + 2 * d * (temperature**0.5 - self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
        )
        return integral_H

    def get_entropy(self, temperature: float) -> float:
        """Calculates the entropy at temperature.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Entropy in J/K.
        """
        S = self.data["S"]  # J/K
        a = self.data["a"]  # J/K           Coeff for calc heat capacity.
        b = self.data["b"]  # J/K^2         Coeff for calc heat capacity.
        c = self.data["c"]  # J K           Coeff for calc heat capacity.
        d = self.data["d"]  # J K^(-1/2)    Coeff for calc heat capacity.

        integral_S: float = (
            S
            + a * np.log(temperature / self.ENTHALPY_REFERENCE_TEMPERATURE)
            + b * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
            - c / 2 * (1 / temperature**2 - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE**2)
            - 2 * d * (1 / temperature**0.5 - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
        )
        return integral_S

    def get_volume_at_temperature(self, temperature: float) -> float:
        """Calculates the volume at temperature.

        The exponential arises from the strict derivation, but often an expansion is performed
        where exp(x) = 1+x as in Holland and Powell (1998). Below the exp term is retained, but
        the equation in Holland and Powell (1998) p311 is expanded.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Volume in J/bar.
        """
        V = self.data["V"]  # J/bar
        alpha0 = self.data["a0"]  # K^(-1), thermal expansivity

        volume_T: float = V * np.exp(
            alpha0 * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
            - 2 * 10.0 * alpha0 * (temperature**0.5 - self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
        )
        return volume_T

    def get_bulk_modulus_at_temperature(self, temperature: float) -> float:
        """Calculates the bulk modulus at temperature.

        Holland and Powell (1998), p312 in the text.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Bulk modulus in bar..
        """
        K = self.data["K"]  # Bulk modulus in bar.
        bulk_modulus_T: float = K * (
            1 + self.dKdT_factor * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
        )
        return bulk_modulus_T

    def get_volume_pressure_integral(self, temperature: float, pressure: float) -> float:
        """Computes the volume-pressure integral.

        Holland and Powell (1998), p312.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            The volume-pressure integral.
        """
        V_T: float = self.get_volume_at_temperature(temperature)
        K_T: float = self.get_bulk_modulus_at_temperature(temperature)
        integral_VP: float = (
            V_T
            * K_T
            / (self.dKdP - 1)
            * ((1 + self.dKdP * (pressure - 1.0) / K_T) ** (1.0 - 1.0 / self.dKdP) - 1)
        )  # J, use P-1.0 instead of P.
        return integral_VP

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure (total) in bar.

        Returns:
            The standard Gibbs free energy of formation in J/mol.
        """

        gibbs: float = self.get_enthalpy(temperature) - temperature * self.get_entropy(temperature)

        if isinstance(self.species, SolidSpecies):
            gibbs += self.get_volume_pressure_integral(temperature, pressure)

        logger.debug(
            "Species = %s, standard Gibbs energy of formation = %f",
            self.species.name_in_thermodynamic_data,
            gibbs,
        )

        return gibbs


class ThermodynamicData(ThermodynamicDataBase):
    """Combines thermodynamic data from multiple datasets.

    Args:
        species: Chemical component.
        datasets: A list of thermodynamic data to use. Defaults to Holland and Powell, and JANAF.
    """

    _DATA_SOURCE: str = "Combined"
    _STANDARD_STATE_PRESSURE: float = 1  # bar
    # We assume the JANAF reference temperature, which is close enough to the reference temperature
    # of Holland and Powell of 298 K (which could in fact be the same, if they simply decided to
    # drop the decimal points when reporting the reference temperature?).
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K

    def __init__(
        self,
        species: ChemicalComponent,
        datasets: Union[list[ThermodynamicDataBase], None] = None,
    ):
        super().__init__(species)
        if datasets is None:
            self.datasets: list[ThermodynamicDataBase] = []
            self.add_dataset(ThermodynamicDataHollandAndPowell(species))
            self.add_dataset(ThermodynamicDataJANAF(species))
        else:
            self.datasets = datasets

    def add_dataset(self, dataset: ThermodynamicDataBase) -> None:
        """Adds a thermodynamic dataset.

        Args:
            dataset: A thermodynamic dataset.
        """
        if len(self.datasets) >= 1:
            logger.warning("Combining different thermodynamic data may result in inconsistencies")
        logger.info("Adding thermodynamic data: %s", dataset.DATA_SOURCE)
        self.datasets.append(dataset)

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """The standard Gibbs free energy of formation in J/mol.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure (total) in bar.

        Returns:
            The standard Gibbs free energy of formation in J/mol.
        """
        for dataset in self.datasets:
            try:
                gibbs: float = dataset.get_formation_gibbs(
                    temperature=temperature, pressure=pressure
                )
                return gibbs
            except KeyError:
                continue

        msg: str = "Thermodynamic data for %s (%s) is not available in any dataset" % (
            self.species.name_in_thermodynamic_data,
            self.species.hill_formula,
        )
        logger.error(msg)
        raise KeyError(msg)


@dataclass(kw_only=True)
class ChemicalComponent(ABC):
    """A chemical component and its properties.

    Args:
        chemical_formula: Chemical formula (e.g., CO2, C, CH4, etc.).
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        thermodynamic_class: The class for thermodynamic data. Defaults to JANAF.

    Attributes:
        chemical_formula: Chemical formula.
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        formula: Formula object derived from the chemical formula.
        thermodynamic_class: The class for thermodynamic data.
        thermodynamic_data: Instance of thermodynamic_class for this chemical component.
    """

    chemical_formula: str
    name_in_thermodynamic_data: str
    thermodynamic_class: Type[ThermodynamicDataBase] = ThermodynamicDataJANAF
    formula: Formula = field(init=False)
    thermodynamic_data: ThermodynamicDataBase = field(init=False)
    output: Any = field(init=False, default=None)

    def __post_init__(self):
        logger.info(
            "Creating a %s: %s (%s)",
            self.__class__.__name__,
            self.name_in_thermodynamic_data,
            self.chemical_formula,
        )
        self.formula = Formula(self.chemical_formula)
        self.thermodynamic_data = self.thermodynamic_class(self)

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        return UnitConversion.g_to_kg(self.formula.mass)

    @property
    def hill_formula(self) -> str:
        """Hill formula."""
        return self.formula.formula

    @property
    def is_homonuclear_diatomic(self) -> bool:
        """True if the species is homonuclear diatomic otherwise False."""

        composition = self.formula.composition()

        if len(list(composition.keys())) == 1 and list(composition.values())[0].count == 2:
            return True
        else:
            return False

    @cached_property
    def modified_hill_formula(self) -> str:
        """Modified Hill formula.

        JANAF uses the modified Hill formula to index its data tables. In short, H, if present,
        should appear after C (if C is present), otherwise it must be the first element.
        """
        elements: dict[str, int] = {
            element: properties.count for element, properties in self.formula.composition().items()
        }

        if "C" in elements:
            ordered_elements: list[str] = ["C"]
        else:
            ordered_elements = []

        if "H" in elements:
            ordered_elements.append("H")

        ordered_elements.extend(sorted(elements.keys() - {"C", "H"}))

        formula_string: str = "".join(
            [
                element + (str(elements[element]) if elements[element] > 1 else "")
                for element in ordered_elements
            ]
        )
        logger.debug("Modified Hill formula = %s", formula_string)

        return formula_string


@dataclass(kw_only=True)
class GasSpecies(ChemicalComponent):
    """A gas species.

    Args:
        chemical_formula: Chemical formula (e.g. CO2, C, CH4, etc.)
        thermodynamic_class: The class for thermodynamic data. Defaults to JANAF
        solubility: Solubility model. Defaults to no solubility
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0
        eos: A gas equation of state. Defaults to an ideal gas.

    Attributes:
        chemical_formula: Chemical formula
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data
        formula: Formula object derived from the chemical formula
        thermodynamic_class: The class for thermodynamic data
        thermodynamic_data: Instance of thermodynamic_class for this chemical component
        solubility: Solubility model
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt
        eos: A gas equation of state
        output: Stores calculated values for output
    """

    name_in_thermodynamic_data: str = field(init=False)
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0
    output: Union[GasSpeciesOutput, None] = field(init=False, default=None)
    eos: RealGasABC = field(default_factory=IdealGas)

    def __post_init__(self):
        self.name_in_thermodynamic_data = self.chemical_formula
        super().__post_init__()

    @_mass_decorator
    def mass(
        self,
        *,
        planet: Planet,
        system: InteriorAtmosphereSystem,
        element: Optional[str] = None,
    ) -> float:
        """Calculates the total mass of the species or element.

        Args:
            planet: Planet properties.
            system: Interior atmosphere system.
            element: Returns the mass for an element. Defaults to None to return the species mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the species (element=None) or element (element=element).
        """

        del element

        pressure: float = system.solution_dict[self.chemical_formula]
        fugacity: float = system.fugacities_dict[self.chemical_formula]
        fugacity_coefficient: float = (
            10 ** system.log10_fugacity_coefficients_dict[self.chemical_formula]
        )

        # Atmosphere.
        mass_in_atmosphere: float = UnitConversion.bar_to_Pa(pressure) / planet.surface_gravity
        mass_in_atmosphere *= (
            planet.surface_area * self.molar_mass / system.atmospheric_mean_molar_mass
        )
        volume_mixing_ratio: float = pressure / system.total_pressure
        moles_in_atmosphere: float = mass_in_atmosphere / self.molar_mass

        # Melt.
        prefactor: float = planet.mantle_mass * planet.mantle_melt_fraction
        ppmw_in_melt: float = self.solubility.get_value(
            fugacity=fugacity,
            temperature=planet.surface_temperature,
            log10_fugacities_dict=system.log10_fugacities_dict,
            pressure=system.total_pressure,
        )
        mass_in_melt: float = prefactor * ppmw_in_melt * UnitConversion.ppm_to_fraction()
        moles_in_melt: float = mass_in_melt / self.molar_mass

        # Solid.
        prefactor: float = planet.mantle_mass * (1 - planet.mantle_melt_fraction)
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass_in_solid: float = prefactor * ppmw_in_solid * UnitConversion.ppm_to_fraction()
        moles_in_solid: float = mass_in_solid / self.molar_mass

        self.output = GasSpeciesOutput(
            mass_in_atmosphere=mass_in_atmosphere,
            mass_in_solid=mass_in_solid,
            mass_in_melt=mass_in_melt,
            moles_in_atmosphere=moles_in_atmosphere,
            moles_in_melt=moles_in_melt,
            moles_in_solid=moles_in_solid,
            ppmw_in_solid=ppmw_in_solid,
            ppmw_in_melt=ppmw_in_melt,
            fugacity=fugacity,
            fugacity_coefficient=fugacity_coefficient,
            pressure_in_atmosphere=pressure,
            volume_mixing_ratio=volume_mixing_ratio,
        )

        return self.output.mass_in_total


@dataclass(kw_only=True)
class SolidSpeciesOutput:
    """Output for a solid species."""

    activity: float


@dataclass(kw_only=True)
class SolidSpecies(ChemicalComponent):
    """A solid species.

    Args:
        chemical_formula: Chemical formula (e.g., CO2, C, CH4, etc.).
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        thermodynamic_class: The class for thermodynamic data. Defaults to JANAF.
        activity: Activity object. Defaults to ideal (i.e. unity).

    Attributes:
        chemical_formula: Chemical formula.
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        formula: Formula object derived from the chemical formula.
        thermodynamic_class: The class for thermodynamic data.
        thermodynamic_data: Instance of thermodynamic_class for this chemical component.
        activity: Activity object.
        output: Stores calculated values for output.
    """

    output: Union[SolidSpeciesOutput, None] = field(init=False, default=None)
    activity: ConstraintABC = field(default_factory=IdealityConstant)
