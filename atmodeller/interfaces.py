# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Interfaces for thermodynamic models and constraints.

This module defines formal protocol classes (via :class:`typing.Protocol`) that specify the
expected interfaces for different thermodynamic components.

It also contains :class:`ChemicalSpeciesData`, a concrete :class:`equinox.Module` that holds
per-species formula and composition data. It lives here rather than in :mod:`atmodeller.containers`
because :mod:`atmodeller.containers` imports from this module. Moving it there would create a
circular import.
"""

from typing import Optional, Protocol, TypeVar, runtime_checkable

import equinox as eqx
from jaxtyping import Array, ArrayLike, Bool
from molmass import Formula

from atmodeller.constants import GAS_SUFFIX
from atmodeller.jax_utils import FloatArray
from atmodeller.sci_utils import unit_conversion


class ChemicalSpeciesData(eqx.Module):
    """General data container for an individual species

    The charge of an ion, such as ``D+`` (``[[2H]]+``), is part of its composition as the element
    ``e-`` (electrons), and its molar mass includes the mass of the electrons. Charge balance is not
    enforced yet, so a model with ions needs another constraint, such as an imposed fugacity, to be
    determined.

    Args:
        formula: Formula
        suffix: Suffix of the name: ``g`` for a gas, ``l`` for a liquid, ``d`` for a dissolved
            species, the phase component for a mineral (e.g. ``quartz-alpha``), ``cd`` for
            stitched phase components, or another name for a chosen stitch
    """

    formula: str
    """Formula"""
    suffix: str
    """Suffix of the name"""
    composition: dict[str, tuple[int, float, float]]
    """Composition"""
    hill_formula: str
    """Hill formula"""
    molar_mass: float = eqx.field(converter=float)
    """Molar mass"""

    def __init__(self, formula: str, suffix: str):
        self.formula = formula
        self.suffix = suffix
        mformula: Formula = Formula(self.formula)
        self.composition = mformula.composition().asdict()
        self.hill_formula = mformula.formula
        self.molar_mass = mformula.mass * unit_conversion.g_to_kg

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    @property
    def name(self) -> str:
        """Unique name from the Hill formula and the suffix, e.g. ``H2O_g``"""
        return f"{self.hill_formula}_{self.suffix}"


def species_name(formula: str, suffix: str = GAS_SUFFIX) -> str:
    """Name of a species, from the Hill formula of its formula and a suffix

    This gives the names used for constraints and in the output, e.g. ``species_name("SO2")`` is
    ``O2S_g`` and ``species_name("SiO2", "l")`` is ``O2Si_l``.

    Args:
        formula: Formula
        suffix: Suffix of the name (see :class:`ChemicalSpeciesData`). Defaults to
            :data:`~atmodeller.constants.GAS_SUFFIX`.

    Returns:
        Name of the species
    """
    return ChemicalSpeciesData(formula, suffix).name


@runtime_checkable
class ActivityProtocol(Protocol):
    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: FloatArray
    ) -> ArrayLike:
        """Log activity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions of all species in the phase (dimensionless)

        Returns:
            Log activity (dimensionless)
        """
        ...


@runtime_checkable
class SpeciesProtocol(Protocol):
    """Protocol for a chemical species that participates in reactions"""

    @property
    def activity(self) -> ActivityProtocol:
        """Activity model"""
        ...

    @property
    def data(self) -> ChemicalSpeciesData:
        """Chemical species data"""
        ...

    @property
    def number_solution(self) -> int:
        """Number of solution quantities"""
        ...

    @property
    def solve_for_stability(self) -> bool:
        """Whether to solve for stability"""
        ...

    @property
    def include_in_phase_mass(self) -> bool:
        """Whether the species is included in phase-level mass, mole, and fraction aggregations"""
        ...


TSpecies_co = TypeVar("TSpecies_co", bound=SpeciesProtocol, covariant=True)


@runtime_checkable
class ActivityConstraintProtocol(Protocol):
    def active(self) -> Bool[Array, "..."]:
        """True if the constraint is active, otherwise False"""
        ...

    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Log activity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            - Log activity (dimensionless)
            - Log fugacity referenced to 1 bar for gaseous species
            - ``NaN`` if the constraint is not active
        """
        ...


@runtime_checkable
class RedoxBufferProtocol(ActivityConstraintProtocol, Protocol):
    evaluation_pressure: Optional[float]
    """Pressure at which to evaluate the buffer, or None to use the total pressure"""

    @property
    def log10_shift(self) -> FloatArray:
        """Log10 shift relative to the buffer"""
        ...

    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Log10 fugacity at the unshifted buffer

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity at the buffer
        """
        ...

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Log10 fugacity including any shift

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log10 fugacity
        """
        ...


@runtime_checkable
class SolubilityProtocol(Protocol):
    """Solubility protocol

    :meth:`~SolubilityProtocol.jax_concentration` is defined in order to allow arguments to be
    passed by position, which is the form used by the dissolution network and by JAX
    transformations.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
        fO2: Optional[ArrayLike] = None,
    ) -> FloatArray:
        r"""Concentration in ppmw

        Args:
            fugacity: Fugacity (bar)
            temperature: Temperature (K). Defaults to ``None`` if unused.
            pressure: Pressure (bar). Defaults to ``None`` if unused.
            fO2: Oxygen fugacity (bar). Defaults to ``None`` if unused.

        Returns:
            Concentration in ppmw
        """
        ...

    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> FloatArray:
        """Wrapper to pass concentration arguments by position

        Args:
            fugacity: Fugacity (bar)
            temperature: Temperature (K)
            pressure: Pressure (bar)
            fO2: Oxygen fugacity (bar)

        Returns:
            Concentration in ppmw
        """
        ...


@runtime_checkable
class HeatCapacityProtocol(Protocol):
    """Protocol for a heat capacity model"""

    def cp(self, temperature: ArrayLike) -> FloatArray:
        r"""Gets heat capacity at constant pressure.

        Args:
            temperature: Temperature (K)

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        ...

    def temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range over which the heat capacity model is valid.

        Returns:
            Minimum and maximum temperature (K)
        """
        ...


@runtime_checkable
class VolumeProtocol(Protocol):
    """Protocol for the volume model of a condensed phase"""

    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Gets the integral of volume with respect to pressure from 1 bar.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Integral of volume with respect to pressure in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        ...
