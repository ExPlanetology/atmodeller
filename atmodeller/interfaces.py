# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

from atmodeller.jax_utils import FloatArray
from atmodeller.sci_utils import unit_conversion


class ChemicalSpeciesData(eqx.Module):
    """General data container for an individual species

    Args:
        formula: Formula
        state: State of aggregation, typically follows JANAF convention: 'g' for gas, 'l' for
            liquid, 's' for solid.
    """

    formula: str
    """Formula"""
    state: str
    """State of aggregation. Defaults to an empty string."""
    composition: dict[str, tuple[int, float, float]]
    """Composition"""
    hill_formula: str
    """Hill formula"""
    molar_mass: float = eqx.field(converter=float)
    """Molar mass"""

    def __init__(self, formula: str, state: str = ""):
        self.formula = formula
        self.state = state
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
        """Unique name by combining Hill notation and state of aggregation"""
        return f"{self.hill_formula}_{self.state}"


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
            - :data:`jax.numpy.nan` if the constraint is not active
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
    passed by position to :func:`jax.lax.switch`.
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
        """Wrapper to pass concentration arguments by position for :func:`jax.lax.switch`

        Args:
            fugacity: Fugacity (bar)
            temperature: Temperature (K)
            pressure: Pressure (bar)
            fO2: Oxygen fugacity (bar)

        Returns:
            Concentration in ppmw
        """
        ...
