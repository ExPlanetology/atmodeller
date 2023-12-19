"""Output

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.constraints import SystemConstraints
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem


@dataclass(kw_only=True)
class GasSpeciesOutput:
    """Output for a gas species"""

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
    pressure: float  # bar
    volume_mixing_ratio: float  # dimensionless
    mass_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


@dataclass(kw_only=True)
class CondensedSpeciesOutput:
    """Output for a condensed species

    These data are not currently output because all condensed phases have an activity of unity
    """

    activity: float


@dataclass
class Output:
    """Stores inputs and outputs of the models.

    Args:
        interior_atmosphere: An interior-atmosphere system
    """

    _interior_atmosphere: InteriorAtmosphereSystem
    _gas_species: dict[str, list[dict]] = field(init=False, default_factory=dict)
    _atmosphere: list[dict[str, float]] = field(init=False, default_factory=list)
    _constraints: list[dict[str, float]] = field(init=False, default_factory=list)
    _planet_properties: list[dict[str, float]] = field(init=False, default_factory=list)
    _solution: list[dict[str, float]] = field(init=False, default_factory=list)

    def __post_init__(self):
        # Initialises the dictionary to store the detailed gas species outputs.
        for species in self._interior_atmosphere.species.gas_species.values():
            self._gas_species[species.chemical_formula] = []

    @property
    def atmosphere(self) -> pd.DataFrame:
        """Atmosphere data"""
        return pd.DataFrame(self._atmosphere)

    @property
    def constraints(self) -> pd.DataFrame:
        """Constraints data"""
        return pd.DataFrame(self._constraints)

    @property
    def gas_species(self) -> dict[str, pd.DataFrame]:
        """Gas species data"""
        gas_species: dict[str, pd.DataFrame] = {
            species: pd.DataFrame(data) for species, data in self._gas_species.items()
        }
        return gas_species

    @property
    def planet(self) -> pd.DataFrame:
        """Planet data"""
        return pd.DataFrame(self._planet_properties)

    @property
    def solution(self) -> pd.DataFrame:
        """Solution data"""
        return pd.DataFrame(self._solution)

    @property
    def size(self) -> int:
        """Number of rows"""
        return len(self._solution)

    def add(self, constraints: SystemConstraints) -> None:
        """Adds all outputs.

        Args:
            constraints: Constraints
        """
        self._add_atmosphere()
        self._add_constraints(constraints)
        self._add_planet()
        self._add_gas_species()
        self._add_solution()

    def _add_atmosphere(self) -> None:
        """Adds atmosphere."""
        atmosphere_dict: dict[str, float] = {}
        atmosphere_dict["total_pressure"] = self._interior_atmosphere.total_pressure
        atmosphere_dict["mean_molar_mass"] = self._interior_atmosphere.atmospheric_mean_molar_mass
        self._atmosphere.append(atmosphere_dict)

    def _add_constraints(self, constraints: SystemConstraints) -> None:
        """Adds constraints.

        Args:
            constraints: Constraints
        """
        evaluate_dict = constraints.evaluate(self._interior_atmosphere)
        self._constraints.append(evaluate_dict)

    def _add_planet(self) -> None:
        """Adds the planetary properties."""
        planet_dict: dict[str, float] = asdict(self._interior_atmosphere.planet)
        self._planet_properties.append(planet_dict)

    def _add_gas_species(self) -> None:
        """Adds gas species."""
        for species in self._interior_atmosphere.species.gas_species.values():
            assert species.output is not None
            self._gas_species[species.chemical_formula].append(asdict(species.output))

    def _add_solution(self):
        """Adds the solution."""
        self._solution.append(self._interior_atmosphere.solution_dict)

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Gets the output dictionary of dataframes.

        Returns:
            An output dictionary
        """
        out: dict[str, pd.DataFrame] = {}
        out["solution"] = self.solution
        out["atmosphere"] = self.atmosphere
        out["constraints"] = self.constraints
        out["planet"] = self.planet
        for gas_species, data in self.gas_species.items():
            out[gas_species] = data

        return out

    def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        out: dict[str, pd.DataFrame] = self.as_dict()
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output data written to %s", output_file)

    def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        out: dict[str, pd.DataFrame] = self.as_dict()
        output_file: Path = Path(f"{file_prefix}.xlsx")

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output data written to %s", output_file)

    def __call__(
        self,
        file_prefix: Path | str = "atmodeller_out",
        to_dict: bool = True,
        to_pickle: bool = False,
        to_excel: bool = False,
    ) -> dict[str, pd.DataFrame] | None:
        """Gets the output and/or optionally write it to a pickle or Excel file.

        Args:
            file_prefix: Prefix of the output file if writing to a pickle or Excel. Defaults to
                'atmodeller_out'
            to_dict: Returns the output data in a dictionary. Defaults to True.
            to_pickle: Writes a pickle file. Defaults to False.
            to_excel: Writes an Excel file. Defaults to False.

        Returns:
            A dictionary of the output if `to_dict = True`, otherwise None.
        """

        if to_pickle:
            self.to_pickle(file_prefix)

        if to_excel:
            self.to_excel(file_prefix)

        if to_dict:
            out: dict[str, pd.DataFrame] = self.as_dict()
            return out
