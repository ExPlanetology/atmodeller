"""Output

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
import pickle
from collections import UserDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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
    moles_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid
        self.moles_in_total = self.moles_in_atmosphere + self.moles_in_melt + self.moles_in_solid


@dataclass(kw_only=True)
class CondensedSpeciesOutput:
    """Output for a condensed species

    These data are not currently output because all condensed phases have an activity of unity
    """

    activity: float


class Output(UserDict):
    """Stores inputs and outputs of the models."""

    def __init__(self, dict=None, /, **kwargs):
        """Init definition from the base class provided for clarity."""
        self.data = {}
        if dict is not None:
            self.update(dict)
        if kwargs:
            self.update(kwargs)

    @property
    def size(self) -> int:
        """Number of rows"""
        try:
            return len(self.data["solution"])
        except KeyError:
            return 0

    @classmethod
    def read_pickle(cls, pickle_file: Path | str) -> Self:
        """Reads output data from a pickle file and creates an Output instance.

        Args:
            pickle_file: Pickle file of the output from a previous (or similar) model run.
                Importantly, the reaction network must be the same (same number of species in the
                same order) and the constraints must be the same (also in the same order).

        Returns:
            Output
        """
        with open(pickle_file, "rb") as handle:
            output_data: dict[str, list[dict[str, float]]] = pickle.load(handle)

        logger.info("%s: Reading data from %s", cls.__name__, pickle_file)

        return cls(output_data)

    def add(
        self, interior_atmosphere: InteriorAtmosphereSystem, extra_output: dict[str, float] | None
    ) -> None:
        """Adds all outputs.

        Args:
            interior_atmosphere: Interior atmosphere system
            extra_output: Extra data to write to the output
        """
        self._add_atmosphere(interior_atmosphere)
        self._add_constraints(interior_atmosphere)
        self._add_planet(interior_atmosphere)
        self._add_gas_species(interior_atmosphere)
        self._add_residual(interior_atmosphere)
        self._add_solution(interior_atmosphere)
        if extra_output is not None:
            data_list: list[dict[str, float]] = self.data.setdefault("extra", [])
            data_list.append(extra_output)

    def _add_atmosphere(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds atmosphere.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        atmosphere_dict: dict[str, float] = {}
        atmosphere_dict["total_pressure"] = interior_atmosphere.total_pressure
        atmosphere_dict["mean_molar_mass"] = interior_atmosphere.atmospheric_mean_molar_mass

        data_list: list[dict[str, float]] = self.data.setdefault("atmosphere", [])
        data_list.append(atmosphere_dict)

    def _add_constraints(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds constraints.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        evaluate_dict: dict[str, float] = interior_atmosphere.constraints.evaluate(
            interior_atmosphere
        )

        data_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
        data_list.append(evaluate_dict)

    def _add_planet(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the planetary properties.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        planet_dict: dict[str, float] = asdict(interior_atmosphere.planet)

        data_list: list[dict[str, float]] = self.data.setdefault("planet", [])
        data_list.append(planet_dict)

    def _add_gas_species(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds gas species.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        for species in interior_atmosphere.species.gas_species.values():
            assert species.output is not None
            data_list: list[dict[str, float]] = self.data.setdefault(species.formula, [])
            data_list.append(asdict(species.output))

    def _add_residual(self, interior_atmosphere: InteriorAtmosphereSystem):
        """Adds the residual.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("residual", [])
        data_list.append(interior_atmosphere.residual_dict)

    def _add_solution(self, interior_atmosphere: InteriorAtmosphereSystem):
        """Adds the solution.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("solution", [])
        data_list.append(interior_atmosphere.solution_dict)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Output as a dictionary of dataframes"""
        out: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(value) for key, value in self.data.items()
        }
        return out

    def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)

    def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        out: dict[str, pd.DataFrame] = self.to_dataframes()
        output_file: Path = Path(f"{file_prefix}.xlsx")

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output written to %s", output_file)

    def __call__(
        self,
        file_prefix: Path | str = "atmodeller_out",
        to_dict: bool = True,
        to_dataframes: bool = False,
        to_pickle: bool = False,
        to_excel: bool = False,
    ) -> dict | None:
        """Gets the output and/or optionally write it to a pickle or Excel file.

        Args:
            file_prefix: Prefix of the output file if writing to a pickle or Excel. Defaults to
                'atmodeller_out'
            to_dict: Returns the output data in a dictionary. Defaults to False.
            to_dataframes: Returns the output data in a dictionary of dataframes. Defaults to
                False.
            to_pickle: Writes a pickle file. Defaults to False.
            to_excel: Writes an Excel file. Defaults to False.

        Returns:
            A dictionary of the output if `to_dict = True`, otherwise None.
        """
        if to_pickle:
            self.to_pickle(file_prefix)

        if to_excel:
            self.to_excel(file_prefix)

        # Acts as an override if to_dict is also set.
        if to_dataframes:
            return self.to_dataframes()

        if to_dict:
            return self.data
