#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Real gas EOSs from :cite:t:`CD21`. 

Examples:
    Evaluate the fugacity coefficient for H2 at 2000 K and 1000 bar::

        from atmodeller.eos.chabrier import H2_CD21
        model = H2_CD21
        fugacity_coefficient = model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)

    Get the preferred EOS models for various species from the Chabrier and colleagues models::

        from atmodeller.eos.chabrier import get_chabrier_eos_models
        models = get_chabrier_eos_models()
        # List the available species
        models.keys()
        # Get the EOS model for H2
        h2_model = models['H2']
        # Determine the fugacity coefficient at 2000 K and 1000 bar
        fugacity_coefficient = h2_model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)
"""

from __future__ import annotations

import importlib.resources
import logging
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline

from atmodeller.eos import DATA_DIRECTORY
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

CHABRIER_DIRECTORY: Path = Path("chabrier")
"""Directory of the Chabrier data within :obj:`~atmodeller.eos.DATA_DIRECTORY`."""


@dataclass(kw_only=True)
class Chabrier(RealGas):
    r"""A real gas EOS from :cite:t:`CD21`

    This uses the rho-T-P tables to lookup density (rho).

    Args:
        filename: Filename of the density-T-P data in :obj:`CHABRIER_DIRECTORY`.
    """

    filename: Path
    """Filename of the density-T-P data"""
    standard_state_pressure: float = field(init=False, default=1)
    """Standard state pressure with the appropriate units. Set to 1 bar"""
    log10density_func: RectBivariateSpline = field(init=False, repr=False)
    """Spline to evaluate the density"""

    def __post_init__(self):
        self._create_spline()

    def _create_spline(self) -> None:
        """Sets spline lookup for density from :cite:t:`CD21` T-P-rhp tables.

        The first 3 columns contain log10 T [K], log10 P [GPa], log10 rho [g/cc].
        """

        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(str(CHABRIER_DIRECTORY.joinpath(self.filename)))
        )
        with data as datapath:
            columns: pd.Index = pd.read_fwf(
                datapath, widths=(16, 15, 15, 15, 16, 15, 15, 15, 15, 15)
            ).columns
            df: pd.DataFrame = pd.read_fwf(
                datapath,
                widths=(16, 15, 15, 15, 16, 15, 15, 15, 15, 15),
                header=None,
                comment="#",
            )
        df.columns = columns
        pivot_table: pd.DataFrame = df.pivot(
            index="#log T [K]", columns="log P [GPa]", values="log rho [g/cc]"
        )
        self.log10density_func: RectBivariateSpline = RectBivariateSpline(
            pivot_table.index.to_numpy(), pivot_table.columns.to_numpy(), pivot_table.to_numpy()
        )

    @override
    def volume(self, temperature: float, pressure: float) -> float:
        # Get log10 (density [g/cm3]) from the Chabrier H2 table
        log10density_gcc = self.log10density_func(
            np.log10(temperature), np.log10(UnitConversion.bar_to_GPa(pressure))
        )
        # Convert units: g/cm3 to mol/cm3 to mol/m3 for H2 (1e6 cm3 = 1 m3; 1 mol H2 = 2.016 g H2)
        molar_density: float = np.power(10, log10density_gcc.item()) / (
            UnitConversion.cm3_to_m3(1) * 2.016
        )
        volume: float = 1 / molar_density

        return volume

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float: # type: ignore
        # Calculate density for the integral
        pressures = np.logspace(
            np.log10(self.standard_state_pressure), np.log10(pressure), num=1000
        )
        log10temperatures = np.full_like(pressures, np.log10(temperature))
        log10pressures_GPa = np.log10(UnitConversion.bar_to_GPa(pressures))
 
        log10densities_gcc = self.log10density_func.ev(log10temperatures, log10pressures_GPa)

        # Convert units: g/cm3 to mol/cm3 to mol/m3 for H2 (1e6 cm3 = 1 m3; 1 mol H2 = 2.016 g H2)
        molar_densities = np.power(10, log10densities_gcc, out=log10densities_gcc) / (
            UnitConversion.cm3_to_m3(1) * 2.016
        )
        
        volumes = np.reciprocal(molar_densities, out=molar_densities)

        volume_integral = trapezoid(volumes, pressures)
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral

# Precompute the spline and store it
precomputed_spline = Chabrier(filename=Path("TABLE_H_TP_v1")).log10density_func

H2_CD21: RealGas = Chabrier(filename=Path("TABLE_H_TP_v1"))
"""H2 :cite:p:`CD21`"""
# Use the precomputed spline for new instances of Chabrier
H2_CD21.log10density_func = precomputed_spline


def get_chabrier_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred Chabrier and colleagues EOS models for each species.

    The latest and/or most sophisticated EOS model is chosen for each species.

    Returns:
        Dictionary of EOS models for each species
    """
    models: dict[str, RealGas] = {}
    models["H2"] = H2_CD21

    return models
