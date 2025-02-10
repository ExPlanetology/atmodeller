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
"""Real gas EOS library

Usage:
    from atmodeller.eos.library import get_eos_models
    eos_models = get_eos_models()
    CH4_beattie = eos_models["CH4_beattie_holley58"]
    # Evaluate fugacity at 10 bar and 800 K
    fugacity = CH4_beattie.fugacity(800, 10)
    print(fugacity)
"""

import logging
from pathlib import Path

from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos._holland_powell import get_holland_eos_models
from atmodeller.eos._holley import get_holley_eos_models
from atmodeller.eos._saxena import get_saxena_eos_models
from atmodeller.eos.classes import Chabrier
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibration

logger: logging.Logger = logging.getLogger(__name__)

calibration_chabrier21: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=100, temperature_max=1.0e8, pressure_min=None, pressure_max=1.0e17
)
"""Calibration for :cite:t:`CD21`"""
H2_chabrier21: RealGasProtocol = Chabrier(Path("TABLE_H_TP_v1"))
"""H2 :cite:p:`CD21`"""
H2_chabrier21_bounded: RealGasProtocol = CombinedRealGas(
    [H2_chabrier21],
    [calibration_chabrier21],
)
"""H2 bounded :cite:p:`CD21`"""
He_chabrier21: RealGasProtocol = Chabrier(Path("TABLE_HE_TP_v1"))
"""He :cite:p:`CD21`"""
He_chabrier21_bounded: RealGasProtocol = CombinedRealGas([He_chabrier21], [calibration_chabrier21])
"""He bounded :cite:p:`CD21`"""
H2_He_Y0275_chabrier21: RealGasProtocol = Chabrier(Path("TABLEEOS_2021_TP_Y0275_v1"))
"""H2HeY0275 :cite:p:`CD21`"""
H2_He_Y0275_chabrier21_bounded: RealGasProtocol = CombinedRealGas(
    [H2_He_Y0275_chabrier21], [calibration_chabrier21]
)
"""H2HeY0275 bounded :cite:p:`CD21`"""
H2_He_Y0292_chabrier21: RealGasProtocol = Chabrier(Path("TABLEEOS_2021_TP_Y0292_v1"))
"""H2HeY0292 :cite:p:`CD21`"""
H2_He_Y0292_chabrier21_bounded: RealGasProtocol = CombinedRealGas(
    [H2_He_Y0292_chabrier21], [calibration_chabrier21]
)
"""H2HeY0292 bounded :cite:p:`CD21`"""
H2_He_Y0297_chabrier21: RealGasProtocol = Chabrier(Path("TABLEEOS_2021_TP_Y0297_v1"))
"""H2HeY0297 :cite:p:`CD21`"""
H2_He_Y0297_chabrier21_bounded: RealGasProtocol = CombinedRealGas(
    [H2_He_Y0297_chabrier21], [calibration_chabrier21]
)
"""H2HeY0297 bounded :cite:p:`CD21`"""


def get_eos_models() -> dict[str, RealGasProtocol]:
    """Gets a dictionary of EOS models

    The naming convention is as follows:
        [species]_[eos model]_[citation]

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGasProtocol] = {}
    eos_models["H2_chabrier21"] = H2_chabrier21_bounded
    eos_models["H2_He_Y0275_chabrier21"] = H2_He_Y0275_chabrier21_bounded
    eos_models["H2_He_Y0292_chabrier21"] = H2_He_Y0292_chabrier21_bounded
    eos_models["H2_He_Y0297_chabrier21"] = H2_He_Y0297_chabrier21_bounded
    eos_models["He_chabrier21"] = He_chabrier21_bounded

    # Merge Holley models
    eos_models |= get_holley_eos_models()
    # Merge Holland and Powell models
    eos_models |= get_holland_eos_models()
    # Merge Saxena models
    eos_models |= get_saxena_eos_models()

    return eos_models
