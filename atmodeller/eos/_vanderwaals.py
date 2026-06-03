# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOS from :cite:t:`Lide2005`"""

from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas, VanderWaals
from atmodeller.sci_utils import ExperimentalCalibration

experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=1)

# van der Waals cefficients from David R. Lide, ed., CRC Handbook of Chemistry and Physics,
# Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005
H2_lide: RealGas = VanderWaals(2.452e-7, 2.65e-5)
"""H2 van der Waals :cite:p:`Lide2005`"""
H2_lide_bounded: RealGas = CombinedRealGas.create([H2_lide], [experimental_calibration])
"""H2 bounded to data range"""
He_lide: RealGas = VanderWaals(3.46e-8, 2.38e-5)
"""He van der Waals :cite:p:`Lide2005`"""
He_lide_bounded: RealGas = CombinedRealGas.create([He_lide], [experimental_calibration])
"""He bounded to data range"""
N2_lide: RealGas = VanderWaals(1.37e-6, 3.87e-5)
"""N2 van der Waals :cite:p:`Lide2005`"""
N2_lide_bounded: RealGas = CombinedRealGas.create([N2_lide], [experimental_calibration])
"""N2 bounded to data range"""
H4Si_lide: RealGas = VanderWaals(4.38e-6, 5.79e-5)
"""SiH4 van der Waals :cite:p:`Lide2005`"""
H4Si_lide_bounded: RealGas = CombinedRealGas.create([H4Si_lide], [experimental_calibration])
"""SiH4 bounded to data range"""
H2O_lide: RealGas = VanderWaals(5.537e-6, 3.05e-5)
"""H2O van der Waals :cite:p:`Lide2005`"""
H2O_lide_bounded: RealGas = CombinedRealGas.create([H2O_lide], [experimental_calibration])
"""H2O bounded to data range"""
CH4_lide: RealGas = VanderWaals(2.303e-6, 4.31e-5)
"""CH4 van der Waals :cite:p:`Lide2005`"""
CH4_lide_bounded: RealGas = CombinedRealGas.create([CH4_lide], [experimental_calibration])
"""CH4 bounded to data range"""
H3N_lide: RealGas = VanderWaals(4.225e-6, 3.71e-5)
"""NH3 van der Waals :cite:p:`Lide2005`"""
H3N_lide_bounded: RealGas = CombinedRealGas.create([H3N_lide], [experimental_calibration])
"""NH3 bounded to data range"""
CHN_lide: RealGas = VanderWaals(1.29e-5, 8.81e-5)
"""HCN van der Waals :cite:p:`Lide2005`"""
CHN_lide_bounded: RealGas = CombinedRealGas.create([CHN_lide], [experimental_calibration])
"""HCN bounded to data range"""
H4Si_isham: RealGas = VanderWaals(2.478e-6, 3.275e-5)
"""SiH4 van der Waals (Isham) :cite:p:`Lide2005`"""
H4Si_isham_bounded: RealGas = CombinedRealGas.create([H4Si_isham], [experimental_calibration])
"""SiH4 (Isham) bounded to data range"""
OSi_isham: RealGas = VanderWaals(8.698e-6, 8.582e-6)
"""OSi van der Waals (Isham) :cite:p:`Lide2005`"""
OSi_isham_bounded: RealGas = CombinedRealGas.create([OSi_isham], [experimental_calibration])
"""OSi (Isham) bounded to data range"""


def get_vanderwaals_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of van der Waals EOS models.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["H2_vdw_lide05"] = H2_lide_bounded
    eos_models["He_vdw_lide05"] = He_lide_bounded
    eos_models["N2_vdw_lide05"] = N2_lide_bounded
    eos_models["H4Si_vdw_lide05"] = H4Si_lide_bounded
    eos_models["H2O_vdw_lide05"] = H2O_lide_bounded
    eos_models["CH4_vdw_lide05"] = CH4_lide_bounded
    eos_models["H3N_vdw_lide05"] = H3N_lide_bounded
    eos_models["CHN_vdw_lide05"] = CHN_lide_bounded

    return eos_models
