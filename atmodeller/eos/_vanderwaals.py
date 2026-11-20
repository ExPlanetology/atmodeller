# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Real gas EOS from :cite:t:`Lide2005`"""

from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.interfaces import CatalogueEntry, ModelInfo
from atmodeller.eos.core import RealGas, VanderWaals
from atmodeller.sci_utils import ExperimentalCalibration

experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=1)

# van der Waals cefficients from David R. Lide, ed., CRC Handbook of Chemistry and Physics,
# Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005
H2_lide: RealGas = VanderWaals(2.452e-7, 2.65e-5)
"""H2 van der Waals :cite:p:`Lide2005`"""
H2_lide_bounded: RealGas = CombinedRealGas.create([H2_lide], [experimental_calibration])
"""H2 bounded: an ideal gas below 1 bar"""
He_lide: RealGas = VanderWaals(3.46e-8, 2.38e-5)
"""He van der Waals :cite:p:`Lide2005`"""
He_lide_bounded: RealGas = CombinedRealGas.create([He_lide], [experimental_calibration])
"""He bounded: an ideal gas below 1 bar"""
N2_lide: RealGas = VanderWaals(1.37e-6, 3.87e-5)
"""N2 van der Waals :cite:p:`Lide2005`"""
N2_lide_bounded: RealGas = CombinedRealGas.create([N2_lide], [experimental_calibration])
"""N2 bounded: an ideal gas below 1 bar"""
H4Si_lide: RealGas = VanderWaals(4.38e-6, 5.79e-5)
"""SiH4 van der Waals :cite:p:`Lide2005`"""
H4Si_lide_bounded: RealGas = CombinedRealGas.create([H4Si_lide], [experimental_calibration])
"""SiH4 bounded: an ideal gas below 1 bar"""
H2O_lide: RealGas = VanderWaals(5.537e-6, 3.05e-5)
"""H2O van der Waals :cite:p:`Lide2005`"""
H2O_lide_bounded: RealGas = CombinedRealGas.create([H2O_lide], [experimental_calibration])
"""H2O bounded: an ideal gas below 1 bar"""
CH4_lide: RealGas = VanderWaals(2.303e-6, 4.31e-5)
"""CH4 van der Waals :cite:p:`Lide2005`"""
CH4_lide_bounded: RealGas = CombinedRealGas.create([CH4_lide], [experimental_calibration])
"""CH4 bounded: an ideal gas below 1 bar"""
H3N_lide: RealGas = VanderWaals(4.225e-6, 3.71e-5)
"""NH3 van der Waals :cite:p:`Lide2005`"""
H3N_lide_bounded: RealGas = CombinedRealGas.create([H3N_lide], [experimental_calibration])
"""NH3 bounded: an ideal gas below 1 bar"""
CHN_lide: RealGas = VanderWaals(1.129e-5, 8.81e-5)
"""HCN van der Waals :cite:p:`Lide2005`"""
CHN_lide_bounded: RealGas = CombinedRealGas.create([CHN_lide], [experimental_calibration])
"""HCN bounded: an ideal gas below 1 bar"""


def get_vanderwaals_eos_catalogue() -> dict[str, CatalogueEntry[RealGas]]:
    """Gets a dictionary of van der Waals EOS models.

    Returns:
        Dictionary of EOS models with their catalogue information
    """
    eos_models: dict[str, CatalogueEntry[RealGas]] = {}
    eos_models["H2_vdw_lide05"] = CatalogueEntry(
        H2_lide_bounded, ModelInfo("H2", sources=("Lide2005",))
    )
    eos_models["He_vdw_lide05"] = CatalogueEntry(
        He_lide_bounded, ModelInfo("He", sources=("Lide2005",))
    )
    eos_models["N2_vdw_lide05"] = CatalogueEntry(
        N2_lide_bounded, ModelInfo("N2", sources=("Lide2005",))
    )
    eos_models["H4Si_vdw_lide05"] = CatalogueEntry(
        H4Si_lide_bounded, ModelInfo("H4Si", sources=("Lide2005",))
    )
    eos_models["H2O_vdw_lide05"] = CatalogueEntry(
        H2O_lide_bounded, ModelInfo("H2O", sources=("Lide2005",))
    )
    eos_models["CH4_vdw_lide05"] = CatalogueEntry(
        CH4_lide_bounded, ModelInfo("CH4", sources=("Lide2005",))
    )
    eos_models["H3N_vdw_lide05"] = CatalogueEntry(
        H3N_lide_bounded, ModelInfo("H3N", sources=("Lide2005",))
    )
    eos_models["CHN_vdw_lide05"] = CatalogueEntry(
        CHN_lide_bounded, ModelInfo("CHN", sources=("Lide2005",))
    )

    return eos_models
