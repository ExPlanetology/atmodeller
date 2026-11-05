# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Real gas EOSs from :cite:t:`RPP87,C16`

The Redlich-Kwong models of H4Si, CHN and H3N use the critical data of :cite:t:`RPP87` (see also
the current edition, :cite:t:`PPO00`). They are not bounded, as in v1 and in
:cite:t:`Hakim2026`, although corresponding states extrapolations are only expected to be accurate
up to about 10 times the critical pressure and 2-3 times the critical temperature
:cite:p:`RPP87`; :data:`experimental_calibration_reid87` records the validity range.
"""

from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.eos.redlich_kwong import RedlichKwong49
from atmodeller.sci_utils import ExperimentalCalibration

experimental_calibration_connolly16: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=1000, temperature_max=10000, pressure_min=1, pressure_max=50e3
)
"""Experimental calibration for :cite:`C16` models"""

experimental_calibration_reid87: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=300, temperature_max=500, pressure_min=1, pressure_max=100
)
"""Validity range of the :cite:t:`RPP87` models (300-500 K, up to 100 bar)

Not applied: the models are registered unbounded, as in v1 and :cite:t:`Hakim2026`. To bound a
model, combine it with this calibration using ``CombinedRealGas.create``.
"""

OSi_rk49_connolly16: RealGas = RedlichKwong49.create("OSi")
"""OSi Redlich-Kwong :cite:p:`C16`"""
H4Si_rk49_reid87: RealGas = RedlichKwong49.create("H4Si")
"""H4Si Redlich-Kwong with the critical data of :cite:t:`RPP87` (unbounded)"""
CHN_rk49_reid87: RealGas = RedlichKwong49.create("CHN")
"""CHN Redlich-Kwong with the critical data of :cite:t:`RPP87` (unbounded)"""
H3N_rk49_reid87: RealGas = RedlichKwong49.create("H3N")
"""H3N Redlich-Kwong with the critical data of :cite:t:`RPP87` (unbounded)"""

OSi_rk49_connolly16_bounded: RealGas = CombinedRealGas.create(
    [OSi_rk49_connolly16], [experimental_calibration_connolly16]
)
"""OSi Redlich-Kwong bounded :cite:p:`C16`"""


def get_reid_connolly_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of EOS models

    The naming convention is as follows:
        [species]_[eos model]_[citation]

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["OSi_rk49_connolly16"] = OSi_rk49_connolly16_bounded
    eos_models["H4Si_rk49_reid87"] = H4Si_rk49_reid87
    eos_models["CHN_rk49_reid87"] = CHN_rk49_reid87
    eos_models["H3N_rk49_reid87"] = H3N_rk49_reid87

    return eos_models
