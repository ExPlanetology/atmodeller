# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOS from :cite:t:`ZD09`"""

from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos._zhang_duan_base import ZhangDuanPureFluid, epsilon_species, sigma_species
from atmodeller.eos.core import RealGas
from atmodeller.sci_utils import ExperimentalCalibration, unit_conversion

CH4_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CH4"], sigma_species["CH4"])
"""CH4 unbounded :cite:p:`ZD09`"""
CH4_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=273,
    temperature_max=2573,
    pressure_min=0.1 * unit_conversion.MPa_to_bar,
    pressure_max=10000 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for CH4 :cite:p:`ZD09{Table 5}`"""
CH4_zhang09_bounded: RealGas = CombinedRealGas.create(
    [CH4_zhang09], [CH4_experimental_calibration]
)
"""CH4 bounded to data range :cite:p:`ZD09{Table 5}`"""

H2O_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["H2O"], sigma_species["H2O"])
"""H2O unbounded :cite:p:`ZD09`"""
H2O_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=673,
    temperature_max=2573,
    pressure_min=0.1 * unit_conversion.MPa_to_bar,
    pressure_max=10000 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for H2O :cite:p:`ZD09{Table 5}`"""
H2O_zhang09_bounded: RealGas = CombinedRealGas.create(
    [H2O_zhang09], [H2O_experimental_calibration]
)
"""H2O bounded to data range :cite:p:`ZD09{Table 5}`"""

CO2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CO2"], sigma_species["CO2"])
"""CO2 unbounded :cite:p:`ZD09`"""
CO2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=473,
    temperature_max=2573,
    pressure_min=0.1 * unit_conversion.MPa_to_bar,
    pressure_max=10000 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for CO2 :cite:p:`ZD09{Table 5}`"""
CO2_zhang09_bounded: RealGas = CombinedRealGas.create(
    [CO2_zhang09], [CO2_experimental_calibration]
)

# Tested boundedness (not the same as physical correctness) for 500<T<10000 K and 0<P<10 GPa
H2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["H2"], sigma_species["H2"])
"""H2 unbounded :cite:p:`ZD09`"""
H2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=250,
    temperature_max=423,
    pressure_min=2 * unit_conversion.MPa_to_bar,
    pressure_max=700 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for H2 :cite:p:`ZD09{Table 5}`"""
H2_zhang09_bounded: RealGas = CombinedRealGas.create([H2_zhang09], [H2_experimental_calibration])
"""H2 bounded to data range :cite:p:`ZD09{Table 5}`"""

CO_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CO"], sigma_species["CO"])
"""CO unbounded :cite:p:`ZD09`"""
CO_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=300,
    temperature_max=573.2,
    pressure_min=10 * unit_conversion.MPa_to_bar,
    pressure_max=1020.6 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for CO :cite:p:`ZD09{Table 5}`"""
CO_zhang09_bounded: RealGas = CombinedRealGas.create([CO_zhang09], [CO_experimental_calibration])
"""CO bounded to data range :cite:p:`ZD09{Table 5}`"""

O2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["O2"], sigma_species["O2"])
"""O2 unbounded :cite:p:`ZD09`"""
O2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=300,
    temperature_max=1000,
    pressure_min=7 * unit_conversion.MPa_to_bar,
    pressure_max=1013.2 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for O2 :cite:p:`ZD09{Table 5}`"""
O2_zhang09_bounded: RealGas = CombinedRealGas.create([O2_zhang09], [O2_experimental_calibration])
"""O2 bounded to data range :cite:p:`ZD09{Table 5}`"""

C2H6_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["C2H6"], sigma_species["C2H6"])
"""C2H6 unbounded :cite:p:`ZD09`"""
C2H6_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=373,
    temperature_max=673,
    pressure_min=30 * unit_conversion.MPa_to_bar,
    pressure_max=900 * unit_conversion.MPa_to_bar,
)
"""Experimental calibration for C2H6 :cite:p:`ZD09{Table 5}`"""
C2H6_zhang09_bounded: RealGas = CombinedRealGas.create(
    [C2H6_zhang09], [C2H6_experimental_calibration]
)
"""C2H6 bounded to data range :cite:p:`ZD09{Table 5}`"""

# Extend the Zhang and Duan (2009) model to more species using Lennard-Jones parameters. These are
# all unbounded and should be used with caution.
Ar_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["Ar"], sigma_species["Ar"])
"""Ar unbounded :cite:p:`ZD09`"""
He_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["He"], sigma_species["He"])
"""He unbounded :cite:p:`ZD09`"""
Kr_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["Kr"], sigma_species["Kr"])
"""Kr unbounded :cite:p:`ZD09`"""
Ne_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["Ne"], sigma_species["Ne"])
"""Ne unbounded :cite:p:`ZD09`"""
Xe_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["Xe"], sigma_species["Xe"])
"""Xe unbounded :cite:p:`ZD09`"""
COS_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["COS"], sigma_species["COS"])
"""COS unbounded :cite:p:`ZD09`"""
Cl2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["Cl2"], sigma_species["Cl2"])
"""Cl2 unbounded :cite:p:`ZD09`"""
F2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["F2"], sigma_species["F2"])
"""F2 unbounded :cite:p:`ZD09`"""
CHN_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["CHN"], sigma_species["CHN"])
"""CHN unbounded :cite:p:`ZD09`"""
ClH_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["ClH"], sigma_species["ClH"])
"""ClH unbounded :cite:p:`ZD09`"""
H2S_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["H2S"], sigma_species["H2S"])
"""H2S unbounded :cite:p:`ZD09`"""
H3N_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["H3N"], sigma_species["H3N"])
"""H3N unbounded :cite:p:`ZD09`"""
NO_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["NO"], sigma_species["NO"])
"""NO unbounded :cite:p:`ZD09`"""
N2_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["N2"], sigma_species["N2"])
"""N2 unbounded :cite:p:`ZD09`"""
N2O_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["N2O"], sigma_species["N2O"])
"""N2O unbounded :cite:p:`ZD09`"""
O2S_zhang09: RealGas = ZhangDuanPureFluid(epsilon_species["O2S"], sigma_species["O2S"])
"""O2S unbounded :cite:p:`ZD09`"""


def get_zhang_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of Zhang and Duan EOS models.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["CH4_zhang09"] = CH4_zhang09_bounded
    eos_models["CH4_zhang09_unbounded"] = CH4_zhang09
    eos_models["H2O_zhang09"] = H2O_zhang09_bounded
    eos_models["H2O_zhang09_unbounded"] = H2O_zhang09
    eos_models["CO2_zhang09"] = CO2_zhang09_bounded
    eos_models["CO2_zhang09_unbounded"] = CO2_zhang09
    eos_models["H2_zhang09"] = H2_zhang09_bounded
    eos_models["H2_zhang09_unbounded"] = H2_zhang09
    eos_models["CO_zhang09"] = CO_zhang09_bounded
    eos_models["CO_zhang09_unbounded"] = CO2_zhang09
    eos_models["O2_zhang09"] = O2_zhang09_bounded
    eos_models["O2_zhang09_unbounded"] = O2_zhang09
    eos_models["C2H6_zhang09"] = C2H6_zhang09_bounded
    eos_models["C2H6_zhang09_unbounded"] = C2H6_zhang09
    eos_models["Ar_zhang09"] = Ar_zhang09
    eos_models["He_zhang09"] = He_zhang09
    eos_models["Kr_zhang09"] = Kr_zhang09
    eos_models["Ne_zhang09"] = Ne_zhang09
    eos_models["Xe_zhang09"] = Xe_zhang09
    eos_models["COS_zhang09"] = COS_zhang09
    eos_models["Cl2_zhang09"] = Cl2_zhang09
    eos_models["F2_zhang09"] = F2_zhang09
    eos_models["CHN_zhang09"] = CHN_zhang09
    eos_models["ClH_zhang09"] = ClH_zhang09
    eos_models["H2S_zhang09"] = H2S_zhang09
    eos_models["H3N_zhang09"] = H3N_zhang09
    eos_models["NO_zhang09"] = NO_zhang09
    eos_models["N2_zhang09"] = N2_zhang09
    eos_models["N2O_zhang09"] = N2O_zhang09
    eos_models["O2S_zhang09"] = O2S_zhang09

    return eos_models
