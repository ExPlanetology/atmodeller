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
"""Thermochemical data for gases from :cite:t:`MZG02`

https://ntrs.nasa.gov/citations/20020085330
"""

from atmodeller.thermodata import CriticalData, SpeciesData, ThermoCoefficients

_C_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (8.545763110e4, 8.410597850e4, 2.355273444e6),
    (4.747924288, 4.130047418, -6.405123160e2),
    (
        (
            6.495031470e2,
            -9.649010860e-1,
            2.504675479,
            -1.281448025e-5,
            1.980133654e-8,
            -1.606144025e-11,
            5.314483411e-15,
        ),
        (
            -1.289136472e5,
            1.719528572e2,
            2.646044387,
            -3.353068950e-4,
            1.742092740e-7,
            -2.902817829e-11,
            1.642182385e-15,
        ),
        (
            4.432528010e8,
            -2.886018412e5,
            7.737108320e1,
            -9.715281890e-3,
            6.649595330e-7,
            -2.230078776e-11,
            2.899388702e-16,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
C_g: SpeciesData = SpeciesData.create(
    "C",
    "g",
    _C_g_coeffs,
)
"Species data for C_g"

_CH4_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-2.331314360e4, 7.532066910e4),
    (8.904322750e1, -1.219124889e2),
    (
        (
            -1.766850998e5,
            2.786181020e3,
            -1.202577850e1,
            3.917619290e-2,
            -3.619054430e-5,
            2.026853043e-8,
            -4.976705490e-12,
        ),
        (
            3.730042760e6,
            -1.383501485e4,
            2.049107091e1,
            -1.961974759e-3,
            4.727313040e-7,
            -3.728814690e-11,
            1.623737207e-15,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
CH4_g: SpeciesData = SpeciesData.create(
    "CH4",
    "g",
    _CH4_g_coeffs,
)
"Species data for CH4_g"

_Cl2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (1.534069331e3, 1.212117724e5),
    (-9.438331107, -1.690778824e2),
    (
        (
            3.462815170e4,
            -5.547126520e2,
            6.207589370,
            -2.989632078e-3,
            3.173027290e-6,
            -1.793629562e-9,
            4.260043590e-13,
        ),
        (
            6.092569420e6,
            -1.949627662e4,
            2.854535795e1,
            -1.449968764e-2,
            4.463890770e-6,
            -6.358525860e-10,
            3.327360290e-14,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
Cl2_g: SpeciesData = SpeciesData.create("Cl2", "g", _Cl2_g_coeffs)
"Species data for Cl2_g"

_CO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-1.303131878e4, -2.466261084e3, 5.701421130e6),
    (
        -7.859241350,
        -1.387413108e1,
        -2.060704786e3,
    ),
    (
        (
            1.489045326e4,
            -2.922285939e2,
            5.724527170,
            -8.176235030e-3,
            1.456903469e-5,
            -1.087746302e-8,
            3.027941827e-12,
        ),
        (
            4.619197250e5,
            -1.944704863e3,
            5.916714180,
            -5.664282830e-4,
            1.398814540e-7,
            -1.787680361e-11,
            9.620935570e-16,
        ),
        (
            8.868662960e8,
            -7.500377840e5,
            2.495474979e2,
            -3.956351100e-2,
            3.297772080e-6,
            -1.318409933e-10,
            1.998937948e-15,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
CO_g: SpeciesData = SpeciesData.create(
    "CO",
    "g",
    _CO_g_coeffs,
)
"Species data for CO_g"

_CO2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-4.528198460e4, -3.908350590e4, -8.043214510e6),
    (-7.048279440, -2.652669281e1, 2.254177493e3),
    (
        (
            4.943650540e4,
            -6.264116010e2,
            5.301725240,
            2.503813816e-3,
            -2.127308728e-7,
            -7.689988780e-10,
            2.849677801e-13,
        ),
        (
            1.176962419e5,
            -1.788791477e3,
            8.291523190,
            -9.223156780e-5,
            4.863676880e-9,
            -1.891053312e-12,
            6.330036590e-16,
        ),
        (
            -1.544423287e9,
            1.016847056e6,
            -2.561405230e2,
            3.369401080e-2,
            -2.181184337e-6,
            6.991420840e-11,
            -8.842351500e-16,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
CO2_g: SpeciesData = SpeciesData.create(
    "CO2",
    "g",
    _CO2_g_coeffs,
)
"Species data for CO2_g"

_H2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (2.682484665e3, 5.339824410e3, 2.488433516e6),
    (-3.043788844e1, -2.202774769, -6.695728110e2),
    (
        (
            4.078323210e4,
            -8.009186040e2,
            8.214702010,
            -1.269714457e-2,
            1.753605076e-5,
            -1.202860270e-8,
            3.368093490e-12,
        ),
        (
            5.608128010e5,
            -8.37150474e2,
            2.975364532,
            1.252249124e-3,
            -3.740716190e-7,
            5.936625200e-11,
            -3.606994100e-15,
        ),
        (
            4.966884120e8,
            -3.147547149e5,
            7.984121880e1,
            -8.414789210e-3,
            4.753248350e-7,
            -1.371873492e-11,
            1.605461756e-16,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
H2_g: SpeciesData = SpeciesData.create("H2", "g", _H2_g_coeffs)
"Species data for H2_g"

_H2O_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-3.303974310e4, -1.384286509e4),
    (1.724205775e1, -7.978148510),
    (
        (
            -3.947960830e4,
            5.755731020e2,
            9.317826530e-1,
            7.222712860e-3,
            -7.342557370e-6,
            4.955043490e-9,
            -1.336933246e-12,
        ),
        (
            1.034972096e6,
            -2.412698562e3,
            4.646110780,
            2.291998307e-3,
            -6.836830480e-7,
            9.426468930e-11,
            -4.822380530e-15,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
H2O_g: SpeciesData = SpeciesData.create(
    "H2O",
    "g",
    _H2O_g_coeffs,
)
"Species data for H2O_g"

_H2S_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-3.278457280e3, 2.908696214e4),
    (1.415194691, -4.349160391e1),
    (
        (
            9.543808810e3,
            -6.875175080e1,
            4.054921960,
            -3.014557336e-4,
            3.768497750e-6,
            -2.239358925e-9,
            3.086859108e-13,
        ),
        (
            1.430040220e6,
            -5.284028650e3,
            1.016182124e1,
            -9.703849960e-4,
            2.154003405e-7,
            -2.169695700e-11,
            9.318163070e-16,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
H2S_g: SpeciesData = SpeciesData.create(
    "H2S",
    "g",
    _H2S_g_coeffs,
)
"Species data for H2S_g"

_HCl_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-1.067782299e4, 5.674958050e3),
    (-7.309305408, -1.642825822e1),
    (
        (
            2.062588287e4,
            -3.093368855e2,
            5.275418850,
            -4.828874220e-3,
            6.195794600e-6,
            -3.040023782e-9,
            4.916790030e-13,
        ),
        (
            9.157749510e5,
            -2.770550211e3,
            5.973539790,
            -3.629810060e-4,
            4.735529190e-8,
            2.810262054e-12,
            -6.656104220e-16,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
HCl_g: SpeciesData = SpeciesData.create(
    "HCl",
    "g",
    _HCl_g_coeffs,
)
"Species data for HCl_g"

_He_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-7.453750000e2, -7.453750000e2, 1.650518960e4),
    (9.287239740e-1, 9.287239740e-1, -4.048814390),
    (
        (0, 0, 2.5, 0, 0, 0, 0),
        (0, 0, 2.5, 0, 0, 0, 0),
        (
            3.396845420e6,
            -2.194037652e3,
            3.080231878,
            -8.068957550e-5,
            6.252784910e-9,
            -2.574990067e-13,
            4.429960218e-18,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
He_g: SpeciesData = SpeciesData.create(
    "He",
    "g",
    _He_g_coeffs,
)
"Species data for He_g"

_N2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (7.108460860e2, 1.283210415e4, 4.938707040e6),
    (-1.076003744e1, -1.586640027e1, -1.672099740e3),
    (
        (
            2.210371497e4,
            -3.818461820e2,
            6.082738360,
            -8.530914410e-03,
            1.384646189e-5,
            -9.625793620e-9,
            2.519705809e-12,
        ),
        (
            5.877124060e5,
            -2.239249073e3,
            6.066949220,
            -6.139685500e-4,
            1.491806679e-7,
            -1.923105485e-11,
            1.061954386e-15,
        ),
        (
            8.310139160e8,
            -6.420733540e5,
            2.020264635e2,
            -3.065092046e-2,
            2.486903333e-6,
            -9.705954110e-11,
            1.437538881e-15,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
N2_g: SpeciesData = SpeciesData.create("N2", "g", _N2_g_coeffs)
"Species data for N2_g"

_NH3_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-1.264886413e4, 4.386191960e4),
    (4.366014588e1, -6.462330602e1),
    (
        (
            -7.681226150e4,
            1.270951578e3,
            -3.893229130,
            2.145988418e-2,
            -2.183766703e-5,
            1.317385706e-8,
            -3.332322060e-12,
        ),
        (
            2.452389535e6,
            -8.040894240e3,
            1.271346201e1,
            -3.980186580e-4,
            3.552502750e-8,
            2.530923570e-12,
            -3.322700530e-16,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
NH3_g: SpeciesData = SpeciesData.create(
    "NH3",
    "g",
    _NH3_g_coeffs,
)
"Species data for NH3_g"

_O2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-3.391454870e3, -1.689010929e4, 2.293554027e6),
    (1.849699470e1, 1.738716506e1, -5.530621610e2),
    (
        (
            -3.425563420e4,
            4.847000970e2,
            1.119010961,
            4.293889240e-3,
            -6.836300520e-7,
            -2.023372700e-9,
            1.039040018e-12,
        ),
        (
            -1.037939022e6,
            2.344830282e3,
            1.819732036,
            1.267847582e-3,
            -2.188067988e-7,
            2.053719572e-11,
            -8.193467050e-16,
        ),
        (
            4.975294300e8,
            -2.866106874e5,
            6.690352250e1,
            -6.169959020e-3,
            3.016396027e-7,
            -7.421416600e-12,
            7.278175770e-17,
        ),
    ),
    (200, 1000, 6000),
    (1000, 6000, 20000),
)
O2_g: SpeciesData = SpeciesData.create("O2", "g", _O2_g_coeffs)
"Species data for O2_g"

_S2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (1.654767715e4, 1.085508427e4),
    (-7.957279032e-1, 1.458544515e1),
    (
        (
            3.528091780e4,
            -4.222156580e2,
            4.677433490,
            1.724046361e-3,
            -3.862208210e-6,
            3.336156340e-9,
            -9.930661540e-13,
        ),
        (
            -1.588128788e4,
            6.315480880e2,
            2.449628069,
            1.986240565e-3,
            -6.507927240e-7,
            1.002813651e-10,
            -5.596990050e-15,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
S2_g: SpeciesData = SpeciesData.create("S2", "g", _S2_g_coeffs)
"Species data for S2_g"

_SO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-3.371292190e3, -2.708838059e4),
    (3.093861963e1, 3.615358329e1),
    (
        (
            -3.342757000e4,
            6.403862500e2,
            -1.006641228,
            1.381512705e-2,
            -1.704486364e-5,
            1.061294930e-8,
            -2.645796205e-12,
        ),
        (
            -1.443410557e6,
            4.113874360e3,
            -5.383695780e-1,
            2.794153269e-3,
            -6.633352260e-7,
            7.838221190e-11,
            -3.560509070e-15,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
SO_g: SpeciesData = SpeciesData.create("SO", "g", _SO_g_coeffs)
"Species data for SO_g"

_SO2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-4.113752080e4, -3.351308690e4),
    (4.045512519e1, -1.655776085e1),
    (
        (
            -5.310842140e4,
            9.090311670e2,
            -2.356891244,
            2.204449885e-2,
            -2.510781471e-5,
            1.446300484e-8,
            -3.369070940e-12,
        ),
        (
            -1.127640116e5,
            -8.252261380e2,
            7.616178630,
            -1.999327610e-4,
            5.655631430e-8,
            -5.454316610e-12,
            2.918294102e-16,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
SO2_g: SpeciesData = SpeciesData.create("SO2", "g", _SO2_g_coeffs)
"Species data for SO2_g"

_SiH4_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (6.269669060e3, 4.766887950e4),
    (4.965461830, -9.801697460e1),
    (
        (
            7.872993290e4,
            -5.526087050e2,
            2.498944303,
            1.442118274e-2,
            -8.467107310e-6,
            2.726164641e-9,
            -5.436754370e-13,
        ),
        (
            1.290378740e6,
            -7.813399780e3,
            1.828851664e1,
            -1.975620946e-3,
            4.156502150e-7,
            -4.596745610e-11,
            2.072777131e-15,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
SiH4_g: SpeciesData = SpeciesData.create("SiH4", "g", _SiH4_g_coeffs)
"Species data for SiH4_g"

_SiO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-1.666585903e4, -1.350842360e4),
    (3.355795700e1, -8.386957330e-1),
    (
        (
            -4.722771050e4,
            8.063137640e2,
            -1.636976133,
            1.454275546e-2,
            -1.723202046e-5,
            1.042397340e-8,
            -2.559365273e-12,
        ),
        (
            -1.7651341625,
            -3.199177090e1,
            4.477441930,
            4.591764710e-6,
            3.558143150e-8,
            -1.327012559e-11,
            1.613253297e-15,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
SiO_g: SpeciesData = SpeciesData.create("SiO", "g", _SiO_g_coeffs)
"Species data for SiO_g"

_Si_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (5.263510310e04, 3.953558760e04, -4.293792120e06),
    (9.698288880e00, 2.679668061e01, 1.086382839e03),
    (
        (
            9.836140810e01,
            1.546544523e02,
            1.876436670e00,
            1.320637995e-03,
            -1.529720059e-06,
            8.950562770e-10,
            -1.952873490e-13,
        ),
        (
            -6.169298850e05,
            2.240683927e03,
            -4.448619320e-01,
            1.710056321e-03,
            -4.107714160e-07,
            4.558884780e-11,
            -1.889515353e-15,
        ),
        (
            -9.286548940e08,
            5.443989890e05,
            -1.206739736e02,
            1.359662698e-02,
            -7.606498660e-07,
            2.149746065e-11,
            -2.474116774e-16,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
Si_g: SpeciesData = SpeciesData.create("Si", "g", _Si_g_coeffs)
"Species data for Si_g"

_SiO2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-4.226487490e04, -3.791834770e04),
    (2.295803206e01, -2.045285414e01),
    (
        (
            -3.362948780e04,
            4.734078920e02,
            2.309770671e-01,
            1.850230806e-02,
            -2.242786671e-05,
            1.364981554e-08,
            -3.351935030e-12,
        ),
        (
            -1.464031193e05,
            -6.261441060e02,
            7.964563710e00,
            -1.854119096e-04,
            4.095214670e-08,
            -4.697206760e-12,
            2.178054280e-16,
        ),
    ),
    (200, 1000),
    (1000, 6000),
)
SiO2_g: SpeciesData = SpeciesData.create("SiO2", "g", _SiO2_g_coeffs)
"Species data for SiO2_g"

_OH_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(2.991214235e03, 2.019640206e04, 1.468393908e06),
    b2=(4.674110790e00, -1.101282337e01, -4.023555580e02),
    cp_coeffs=(
        (
            -1.998858990e03,
            9.300136160e01,
            3.050854229e00,
            1.529529288e-03,
            -3.157890998e-06,
            3.315446180e-09,
            -1.138762683e-12,
        ),
        (
            1.017393379e06,
            -2.509957276e03,
            5.116547860e00,
            1.305299930e-04,
            -8.284322260e-08,
            2.006475941e-11,
            -1.556993656e-15,
        ),
        (
            2.847234193e08,
            -1.859532612e05,
            5.008240900e01,
            -5.142374980e-03,
            2.875536589e-07,
            -8.228817960e-12,
            9.567229020e-17,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
OH_g: SpeciesData = SpeciesData.create("OH", "g", _OH_g_coeffs)
"Species data for OH_g"

_MgO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(2.790679519e04, -2.300504434e05, 1.490218815e05),
    b2=(-1.624886199e02, 1.738984472e02, -8.007281730e01),
    cp_coeffs=(
        (
            3.513659740e05,
            -5.287197160e03,
            3.382060060e01,
            -8.400489630e-02,
            1.210016160e-04,
            -7.630795020e-08,
            1.701022862e-11,
        ),
        (
            -1.586738367e07,
            3.420468100e04,
            -1.774087677e01,
            7.004963050e-03,
            -1.104138249e-06,
            8.957488530e-11,
            -3.052513649e-15,
        ),
        (
            2.290059050e06,
            -2.073499632e04,
            1.444150005e01,
            -1.490609900e-03,
            1.052119343e-07,
            -3.523030610e-12,
            4.613111760e-17,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
MgO_g: SpeciesData = SpeciesData.create("MgO", "g", _MgO_g_coeffs)
"Species data for MgO_g"

_Mg_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(1.694658761e04, 4.829188110e03, 8.349525900e06),
    b2=(3.634330140e00, 2.339104998e01, -1.469355261e03),
    cp_coeffs=(
        (0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0),
        (
            -5.364831550e05,
            1.973709576e03,
            -3.633776900e-01,
            2.071795561e-03,
            -7.738051720e-07,
            1.359277788e-10,
            -7.766898397e-15,
        ),
        (
            2.166012586e09,
            -1.008355665e06,
            1.619680021e02,
            -8.790130350e-03,
            -1.925690961e-08,
            1.725045214e-11,
            -4.234946112e-16,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
Mg_g: SpeciesData = SpeciesData.create("Mg", "g", _Mg_g_coeffs)
"Species data for Mg_g"

_SH_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(1.742902395e04, 4.899214490e04),
    b2=(-1.760761843e01, -3.770400275e01),
    cp_coeffs=(
        (
            6.389434680e03,
            -3.747960920e02,
            7.548145770e00,
            -1.288875477e-02,
            1.907786343e-05,
            -1.265033728e-08,
            3.235158690e-12,
        ),
        (
            1.682631601e06,
            -5.177152210e03,
            9.198168520e00,
            -2.323550224e-03,
            6.543914780e-07,
            -8.468470420e-11,
            3.864741550e-15,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
SH_g: SpeciesData = SpeciesData.create("SH", "g", _SH_g_coeffs)
"Species data for SH_g"

_C2H2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(3.712619060e04, 6.266578970e04),
    b2=(-5.244338900e01, -5.818960590e01),
    cp_coeffs=(
        (
            1.598112089e05,
            -2.216644118e03,
            1.265707813e01,
            -7.979651080e-03,
            8.054992750e-06,
            -2.433307673e-09,
            -7.529233180e-14,
        ),
        (
            1.713847410e06,
            -5.929106660e03,
            1.236127943e01,
            1.314186993e-04,
            -1.362764431e-07,
            2.712655786e-11,
            -1.302066204e-15,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
C2H2_g: SpeciesData = SpeciesData.create("C2H2", "g", _C2H2_g_coeffs)
"Species data for C2H2_g"

_HCN_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(2.098915450e04, 4.221513770e04),
    b2=(-2.746678076e01, -4.005774072e01),
    cp_coeffs=(
        (
            9.098286930e04,
            -1.238657512e03,
            8.721307870e00,
            -6.528242940e-03,
            8.872700830e-06,
            -4.808886670e-09,
            9.317898500e-13,
        ),
        (
            1.236889278e06,
            -4.446732410e03,
            9.738874850e00,
            -5.855182640e-04,
            1.072791440e-07,
            -1.013313244e-11,
            3.348247980e-16,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
HCN_g: SpeciesData = SpeciesData.create("HCN", "g", _HCN_g_coeffs)
"Species data for HCN_g"

_SO3_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-5.184106170e04, -4.398283990e04),
    b2=(3.391331216e01, -3.655217314e01),
    cp_coeffs=(
        (
            -3.952855290e04,
            6.208572570e02,
            -1.437731716e00,
            2.764126467e-02,
            -3.144958662e-05,
            1.792798000e-08,
            -4.126386660e-12,
        ),
        (
            -2.166923781e05,
            -1.301022399e03,
            1.096287985e01,
            -3.837100020e-04,
            8.466889040e-08,
            -9.705399290e-12,
            4.498397540e-16,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
SO3_g: SpeciesData = SpeciesData.create("SO3", "g", _SO3_g_coeffs)
"Species data for SO3_g"

_FeO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(2.964572665e04, 3.037985806e04),
    b2=(1.326115545e01, -3.633655420e00),
    cp_coeffs=(
        (
            1.569282213e04,
            -6.460188880e01,
            2.458925470e00,
            7.016047360e-03,
            -1.021405947e-05,
            7.179297870e-09,
            -1.978966365e-12,
        ),
        (
            -1.195971480e05,
            -3.624864780e02,
            5.518880750e00,
            -9.978856890e-04,
            4.376913830e-07,
            -6.790629460e-11,
            3.639292680e-15,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
FeO_g: SpeciesData = SpeciesData.create("FeO", "g", _FeO_g_coeffs)
"Species data for FeO_g"

_Fe_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(5.466995940e04, 7.137370060e03, 4.847648290e06),
    b2=(-3.383946260e01, 6.504979860e01, -8.697289770e02),
    cp_coeffs=(
        (
            6.790822660e04,
            -1.197218407e03,
            9.843393310e00,
            -1.652324828e-02,
            1.917939959e-05,
            -1.149825371e-08,
            2.832773807e-12,
        ),
        (
            -1.954923682e06,
            6.737161100e03,
            -5.486410970e00,
            4.378803450e-03,
            -1.116286672e-06,
            1.544348856e-10,
            -8.023578182e-15,
        ),
        (
            1.216352511e09,
            -5.828563930e05,
            9.789634510e01,
            -5.370704430e-03,
            3.192037920e-08,
            6.267671430e-12,
            -1.480574914e-16,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
Fe_g: SpeciesData = SpeciesData.create("Fe", "g", _Fe_g_coeffs)
"Species data for Fe_g"

_NO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(9.098214410e03, 1.750317656e04, -4.677501240e06),
    b2=(6.728725490e00, -8.501669090e00, 1.242081216e03),
    cp_coeffs=(
        (
            -1.143916503e04,
            1.536467592e02,
            3.431468730e00,
            -2.668592368e-03,
            8.481399120e-06,
            -7.685111050e-09,
            2.386797655e-12,
        ),
        (
            2.239018716e05,
            -1.289651623e03,
            5.433936030e00,
            -3.656034900e-04,
            9.880966450e-08,
            -1.416076856e-11,
            9.380184620e-16,
        ),
        (
            -9.575303540e08,
            5.912434480e05,
            -1.384566826e02,
            1.694339403e-02,
            -1.007351096e-06,
            2.912584076e-11,
            -3.295109350e-16,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
NO_g: SpeciesData = SpeciesData.create("NO", "g", _NO_g_coeffs)
"Species data for NO_g"

_COS_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-1.191657685e04, -8.927096690e03),
    b2=(-2.991988593e01, -2.636328016e01),
    cp_coeffs=(
        (
            8.547876430e04,
            -1.319464821e03,
            9.735257240e00,
            -6.870830960e-03,
            1.082331416e-05,
            -7.705597340e-09,
            2.078570344e-12,
        ),
        (
            1.959098567e05,
            -1.756167688e03,
            8.710430340e00,
            -4.139424960e-04,
            1.015243648e-07,
            -1.159609663e-11,
            5.691053860e-16,
        ),
    ),
    T_min=(200, 1000),
    T_max=(1000, 6000),
)
COS_g: SpeciesData = SpeciesData.create("COS", "g", _COS_g_coeffs)
"Species data for COS_g"

_Ar_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.449939610e02, -5.078300340e06),
    b2=(4.379674910e00, 4.379180110e00, 1.465298484e03),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.010538475e01,
            -5.992661070e-02,
            2.500069401e00,
            -3.992141160e-08,
            1.205272140e-11,
            -1.819015576e-15,
            1.078576636e-19,
        ),
        (
            -9.951265080e08,
            6.458887260e05,
            -1.675894697e02,
            2.319933363e-02,
            -1.721080911e-06,
            6.531938460e-11,
            -9.740147729e-16,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
Ar_g: SpeciesData = SpeciesData.create("Ar", "g", _Ar_g_coeffs)
"Species data for Ar_g"

_He_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.453750000e02, 1.650518960e04),
    b2=(9.287239740e-01, 9.287239740e-01, -4.048814390e00),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            3.396845420e06,
            -2.194037652e03,
            3.080231878e00,
            -8.068957550e-05,
            6.252784910e-09,
            -2.574990067e-13,
            4.429960218e-18,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
He_g: SpeciesData = SpeciesData.create("He", "g", _He_g_coeffs)
"Species data for He_g"

_Ne_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.453750000e02, -5.663933630e04),
    b2=(3.355322720e00, 3.355322720e00, 1.648438697e01),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -1.238252746e07,
            6.958579580e03,
            1.016709287e00,
            1.424664555e-04,
            -4.803933930e-09,
            -1.170213183e-13,
            8.415153652e-18,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
Ne_g: SpeciesData = SpeciesData.create("Ne", "g", _Ne_g_coeffs)
"Species data for Ne_g"

_Kr_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.403488940e02, -7.111667370e06),
    b2=(5.490956510e00, 5.484398150e00, 2.086866326e03),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.643639057e02,
            -7.910050820e-01,
            2.500920585e00,
            -5.328164110e-07,
            1.620730161e-10,
            -2.467898017e-14,
            1.478585040e-18,
        ),
        (
            -1.375531087e09,
            9.064030530e05,
            -2.403481435e02,
            3.378312030e-02,
            -2.563103877e-06,
            9.969787790e-11,
            -1.521249677e-15,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
Kr_g: SpeciesData = SpeciesData.create("Kr", "g", _Kr_g_coeffs)
"Species data for Kr_g"

_Xe_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -6.685800730e02, 9.285443830e05),
    b2=(6.164454205e00, 6.063710715e00, -1.109834556e02),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.025226680e03,
            -1.209507521e01,
            2.514153347e00,
            -8.248102080e-06,
            2.530232618e-09,
            -3.892333230e-13,
            2.360439138e-17,
        ),
        (
            2.540397456e08,
            -1.105373774e05,
            1.382644099e01,
            1.500614606e-03,
            -3.935359030e-07,
            2.765790584e-11,
            -5.943990574e-16,
        ),
    ),
    T_min=(200, 1000, 6000),
    T_max=(1000, 6000, 20000),
)
Xe_g: SpeciesData = SpeciesData.create("Xe", "g", _Xe_g_coeffs)
"Species data for Xe_g"

_critical_data_H2O_g: CriticalData = CriticalData(647.25, 221.1925)
"""Critical parameters for H2O_g :cite:p:`SS92{Table 2}`"""
_critical_data_CO2_g: CriticalData = CriticalData(304.15, 73.8659)
"""Critical parameters for CO2_g :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 304.2 K and 73.8 bar
"""
_critical_data_CH4_g: CriticalData = CriticalData(191.05, 46.4069)
"""Critical parameters for CH4_g :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 190.6 K and 46 bar
"""
_critical_data_CO_g: CriticalData = CriticalData(133.15, 34.9571)
"""Critical parameters for CO :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 132.9 K and 35 bar
"""
_critical_data_O2_g: CriticalData = CriticalData(154.75, 50.7638)
"""Critical parameters for O2 :cite:p:`SS92{Table 2}`"""
_critical_data_H2_g: CriticalData = CriticalData(33.25, 12.9696)
"""Critical parameters for H2 :cite:p:`SS92{Table 2}`"""
_critical_data_H2_g_holland: CriticalData = CriticalData(41.2, 21.1)
"""Critical parameters for H2 :cite:p:`HP91`"""
_critical_data_S2_g: CriticalData = CriticalData(208.15, 72.954)
"""Critical parameters for S2 :cite:p:`SS92{Table 2}`

http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

:cite:p:`HP11` state that the critical parameters are from :cite:t:`RPS77`. However, in the fifth
edition of this book (:cite:t:`PPO00`) S2 is not given (only S is).
"""
_critical_data_SO2_g: CriticalData = CriticalData(430.95, 78.7295)
"""Critical parameters for SO2 :cite:p:`SS92{Table 2}`"""
_critical_data_COS_g: CriticalData = CriticalData(377.55, 65.8612)
"""Critical parameters for COS :cite:p:`SS92{Table 2}`"""
_critical_data_H2S_g: CriticalData = CriticalData(373.55, 90.0779)
"""Critical parameters for H2S :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 373.4 K and 0.08963 bar
"""
_critical_data_N2_g: CriticalData = CriticalData(126.2, 33.9)
"""Critical parameters for N2 :cite:p:`SF87{Table 1}`"""
_critical_data_Ar_g: CriticalData = CriticalData(151, 48.6)
"""Critical parameters for Ar :cite:p:`SF87{Table 1}`"""
_critical_data_He_g: CriticalData = CriticalData(5.2, 2.274)
"""Critical parameters for He :cite:p:`ADM77"""
_critical_data_Ne_g: CriticalData = CriticalData(44.49, 26.8)
"""Critical paramters for Ne :cite:p:`KJS86{Table 4}`"""
_critical_data_Kr_g: CriticalData = CriticalData(209.46, 55.2019)
"""Critical parameters for Kr :cite:p:`TB70`"""
_critical_data_Xe_g: CriticalData = CriticalData(289.765, 5.8415)
"""Critical parameters for Xe :cite:p:`SK94`"""

critical_data: dict[str, CriticalData] = {
    "Ar_g": _critical_data_Ar_g,
    "CH4_g": _critical_data_CH4_g,
    "CO_g": _critical_data_CO_g,
    "CO2_g": _critical_data_CO2_g,
    "COS_g": _critical_data_COS_g,
    "H2_g": _critical_data_H2_g,
    "H2_g_Holland": _critical_data_H2_g_holland,
    "H2O_g": _critical_data_H2O_g,
    "H2S_g": _critical_data_H2S_g,
    "N2_g": _critical_data_N2_g,
    "O2_g": _critical_data_O2_g,
    "S2_g": _critical_data_S2_g,
    "SO2_g": _critical_data_SO2_g,
    "He_g": _critical_data_He_g,
    "Ne_g": _critical_data_Ne_g,
    "Kr_g": _critical_data_Kr_g,
    "Xe_g": _critical_data_Xe_g,
}
"""Critical parameters for gases

These critical data could be extended to more species using :cite:t:`PPO00{Appendix A.19}`
"""
