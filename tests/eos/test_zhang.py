# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`ZD09`"""

import importlib.resources
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pandas as pd
from matplotlib.axes import Axes
from scipy.constants import kilo

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies
from atmodeller.eos import DATA_DIRECTORY, get_eos_models
from atmodeller.eos._zhang_duan import ZhangDuanMixture
from atmodeller.eos.core import RealGas
from atmodeller.jax_utils import NpArray
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import unit_conversion
from atmodeller.state import ThermodynamicState

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
MODEL_SUFFIX: str = "zhang09"
"""Suffix of the :cite:t:`ZD09` models"""

ZHANG_DUAN_DIRECTORY: Path = Path("zhang_duan_2009")

debug_logger()


def test_H2O_vol_low_TP(check_values) -> None:
    """Tests H2O volume at 1203.15 K and 950 MPa :cite:p:`ZD09{Table 6}`"""
    expected: float = 22.20343433408026
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1203.15,
        9500,  # 950 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_H2O_vol_high_TP(check_values) -> None:
    """Tests H2O volume at 1873.15 K and 2500 MPa :cite:p:`ZD09{Table 6}`"""
    expected: float = 19.41089977577485
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1873.15,
        25000,  # 2500 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_H2O_vol_high_TP2(check_values) -> None:
    """Tests H2O volume at 1373.15 K and 3500 MPa :cite:p:`ZD09{Table 6}`"""
    expected: float = 16.02290245403692
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1373.15,
        35000,  # 3500 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_volume_with_broadcasting(check_values) -> None:
    """Tests volume with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2O", MODEL_SUFFIX)
    check_values.check_broadcasting("volume", model)


def test_fugacity_with_broadcasting(check_values) -> None:
    """Tests volume with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2O", MODEL_SUFFIX)
    check_values.check_broadcasting("fugacity", model)


def test_H2O_CO2_vol_mix_low_H2O() -> None:
    """Experimental molar volume in the H2O-CO2 system

    Tests H2O-CO2 mixture at 1573.15 K and 14500 MPa :cite:p:`ZD09{Figure 1}` using data from
    :cite:t:`Frost1997`.
    """
    species: tuple[str, ...] = ("H2O", "CO2")
    model = ZhangDuanMixture(species)

    mole_fractions = jnp.array([0.6125, 0.3875])

    expected: float = 28.31805
    expected *= unit_conversion.cm3_to_m3
    volume = model.volume(1573.15, 14500, mole_fractions)

    nptest.assert_allclose(volume, expected, rtol=RTOL, atol=ATOL)


def test_H2O_CO2_vol_mix_high_H2O() -> None:
    """Experimental molar volume in the H2O-CO2 system

    Tests H2O-CO2 mixture at 1573.15 K and 14500 MPa :cite:p:`ZD09{Figure 1}` using data from
    :cite:t:`Frost1997`.
    """
    species: tuple[str, ...] = ("H2O", "CO2")
    model = ZhangDuanMixture(species)

    mole_fractions = jnp.array([0.875, 0.125])

    expected: float = 23.93116
    expected *= unit_conversion.cm3_to_m3
    volume = model.volume(1573.15, 14500, mole_fractions)

    nptest.assert_allclose(volume, expected, rtol=RTOL, atol=ATOL)


def test_RTlnfCO2() -> None:
    """Comparison of RTlnf for CO2

    Tests based on the data for MgCO3 --> MgO + CO2 :cite:p:`ZD09{Table 8}`.
    """
    # MPa to bar
    unit_scale = 10

    pressure = np.array([500, 700, 1000, 1630, 1920, 2200]) * unit_scale
    temperature = np.array([1298, 1278, 1373, 1600, 1700, 1800])

    model_CO2: RealGas = cast(RealGas, get_eos_models()["CO2_zhang09"])
    RTlnfCO2 = model_CO2.volume_integral_J(temperature, pressure) / kilo  # convert J->kJ

    expected: NpArray = np.array([106.5, 114.7, 135.1, 178.1, 197.1, 215.6])

    # rtol chosen to account for rounding in the expected values, which are only given to 1
    # decimal place in the paper. The expected values are likely rounded, but the model is not,
    # so we need to allow for some tolerance in the comparison.
    nptest.assert_allclose(RTlnfCO2, expected, rtol=3.3e-4, atol=ATOL)


def test_log_activity() -> None:
    """Test log activity"""
    species: tuple[str, ...] = ("H2O", "CO2", "CH4")
    eos_H2O = ZhangDuanMixture(species, "H2O")
    eos_CO2 = ZhangDuanMixture(species, "CO2")
    eos_CH4 = ZhangDuanMixture(species, "CH4")

    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", activity=eos_H2O)
    CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2", activity=eos_CO2)
    CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4", activity=eos_CH4)

    gas_species = (H2O_g, CO2_g, CH4_g)

    pressure = 2.4e3 * 10
    temperature = 1273

    mole_fractions = jnp.array([0.5, 0.25, 0.25])

    H2O_log_activity = eos_H2O.log_activity(temperature, pressure, mole_fractions)
    CO2_log_activity = eos_CO2.log_activity(temperature, pressure, mole_fractions)
    CH4_log_activity = eos_CH4.log_activity(temperature, pressure, mole_fractions)

    print("H2O activity:", jnp.exp(H2O_log_activity))
    print("CO2 activity:", jnp.exp(CO2_log_activity))
    print("CH4 activity:", jnp.exp(CH4_log_activity))

    # state = ThermodynamicState.from_species(
    #    gas_species, pressure=pressure, temperature=temperature
    # )

    # mass_constraints = {"O": 0.25, "C": 0.5, "H": 0.25}
    # parameters = Parameters(state, mass_constraints=mass_constraints, mass_units="moles")

    # model: EquilibriumModel = EquilibriumModel(parameters)

    # output: Output = model.solve_with_default()


def plot_ZD09_fig2a(ax: Axes) -> None:
    """Plots the data from :cite:t:`ZD09{Figure 2a}`.

    Args:
        ax: The axes to plot on
    """
    filename: Path = Path("ZD09_fig2a_data.csv")
    data: AbstractContextManager[Path] = importlib.resources.as_file(
        DATA_DIRECTORY.joinpath(str(ZHANG_DUAN_DIRECTORY.joinpath(filename)))
    )
    with data as datapath:
        df: pd.DataFrame = pd.read_csv(datapath)

    ax.plot(df["X_O"], df["X_H2O"], "s", label="Data from ZD09 Fig. 2a")
    ax.set_xlabel("X_O")
    ax.set_ylabel("X_H2O")
    ax.set_title("X_H2O vs X_O")
    ax.set_xlim(0.15, 0.4)
    ax.set_ylim(0.5, 1.0)


def plot_ZD09_fig2b(ax: Axes) -> None:
    """Plots the data from :cite:t:`ZD09{Figure 2b}`.

    Args:
        ax: The axes to plot on
    """
    filename: Path = Path("ZD09_fig2b_data.csv")
    data: AbstractContextManager[Path] = importlib.resources.as_file(
        DATA_DIRECTORY.joinpath(str(ZHANG_DUAN_DIRECTORY.joinpath(filename)))
    )
    with data as datapath:
        df: pd.DataFrame = pd.read_csv(datapath)

    ax.plot(df["X_O"], df["X_CH4"], "s", label="Data from ZD09 Fig. 2b")
    ax.set_xlabel("X_O")
    ax.set_ylabel("X_CH4")
    ax.set_title("X_CH4 vs X_O")
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0.5, 1.0)


def get_O_mole_fraction(test: dict) -> float:
    """Gets element mole fraction of O in the gas phase

    Returns:
        O mole fraction
    """
    # System is carbon saturated
    # moles_carbon = test["gas"]["elements"]["number_moles"]["C"]
    moles_hydrogen = test["gas"]["elements"]["number_moles"]["H"]
    moles_oxygen = test["gas"]["elements"]["number_moles"]["O"]
    moles_total = moles_hydrogen + moles_oxygen  # + moles_carbon
    O_ratio = moles_oxygen / moles_total

    return O_ratio


def test_autodiff() -> None:
    """TODO"""

    # Mixture EOS
    species: tuple[str, ...] = ("H2O", "CO2", "CH4", "O2", "CO", "H2", "C2H6")

    eos_H2O = ZhangDuanMixture(species, "H2O")
    eos_CO2 = ZhangDuanMixture(species, "CO2")
    eos_CH4 = ZhangDuanMixture(species, "CH4")
    eos_O2 = ZhangDuanMixture(species, "O2")
    eos_CO = ZhangDuanMixture(species, "CO")
    eos_H2 = ZhangDuanMixture(species, "H2")
    eos_C2H6 = ZhangDuanMixture(species, "C2H6")

    # H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", activity=eos_H2O)
    # CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2", activity=eos_CO2)
    # CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4", activity=eos_CH4)
    # O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2", activity=eos_O2)
    # CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO", activity=eos_CO)
    # H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_H2)
    # C2H6_g: ChemicalSpecies = ChemicalSpecies.create_gas("C2H6", activity=eos_C2H6)

    # mixture_gas_species = (H2O_g, CO2_g, CH4_g, O2_g, CO_g, H2_g, C2H6_g)

    pressure = 2.4e3 * 10
    temperature = 1273

    moles_in = jnp.array([0.05, 0.05, 0.05, 0.1, 0.25, 0.25, 0.25])

    out_autodiff = eos_H2O.log_fugacity(temperature, pressure, moles_in)

    out_func = eos_H2O.log_fugacity_original(temperature, pressure, moles_in)

    print("out_autodiff:", out_autodiff)
    print("out_func:", out_func)


def test_compare_fig2() -> None:
    """Comparison of the model output to the data in :cite:t:`ZD09{Figure 2}`"""

    PURE: bool = True
    MIXTURE: bool = True

    if PURE:
        # Pure EOS
        pure_eos_H2O = get_eos_models()["H2O_zhang09"]
        pure_eos_CO2 = get_eos_models()["CO2_zhang09"]
        pure_eos_CH4 = get_eos_models()["CH4_zhang09"]
        pure_eos_O2 = get_eos_models()["O2_zhang09"]
        pure_eos_CO = get_eos_models()["CO_zhang09"]
        pure_eos_H2 = get_eos_models()["H2_zhang09"]
        pure_eos_C2H6 = get_eos_models()["C2H6_zhang09"]

        H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", activity=pure_eos_H2O)
        CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2", activity=pure_eos_CO2)
        CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4", activity=pure_eos_CH4)
        O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2", activity=pure_eos_O2)
        CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO", activity=pure_eos_CO)
        H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=pure_eos_H2)
        C2H6_g: ChemicalSpecies = ChemicalSpecies.create_gas("C2H6", activity=pure_eos_C2H6)

        pure_gas_species = (H2O_g, CO2_g, CH4_g, O2_g, CO_g, H2_g, C2H6_g)

    if MIXTURE:
        # Mixture EOS
        species: tuple[str, ...] = ("H2O", "CO2", "CH4", "O2", "CO", "H2", "C2H6")

        eos_H2O = ZhangDuanMixture(species, "H2O")
        eos_CO2 = ZhangDuanMixture(species, "CO2")
        eos_CH4 = ZhangDuanMixture(species, "CH4")
        eos_O2 = ZhangDuanMixture(species, "O2")
        eos_CO = ZhangDuanMixture(species, "CO")
        eos_H2 = ZhangDuanMixture(species, "H2")
        eos_C2H6 = ZhangDuanMixture(species, "C2H6")

        H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O", activity=eos_H2O)
        CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2", activity=eos_CO2)
        CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4", activity=eos_CH4)
        O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2", activity=eos_O2)
        CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO", activity=eos_CO)
        H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_H2)
        C2H6_g: ChemicalSpecies = ChemicalSpecies.create_gas("C2H6", activity=eos_C2H6)

        mixture_gas_species = (H2O_g, CO2_g, CH4_g, O2_g, CO_g, H2_g, C2H6_g)

    # Experiments are buffered by graphite
    graphite: PurePhase = PurePhase.from_species("C", state="s", solve_for_stability=False)
    condensates: tuple[PurePhase, ...] = (graphite,)
    pressure = 2.4e3 * 10
    temperature = 1273

    if PURE:
        pure_eos_state = ThermodynamicState.from_species(
            pure_gas_species, pressure=pressure, temperature=temperature, condensates=condensates
        )
    if MIXTURE:
        mixture_eos_state = ThermodynamicState.from_species(
            mixture_gas_species,
            pressure=pressure,
            temperature=temperature,
            condensates=condensates,
        )

    # Normalise to 1 mole of H
    mass_constraints = {"H": 1.0, "O": np.logspace(-2, 6, 200)}

    # Define log10 shifts for the IW buffer
    # shifts = np.linspace(-2, 6, 50)
    # IW_arg = np.argmin(np.abs(shifts))

    # fugacity_IW = IronWustiteBuffer(shifts, evaluation_pressure=None)
    # fugacity_IW_value = np.exp(fugacity_IW.log_activity(temperature, pressure))
    # print("O2 fugacity (IW at P):", fugacity_IW_value)
    # fugacity_IW_value_1bar = np.exp(fugacity_IW.log_activity(temperature, 1.0))
    # print("O2 fugacity (IW at 1 bar):", fugacity_IW_value_1bar)

    # IW_value = IronWustiteBuffer(shifts[IW_arg], evaluation_pressure=1)
    # IW_value_at_P = np.exp(IW_value.log_activity(temperature, pressure))

    # activity_constraints: dict[str, ActivityConstraintProtocol] = {"O2_g": fugacity_IW}

    # initial_guess = jnp.array(
    #    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, -50, -50, -50, -50, -50, -50, -50, -50]
    # )

    if PURE:
        # Pure model
        pure_parameters = Parameters(
            pure_eos_state,
            mass_constraints=mass_constraints,
            # activity_constraints=activity_constraints,
            mass_units="moles",
        )
        pure_model: EquilibriumModel = EquilibriumModel(pure_parameters)
        pure_output: Output = pure_model.solve_with_default()
        pure_dict = pure_output.to_dict()
        pure_H2O = pure_dict["gas"]["species"]["mole_fraction"]["H2O_g"]
        pure_H2 = pure_dict["gas"]["species"]["mole_fraction"]["H2_g"]
        pure_CH4 = pure_dict["gas"]["species"]["mole_fraction"]["CH4_g"]
        pure_CO2 = pure_dict["gas"]["species"]["mole_fraction"]["CO2_g"]
        pure_O = get_O_mole_fraction(pure_dict)
        pure_output.to_excel("pure_output")

    if MIXTURE:
        # Mixture model
        mixture_parameters = Parameters(
            mixture_eos_state,
            mass_constraints=mass_constraints,
            # activity_constraints=activity_constraints,
            mass_units="moles",
        )
        mixture_model: EquilibriumModel = EquilibriumModel(mixture_parameters)
        mixture_output: Output = mixture_model.solve_with_default()
        mixture_dict = mixture_output.to_dict()
        mixture_H2O = mixture_dict["gas"]["species"]["mole_fraction"]["H2O_g"]
        mixture_H2 = mixture_dict["gas"]["species"]["mole_fraction"]["H2_g"]
        mixture_CH4 = mixture_dict["gas"]["species"]["mole_fraction"]["CH4_g"]
        mixture_CO2 = mixture_dict["gas"]["species"]["mole_fraction"]["CO2_g"]
        mixture_O = get_O_mole_fraction(mixture_dict)
        mixture_output.to_excel("mixture_output")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_ZD09_fig2a(axes[0])
    if PURE:
        axes[0].plot(pure_O, pure_H2O, linestyle="--", color="grey", label="Pure output")
    if MIXTURE:
        axes[0].plot(mixture_O, mixture_H2O, linestyle="-", color="black", label="Mixture output")
    axes[0].legend()

    plot_ZD09_fig2b(axes[1])
    if PURE:
        axes[1].plot(pure_O, pure_CH4, linestyle="--", color="grey", label="Pure output")
    if MIXTURE:
        axes[1].plot(mixture_O, mixture_CH4, linestyle="-", color="black", label="Mixture output")
        IW_shifts = mixture_dict["gas"]["phase"]["log10dIW_P"]
        IW_arg = np.argmin(np.abs(IW_shifts))
        axes[1].axvline(
            mixture_O[IW_arg],
            linestyle=":",
            color="red",
            label="IW buffer at P",
        )
    axes[1].legend()

    fig, axes2 = plt.subplots(1, 1, figsize=(12, 6))
    if PURE:
        axes2.plot(pure_O, pure_H2O, linestyle="--", color="blue", label="H2O (pure)")
        axes2.plot(pure_O, pure_CH4, linestyle="--", color="purple", label="CH4 (pure)")
        axes2.plot(pure_O, pure_H2, linestyle="--", color="green", label="H2 (pure)")
        axes2.plot(pure_O, pure_CO2, linestyle="--", color="orange", label="CO2 (pure)")
    if MIXTURE:
        axes2.plot(mixture_O, mixture_H2O, linestyle="-", color="blue", label="H2O (mix)")
        axes2.plot(mixture_O, mixture_CH4, linestyle="-", color="purple", label="CH4 (mix)")
        axes2.plot(mixture_O, mixture_H2, linestyle="-", color="green", label="H2 (mix)")
        axes2.plot(mixture_O, mixture_CO2, linestyle="-", color="orange", label="CO2 (mix)")
    axes2.axvline(
        mixture_O[IW_arg],
        linestyle=":",
        color="black",
        label="IW buffer at P",
    )
    axes2.set_xlim(0, 1)
    axes2.set_ylim(0, 1)

    axes2.legend()

    plt.show()
