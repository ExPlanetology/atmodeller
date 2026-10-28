# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the integrated thermodynamic properties, the diamond of Gustafson (1986), the Hill
formulae of the NASA Glenn data and the data outside their temperature ranges with and without
extrapolation, and the alternative carbon and silicon models of Vassiliev (2021) and Holland and
Powell"""

from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from molmass import Formula
from scipy.optimize import brentq

from atmodeller.constants import NO_DATA_GIBBS_OVER_RT, TEMPERATURE_REFERENCE
from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.database import DataBase
from atmodeller.sci_utils import GAS_CONSTANT
from atmodeller.thermodata import (
    IntegratedThermodynamicProperties,
    ThermodynamicProperties,
)
from atmodeller.thermodata import holland_powell, vassiliev
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.glenn import (
    NasaGlennThermodynamicProperties,
    glenn_properties,
)

graphite: NasaGlennThermodynamicProperties = glenn_properties["C"]["graphite"]
"""Graphite from the NASA Glenn coefficients"""

TEMPERATURE: Array = jnp.linspace(200.0, 6000.0, 2901)
"""Temperatures in K spanning the NASA Glenn data for graphite"""


class GraphiteHeatCapacity(eqx.Module):
    """Heat capacity of graphite from the NASA Glenn coefficients, as a heat capacity model"""

    def cp(self, temperature):
        return graphite.cp(temperature)

    def temperature_range(self) -> tuple[float, float]:
        return graphite.temperature_range()


def test_integrated_matches_analytical() -> None:
    """Integrating the NASA Glenn heat capacity of graphite recovers its analytical H and S"""
    integrated: ThermodynamicProperties = IntegratedThermodynamicProperties(
        GraphiteHeatCapacity(),
        float(graphite.enthalpy(TEMPERATURE_REFERENCE)),
        float(graphite.entropy(TEMPERATURE_REFERENCE)),
    )

    np.testing.assert_allclose(
        integrated.enthalpy(TEMPERATURE), graphite.enthalpy(TEMPERATURE), rtol=0, atol=0.05
    )
    np.testing.assert_allclose(
        integrated.entropy(TEMPERATURE), graphite.entropy(TEMPERATURE), rtol=0, atol=1e-4
    )


def test_gustafson_gibbs_energy_exact() -> None:
    """Diamond of Gustafson is NASA Glenn graphite plus the Gibbs energy of diamond relative to
    graphite printed by Dinsdale (1991, p. 338), without quadrature error"""
    temperature: Array = jnp.linspace(TEMPERATURE_REFERENCE, 6000.0, 50)
    relative_printed: Array = (
        1009
        + 4.88 * temperature
        - 0.01 * temperature * jnp.log(temperature)
        + 135400 / temperature
        + 33.0e5 / temperature**2
        - 9e8 / temperature**3
    )
    np.testing.assert_allclose(
        diamond_gustafson.gibbs_energy(temperature) - graphite.gibbs_energy(temperature),
        relative_printed,
        atol=1e-6,
    )


def test_gustafson_reproduces_sgte_boundary() -> None:
    """Diamond and graphite with the SGTE volumes reproduce the transition of Gustafson (1986)"""
    graphite_gustafson: ThermodynamicProperties = graphite.with_volume(GRAPHITE_VOLUME_GUSTAFSON)

    def gibbs_difference(pressure: float, temperature: float) -> float:
        return float(
            diamond_gustafson.standard_gibbs_over_RT(temperature, pressure)
            - graphite_gustafson.standard_gibbs_over_RT(temperature, pressure)
        )

    # In K and GPa, from the SGTE Gibbs energies of graphite and diamond (Dinsdale 1991). At
    # 4710 K this is the graphite-diamond-liquid triple point at 12.4 GPa (Gustafson 1986)
    for temperature, pressure in (
        (298.15, 1.634),
        (1000.0, 3.360),
        (2000.0, 5.984),
        (3000.0, 8.502),
        (4000.0, 10.865),
        (4710.0, 12.447),
        (6000.0, 15.135),
    ):
        boundary: float = cast(float, brentq(gibbs_difference, 1.0, 1e6, args=(temperature,)))
        np.testing.assert_allclose(boundary / 1e4, pressure, atol=0.005)


def test_glenn_formulae_are_hill() -> None:
    """Every formula in the NASA Glenn data is in Hill order, since species are looked up by the
    Hill formula of their input formula"""
    formulae: set[str] = set(glenn_properties)
    not_hill: list[str] = sorted(f for f in formulae if Formula(f).formula != f)

    assert not not_hill, f"Not in Hill order: {not_hill}"
    assert ChemicalSpecies.create_gas("NH").name == "HN_g"


def test_phase_components_and_stitching() -> None:
    """Stitching the condensed phase components of a formula in order of temperature gives each
    phase component in its range; overlaps, gaps and volume models cannot be stitched"""
    stitch = NasaGlennThermodynamicProperties.stitch
    silica: dict[str, NasaGlennThermodynamicProperties] = glenn_properties["O2Si"]
    stitched = stitch(
        {
            name: silica[name]
            for name in ("quartz-beta", "liquid", "cristobalite-beta", "quartz-alpha")
        }
    )
    for temperature, name in (
        (300.0, "quartz-alpha"),
        (1000.0, "quartz-beta"),
        (1500.0, "cristobalite-beta"),
        (4000.0, "liquid"),
    ):
        np.testing.assert_array_equal(
            stitched.standard_gibbs_over_RT(temperature),
            silica[name].standard_gibbs_over_RT(temperature),
        )
    # A single phase component is returned as it is, so graphite keeps its volume
    graphite_volume = cast(
        NasaGlennThermodynamicProperties,
        DataBase.default().get_phase_component("C", "graphite"),
    )
    assert stitch({"graphite": graphite_volume}) is graphite_volume

    hematite: dict[str, NasaGlennThermodynamicProperties] = glenn_properties["Fe2O3"]
    with pytest.raises(ValueError, match="overlaps"):
        stitch(hematite)
    with pytest.raises(ValueError, match="gap"):
        stitch({name: silica[name] for name in ("quartz-alpha", "liquid")})
    with pytest.raises(ValueError, match="volume model"):
        stitch({"graphite": graphite_volume, "other": graphite})


def test_ions_can_be_selected() -> None:
    """Ions have the electron as an element, with the molar mass of the electron"""
    ions = SpeciesCollection([ChemicalSpecies.create_gas("D+"), ChemicalSpecies.create_gas("D-")])

    assert ions.unique_elements == ("2H", "e-")
    np.testing.assert_allclose(ions.element_molar_masses, [2.01410177812e-3, 5.48579909065e-7])


def test_glenn_extrapolation_uses_nearest_range() -> None:
    """Outside its ranges, the NASA Glenn data use the nearest range, so G is continuous"""
    temperature: Array = jnp.array([199.0, 200.0, 6000.0, 6001.0, 7000.0])

    np.testing.assert_array_equal(graphite.get_index(temperature), [0, 0, 2, 2, 2])
    np.testing.assert_allclose(
        graphite.standard_gibbs_over_RT(6000.0 + 1e-6, 1.0),
        graphite.standard_gibbs_over_RT(6000.0, 1.0),
        rtol=1e-8,
    )


def test_no_extrapolation_holds_gibbs_and_adds_penalty() -> None:
    """Without extrapolation, G is held at the bound and the penalty is added outside the range.
    Condensates do not extrapolate by default, whereas gases keep the shared default data."""
    water: ThermodynamicProperties = glenn_properties["H2O"]["liquid"]
    no_extrapolation: ThermodynamicProperties = water.with_extrapolation(False, False)
    temperature_min, temperature_max = water.temperature_range()

    def held(temperature_bound: float, temperature: float) -> Array:
        return water.gibbs_energy(temperature_bound) / (GAS_CONSTANT * temperature)

    np.testing.assert_array_equal(
        no_extrapolation.standard_gibbs_over_RT(500.0), water.standard_gibbs_over_RT(500.0)
    )
    np.testing.assert_allclose(
        no_extrapolation.standard_gibbs_over_RT(jnp.array([200.0, 1000.0])),
        [
            held(temperature_min, 200.0) + NO_DATA_GIBBS_OVER_RT,
            held(temperature_max, 1000.0) + NO_DATA_GIBBS_OVER_RT,
        ],
        rtol=1e-12,
    )
    # Extrapolating below only
    np.testing.assert_array_equal(
        water.with_extrapolation(True, False).standard_gibbs_over_RT(200.0),
        water.standard_gibbs_over_RT(200.0),
    )

    condensed: ThermodynamicProperties = ChemicalSpecies.from_component(
        "liquid", formula="H2O"
    ).thermo
    assert not condensed.extrapolate_below and not condensed.extrapolate_above
    assert ChemicalSpecies.create_gas("H2O").thermo is glenn_properties["H2O"]["gas"]


def test_vassiliev_heat_capacity() -> None:
    """The Vassiliev (2021) heat capacities agree with standard values at 298.15 K, and are pinned
    at 1000 K"""
    # Standard heat capacities at 298.15 K (J/mol/K): graphite from the NASA Glenn data, diamond
    # and silicon from NIST-JANAF
    reference: dict[str, float] = {
        "diamond": 6.115,
        "graphite": float(graphite.cp(TEMPERATURE_REFERENCE)),
        "silicon": 20.00,
    }
    models: dict[str, list[vassiliev.VassilievHeatCapacity]] = {
        "diamond": [
            vassiliev.cp_diamond_1a,
            vassiliev.cp_diamond_1b,
            vassiliev.cp_diamond_1c,
            vassiliev.cp_diamond_1d,
        ],
        "graphite": [vassiliev.cp_graphite_2a, vassiliev.cp_graphite_2b, vassiliev.cp_graphite_2c],
        "silicon": [vassiliev.cp_silicon_3a, vassiliev.cp_silicon_3b],
    }
    for material, material_models in models.items():
        for model in material_models:
            np.testing.assert_allclose(
                model.cp(TEMPERATURE_REFERENCE), reference[material], rtol=0.04
            )

    pinned_1000: list[tuple[vassiliev.VassilievHeatCapacity, float]] = [
        (vassiliev.cp_diamond_1a, 21.1462),
        (vassiliev.cp_graphite_2a, 21.7438),
        (vassiliev.cp_silicon_3a, 26.5799),
    ]
    for model, expected in pinned_1000:
        np.testing.assert_allclose(model.cp(1000.0), expected, rtol=1e-5)


def test_holland_powell_volume() -> None:
    """The Holland and Powell Murnaghan and Tait volume models of graphite and diamond: the
    reference volume at 298.15 K and 1 bar, a zero volume integral at 1 bar, agreement between the
    two forms at 50 kbar, and pinned integrals"""
    pairs: dict[str, tuple[holland_powell.MurnaghanEOS, holland_powell.TaitEOS, float]] = {
        "diamond": (
            holland_powell.DIAMOND_VOLUME_MURNAGHAN,
            holland_powell.DIAMOND_VOLUME_TAIT,
            0.342,
        ),
        "graphite": (
            holland_powell.GRAPHITE_VOLUME_MURNAGHAN,
            holland_powell.GRAPHITE_VOLUME_TAIT,
            0.530,
        ),
    }
    pinned: dict[str, tuple[float, float]] = {
        "diamond": (17229.528855, 17276.191681),
        "graphite": (25777.298750, 25504.057812),
    }
    for material, (murnaghan, tait, volume_reference) in pairs.items():
        np.testing.assert_allclose(murnaghan.volume_1bar(TEMPERATURE_REFERENCE), volume_reference)
        np.testing.assert_allclose(tait.volume(TEMPERATURE_REFERENCE, 1.0), volume_reference)
        for model in (murnaghan, tait):
            np.testing.assert_allclose(model.volume_integral(1000.0, 1.0), 0.0, atol=1e-12)

        integral_murnaghan = murnaghan.volume_integral(1500.0, 5.0e4)
        integral_tait = tait.volume_integral(1500.0, 5.0e4)
        np.testing.assert_allclose(integral_murnaghan, integral_tait, rtol=0.015)
        np.testing.assert_allclose(
            [integral_murnaghan, integral_tait], pinned[material], rtol=1e-6
        )

    # The linear approximation of the 1-bar volume and the Einstein temperature
    np.testing.assert_allclose(
        holland_powell.DIAMOND_VOLUME_MURNAGHAN.volume_1bar_linear(1000.0),
        holland_powell.DIAMOND_VOLUME_MURNAGHAN.volume_1bar(1000.0),
        rtol=1e-4,
    )
    np.testing.assert_allclose(holland_powell.einstein_temperature(2.38, 1), 10636 / (2.38 + 6.44))
