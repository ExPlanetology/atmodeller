# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the initializers in :mod:`atmodeller.initializer`.

The cold initial guess of each initializer is compared against pinned values for a small H-O
system, without solving it. A magma ocean with a primordial atmosphere checks that the
element-potential guess converges where dissolution dominates the budgets of the minor elements.
"""

import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.initializer import (
    ElementPotentialInitializer,
    HeuristicInitializer,
    InitializerProtocol,
)
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
from atmodeller.solubility import get_solubility_models
from atmodeller.state import Planet


@pytest.fixture(scope="module")
def parameters() -> Parameters:
    """Parameters for H2O, H2, and O2 gas with H and O mass constraints"""
    species: tuple[ChemicalSpecies, ...] = tuple(
        ChemicalSpecies.create_gas(formula) for formula in ("H2O", "H2", "O2")
    )
    h_kg: ArrayLike = earth.oceans_to_hydrogen_mass(1)

    return Parameters(
        state=Planet.from_species(species, temperature=1500),
        mass_constraints={"H": h_kg, "O": 4 * h_kg},
    )


# Log number of moles of H2O, H2, and O2, followed by their log stabilities
@pytest.mark.parametrize(
    "initializer_class, expected",
    [
        (
            HeuristicInitializer,
            [52.00350906460628, 52.00350906460628, 46.47935504859353]
            + [-57.55668050046958, -56.17833296373126, -52.03252648445684],
        ),
        (
            ElementPotentialInitializer,
            [52.011455888987726, 51.99549858198854, 21.364412924513196]
            + [-57.56462732485103, -56.17032248111352, -26.9175843603765],
        ),
    ],
    ids=["heuristic", "element_potential"],
)
def test_initial_guess(
    parameters: Parameters, initializer_class: type[InitializerProtocol], expected: list[float]
) -> None:
    """The cold initial guess matches the pinned values, and the heuristic initializer warns
    that it is deprecated."""
    if initializer_class is HeuristicInitializer:
        with pytest.warns(FutureWarning, match="removed in version 3"):
            initializer: InitializerProtocol = initializer_class()
    else:
        initializer = initializer_class()

    np.testing.assert_allclose(initializer.start(parameters), expected, rtol=1e-8)


@pytest.fixture(scope="module")
def magma_ocean_parameters() -> Parameters:
    """Parameters for a sub-Neptune magma ocean with a primordial atmosphere

    The first re-equilibration of an IsoFATE run (LHS 1140 b): a planet of about 5.6 Earth masses
    at 6000 K with a mantle melt fraction of 0.92, and an H/He atmosphere of protosolar
    composition with trace C, N, O and S, in 11 gas and 8 dissolved species. The melt holds most
    of the minor elements, at a pressure of about 5e4 bar. An even split of the total over the gas
    species puts the dissolved species of C, N, O and S above their whole budgets, and the
    element-potential guess then failed to converge (Atmodeller 2.0.0).
    """
    solubility = get_solubility_models()
    gas_species: tuple[ChemicalSpecies, ...] = tuple(
        ChemicalSpecies.create_gas(formula)
        for formula in ("He", "H2", "H2O", "O2", "CO2", "CO", "CH4", "N2", "S2", "H2O4S", "SO2")
    )
    melt_species: tuple[ReservoirSpecies, ...] = tuple(
        ReservoirSpecies.create_dissolved(formula, solubility=solubility[model])
        for formula, model in (
            ("He", "He_basalt_jambon86"),
            ("H2", "H2_basalt_hirschmann12"),
            ("H2O", "H2O_basalt_dixon95"),
            ("CO2", "CO2_basalt_dixon95"),
            ("CO", "CO_basalt_yoshioka19"),
            ("CH4", "CH4_basalt_ardia13"),
            ("N2", "N2_basalt_libourel03"),
            ("S2", "S2_sulfide_basalt_boulliung23"),
        )
    )
    planet: Planet = Planet.from_species(
        gas_species,
        planet_mass=3.344432e25,
        core_mass_fraction=0.295334691460966,
        surface_radius=9811412.196650479,
        temperature=6000.0,
        silicate_melt_species=melt_species,
        mantle_melt_fraction=0.9192292610002111,
    )
    mass_constraints: dict[str, ArrayLike] = {
        "H": 2.3890350796749377e23,
        "He": 9.210769185275735e22,
        "O": 2.1994333998668505e21,
        "C": 8.255564101639023e20,
        "N": 2.655858281989794e20,
        "S": 1.3679808905834534e20,
    }

    return Parameters(state=planet, mass_constraints=mass_constraints)


def test_element_potential_initializer_magma_ocean(magma_ocean_parameters: Parameters) -> None:
    """The element-potential guess for a magma ocean with a primordial atmosphere satisfies every
    element budget (the initializer converges), and the solve succeeds from it on its first
    attempt."""
    parameters: Parameters = magma_ocean_parameters
    number_species: int = parameters.species.number_species
    guess: np.ndarray = np.asarray(ElementPotentialInitializer().start(parameters))

    formula_matrix: np.ndarray = np.asarray(parameters.reaction_system.formula_matrix, dtype=float)
    budget: np.ndarray = np.ravel(np.asarray(parameters.mass_constraints.abundance()))
    element_moles: np.ndarray = formula_matrix @ np.exp(guess[:number_species])
    np.testing.assert_allclose(element_moles, budget, rtol=1e-6)

    output: Output = EquilibriumModel(parameters).solve_with_default()
    assert bool(np.all(output.checked_solution.success))
    assert int(output.checked_solution.successful_attempt[0]) == 1
