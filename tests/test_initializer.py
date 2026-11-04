# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the initializers in :mod:`atmodeller.initializer`.

The cold initial guess of each initializer is compared against pinned values for a small H-O
system, without solving it.
"""

import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller.containers import ChemicalSpecies
from atmodeller.initializer import (
    ElementPotentialInitializer,
    HeuristicInitializer,
    InitializerProtocol,
)
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth
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
