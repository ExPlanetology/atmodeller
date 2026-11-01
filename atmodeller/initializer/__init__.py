# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Initializers for the solver's initial guess

An initializer provides the cold initial guess for a single system, and is vmapped over the batch
by the solver (see :func:`~atmodeller.initializer.batch.make_batch_initializer`).

- :class:`~atmodeller.initializer.core.InitializerProtocol`: The interface
- :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer`: Solves an
  ideal-gas equilibrium in the element potentials; the default
- :class:`~atmodeller.initializer.heuristic.HeuristicInitializer`: Allocates element budgets
  heuristically
- :func:`~atmodeller.initializer.batch.make_batch_initializer`: Completes a batch of initial
  guesses with an initializer's cold guess
"""

from atmodeller.initializer.core import InitializerProtocol  # noqa: F401
from atmodeller.initializer.element_potential import ElementPotentialInitializer  # noqa: F401
from atmodeller.initializer.heuristic import HeuristicInitializer  # noqa: F401
from atmodeller.initializer.batch import make_batch_initializer  # noqa: F401
