# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Solubility package"""

# Expose the public API
from atmodeller.solubility.core import (  # noqa: E402, F401
    NoSolubility,
    Solubility,
    SolubilityPowerLaw,
    SolubilityPowerLawLog10,
)
from atmodeller.solubility.library import (  # noqa: E402, F401
    get_solubility_catalogue,
    get_solubility_models,
)

__all__ = [
    "NoSolubility",
    "Solubility",
    "SolubilityPowerLaw",
    "SolubilityPowerLawLog10",
    "get_solubility_catalogue",
    "get_solubility_models",
]
