# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Custom activity models for specific species in specific phases."""

from dataclasses import KW_ONLY

import jax.numpy as jnp
from jaxtyping import ArrayLike

from atmodeller.jax_utils import FloatArray
from atmodeller.thermodata.core import ActivityCoefficient


class Si_iron_badro15(ActivityCoefficient):
    """Activity coefficient of Si in Fe. # TODO: Add citation for Badro 2015

    Args:
        gamma: Activity coefficient. Defaults to 1 (ideal).
        Si_index: Index of silicon in the mole fraction array for the metal species
    """

    _: KW_ONLY
    Si_index: int
    """Index of silicon in the mole fraction array for the metal species"""

    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike, mole_fractions: FloatArray | None = None
    ) -> FloatArray:
        """Log of the activity coefficient (dimensionless).

        This is the primary access point for calling the EOS within the main engine so must adhere
        to the expected interface for activity.

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)
            mole_fractions: Mole fractions. Defaults to ``None`` if unused.

        Returns:
            Log activity coefficient
        """
        del temperature
        del pressure

        if mole_fractions is None:
            raise ValueError("Mole fractions must be provided for this activity model.")

        x_si = mole_fractions[..., self.Si_index]

        return jnp.log(self.gamma * 0.3 * x_si)
