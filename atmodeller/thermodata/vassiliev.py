# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Heat capacity models for graphite and diamond from :cite:t:`Vassiliev2021`"""

import equinox as eqx
from jaxtyping import ArrayLike

from atmodeller.jax_utils import FloatArray


class VassilievThermodynamicModel(eqx.Module):
    def _cp_over_R(self, temperature: ArrayLike) -> FloatArray: ...

    def cp(self, temperature: ArrayLike) -> FloatArray: ...
