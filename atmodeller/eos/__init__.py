# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""EOS package"""

import importlib.resources
from importlib.resources.abc import Traversable

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory, which is the same as the package directory"""
# ABSOLUTE_TOLERANCE is less than RELATIVE_TOLERANCE because typical volumes are around 1e-6
ABSOLUTE_TOLERANCE: float = 1.0e-12
r"""Absolute tolerance when solving for the volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`"""
RELATIVE_TOLERANCE: float = 1.0e-6
r"""Relative tolerance when solving for the volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`"""
THROW: bool = False
"""Whether to throw errors. Change to ``True`` for debugging purposes."""
VOLUME_EPSILON: float = 1.0e-12
r"""Small volume offset in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`"""

# Expose the public API
from atmodeller.eos._aggregators import CombinedRealGas  # noqa: E402, F401
from atmodeller.eos._zhang_duan import ZhangDuanMixture  # noqa: E402, F401
from atmodeller.eos.core import IdealGas, RealGas  # noqa: E402, F401
from atmodeller.eos.library import get_eos_models  # noqa: E402, F401
