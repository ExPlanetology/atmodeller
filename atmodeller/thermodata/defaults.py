# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Default thermodynamic data

The default data are the packaged NASA Glenn coefficients
(:data:`~atmodeller.thermodata.glenn.glenn_properties`) with the following changes:

    - Graphite (``["C"]["graphite"]``) includes the volume
      :data:`~atmodeller.thermodata.gustafson.GRAPHITE_VOLUME_GUSTAFSON`, so that its Gibbs energy,
      and hence the equilibrium constant of any reaction involving it, depends on pressure.
    - Diamond (``["C"]["diamond"]``) is added as
      :data:`~atmodeller.thermodata.gustafson.diamond_gustafson`, with the Gibbs energy of
      graphite plus the Gibbs energy of diamond relative to graphite of :cite:t:`Gustafson1986`
      and its own volume.

The volumes change the Gibbs energies of graphite and diamond, not their activities. Gases and
other condensates are unchanged.
"""

from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.glenn import glenn_properties

default_properties: dict[str, dict[str, ThermodynamicProperties]] = {
    **{
        formula: dict[str, ThermodynamicProperties](components)
        for formula, components in glenn_properties.items()
    },
    "C": {
        **glenn_properties["C"],
        "graphite": glenn_properties["C"]["graphite"].with_volume(GRAPHITE_VOLUME_GUSTAFSON),
        "diamond": diamond_gustafson,
    },
}
"""Default thermodynamic properties of each phase component, keyed by Hill formula and then phase
component, e.g. ``default_properties["C"]["graphite"]``

:meta hide-value:
"""
