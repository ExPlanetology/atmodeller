# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Default thermodynamic data

The default data are the packaged NASA Glenn coefficients
(:data:`~atmodeller.thermodata.janaf.glenn_properties`) with the following changes:

    - Graphite (``C_s``) includes the volume
      :data:`~atmodeller.thermodata.gustafson.GRAPHITE_VOLUME_GUSTAFSON`, so that its Gibbs energy,
      and hence the equilibrium constant of any reaction involving it, depends on pressure.
    - Diamond (``C_diamond``) is added as
      :data:`~atmodeller.thermodata.gustafson.diamond_gustafson`, with the Gibbs energy of
      graphite plus the Gibbs energy of diamond relative to graphite of :cite:t:`Gustafson1986`
      and its own volume.

The volumes change the Gibbs energies of graphite and diamond, not their activities. Gases and
other condensates are unchanged.
"""

from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson
from atmodeller.thermodata.janaf import glenn_properties

default_properties: dict[str, ThermodynamicProperties] = {
    **glenn_properties,
    "C_s": glenn_properties["C_s"].with_volume(GRAPHITE_VOLUME_GUSTAFSON),
    "C_diamond": diamond_gustafson,
}
"""Default thermodynamic properties of each species, keyed by name"""
