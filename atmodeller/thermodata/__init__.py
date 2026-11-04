# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Thermodata package level variables"""

# Expose public API
from atmodeller.thermodata._redox_buffers import (  # noqa: E402, F401
    IronWustiteBuffer,
    IronWustiteBufferHirschmann,
    IronWustiteBufferHirschmann08,
    IronWustiteBufferHirschmann21,
    RedoxBuffer,
)
from atmodeller.thermodata.activity_models import Si_iron_badro15  # noqa: E402, F401
from atmodeller.thermodata.core import (  # noqa: E402, F401
    ActivityCoefficient,
    CriticalData,
    GibbsThermodynamicProperties,
    IntegratedThermodynamicProperties,
    RelativeThermodynamicProperties,
    ThermodynamicProperties,
    critical_data_dictionary,
)
from atmodeller.thermodata.holland_powell import MurnaghanEOS, TaitEOS  # noqa: E402, F401
from atmodeller.thermodata.glenn import (  # noqa: E402, F401
    NasaGlennThermodynamicProperties,
    glenn_properties,
)

__all__ = [
    "ActivityCoefficient",
    "CriticalData",
    "GibbsThermodynamicProperties",
    "IntegratedThermodynamicProperties",
    "IronWustiteBuffer",
    "IronWustiteBufferHirschmann",
    "IronWustiteBufferHirschmann08",
    "IronWustiteBufferHirschmann21",
    "MurnaghanEOS",
    "NasaGlennThermodynamicProperties",
    "RedoxBuffer",
    "RelativeThermodynamicProperties",
    "Si_iron_badro15",
    "TaitEOS",
    "ThermodynamicProperties",
    "critical_data_dictionary",
    "glenn_properties",
]
