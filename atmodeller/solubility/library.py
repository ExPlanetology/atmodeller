#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Solubility library

Usage:
    from atmodeller.solubility.library import get_solubility_models
    sol_models = get_solubility_models()
    H2O_peridotite = sol_models["H2O_peridotite_sossi23"]
    # Evaluate solubility (concentration) at 2 bar fH2O and 2000 K
    concentration = H2O_peridotite.concentration(2, temperature=2000)
    print(concentration)
"""

from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility._carbon_species import (
    CH4_basalt_ardia13,
    CO2_basalt_dixon95,
    CO_basalt_armstrong15,
    CO_basalt_yoshioka19,
    CO_rhyolite_yoshioka19,
)
from atmodeller.solubility._hydrogen_species import (
    H2_andesite_hirschmann12,
    H2_basalt_hirschmann12,
    H2_silicic_melts_gaillard03,
    H2O_ano_dio_newcombe17,
    H2O_basalt_dixon95,
    H2O_basalt_mitchell17,
    H2O_lunar_glass_newcombe17,
    H2O_peridotite_sossi23,
)
from atmodeller.solubility._other_species import (
    Ar_basalt_jambon86,
    Cl2_ano_dio_for_thomas21,
    Cl2_basalt_thomas21,
    He_basalt_jambon86,
    Kr_basalt_jambon86,
    N2_basalt_bernadou21,
    N2_basalt_dasgupta22,
    N2_basalt_libourel03,
    Ne_basalt_jambon86,
    Xe_basalt_jambon86,
)
from atmodeller.solubility._sulfur_species import (
    S2_andesite_boulliung23,
    S2_basalt_boulliung23,
    S2_sulfate_andesite_boulliung23,
    S2_sulfate_basalt_boulliung23,
    S2_sulfate_trachybasalt_boulliung23,
    S2_sulfide_andesite_boulliung23,
    S2_sulfide_basalt_boulliung23,
    S2_sulfide_trachybasalt_boulliung23,
    S2_trachybasalt_boulliung23,
)
from atmodeller.solubility.classes import NoSolubility


def get_solubility_models() -> dict[str, SolubilityProtocol]:
    """Gets a dictionary of solubility models

    Returns:
        Dictionary of solubility models
    """
    models: dict[str, SolubilityProtocol] = {}

    # Carbon species
    models["CH4_basalt_ardia13"] = CH4_basalt_ardia13
    models["CO2_basalt_dixon95"] = CO2_basalt_dixon95
    models["CO_basalt_armstrong15"] = CO_basalt_armstrong15
    models["CO_basalt_yoshioka19"] = CO_basalt_yoshioka19
    models["CO_rhyolite_yoshioka19"] = CO_rhyolite_yoshioka19

    # Hydrogen species
    models["H2_andesite_hirschmann12"] = H2_andesite_hirschmann12
    models["H2_basalt_hirschmann12"] = H2_basalt_hirschmann12
    models["H2_silicic_melts_gaillard03"] = H2_silicic_melts_gaillard03
    models["H2O_ano_dio_newcombe17"] = H2O_ano_dio_newcombe17
    models["H2O_basalt_dixon95"] = H2O_basalt_dixon95
    models["H2O_basalt_mitchell17"] = H2O_basalt_mitchell17
    models["H2O_lunar_glass_newcombe17"] = H2O_lunar_glass_newcombe17
    models["H2O_peridotite_sossi23"] = H2O_peridotite_sossi23

    # Sulfur species
    models["S2_andesite_boulliung23"] = S2_andesite_boulliung23
    models["S2_basalt_boulliung23"] = S2_basalt_boulliung23
    models["S2_sulfate_andesite_boulliung23"] = S2_sulfate_andesite_boulliung23
    models["S2_sulfate_basalt_boulliung23"] = S2_sulfate_basalt_boulliung23
    models["S2_sulfate_trachybasalt_boulliung23"] = S2_sulfate_trachybasalt_boulliung23
    models["S2_sulfide_andesite_boulliung23"] = S2_sulfide_andesite_boulliung23
    models["S2_sulfide_basalt_boulliung23"] = S2_sulfide_basalt_boulliung23
    models["S2_sulfide_trachybasalt_boulliung23"] = S2_sulfide_trachybasalt_boulliung23
    models["S2_trachybasalt_boulliung23"] = S2_trachybasalt_boulliung23

    # Other species
    models["Cl2_ano_dio_for_thomas21"] = Cl2_ano_dio_for_thomas21
    models["Cl2_basalt_thomas21"] = Cl2_basalt_thomas21
    models["He_basalt_jambon86"] = He_basalt_jambon86
    models["N2_basalt_bernadou21"] = N2_basalt_bernadou21
    models["N2_basalt_dasgupta22"] = N2_basalt_dasgupta22
    models["N2_basalt_libourel03"] = N2_basalt_libourel03
    models["Ne_basalt_jambon86"] = Ne_basalt_jambon86
    models["Ar_basalt_jambon86"] = Ar_basalt_jambon86
    models["Xe_basalt_jambon86"] = Xe_basalt_jambon86
    models["Kr_basalt_jambon86"] = Kr_basalt_jambon86

    # Sort the dictionary by keys
    sorted_models = {k: models[k] for k in sorted(models)}

    # For completeness add the no solubility model at the end of the dictionary
    sorted_models["NO_SOLUBILITY"] = NoSolubility()

    return sorted_models
