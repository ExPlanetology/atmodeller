# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Solubility library

.. code-block::
   :caption: Usage

        from atmodeller.solubility.library import get_solubility_models

        sol_models = get_solubility_models()
        H2O_peridotite = sol_models["H2O_peridotite_sossi23"]
        # Evaluate solubility (concentration) at 2 bar fH2O and 2000 K
        concentration = H2O_peridotite.concentration(2, temperature=2000)
        print(concentration)
"""

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
    H2_chachan18,
    H2_kite19,
    H2_silicic_melts_gaillard03,
    H2O_ano_dio_newcombe17,
    H2O_basalt_dixon95,
    H2O_basalt_dixon95_table5,
    H2O_basalt_mitchell17,
    H2O_basalt_moore98,
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
from atmodeller.interfaces import CatalogueEntry, ModelInfo
from atmodeller.solubility.core import NoSolubility, Solubility


def get_solubility_catalogue() -> dict[str, CatalogueEntry[Solubility]]:
    """Gets the solubility registry: each solubility law with its catalogue information

    The catalogue information (species, melt and sources) lists the laws in the documentation.
    ``NO_SOLUBILITY`` has none, so it is not listed.

    Returns:
        Dictionary of catalogue entries, keyed as :func:`get_solubility_models`
    """
    models: dict[str, CatalogueEntry[Solubility]] = {}

    # Carbon species
    models["CH4_basalt_ardia13"] = CatalogueEntry(
        CH4_basalt_ardia13, ModelInfo("CH4", "basalt", ("AHW13",))
    )
    models["CO2_basalt_dixon95"] = CatalogueEntry(
        CO2_basalt_dixon95, ModelInfo("CO2", "basalt", ("DSH95",))
    )
    models["CO_basalt_armstrong15"] = CatalogueEntry(
        CO_basalt_armstrong15, ModelInfo("CO", "basalt", ("AHS15", "SHW14"))
    )
    models["CO_basalt_yoshioka19"] = CatalogueEntry(
        CO_basalt_yoshioka19, ModelInfo("CO", "basalt", ("YNN19",))
    )
    models["CO_rhyolite_yoshioka19"] = CatalogueEntry(
        CO_rhyolite_yoshioka19, ModelInfo("CO", "rhyolite", ("YNN19",))
    )

    # Hydrogen species
    models["H2_andesite_hirschmann12"] = CatalogueEntry(
        H2_andesite_hirschmann12, ModelInfo("H2", "andesite", ("HWA12",))
    )
    models["H2_basalt_hirschmann12"] = CatalogueEntry(
        H2_basalt_hirschmann12, ModelInfo("H2", "basalt", ("HWA12",))
    )
    models["H2_silicic_melts_gaillard03"] = CatalogueEntry(
        H2_silicic_melts_gaillard03, ModelInfo("H2", "silicic melts", ("GSM03",))
    )
    models["H2_chachan18"] = CatalogueEntry(H2_chachan18, ModelInfo("H2", "", ("CS18",)))
    models["H2_kite19"] = CatalogueEntry(H2_kite19, ModelInfo("H2", "", ("KFS19",)))
    models["H2O_ano_dio_newcombe17"] = CatalogueEntry(
        H2O_ano_dio_newcombe17, ModelInfo("H2O", "anorthite–diopside", ("NBB17",))
    )
    models["H2O_basalt_dixon95"] = CatalogueEntry(
        H2O_basalt_dixon95, ModelInfo("H2O", "basalt", ("DSH95",))
    )
    models["H2O_basalt_dixon95_table5"] = CatalogueEntry(
        H2O_basalt_dixon95_table5, ModelInfo("H2O", "basalt", ("DSH95", "SS89"))
    )
    models["H2O_basalt_mitchell17"] = CatalogueEntry(
        H2O_basalt_mitchell17, ModelInfo("H2O", "basalt", ("MGO17",))
    )
    models["H2O_basalt_moore98"] = CatalogueEntry(
        H2O_basalt_moore98, ModelInfo("H2O", "basalt", ("MVC98",))
    )
    models["H2O_lunar_glass_newcombe17"] = CatalogueEntry(
        H2O_lunar_glass_newcombe17, ModelInfo("H2O", "lunar glass", ("NBB17",))
    )
    models["H2O_peridotite_sossi23"] = CatalogueEntry(
        H2O_peridotite_sossi23, ModelInfo("H2O", "peridotite", ("STB23",))
    )

    # Sulfur species
    models["S2_andesite_boulliung23"] = CatalogueEntry(
        S2_andesite_boulliung23, ModelInfo("S2", "andesite", ("BW22", "BW23corr", "BW23"))
    )
    models["S2_basalt_boulliung23"] = CatalogueEntry(
        S2_basalt_boulliung23, ModelInfo("S2", "basalt", ("BW22", "BW23corr", "BW23"))
    )
    models["S2_sulfate_andesite_boulliung23"] = CatalogueEntry(
        S2_sulfate_andesite_boulliung23, ModelInfo("S2", "andesite", ("BW22", "BW23corr"))
    )
    models["S2_sulfate_basalt_boulliung23"] = CatalogueEntry(
        S2_sulfate_basalt_boulliung23, ModelInfo("S2", "basalt", ("BW22", "BW23corr"))
    )
    models["S2_sulfate_trachybasalt_boulliung23"] = CatalogueEntry(
        S2_sulfate_trachybasalt_boulliung23, ModelInfo("S2", "trachybasalt", ("BW22", "BW23corr"))
    )
    models["S2_sulfide_andesite_boulliung23"] = CatalogueEntry(
        S2_sulfide_andesite_boulliung23, ModelInfo("S2", "andesite", ("BW23", "BW22"))
    )
    models["S2_sulfide_basalt_boulliung23"] = CatalogueEntry(
        S2_sulfide_basalt_boulliung23, ModelInfo("S2", "basalt", ("BW23", "BW22"))
    )
    models["S2_sulfide_trachybasalt_boulliung23"] = CatalogueEntry(
        S2_sulfide_trachybasalt_boulliung23, ModelInfo("S2", "trachybasalt", ("BW23", "BW22"))
    )
    models["S2_trachybasalt_boulliung23"] = CatalogueEntry(
        S2_trachybasalt_boulliung23, ModelInfo("S2", "trachybasalt", ("BW22", "BW23corr", "BW23"))
    )

    # Other species
    models["Cl2_ano_dio_for_thomas21"] = CatalogueEntry(
        Cl2_ano_dio_for_thomas21, ModelInfo("Cl2", "anorthite–diopside–forsterite", ("TW21",))
    )
    models["Cl2_basalt_thomas21"] = CatalogueEntry(
        Cl2_basalt_thomas21, ModelInfo("Cl2", "basalt", ("TW21",))
    )
    models["He_basalt_jambon86"] = CatalogueEntry(
        He_basalt_jambon86, ModelInfo("He", "basalt", ("JWB86",))
    )
    models["N2_basalt_bernadou21"] = CatalogueEntry(
        N2_basalt_bernadou21, ModelInfo("N2", "basalt", ("BGF21",))
    )
    models["N2_basalt_dasgupta22"] = CatalogueEntry(
        N2_basalt_dasgupta22, ModelInfo("N2", "basalt", ("DFP22",))
    )
    models["N2_basalt_libourel03"] = CatalogueEntry(
        N2_basalt_libourel03, ModelInfo("N2", "basalt", ("LMH03",))
    )
    models["Ne_basalt_jambon86"] = CatalogueEntry(
        Ne_basalt_jambon86, ModelInfo("Ne", "basalt", ("JWB86",))
    )
    models["Ar_basalt_jambon86"] = CatalogueEntry(
        Ar_basalt_jambon86, ModelInfo("Ar", "basalt", ("JWB86",))
    )
    models["Xe_basalt_jambon86"] = CatalogueEntry(
        Xe_basalt_jambon86, ModelInfo("Xe", "basalt", ("JWB86",))
    )
    models["Kr_basalt_jambon86"] = CatalogueEntry(
        Kr_basalt_jambon86, ModelInfo("Kr", "basalt", ("JWB86",))
    )

    # Sort the dictionary by keys
    sorted_models = {k: models[k] for k in sorted(models)}

    # For completeness add the no solubility model at the end of the dictionary
    sorted_models["NO_SOLUBILITY"] = CatalogueEntry(NoSolubility())

    return sorted_models


def get_solubility_models() -> dict[str, Solubility]:
    """Gets a dictionary of solubility models

    Returns:
        Dictionary of solubility models
    """
    return {key: entry.model for key, entry in get_solubility_catalogue().items()}
