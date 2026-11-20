# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Real gas EOS library

.. code-block::
   :caption: Usage

        from atmodeller.eos.library import get_eos_models

        eos_models = get_eos_models()
        CH4_beattie = eos_models["CH4_beattie_holley58"]
        # Evaluate fugacity at 10 bar and 800 K
        fugacity = CH4_beattie.fugacity(800, 10)
        print(fugacity)
"""

from atmodeller.eos._chabrier import get_chabrier_eos_catalogue
from atmodeller.eos._holland_powell import get_holland_eos_catalogue
from atmodeller.eos._holley import get_holley_eos_catalogue
from atmodeller.eos._reid_connolly import get_reid_connolly_eos_catalogue
from atmodeller.eos._saxena import get_saxena_eos_catalogue
from atmodeller.eos._vanderwaals import get_vanderwaals_eos_catalogue
from atmodeller.eos._wang import get_wang_eos_catalogue
from atmodeller.eos._zhang_duan import get_zhang_eos_catalogue
from atmodeller.eos.core import RealGasBase
from atmodeller.interfaces import CatalogueEntry


def get_eos_catalogue() -> dict[str, CatalogueEntry[RealGasBase]]:
    """Gets the EOS registry: each EOS model with its catalogue information

    The catalogue information (species and sources) lists the models in the documentation.

    Returns:
        Dictionary of catalogue entries, keyed as :func:`get_eos_models`
    """
    catalogue: dict[str, CatalogueEntry[RealGasBase]] = {}
    for part in (
        get_chabrier_eos_catalogue,
        get_holley_eos_catalogue,
        get_holland_eos_catalogue,
        get_reid_connolly_eos_catalogue,
        get_saxena_eos_catalogue,
        get_vanderwaals_eos_catalogue,
        get_wang_eos_catalogue,
        get_zhang_eos_catalogue,
    ):
        catalogue |= part()

    return catalogue


def get_eos_models() -> dict[str, RealGasBase]:
    """Gets a dictionary of EOS models

    Returns:
        Dictionary of EOS models
    """
    return {key: entry.model for key, entry in get_eos_catalogue().items()}
