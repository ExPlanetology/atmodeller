"""Reproducing the results of Tian and Heng (2023).

See the LICENSE file for licensing information.
"""


import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintBallhaus,
    SystemConstraints,
)
from atmodeller.interfaces import (
    GasSpecies,
    SolidSpecies,
    ThermodynamicDatasetABC,
    ThermodynamicDatasetJANAF,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species

thermodynamic_data: ThermodynamicDatasetABC = ThermodynamicDatasetJANAF()

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_graphite() -> None:
    """Tests including graphite.

    Note that the thermodynamic data must be assigned independently to each species.
    """

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(chemical_formula="H2O", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(chemical_formula="CO", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(chemical_formula="CO2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(chemical_formula="CH4", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(chemical_formula="O2", thermodynamic_dataset=thermodynamic_data),
            SolidSpecies(
                chemical_formula="C",
                name_in_thermodynamic_data="graphite",
                thermodynamic_dataset=thermodynamic_data,
            ),  # Ideal activity by default.
        ]
    )

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    # This is comparable to the constraints imposed by Meng.
    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=44.49334998176607),
        ]
    )

    target_pressures: dict[str, float] = {
        "C": 1.0,
        "CH4": 900.3912797397132,
        "CO": 0.07741709702165529,
        "CO2": 0.0685518295157825,
        "H2": 44.493349981766045,
        "H2O": 14.708340036418534,
        "O2": 1.4458158511932372e-25,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)
