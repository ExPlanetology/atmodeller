<p align="center">
<img src="https://raw.githubusercontent.com/ExPlanetology/atmodeller/main/docs/logo.png" alt="atmodeller logo" width="300"/>
</p>

# Atmodeller

[![Release 1.0.2](https://img.shields.io/badge/Release-1.0.2-blue.svg)](https://github.com/ExPlanetology/atmodeller/releases/tag/v1.0.2)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/ExPlanetology/atmodeller/actions/workflows/ci.yml/badge.svg)](https://github.com/ExPlanetology/atmodeller/actions/workflows/ci.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Test coverage](https://img.shields.io/badge/Coverage-93%25-brightgreen)](https://github.com/ExPlanetology/atmodeller)

## About
Atmodeller is a Python package that uses [JAX](https://jax.readthedocs.io/en/latest/index.html) to compute the partitioning of volatiles between a planetary atmosphere and its rocky interior. It is released under [The GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Documentation

The documentation is available online, with options to download it in EPUB or PDF format:

[https://atmodeller.readthedocs.io/en/latest/](https://atmodeller.readthedocs.io/en/latest/)

## Quick install

Atmodeller is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux). It is recommended to install Atmodeller in a dedicated Python environment. Before installation, create and activate the environment, then run:

```pip install atmodeller```

Downloading the source code is also recommended if you'd like access to the example notebooks in `notebooks/`.

## Citation

If you use Atmodeller, or data from Atmodeller, please cite:

- Bower, D. J., Thompson, M. A., Hakim, K., Tian, M., and Sossi, P. A. (2025), Diversity of low-mass planet atmospheres in the C&ndash;H&ndash;O&ndash;N&ndash;S&ndash;Cl system with interior dissolution, nonideality and condensation: Application to TRAPPIST-1e and sub-Neptunes, The Astrophysical Journal, 995(1), 59, doi: <https://www.doi.org/10.3847/1538-4357/ae1479>. ArXiv e-print [2507.00499](https://arxiv.org/abs/2507.00499).

The data from the above study are also available for download at <https://doi.org/10.17605/OSF.IO/PC5TD>.

## Basic usage

Several Jupyter notebooks providing examples are in the `notebooks/` directory. A simple example of how to use Atmodeller is provided below:

```
from atmodeller import (
    EquilibriumModel,
    Planet,
    ChemicalSpecies,
    SpeciesNetwork,
    earth_oceans_to_hydrogen_mass,
)
from atmodeller.solubility import get_solubility_models

solubility_models = get_solubility_models()
# Get the available solubility models
print("solubility models = ", solubility_models.keys())

H2_g = ChemicalSpecies.create_gas("H2")
H2O_g = ChemicalSpecies.create_gas("H2O", solubility=solubility_models["H2O_peridotite_sossi23"])
O2_g = ChemicalSpecies.create_gas("O2")

species = SpeciesNetwork((H2_g, H2O_g, O2_g))
planet = Planet()
interior_atmosphere = EquilibriumModel(species)

oceans = 1
h_kg = earth_oceans_to_hydrogen_mass(oceans)
o_kg = 6.25774e20
mass_constraints = {"H": h_kg, "O": o_kg}

# If you do not specify an initial solution guess then a default will be used
# Initial solution guess number moles
initial_log_number_moles = 50

interior_atmosphere.solve(
    planet=planet,
    initial_log_number_moles=initial_log_number_moles,
    mass_constraints=mass_constraints,
)
output = interior_atmosphere.output

# Quick look at the solution
solution = output.quick_look()

# Get complete solution as a dictionary
solution_asdict = output.asdict()
print("solution_asdict =", solution_asdict)

# Write the complete solution to Excel
output.to_excel("example_single")
```

## Funding
Atmodeller was created as part of a SERI-funded ERC Starting grant '2ATMO' granted to P. Sossi (Contract no. MB22.00033), with additional funding provided through a Swiss National Science Foundation (SNSF) Eccellenza Professorship (#203668).

K\. Hakim acknowledges the FED-tWIN research program STELLA (Prf-2021-022) funded by the Belgian Science Policy Office (BELSPO) and the research grant G014425N funded by the Research Foundation Flanders (FWO).