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
"""Package level variables and initialises the package logger"""

from __future__ import annotations

__version__: str = "0.4.0"

import logging

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=15)  # For better clarity in printed output
print("Atmodeller initialized with double precision (float64)")

# For debugging
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", False)
# jax.config.update("jax_disable_jit", True)

# Thermodynamic standard state
ENTHALPY_REFERENCE: float = 298.15
"""Enthalpy reference temperature in K"""
PRESSURE_REFERENCE: float = 1.0
"""Standard state pressure in bar"""

# Initial solution guess
INITIAL_LOG_NUMBER_DENSITY: float = 50.0
"""Initial log number density

Empiricially determined. This value is mid-range for Earth-like planets.
"""
INITIAL_LOG_STABILITY: float = -140.0
"""Initial log stability

Empirically determined. Preliminary testing seems to reveal that starting from a large negative 
value improves the performance of the solver. This asserts that all species are stable.
"""

# Maximum x for which exp(x) is finite in 64-bit precision (to prevent overflow)
max_exp_input = np.log(np.finfo(np.float64).max)
# Minimum x for which exp(x) is non-zero in 64-bit precision
min_exp_input = np.log(jnp.finfo(np.float64).tiny)

# Lower and upper bounds on the hypercube which contains the root
LOG_NUMBER_DENSITY_LOWER: float = -170.0
"""Lower log number density for a species

At 3000 K this corresponds to 3.17E-77 bar and at 298 K this corresponds to 3.16E-78 bar.
"""
LOG_NUMBER_DENSITY_UPPER: float = 80.0
"""Upper log number density for a species

TODO: The summary values below are for a value of 70.0, but it has been increased to 80 since this
cut-off also affects condensates, which can extend to large values. The hypercube specification
can take a pytree structurs, so a more sophisticated approach can be implemented where gas species
and condensate are treated differently.

At 3000 K this corresponds to 1041881 bar (104 GPa) and at 298 K this corresponds to 103494 bar
(10.3 GPa).
"""
LOG_STABILITY_LOWER: float = -700.0  # basically the same as min_exp_input
"""Lower stability for a species

Derived to ensure that the exponential function exp(x) does not underflow to zero
"""
LOG_STABILITY_UPPER: float = 35.0
"""Upper stability for a species

Empirically determined.
"""
TAU: float = 1.0e-25
"""Tau scaling factor for species stability :cite:p:`LKK16`.

Tau effectively controls the minimum non-zero number density of unstable species. Formally, it
defines the number density of an unstable pure condensate with an activity of 1/e, which
corresponds to a log stability of zero.

This value is typically appropriate for condensate stability only, but if you additionally apply 
stability criteria to gas species you should reduce this value, maybe as low as 1e-60 to 1e-72 if
you want to ensure you do not truncated O2 at low temperatures. Hence you can override this default
using an argument to :class:`atmodeller.classes.InteriorAtmosphere`.
"""

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter"""
    fmt: str = "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()]"
    fmt += " - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter for logging

    Returns:
        Formatter for logging
    """
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Sets up debug logging to the console.

    Returns:
        A logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return package_logger


def debug_file_logger() -> logging.Logger:
    """Sets up info logging to the console and debug logging to a file.

    Returns:
        A logger
    """
    # Console logger
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    package_logger.addHandler(console_handler)
    # File logger
    file_handler: logging.Handler = logging.FileHandler(f"{__package__}.log")
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(file_handler)

    return package_logger


from atmodeller.classes import InteriorAtmosphere  # noqa: E402, F401
from atmodeller.containers import (  # noqa: E402, F401
    Planet,
    SolverParameters,
    Species,
    SpeciesCollection,
)
from atmodeller.utilities import (  # noqa: E402, F401
    bulk_silicate_earth_abundances,
    earth_oceans_to_hydrogen_mass,
)
