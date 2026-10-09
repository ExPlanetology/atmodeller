# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Package-level exports and logger initialization."""

__version__: str = "2.0.0"

import logging
import os

import jax
import jax.numpy as jnp

# from beartype.claw import beartype_this_package
# beartype_this_package()

try:
    from typing import override as _override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override as _override  # Python 3.11 and earlier

override = _override

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=15)  # For better clarity in printed output

# For debugging
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)

# This prevents error_if from throwing an error when encountering nan or inf values. To actually
# find the root cause of nan or inf values, you should set this to "raise" or "breakpoint" as per
# https://docs.kidger.site/equinox/api/errors/
os.environ["EQX_ON_ERROR"] = "off"

# Suppress warnings (notably from Equinox about static JAX arrays)
# if not sys.warnoptions:
#     import warnings

#     warnings.simplefilter("ignore")


# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.debug("Initialized with double precision (float64)")


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
    package_logger.addHandler(console_handler)

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


from atmodeller.classes import EquilibriumModel  # noqa: E402, F401
from atmodeller.containers import (  # noqa: E402, F401
    ChemicalSpecies,
    ReservoirSpecies,
    SolverParameters,
)
from atmodeller.database import (  # noqa: E402, F401
    DataBase,
    GlennDataBase,
    get_default_database,
)
from atmodeller.output import Output  # noqa: E402, F401
from atmodeller.parameters import FixedActivityConstraint, Parameters  # noqa: E402, F401
from atmodeller.phases import CondensedPhase, GasPhase, PurePhase  # noqa: E402, F401
from atmodeller.sci_utils import (  # noqa: E402, F401
    bulk_silicate_earth_abundances,
    earth,
    unit_conversion,
)
from atmodeller.solubility import get_solubility_models  # noqa: E402, F401
from atmodeller.solvers import (  # noqa: E402, F401
    make_solver,
    make_solver_with_jit,  # Alias for the default solver
    make_solver_with_jit_batch_only,
    make_solver_with_jit_single_path,
)
from atmodeller.state import (  # noqa: E402, F401
    Planet,
    PressureScalingLawPlanet,
    ThermodynamicState,
    ThinAtmospherePlanet,
)
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer  # noqa: E402, F401
from atmodeller.thermodata.core import ActivityCoefficient  # noqa: E402, F401
