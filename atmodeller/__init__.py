# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Package-level exports and logger initialisation."""

__version__: str = "2.0.0"

import logging
import os
from datetime import datetime
from pathlib import Path

import jax

# from beartype.claw import beartype_this_package
# beartype_this_package()

try:
    from typing import override as _override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override as _override  # Python 3.11 and earlier

override = _override

jax.config.update("jax_enable_x64", True)

# For debugging
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)

# This prevents error_if from throwing an error when encountering nan or inf values. To actually
# find the root cause of nan or inf values, set EQX_ON_ERROR to "raise" or "breakpoint" before
# importing atmodeller, as per https://docs.kidger.site/equinox/api/errors/. A value that is already
# set is kept.
os.environ.setdefault("EQX_ON_ERROR", "off")

# Suppress warnings (notably from Equinox about static JAX arrays)
# if not sys.warnoptions:
#     import warnings

#     warnings.simplefilter("ignore")


# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.debug("Initialised with double precision (float64)")


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


_HANDLER_MARKER: str = "_atmodeller_debug_handler"
"""Attribute that marks the handlers added by :func:`debug_logger` and :func:`debug_file_logger`"""


def _add_handler(
    package_logger: logging.Logger,
    handler: logging.Handler,
    formatter: logging.Formatter,
    level: int,
) -> None:
    """Adds a handler to the package logger, marked as added by the debug helpers.

    Args:
        package_logger: The package logger
        handler: The handler to add
        formatter: The formatter of the handler
        level: The level of the handler
    """
    handler.setFormatter(formatter)
    handler.setLevel(level)
    setattr(handler, _HANDLER_MARKER, True)
    package_logger.addHandler(handler)


def _remove_handlers(package_logger: logging.Logger) -> None:
    """Removes and closes the handlers added earlier by the debug helpers.

    Handlers that the user added are kept.

    Args:
        package_logger: The package logger
    """
    for handler in list(package_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            package_logger.removeHandler(handler)
            handler.close()


def debug_logger(level: int = logging.INFO) -> logging.Logger:
    """Sets up logging to the console.

    Calling it again replaces the handlers added by an earlier call to this function or
    :func:`debug_file_logger`, and keeps any handler that you added yourself. Messages also
    propagate to the root logger, so if you have configured root logging as well (e.g. with
    :func:`logging.basicConfig`), they are shown twice.

    Args:
        level: Lowest level of the messages that are shown, e.g. ``logging.DEBUG`` to also show
            the debug messages. Defaults to ``logging.INFO``.

    Returns:
        The package logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(level)
    _remove_handlers(package_logger)
    _add_handler(package_logger, logging.StreamHandler(), simple_formatter(), level)

    return package_logger


def debug_file_logger(log_directory: str | Path = "logs") -> logging.Logger:
    """Sets up info logging to the console and debug logging to a file.

    Each call writes to a new file, ``atmodeller_<date>_<time>.log``, in ``log_directory``, which
    is created if needed, so earlier logs are kept. Calling it again replaces the handlers added
    by an earlier call to this function or :func:`debug_logger`, and keeps any handler that you
    added yourself. Messages also propagate to the root logger, so if you have configured root
    logging as well (e.g. with :func:`logging.basicConfig`), they are shown twice.

    Args:
        log_directory: Directory of the log files. Defaults to ``logs`` in the current working
            directory.

    Returns:
        The package logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    _remove_handlers(package_logger)
    _add_handler(package_logger, logging.StreamHandler(), simple_formatter(), logging.INFO)

    log_path: Path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file: Path = log_path / f"{__package__}_{timestamp}.log"
    _add_handler(package_logger, logging.FileHandler(log_file), complex_formatter(), logging.DEBUG)
    package_logger.info("Writing the debug log to %s", log_file)

    return package_logger


from atmodeller.classes import EquilibriumModel  # noqa: E402, F401
from atmodeller.containers import (  # noqa: E402, F401
    ChemicalSpecies,
    ReservoirSpecies,
    SolverParameters,
)
from atmodeller.database import (  # noqa: E402, F401
    Database,
)
from atmodeller.interfaces import species_name  # noqa: E402, F401
from atmodeller.output import Output  # noqa: E402, F401
from atmodeller.parameters import FixedActivityConstraint, Parameters  # noqa: E402, F401
from atmodeller.phases import CondensedPhase, GasPhase, PurePhase  # noqa: E402, F401
from atmodeller.sci_utils import (  # noqa: E402, F401
    bulk_silicate_earth_abundances,
    earth,
    unit_conversion,
)
from atmodeller.eos import get_eos_models  # noqa: E402, F401
from atmodeller.solubility import get_solubility_models  # noqa: E402, F401
from atmodeller.solver_utils import RESULTS  # noqa: E402, F401
from atmodeller.state import (  # noqa: E402, F401
    Planet,
    PressureScalingLawPlanet,
    ThermodynamicState,
    ThinAtmospherePlanet,
)
from atmodeller.thermodata import (  # noqa: E402, F401
    ActivityCoefficient,
    IronWustiteBuffer,
    RedoxBuffer,
)

__all__ = [
    "ActivityCoefficient",
    "ChemicalSpecies",
    "CondensedPhase",
    "Database",
    "EquilibriumModel",
    "FixedActivityConstraint",
    "GasPhase",
    "IronWustiteBuffer",
    "Output",
    "Parameters",
    "Planet",
    "PressureScalingLawPlanet",
    "PurePhase",
    "RESULTS",
    "RedoxBuffer",
    "ReservoirSpecies",
    "SolverParameters",
    "ThermodynamicState",
    "ThinAtmospherePlanet",
    "__version__",
    "bulk_silicate_earth_abundances",
    "debug_file_logger",
    "debug_logger",
    "earth",
    "get_eos_models",
    "get_solubility_models",
    "species_name",
    "unit_conversion",
]
