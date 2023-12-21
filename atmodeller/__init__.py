"""Package level variables and initialises the package logger.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

__version__: str = "0.1.0"

import importlib.resources
import logging

from scipy import constants

# Module constants
GAS_CONSTANT: float = constants.gas_constant  # J/K/mol
GAS_CONSTANT_BAR: float = GAS_CONSTANT * 1.0e-5  # m^3 bar/K/mol
GRAVITATIONAL_CONSTANT: float = constants.gravitational_constant  # m^3/kg/s^2
ATMOSPHERE: float = constants.atmosphere / constants.bar  # bar

# Used to determine the JANAF reference state
NOBLE_GASES: list[str] = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]

OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.

DATA_ROOT_PATH = importlib.resources.files("%s.data" % __package__)

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter."""
    fmt: str = "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()] - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter."""
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
    return formatter


def debug_logger() -> logging.Logger:
    """Setup the logging for debugging: DEBUG to the console."""
    # Console logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def debug_file_logger() -> logging.Logger:
    """Setup the logging to a file (DEBUG) and to the console (INFO)."""
    # Console logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    # File logger
    file_handler: logging.Handler = logging.FileHandler("%s.log" % __package__)
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger
