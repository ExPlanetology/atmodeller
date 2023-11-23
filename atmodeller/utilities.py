"""Utilities.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict, abc
from typing import Any, Callable, Type, TypeVar

from molmass import Formula
from scipy.constants import kilo, mega

from atmodeller import OCEAN_MOLES

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def debug_decorator(logger: logging.Logger) -> Callable:
    """A decorator to print the result of a function to a debug logger."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # logger.info(f"Executing {func.__name__}")
            result: Any = func(*args, **kwargs)
            logger.debug("%s = %s", func.__name__, result)
            # logger.info(f"Finished executing {func.__name__}")
            return result

        return wrapper

    return decorator


def filter_by_type(some_collection: abc.Collection, class_type: Type[T]) -> dict[int, T]:
    """Filters entries by the given class type and maintains order.

    Args:
        some_collection: A collection (e.g. a list) to filter.
        class_type: Class type to filter.

    Returns:
        An OrderedDict with indices in some_collection as keys and filtered entries as values.
    """

    ordered_dict = OrderedDict()
    for index, entry in enumerate(some_collection):
        if isinstance(entry, class_type):
            ordered_dict[index] = entry

    return ordered_dict


def earth_oceans_to_kg(number_of_earth_oceans: float = 1) -> float:
    h_grams: float = number_of_earth_oceans * OCEAN_MOLES * Formula("H2").mass
    h_kg: float = UnitConversion().g_to_kg(h_grams)
    return h_kg


class UnitConversion:
    """Unit conversions."""

    @staticmethod
    def bar_to_Pa(value_bar: float = 1) -> float:
        """bar to Pa."""
        return value_bar * 1e5

    @classmethod
    def bar_to_GPa(cls, value_bar: float = 1) -> float:
        """Bar to GPa."""
        return cls.bar_to_Pa(value_bar) * 1.0e-9

    @staticmethod
    def fraction_to_ppm(value_fraction: float = 1) -> float:
        """Mole or mass fraction to parts-per-million by mole or mass, respectively."""
        return value_fraction * mega

    @staticmethod
    def g_to_kg(value_grams: float = 1) -> float:
        """Grams to kilograms."""
        return value_grams / kilo

    @classmethod
    def ppm_to_fraction(cls, value_ppm: float = 1) -> float:
        """Parts-per-million by mole or mass to mole or mass fraction, respectively."""
        return value_ppm / cls.fraction_to_ppm()

    @classmethod
    def J_per_bar_to_cm3(cls, J_per_bar: float = 1) -> float:
        """J/bar (or kJ/kbar) to cm^3."""
        return J_per_bar * 10

    @classmethod
    def cm3_to_J_per_bar(cls, cm_cubed: float = 1) -> float:
        """cm^3 to J/bar"""
        return cm_cubed / cls.J_per_bar_to_cm3()

    @staticmethod
    def weight_percent_to_ppmw(value_weight_percent: float = 1) -> float:
        """Weight percent to parts-per-million by weight"""
        return value_weight_percent * 1.0e4
