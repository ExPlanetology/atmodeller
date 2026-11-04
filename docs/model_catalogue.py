# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Generates the tables of the model catalogue (docs/models.rst) from the model registries

Run by ``conf.py`` at the start of each docs build, so the catalogue always lists the models of
:func:`~atmodeller.eos.get_eos_models` and :func:`~atmodeller.solubility.get_solubility_models`.
The source of each model is read from the ``:cite:`` keys in the docstring of the module-level
object that is registered, or else of its class.
"""

import importlib
import pkgutil
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from sphinx.pycode import ModuleAnalyzer

from atmodeller import eos, solubility
from atmodeller.eos import get_eos_models
from atmodeller.solubility import get_solubility_models

CITE: re.Pattern = re.compile(r":cite:[a-z]*:`([^`]+)`")


def _attribute_docstrings(package: ModuleType) -> dict[int, str]:
    """Docstrings of the module-level objects of a package, keyed by the id of the object"""
    docstrings: dict[int, str] = {}
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module: ModuleType = importlib.import_module(info.name)
        try:
            attribute_docs = ModuleAnalyzer.for_module(info.name).find_attr_docs()
        except Exception:  # noqa: BLE001
            continue
        for (scope, name), lines in attribute_docs.items():
            if scope == "" and hasattr(module, name):
                docstrings.setdefault(id(getattr(module, name)), "\n".join(lines))

    return docstrings


def _citation_keys(obj: Any, docstrings: dict[int, str]) -> list[str]:
    """The citation keys of a model, from its docstring or, for a bounded model without one, from
    the models that it combines"""
    text: str = docstrings.get(id(obj)) or (type(obj).__doc__ or "")
    keys: list[str] = []
    for match in CITE.findall(text):
        # Drop the parts in braces first, which can contain commas
        for key in re.sub(r"\{.*?\}", "", match).split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    if not keys:
        for real_gas in getattr(obj, "real_gases", ()):
            keys.extend(k for k in _citation_keys(real_gas, docstrings) if k not in keys)

    return keys


def _source(obj: Any, docstrings: dict[int, str]) -> str:
    """The citations of a model, as reST"""
    return ", ".join(f":cite:t:`{key}`" for key in _citation_keys(obj, docstrings)) or "—"


def _summary(obj: Any, docstrings: dict[int, str]) -> str:
    """The first line of the docstring of a model, without citations"""
    text: str = docstrings.get(id(obj)) or (type(obj).__doc__ or "")
    first: str = text.strip().splitlines()[0] if text.strip() else ""

    return CITE.sub("", first).strip().rstrip(".") or "—"


def _number(value: float) -> str:
    """A compact number for a table"""
    return f"{value:g}"


def _calibrated_range(model: Any) -> tuple[str, str]:
    """Calibrated temperature (K) and pressure (bar) range of an EOS

    A bounded model combines EOS for pressures below, within and above the calibration. The
    calibrated part is the one that is neither an ideal gas nor an upper bound.
    """
    calibrations = getattr(model, "calibrations", None)
    if not calibrations:
        return "not bounded", "not bounded"

    calibration = calibrations[0]
    for real_gas, candidate in zip(getattr(model, "real_gases", ()), calibrations):
        if type(real_gas).__name__ not in ("IdealGas", "UpperBoundRealGas"):
            calibration = candidate
            break

    def span(low: float | None, high: float | None) -> str:
        if low is None and high is None:
            return "—"
        return f"{_number(low) if low is not None else ''}–{_number(high) if high is not None else ''}"

    return (
        span(calibration.temperature_min, calibration.temperature_max),
        span(calibration.pressure_min, calibration.pressure_max),
    )


def _table(title: str, header: list[str], rows: list[list[str]], widths: str) -> str:
    """A reST list-table"""
    lines: list[str] = [f".. list-table:: {title}", f"   :widths: {widths}", "   :header-rows: 1", ""]
    for row in [header] + rows:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])

    return "\n".join(lines) + "\n"


def eos_table() -> str:
    """Table of the real-gas EOS"""
    docstrings: dict[int, str] = _attribute_docstrings(eos)
    rows: list[list[str]] = []
    for key, model in sorted(get_eos_models().items(), key=lambda item: item[0].lower()):
        temperature, pressure = _calibrated_range(model)
        rows.append([f"``{key}``", key.split("_")[0], temperature, pressure, _source(model, docstrings)])

    return _table(
        "Real-gas equations of state (``get_eos_models()``)",
        ["Key", "Species", "Calibrated T (K)", "Calibrated P (bar)", "Source"],
        rows,
        "30 10 15 20 25",
    )


def solubility_table() -> str:
    """Table of the solubility laws"""
    docstrings: dict[int, str] = _attribute_docstrings(solubility)
    rows: list[list[str]] = []
    for key, law in sorted(get_solubility_models().items(), key=lambda item: item[0].lower()):
        if key == "NO_SOLUBILITY":
            continue
        parts: list[str] = key.split("_")
        melt: str = " ".join(parts[1:-1]) or "—"
        rows.append(
            [f"``{key}``", parts[0], melt, _summary(law, docstrings), _source(law, docstrings)]
        )

    return _table(
        "Solubility laws (``get_solubility_models()``)",
        ["Key", "Species", "Melt", "Description", "Source"],
        rows,
        "30 8 14 28 20",
    )


def generate(directory: Path) -> None:
    """Writes the tables to ``directory``"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "models_eos.rst").write_text(eos_table())
    (directory / "models_solubility.rst").write_text(solubility_table())


if __name__ == "__main__":
    generate(Path(__file__).parent / "_generated")
