# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Generates the tables of the model catalogue (docs/models.rst) from the model registries

Run by ``conf.py`` at the start of each docs build, so the catalogue always lists the models of
:func:`~atmodeller.eos.get_eos_catalogue` and :func:`~atmodeller.solubility.get_solubility_catalogue`.
The species, phase and sources of each model come from its catalogue information
(:class:`~atmodeller.interfaces.ModelInfo`); a model without it is not listed. The description is
the first line of the docstring of the registered object (or of its class), and the calibrated
range of an EOS comes from its calibrations.
"""

import importlib
import pkgutil
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from sphinx.pycode import ModuleAnalyzer

from atmodeller import solubility
from atmodeller.eos import get_eos_catalogue
from atmodeller.interfaces import CatalogueEntry, ModelInfo
from atmodeller.solubility import get_solubility_catalogue

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


def _source(info: ModelInfo) -> str:
    """The sources of a model, as reST citations"""
    return ", ".join(f":cite:t:`{key}`" for key in info.sources) or "—"


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

    A bounded model combines EOS for pressures below, within and above the calibration: the
    calibrated segments are those that are neither an ideal gas nor an upper bound. The range spans
    all of them, and a bound that none of them sets is open.
    """
    calibrations = getattr(model, "calibrations", None)
    if not calibrations:
        return "not bounded", "not bounded"

    segments = [
        calibration
        for real_gas, calibration in zip(getattr(model, "real_gases", ()), calibrations)
        if type(real_gas).__name__ not in ("IdealGas", "UpperBoundRealGas")
    ] or list(calibrations)

    def span(lows: list[float | None], highs: list[float | None]) -> str:
        low: float | None = None if None in lows else min(lows)  # type: ignore[type-var]
        high: float | None = None if None in highs else max(highs)  # type: ignore[type-var]
        if low is None and high is None:
            return "—"
        return f"{_number(low) if low is not None else ''}–{_number(high) if high is not None else ''}"

    return (
        span([c.temperature_min for c in segments], [c.temperature_max for c in segments]),
        span([c.pressure_min for c in segments], [c.pressure_max for c in segments]),
    )


def _table(title: str, header: list[str], rows: list[list[str]], widths: str) -> str:
    """A reST list-table"""
    lines: list[str] = [
        f".. list-table:: {title}",
        f"   :widths: {widths}",
        "   :header-rows: 1",
        "",
    ]
    for row in [header] + rows:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])

    return "\n".join(lines) + "\n"


def _listed(catalogue: dict[str, CatalogueEntry]) -> list[tuple[str, CatalogueEntry]]:
    """The entries with catalogue information, sorted by key"""
    return sorted(
        ((key, entry) for key, entry in catalogue.items() if entry.info != ModelInfo()),
        key=lambda item: item[0].lower(),
    )


def eos_table() -> str:
    """Table of the real-gas EOS"""
    rows: list[list[str]] = []
    for key, entry in _listed(get_eos_catalogue()):
        temperature, pressure = _calibrated_range(entry.model)
        rows.append([f"``{key}``", entry.info.species, temperature, pressure, _source(entry.info)])

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
    for key, entry in _listed(get_solubility_catalogue()):
        rows.append(
            [
                f"``{key}``",
                entry.info.species,
                entry.info.phase or "—",
                _summary(entry.model, docstrings),
                _source(entry.info),
            ]
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
