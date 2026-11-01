# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that the basic usage example in the README runs."""

import re
from pathlib import Path

import numpy as np
import pytest

README: Path = Path(__file__).parents[1] / "README.md"


def _basic_usage_code() -> str:
    """Code block of the "Basic usage" section of the README"""
    section: str = README.read_text().split("## Basic usage", 1)[1]
    match: re.Match[str] | None = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match is not None, "No python code block in the Basic usage section of the README"

    return match.group(1)


def test_readme_basic_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The README example solves successfully and writes its Excel file"""
    # The example writes an Excel file and a log file to the working directory
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(compile(_basic_usage_code(), str(README), "exec"), namespace)

    assert bool(np.all(namespace["output"].checked_solution.success))
    assert list(tmp_path.glob("example_single*.xlsx"))
