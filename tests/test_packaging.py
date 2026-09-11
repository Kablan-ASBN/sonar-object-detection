"""Checks that what is on disk is what actually ships.

An unanchored `data/` line in .gitignore once matched `src/sonar/data/` as well as the dataset
at the repository root, so six package modules were never committed. Every local run stayed green
because the files were present in the working tree the whole time; only a fresh checkout could
see it. These tests are the cheap version of a fresh checkout.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

SUBPACKAGES = ["sonar", "sonar.data", "sonar.models", "sonar.engine", "sonar.audit", "sonar.utils"]


def tracked_files() -> set[Path] | None:
    """Paths git knows about under src/, or None when this is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "src"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {REPO / line for line in result.stdout.splitlines() if line}


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_subpackage_imports(name: str) -> None:
    """A missing __init__.py makes setuptools skip the subpackage without any error."""
    assert importlib.import_module(name) is not None


def test_every_source_module_is_tracked_by_git() -> None:
    tracked = tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout")

    on_disk = {p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts}
    untracked = sorted(str(p.relative_to(REPO)) for p in on_disk - tracked)

    assert not untracked, (
        f"{len(untracked)} module(s) exist locally but are not in the repository, so a clone "
        f"would get a broken package: {untracked}. Check .gitignore for an unanchored pattern."
    )


def test_gitignore_does_not_shadow_the_package() -> None:
    """`data/` must be anchored; unanchored, it matches src/sonar/data/ at any depth."""
    if tracked_files() is None:
        pytest.skip("not a git checkout")

    probe = "src/sonar/data/voc.py"
    # --no-index is what makes this meaningful: without it git consults the index first and
    # reports an already-tracked file as not ignored, so the rule could come back unnoticed.
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--no-index", "-v", probe],
        capture_output=True,
        text=True,
    )
    # check-ignore exits 0 only when something matched, and prints the offending rule.
    assert result.returncode != 0, f"{probe} is ignored by {result.stdout.strip()}"
