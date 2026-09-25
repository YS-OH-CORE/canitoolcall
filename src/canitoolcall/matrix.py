"""Static matrix site generator (``canitoolcall matrix``).

Reads ``results/*.json`` (spec/results.schema.json) and renders a static site
of family x engine x version, with per-cell drill-down to failing fixtures,
the check that failed, the strategy, and the observed vs expected parse.

The site is built ONLY from real run files; there is no other data source.
Templates live in ``site/templates/`` (Jinja2); output goes to ``--out``
(default ``site/_build``) and is published to GitHub Pages by CI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from canitoolcall.results import RunResults, Status


@dataclass(frozen=True)
class Cell:
    """One (family, engine@version) cell."""

    family: str
    engine: str
    engine_version: str
    counts: dict[str, int]
    status: Status
    """Headline status: worst over supported cases, or UNSUPPORTED if none."""
    run_at: str


@dataclass(frozen=True)
class Matrix:
    families: tuple[str, ...]
    engines: tuple[tuple[str, str], ...]
    """(engine, version) columns, sorted."""
    cells: tuple[Cell, ...]

    def cell(self, family: str, engine: str, version: str) -> Cell | None:
        raise NotImplementedError


def load_results(results_dir: Path) -> list[RunResults]:
    """Load and schema-check every ``*.json`` in ``results_dir``."""
    raise NotImplementedError


def build_matrix(runs: Sequence[RunResults]) -> Matrix:
    """Aggregate runs into a Matrix (latest run wins per engine@version)."""
    raise NotImplementedError


def render_site(results_dir: Path, out_dir: Path, templates_dir: Path | None = None) -> Path:
    """Render the static site; returns the path of ``index.html``."""
    raise NotImplementedError
