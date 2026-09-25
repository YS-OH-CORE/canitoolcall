"""Offline suite runner: drives an adapter worker and applies the checks.

Flow for ``canitoolcall run --engine E``:

1. load + validate fixtures and families (main env)
2. start ``<engine-python> -m canitoolcall.adapters.worker E`` with
   ``PYTHONPATH=<repo>/src`` (:class:`WorkerClient`), send ``hello``
3. for each fixture send ``replay`` with the strategy ids, receive the
   observations, run :func:`canitoolcall.checks.run_checks`
4. assemble :class:`~canitoolcall.results.RunResults` and write
   ``results/<engine>-<version>.json``

The runner never imports an engine. Harness failures (worker ``ok: false``,
crash, timeout) become ``error`` cases with ``harness_error``; the worker is
restarted after a crash and the run continues.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from canitoolcall.chunking import DEFAULT_STRATEGIES, ChunkStrategy
from canitoolcall.fixtures import Family, Fixture
from canitoolcall.results import CaseResult, RunResults


@dataclass(frozen=True)
class RunConfig:
    engine: str
    fixtures: Sequence[Path] = ()
    """Files/dirs to load; empty -> the default fixtures root."""
    families: Sequence[str] = ()
    """Restrict to these family slugs; empty -> all."""
    strategies: Sequence[ChunkStrategy] = DEFAULT_STRATEGIES
    python: Path | None = None
    """Engine interpreter; None -> :func:`canitoolcall.adapters.engine_python`."""
    out_dir: Path = Path("results")
    include_observed: Literal["all", "failures", "none"] = "failures"
    timeout_s: float = 120.0
    """Per-message worker timeout."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for the worker (e.g. HF_HUB_OFFLINE=1)."""


class WorkerError(RuntimeError):
    """The worker process crashed, timed out, or answered ``ok: false``."""


class WorkerClient:
    """Talks the JSON-lines protocol of :mod:`canitoolcall.adapters.worker`.

    Usage::

        with WorkerClient("vllm", python=...) as w:
            info = w.hello()
            reply = w.replay(fixture, family, strategies)
    """

    def __init__(
        self, engine: str, python: Path, env: Mapping[str, str] | None = None, timeout_s: float = 120.0
    ) -> None:
        raise NotImplementedError

    def __enter__(self) -> WorkerClient:
        raise NotImplementedError

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError

    def hello(self) -> dict[str, Any]:
        """Send ``hello``; returns engine name/version/commit/details."""
        raise NotImplementedError

    def replay(self, fixture: Fixture, family: Family | None, strategies: Sequence[ChunkStrategy]) -> dict[str, Any]:
        """Send ``replay``; returns the worker reply (raises WorkerError if not ok)."""
        raise NotImplementedError

    def restart(self) -> None:
        """Kill and relaunch the worker process (after a crash or timeout)."""
        raise NotImplementedError

    def close(self) -> None:
        """Send ``shutdown`` and wait; kill on timeout."""
        raise NotImplementedError


def evaluate(
    fixture: Fixture,
    family: Family | None,
    reply: Mapping[str, Any],
    include_observed: Literal["all", "failures", "none"] = "failures",
) -> CaseResult:
    """Turn one worker ``replay`` reply into a CaseResult (runs the checks)."""
    raise NotImplementedError


def run(config: RunConfig) -> RunResults:
    """Run the offline suite for one engine and return the results (not written)."""
    raise NotImplementedError


def run_and_write(config: RunConfig) -> Path:
    """:func:`run`, then write ``<out_dir>/<engine>-<version>.json``; returns the path."""
    raise NotImplementedError
