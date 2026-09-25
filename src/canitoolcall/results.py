"""Result data model and the results JSON format (spec/results.schema.json).

* :class:`ParseResult` is what an adapter returns for one parse.
* :class:`CheckResult` / :class:`CaseResult` are what the checks produce.
* :class:`RunResults` is one engine run, serialized to
  ``results/<engine>-<version>.json`` and consumed by the matrix site.

Standard library only (ParseResult crosses the worker boundary as JSON).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

RESULTS_SCHEMA_VERSION = "0.1"


class Status(str, Enum):
    PASS = "pass"
    SOFT_PASS = "soft_pass"
    FAIL = "fail"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


_SEVERITY = {Status.PASS: 0, Status.SOFT_PASS: 1, Status.ERROR: 2, Status.FAIL: 3, Status.UNSUPPORTED: -1}


def worst_status(statuses: Iterable[Status]) -> Status:
    """Worst status: fail > error > soft_pass > pass. Empty input is ``pass``."""
    worst = Status.PASS
    for s in statuses:
        if _SEVERITY[s] > _SEVERITY[worst]:
            worst = s
    return worst


@dataclass(frozen=True)
class ParsedToolCall:
    """One tool call as the engine returned it."""

    name: str
    arguments_raw: str
    """Arguments exactly as returned (OpenAI format: JSON text)."""

    def arguments(self) -> Any:
        """Decode ``arguments_raw``; raises ``json.JSONDecodeError`` if invalid.

        No leniency here: an empty string is invalid JSON. Whether an engine
        may return ``""`` for a no-argument call is a policy of the checks.
        """
        return json.loads(self.arguments_raw)


@dataclass(frozen=True)
class ParseResult:
    """What an engine's parser produced for one (non-)streaming replay.

    For streams, adapters accumulate deltas exactly as an OpenAI client would:
    content and reasoning deltas are concatenated, tool-call deltas are merged
    by index (name set once, argument fragments concatenated).

    ``exception`` is set when the ENGINE's parser raised (``"Type: msg"``);
    that is a parse outcome, not a harness error.
    """

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ParsedToolCall, ...] = ()
    exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": [{"name": t.name, "arguments_raw": t.arguments_raw} for t in self.tool_calls],
            "exception": self.exception,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ParseResult:
        return cls(
            content=d.get("content"),
            reasoning_content=d.get("reasoning_content"),
            tool_calls=tuple(ParsedToolCall(t["name"], t["arguments_raw"]) for t in d.get("tool_calls", [])),
            exception=d.get("exception"),
        )


@dataclass(frozen=True)
class Observation:
    """All parses of one fixture by one engine."""

    nonstream: ParseResult
    streams: Mapping[str, ParseResult]
    """Keyed by chunking strategy id."""

    def to_dict(self) -> dict[str, Any]:
        return {"nonstream": self.nonstream.to_dict(), "streams": {k: v.to_dict() for k, v in self.streams.items()}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Observation:
        return cls(
            nonstream=ParseResult.from_dict(d["nonstream"]),
            streams={k: ParseResult.from_dict(v) for k, v in d.get("streams", {}).items()},
        )


@dataclass(frozen=True)
class CheckResult:
    check: str
    strategy: str
    """``"nonstream"`` or a chunking strategy id."""
    status: Status
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "strategy": self.strategy, "status": self.status.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CheckResult:
        return cls(d["check"], d["strategy"], Status(d["status"]), d.get("detail"))


@dataclass(frozen=True)
class CaseResult:
    fixture_id: str
    family: str
    status: Status
    checks: tuple[CheckResult, ...] = ()
    parser_config: Mapping[str, Any] | None = None
    observed: Observation | None = None
    harness_error: str | None = None
    reason: str | None = None
    """Why the adapter declined (set when status is ``unsupported``)."""
    skipped_strategies: Mapping[str, str] = field(default_factory=dict)
    """Requested strategies the worker could not run, with the reason."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "fixture_id": self.fixture_id,
            "family": self.family,
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
        }
        if self.reason is not None:
            out["reason"] = self.reason
        if self.skipped_strategies:
            out["skipped_strategies"] = dict(self.skipped_strategies)
        if self.parser_config is not None:
            out["parser_config"] = dict(self.parser_config)
        if self.observed is not None:
            out["observed"] = self.observed.to_dict()
        if self.harness_error is not None:
            out["harness_error"] = self.harness_error
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CaseResult:
        return cls(
            fixture_id=d["fixture_id"],
            family=d["family"],
            status=Status(d["status"]),
            checks=tuple(CheckResult.from_dict(c) for c in d.get("checks", [])),
            parser_config=d.get("parser_config"),
            observed=Observation.from_dict(d["observed"]) if d.get("observed") else None,
            harness_error=d.get("harness_error"),
            reason=d.get("reason"),
            skipped_strategies=dict(d.get("skipped_strategies") or {}),
        )


@dataclass(frozen=True)
class EngineInfo:
    name: str
    version: str
    commit: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunInfo:
    started_at: str
    finished_at: str
    platform: str
    python: str
    fixtures_digest: str
    strategies: tuple[str, ...]
    normalization: str


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._+-]")


def _counts(cases: Iterable[CaseResult]) -> dict[str, int]:
    c = {s.value: 0 for s in Status}
    for case in cases:
        c[case.status.value] += 1
    return c


@dataclass(frozen=True)
class RunResults:
    """One engine run; the unit the matrix site is built from."""

    canitoolcall_version: str
    engine: EngineInfo
    run: RunInfo
    cases: tuple[CaseResult, ...]

    def summary(self) -> dict[str, Any]:
        fams = sorted({c.family for c in self.cases})
        return {
            "totals": _counts(self.cases),
            "by_family": {f: _counts(c for c in self.cases if c.family == f) for f in fams},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "canitoolcall_version": self.canitoolcall_version,
            "engine": {
                "name": self.engine.name,
                "version": self.engine.version,
                "commit": self.engine.commit,
                "details": dict(self.engine.details),
            },
            "run": {**vars(self.run), "strategies": list(self.run.strategies)},
            "cases": [c.to_dict() for c in self.cases],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunResults:
        if str(d.get("schema_version", "")).split(".")[0] != RESULTS_SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"unsupported results schema_version {d.get('schema_version')!r}")
        e = d["engine"]
        r = d["run"]
        return cls(
            canitoolcall_version=d["canitoolcall_version"],
            engine=EngineInfo(e["name"], e["version"], e.get("commit"), e.get("details") or {}),
            run=RunInfo(**{**r, "strategies": tuple(r["strategies"])}),
            cases=tuple(CaseResult.from_dict(c) for c in d["cases"]),
        )

    def default_filename(self) -> str:
        """``<engine>-<version>.json`` with filesystem-unsafe characters replaced by ``_``."""
        return _SAFE_NAME.sub("_", f"{self.engine.name}-{self.engine.version}") + ".json"

    def write(self, path: Path) -> Path:
        """Write pretty, key-stable JSON atomically; returns ``path``.

        The file is written to a temporary sibling and renamed into place, so
        concurrent runs and readers never see a partial file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: Path) -> RunResults:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
