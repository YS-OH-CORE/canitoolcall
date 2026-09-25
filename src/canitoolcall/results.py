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
    arguments_raw: str | None
    """Arguments exactly as returned (OpenAI format: JSON text). ``None`` only
    when an engine returned no arguments at all; the checks treat that as
    invalid JSON."""

    def arguments(self) -> Any:
        """Decode ``arguments_raw``; raises ``json.JSONDecodeError`` if invalid.

        No leniency here: an empty string is invalid JSON. Whether an engine
        may return ``""`` for a no-argument call is a policy of the checks.
        """
        if self.arguments_raw is None:
            raise json.JSONDecodeError("no arguments", "", 0)
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
        """Decode a parse (e.g. from a worker reply); raises ``TypeError`` on wrong field types."""
        if not isinstance(d, Mapping):
            raise TypeError(f"parse result must be an object, got {type(d).__name__}")
        for key in ("content", "reasoning_content", "exception"):
            if d.get(key) is not None and not isinstance(d[key], str):
                raise TypeError(f"{key} must be a string or null, got {type(d[key]).__name__}")
        calls = d.get("tool_calls") or []
        if not isinstance(calls, list):
            raise TypeError(f"tool_calls must be a list, got {type(calls).__name__}")
        tool_calls: list[ParsedToolCall] = []
        for i, t in enumerate(calls):
            if not isinstance(t, Mapping):
                raise TypeError(f"tool_calls[{i}] must be an object, got {type(t).__name__}")
            name, args = t.get("name"), t.get("arguments_raw")
            if not isinstance(name, str):
                raise TypeError(f"tool_calls[{i}].name must be a string, got {type(name).__name__}")
            # A non-string arguments_raw (null from an engine's Optional field) is kept
            # as is: the checks report it as invalid arguments instead of crashing.
            tool_calls.append(ParsedToolCall(name, args))
        return cls(
            content=d.get("content"),
            reasoning_content=d.get("reasoning_content"),
            tool_calls=tuple(tool_calls),
            exception=d.get("exception"),
        )


class StreamAccumulator:
    """Accumulates streamed deltas the way an OpenAI client does (DESIGN.md rule 8).

    Mirrors openai-python's ``accumulate_delta``: content and reasoning deltas
    are concatenated; tool-call deltas merge by ``index`` and their string
    fields (``name``, ``arguments``) are concatenated. A name sent once stays
    as is; a name the engine re-sends shows up repeated, as a client sees it.
    Every adapter uses this one implementation.
    """

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, list[str]] = {}

    def add_content(self, text: str | None) -> None:
        if text:
            self._content.append(text)

    def add_reasoning(self, text: str | None) -> None:
        if text:
            self._reasoning.append(text)

    def add_tool_call(self, index: int, name: str | None = None, arguments: str | None = None) -> None:
        """Merge one tool-call delta into the call at ``index``."""
        slot = self._calls.setdefault(int(index), ["", ""])
        if name:
            slot[0] += name
        if arguments:
            slot[1] += arguments

    def append_tool_call(self, name: str, arguments: str) -> None:
        """A complete call sent as one delta with the next free index."""
        self.add_tool_call(max(self._calls, default=-1) + 1, name, arguments)

    def add_openai_delta(self, delta: Mapping[str, Any]) -> None:
        """Merge one ``choices[].delta`` object of an OpenAI chat-completion chunk."""
        self.add_content(delta.get("content"))
        self.add_reasoning(delta.get("reasoning_content") or delta.get("reasoning"))
        for tc in delta.get("tool_calls") or ():
            fn = tc.get("function") or {}
            self.add_tool_call(int(tc.get("index") or 0), fn.get("name"), fn.get("arguments"))

    def result(self, exception: str | None = None) -> ParseResult:
        return ParseResult(
            content="".join(self._content) or None,
            reasoning_content="".join(self._reasoning) or None,
            tool_calls=tuple(ParsedToolCall(n, a) for _, (n, a) in sorted(self._calls.items())),
            exception=exception,
        )


@dataclass(frozen=True)
class Observation:
    """All parses of one fixture by one engine."""

    nonstream: ParseResult
    streams: Mapping[str, ParseResult]
    """Keyed by chunking strategy id."""
    synthetic: frozenset[str] = field(default=frozenset(), compare=False)
    """Strategy ids in ``streams`` this engine cannot produce (``char:*``, and
    multi-token deltas on one-token-per-step engines). They are checked and
    reported but never count toward the case status. Not serialized: the run
    records them in ``run.synthetic_strategies``."""

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
    synthetic_strategies: tuple[str, ...] = ()
    """Strategies that ran but do not count toward case status for this engine
    (see :func:`canitoolcall.chunking.synthetic_strategies`)."""


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


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
            "run": {
                **vars(self.run),
                "strategies": list(self.run.strategies),
                "synthetic_strategies": list(self.run.synthetic_strategies),
            },
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
            run=RunInfo(
                **{
                    **r,
                    "strategies": tuple(r["strategies"]),
                    "synthetic_strategies": tuple(r.get("synthetic_strategies") or ()),
                }
            ),
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
            # mkstemp creates 0600; results are meant to be shared, so use the umask like open() would.
            os.chmod(tmp, 0o666 & ~_umask())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: Path) -> RunResults:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
