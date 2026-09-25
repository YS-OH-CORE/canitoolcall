"""Live probe: check a running OpenAI-compatible endpoint in about a minute.

``canitoolcall probe --base-url http://localhost:8000/v1 --model M`` sends a
small set of scripted tool-use scenarios (single call, parallel calls, unicode
and nested arguments, no-call answer, reasoning + call, multi-turn with a tool
result), each non-streaming and streaming, and reports pass/fail per scenario.

Unlike the offline suite this exercises the whole stack (template, sampling,
parser, server), so outcomes depend on the model's behaviour. Checks are
therefore structural (a call to the right tool with schema-valid arguments,
stream == non-stream shape, no marker leakage) rather than exact-text.

Uses only the standard library HTTP client (``urllib``) so it runs anywhere;
SSE is parsed by hand. No API key is ever logged or written to reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from canitoolcall.results import ParseResult

ProbeStatus = Literal["pass", "fail", "error", "skip"]


@dataclass(frozen=True)
class ProbeExpectation:
    """Structural expectation for a scenario."""

    tool_names: tuple[str, ...] = ()
    """Tools that must be called, in order; empty means no call expected."""
    min_calls: int = 0
    required_argument_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    expect_reasoning: bool | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    expect: ProbeExpectation
    tool_choice: str | Mapping[str, Any] = "auto"
    extra_body: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeOutcome:
    scenario_id: str
    stream: bool
    status: ProbeStatus
    detail: str | None
    latency_s: float
    observed: ParseResult | None = None


@dataclass(frozen=True)
class ProbeReport:
    base_url: str
    model: str
    started_at: str
    outcomes: tuple[ProbeOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def render_text(self) -> str:
        """Human-readable table for the terminal."""
        raise NotImplementedError


BUILTIN_SCENARIOS: tuple[Scenario, ...] = ()
"""The default scenario set (filled in by the probe builder)."""


def chat(
    base_url: str, model: str, scenario: Scenario, *, stream: bool, api_key: str | None = None, timeout_s: float = 60.0
) -> tuple[ParseResult, float]:
    """One /chat/completions request; returns the accumulated parse and latency.

    Streaming responses are accumulated OpenAI-client style (content and
    reasoning deltas concatenated; tool-call deltas merged by index).
    """
    raise NotImplementedError


def evaluate(scenario: Scenario, result: ParseResult) -> tuple[ProbeStatus, str | None]:
    """Judge one observed result against ``scenario.expect``."""
    raise NotImplementedError


def probe(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    scenarios: Sequence[Scenario] | None = None,
    stream_modes: Sequence[bool] = (False, True),
    timeout_s: float = 60.0,
) -> ProbeReport:
    """Run every scenario in every stream mode against the endpoint."""
    raise NotImplementedError
