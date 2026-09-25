"""Build the ``qwen3-hermes`` fixture corpus.

Every fixture written by this script has one of three provenances:

* ``template_render``: an assistant message rendered through the OFFICIAL Qwen3
  chat template with ``apply_chat_template(tokenize=True)``. The prompt ids are
  sliced off, the output is cut at the first stop id from
  ``generation_config.json``, and ``raw_output`` is those ids decoded with
  ``skip_special_tokens=False`` (spec/README.md, "Token ids come first").
  Truncated fixtures are a prefix of such a render, cut at a token boundary,
  which is exactly what ``max_tokens`` produces.
* ``engine_test``: raw strings copied verbatim from vLLM / SGLang test suites
  (Apache-2.0), see ``imported.py``.
* ``bug_report``: raw strings quoted from public issues, see ``imported.py``.

Only tokenizer/template files are downloaded (never weights), each at a pinned
revision. Run it from the repo root inside the transformers engine venv::

    .venvs/transformers/bin/python scripts/fixtures/qwen3-hermes/build.py

The output is deterministic, so re-running it must leave ``git diff`` clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imported  # noqa: E402

REPO_ROOT = HERE.parents[2]
SLUG = "qwen3-hermes"
OUT_DIR = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    """A reference model: tokenizer + official template at a pinned revision."""

    key: str
    repo: str
    revision: str
    variant: str
    siblings: tuple[str, ...]
    """Other repos whose chat template is byte-identical (checked 2026-09-25)."""


HYBRID = Model(
    "hybrid",
    "Qwen/Qwen3-0.6B",
    "c1899de289a04d12100db370d81485cdf75e47ca",
    "qwen3-hybrid",
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B"),
)
THINKING = Model(
    "thinking",
    "Qwen/Qwen3-4B-Thinking-2507",
    "768f209d9ea81521153ed38c47d515654e938aea",
    "qwen3-thinking-2507",
    ("Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507"),
)
INSTRUCT = Model(
    "instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "cdbee75f17c01a7cc42f958dc650907174af0554",
    "qwen3-instruct-2507",
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
)
MODELS = {m.key: m for m in (HYBRID, THINKING, INSTRUCT)}


class Renderer:
    """Render assistant messages through one model's official chat template."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.tok = AutoTokenizer.from_pretrained(model.repo, revision=model.revision)
        gen_cfg = json.loads(
            Path(hf_hub_download(model.repo, "generation_config.json", revision=model.revision)).read_text()
        )
        eos = gen_cfg["eos_token_id"]
        self.stop_ids: set[int] = set(eos if isinstance(eos, list) else [eos])
        self.template_sha256 = hashlib.sha256(self.tok.chat_template.encode("utf-8")).hexdigest()

    def _ids(
        self, messages: list[dict[str, Any]], tools: Sequence[Mapping[str, Any]], gen: bool, **kw: Any
    ) -> list[int]:
        out = self.tok.apply_chat_template(messages, tools=list(tools), add_generation_prompt=gen, tokenize=True, **kw)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def _text(self, messages: list[dict[str, Any]], tools: Sequence[Mapping[str, Any]], gen: bool, **kw: Any) -> str:
        return str(
            self.tok.apply_chat_template(messages, tools=list(tools), add_generation_prompt=gen, tokenize=False, **kw)
        )

    def render(
        self, history: list[dict[str, Any]], assistant: dict[str, Any], tools: Sequence[Mapping[str, Any]], **kw: Any
    ) -> tuple[list[int], str]:
        """Return ``(output_ids, generation_prompt)`` for ``assistant`` after ``history``."""
        prompt = self._ids(history, tools, True, **kw)
        full = self._ids([*history, assistant], tools, False, **kw)
        if full[: len(prompt)] != prompt:
            raise AssertionError(f"{self.model.repo}: generation prompt is not a prefix of the rendered turn")
        out = full[len(prompt) :]
        cut = next(i for i, t in enumerate(out) if t in self.stop_ids)
        gen_prompt = self._text(history, tools, True, **kw)[len(self._text(history, tools, False, **kw)) :]
        return out[:cut], gen_prompt

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.tok.decode(list(ids), skip_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        """Encode raw text (imported fixtures); asserts a lossless round trip."""
        ids = list(self.tok.encode(text, add_special_tokens=False))
        if self.decode(ids) != text:
            raise AssertionError(f"{self.model.repo}: lossy round trip for {text!r}")
        return ids

    def truncate(self, ids: list[int], after: str) -> list[int]:
        """Shortest token prefix whose decoded text contains ``after`` (a max_tokens cut)."""
        for n in range(1, len(ids) + 1):
            if after in self.decode(ids[:n]):
                return ids[:n]
        raise AssertionError(f"{after!r} not found in render")


# --------------------------------------------------------------------------- tools


def _tool(name: str, description: str, properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    params: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params["required"] = list(required)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


TOOLS: dict[str, dict[str, Any]] = {
    t["function"]["name"]: t
    for t in [
        _tool(
            "get_weather",
            "Get the current weather for a city.",
            {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            ["city"],
        ),
        _tool(
            "search",
            "Search the web.",
            {"query": {"type": "string"}, "filters": {"type": "object"}},
            ["query"],
        ),
        _tool("get_time", "Get the current UTC time. Takes no arguments.", {}),
        _tool(
            "create_event",
            "Create a calendar event.",
            {
                "title": {"type": "string"},
                "when": {"type": "object"},
                "attendees": {"type": "array", "items": {"type": "object"}},
                "reminders": {"type": "array"},
                "location": {"type": ["object", "null"]},
            },
            ["title", "when"],
        ),
        _tool(
            "translate",
            "Translate text.",
            {"text": {"type": "string"}, "target": {"type": "string"}},
            ["text", "target"],
        ),
        _tool("echo", "Echo a string back verbatim.", {"text": {"type": "string"}}, ["text"]),
        _tool(
            "write_file",
            "Write a text file.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        _tool(
            "calculate",
            "Evaluate numbers.",
            {
                "x": {"type": "integer"},
                "y": {"type": "number"},
                "tolerance": {"type": "number"},
                "exact": {"type": "boolean"},
                "limit": {"type": ["integer", "null"]},
                "values": {"type": "array", "items": {"type": "number"}},
            },
            ["x", "y"],
        ),
    ]
}


def tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


# --------------------------------------------------------------------------- cases

Call = tuple[str, dict[str, Any]]
LONG_CONTENT = "".join(
    f"{i:03d}: The quick brown fox jumps over the lazy dog; café, naïve, 東京.\n" for i in range(1, 61)
)
STD_CALLS: list[Call] = [
    ("get_weather", {"city": "Zürich", "unit": "c"}),
    ("search", {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}),
]


@dataclass
class Case:
    name: str
    file: str
    model: str
    user: str
    tools: list[str]
    calls: list[Call] = field(default_factory=list)
    content: str | None = None
    reasoning: str | None = None
    thinking: bool | None = None
    tags: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    """Messages between the user turn and the rendered assistant turn (multi-turn)."""
    truncate_after: str | None = None
    expected: dict[str, Any] | None = None
    """Override for truncated cases (a dict for ``expected`` or ``expected_error``)."""
    notes: str | None = None


def assistant_msg(calls: Sequence[Call], content: str | None, reasoning: str | None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(calls)
        ]
    return msg


def tool_turn(calls: Sequence[Call], results: Sequence[str], reasoning: str | None = None) -> list[dict[str, Any]]:
    """An earlier assistant tool-call turn plus its tool results (for multi-turn cases)."""
    msgs = [assistant_msg(calls, None, reasoning)]
    msgs += [
        {"role": "tool", "tool_call_id": f"call_{i}", "name": calls[i][0], "content": r} for i, r in enumerate(results)
    ]
    return msgs


TRUNCATED = {
    "reason": "Output cut by max_tokens inside the tool-call JSON: no complete call exists.",
    "accept": ["no_tool_calls", "content_passthrough", "exception"],
}

CASES: list[Case] = [
    # ---- basic: Qwen3 hybrid (2504), thinking on and off
    Case(
        "single-call-thinking",
        "basic",
        "hybrid",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        reasoning="The user wants the current weather in Paris. I'll call get_weather.",
        thinking=True,
        tags=["single-call", "reasoning"],
    ),
    Case(
        "single-call-no-thinking",
        "basic",
        "hybrid",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        thinking=False,
        tags=["single-call"],
        notes="enable_thinking=false: the template pre-fills an empty think block in the generation prompt.",
    ),
    Case(
        "no-call-thinking",
        "basic",
        "hybrid",
        "Hi there!",
        ["get_weather"],
        content="Hello! How can I help you today?",
        reasoning="The user is just greeting me. No tool is needed.",
        thinking=True,
        tags=["no-call", "reasoning"],
    ),
    Case(
        "no-call-no-thinking",
        "basic",
        "hybrid",
        "Hi there!",
        ["get_weather"],
        content="Hello! How can I help you today?",
        thinking=False,
        tags=["no-call"],
    ),
    Case(
        "text-before-call-thinking",
        "basic",
        "hybrid",
        "Is it raining in London?",
        ["get_weather"],
        [("get_weather", {"city": "London"})],
        content="Let me check the current conditions in London.",
        reasoning="I need live weather data for London.",
        thinking=True,
        tags=["single-call", "text-before-call", "reasoning"],
    ),
    Case(
        "text-before-call-no-thinking",
        "basic",
        "hybrid",
        "Is it raining in London?",
        ["get_weather"],
        [("get_weather", {"city": "London"})],
        content="Let me check the current conditions in London.",
        thinking=False,
        tags=["single-call", "text-before-call"],
    ),
    Case(
        "empty-arguments-no-thinking",
        "basic",
        "hybrid",
        "What time is it?",
        ["get_time"],
        [("get_time", {})],
        thinking=False,
        tags=["single-call", "empty-arguments"],
        notes="Parameterless tool, arguments {} (cf. https://github.com/vllm-project/vllm/issues/28806).",
    ),
    Case(
        "empty-arguments-thinking",
        "basic",
        "hybrid",
        "What time is it?",
        ["get_time"],
        [("get_time", {})],
        reasoning="get_time takes no arguments.",
        thinking=True,
        tags=["single-call", "empty-arguments", "reasoning"],
    ),
    # ---- parallel
    Case(
        "parallel-two-calls-thinking",
        "parallel",
        "hybrid",
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS,
        reasoning="I should call the tools.",
        thinking=True,
        tags=["parallel-calls", "reasoning", "unicode", "nested-json", "string-escapes"],
    ),
    Case(
        "parallel-three-calls-no-thinking",
        "parallel",
        "hybrid",
        "Weather in Paris, London and Tokyo?",
        ["get_weather"],
        [
            ("get_weather", {"city": "Paris"}),
            ("get_weather", {"city": "London"}),
            ("get_weather", {"city": "Tokyo", "unit": "c"}),
        ],
        thinking=False,
        tags=["parallel-calls"],
    ),
    Case(
        "parallel-mixed-tools-text-before-no-thinking",
        "parallel",
        "hybrid",
        "What time is it, and what's the weather in Oslo?",
        ["get_time", "get_weather"],
        [("get_time", {}), ("get_weather", {"city": "Oslo", "unit": "c"})],
        content="I'll fetch both for you.",
        thinking=False,
        tags=["parallel-calls", "text-before-call", "empty-arguments"],
    ),
    # ---- edge values
    Case(
        "nested-json-deep",
        "edge",
        "hybrid",
        "Set up the design review.",
        ["create_event"],
        [
            (
                "create_event",
                {
                    "title": "Design review",
                    "when": {"start": "2026-10-01T09:00:00Z", "end": "2026-10-01T10:00:00Z", "tz": "UTC"},
                    "attendees": [
                        {"name": "Ana", "email": "ana@example.com", "optional": False},
                        {"name": "Bo", "email": "bo@example.com", "optional": True, "roles": ["notes", "timekeeper"]},
                    ],
                    "reminders": [[10, "email"], [1, "popup"]],
                    "location": None,
                },
            )
        ],
        thinking=False,
        tags=["single-call", "nested-json"],
    ),
    Case(
        "unicode-emoji-thinking",
        "edge",
        "hybrid",
        "Translate this greeting to German.",
        ["translate"],
        [("translate", {"text": "こんにちは 🌸 مرحبا — naïve café 👩‍💻", "target": "de"})],
        reasoning="Translate the mixed-script greeting (日本語, العربية, emoji 🌸).",
        thinking=True,
        tags=["single-call", "unicode", "reasoning"],
    ),
    Case(
        "marker-in-arguments",
        "edge",
        "hybrid",
        "Echo the markup back to me.",
        ["echo"],
        [("echo", {"text": 'Use <tool_call> ... </tool_call> tags, or "<think>" blocks.'})],
        thinking=False,
        tags=["single-call", "marker-in-arguments", "string-escapes"],
        notes=(
            "The argument string contains the literal text of the format markers. The official tokenizer "
            "maps them to their added-token ids (151657/151658/151667) inside the JSON string; a correct "
            "parser must not end the call at the inner </tool_call>, because the JSON is still open there."
        ),
    ),
    Case(
        "string-escapes-code",
        "edge",
        "hybrid",
        "Save a hello-world script.",
        ["write_file"],
        [
            (
                "write_file",
                {
                    "path": "C:\\Users\\me\\hello.py",
                    "content": 'def main():\n\tprint("hi \\u00e9")\n\treturn {"ok": True}\n',
                },
            )
        ],
        thinking=False,
        tags=["single-call", "string-escapes"],
    ),
    Case(
        "json-string-argument",
        "edge",
        "hybrid",
        "Echo this JSON text.",
        ["echo"],
        [("echo", {"text": '{"name": "x", "arguments": {"a": 1}}'})],
        thinking=False,
        tags=["single-call", "string-escapes", "x-json-in-string"],
        notes="A string argument that itself looks like a Hermes tool-call payload; it must stay a string.",
    ),
    Case(
        "numeric-arguments",
        "edge",
        "hybrid",
        "Compute it.",
        ["calculate"],
        [
            (
                "calculate",
                {"x": -3, "y": 2.5, "tolerance": 1e-05, "exact": True, "limit": None, "values": [0, -0.5, 1e21]},
            )
        ],
        thinking=False,
        tags=["single-call", "numeric-arguments"],
        notes="Types matter: integers, floats, exponent notation, booleans and null must round-trip.",
    ),
    Case(
        "long-arguments",
        "edge",
        "hybrid",
        "Write the sample file.",
        ["write_file"],
        [("write_file", {"path": "notes/sample.txt", "content": LONG_CONTENT})],
        thinking=False,
        tags=["single-call", "long-arguments", "unicode"],
    ),
    # ---- multi-turn
    Case(
        "multi-turn-second-call-thinking",
        "multiturn",
        "hybrid",
        "Weather in Zürich, then find a café there.",
        ["get_weather", "search"],
        [("search", {"query": "café Zürich", "filters": {"open_now": True}})],
        reasoning="It is 20C in Zürich. Now search for cafés.",
        thinking=True,
        history=tool_turn(
            [("get_weather", {"city": "Zürich", "unit": "c"})], ['{"temp": 20}'], reasoning="First, the weather."
        ),
        tags=["single-call", "multi-turn", "reasoning", "unicode"],
    ),
    Case(
        "multi-turn-final-answer-thinking",
        "multiturn",
        "hybrid",
        "Weather in Zürich?",
        ["get_weather"],
        content="It is 20 °C in Zürich right now.",
        reasoning="The tool returned 20C.",
        thinking=True,
        history=tool_turn(
            [("get_weather", {"city": "Zürich", "unit": "c"})], ['{"temp": 20}'], reasoning="Call the weather tool."
        ),
        tags=["no-call", "multi-turn", "reasoning", "unicode"],
    ),
    # ---- Qwen3-*-Thinking-2507: the generation prompt pre-fills "<think>\n"
    Case(
        "thinking2507-single-call",
        "thinking-2507",
        "thinking",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        reasoning="The user wants the current weather in Paris. I'll call get_weather.",
        tags=["single-call", "reasoning", "reasoning-prefilled"],
    ),
    Case(
        "thinking2507-parallel-calls",
        "thinking-2507",
        "thinking",
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS,
        reasoning="I should call the tools.",
        tags=["parallel-calls", "reasoning", "reasoning-prefilled", "unicode", "nested-json", "string-escapes"],
    ),
    Case(
        "thinking2507-no-call",
        "thinking-2507",
        "thinking",
        "What is 2 + 2?",
        ["get_weather"],
        content="2 + 2 = 4.",
        reasoning="Simple arithmetic, no tool needed.",
        tags=["no-call", "reasoning", "reasoning-prefilled"],
    ),
    Case(
        "thinking2507-text-before-call",
        "thinking-2507",
        "thinking",
        "Is it raining in London?",
        ["get_weather"],
        [("get_weather", {"city": "London"})],
        content="Let me check the current conditions in London.",
        reasoning="I need live weather data for London.",
        tags=["single-call", "text-before-call", "reasoning", "reasoning-prefilled"],
    ),
    Case(
        "thinking2507-multi-turn",
        "thinking-2507",
        "thinking",
        "Weather in Zürich, then find a café there.",
        ["get_weather", "search"],
        [("search", {"query": "café Zürich"})],
        reasoning="It is 20C. Now the café search.",
        history=tool_turn(
            [("get_weather", {"city": "Zürich", "unit": "c"})], ['{"temp": 20}'], reasoning="First, the weather."
        ),
        tags=["single-call", "multi-turn", "reasoning", "reasoning-prefilled", "unicode"],
    ),
    # ---- Qwen3-*-Instruct-2507: no think handling at all
    Case(
        "instruct2507-single-call",
        "instruct-2507",
        "instruct",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        tags=["single-call"],
    ),
    Case(
        "instruct2507-parallel-calls",
        "instruct-2507",
        "instruct",
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS,
        tags=["parallel-calls", "unicode", "nested-json", "string-escapes"],
    ),
    Case(
        "instruct2507-no-call",
        "instruct-2507",
        "instruct",
        "Hi there!",
        ["get_weather"],
        content="Hello! How can I help you today?",
        tags=["no-call"],
    ),
    Case(
        "instruct2507-empty-arguments",
        "instruct-2507",
        "instruct",
        "What time is it?",
        ["get_time"],
        [("get_time", {})],
        tags=["single-call", "empty-arguments"],
    ),
    Case(
        "instruct2507-text-before-call",
        "instruct-2507",
        "instruct",
        "Is it raining in London?",
        ["get_weather"],
        [("get_weather", {"city": "London"})],
        content="Let me check the current conditions in London.",
        tags=["single-call", "text-before-call"],
    ),
    # ---- truncated (token-prefix of a render, as max_tokens produces)
    Case(
        "truncated-mid-arguments-no-thinking",
        "truncated",
        "hybrid",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        thinking=False,
        truncate_after='"city": "Par',
        expected={"expected_error": TRUNCATED},
        tags=["truncated"],
    ),
    Case(
        "truncated-mid-arguments-thinking",
        "truncated",
        "hybrid",
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS[:1],
        reasoning="I should call the tool.",
        thinking=True,
        truncate_after='"arguments": {"ci',
        expected={"expected_error": TRUNCATED},
        tags=["truncated", "reasoning"],
    ),
    Case(
        "truncated-after-open-tag",
        "truncated",
        "hybrid",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris"})],
        thinking=False,
        truncate_after="<tool_call>\n",
        expected={
            "expected_error": {
                "reason": "Output cut by max_tokens right after <tool_call>: there is no call name or arguments.",
                "accept": ["no_tool_calls", "content_passthrough", "exception"],
            }
        },
        tags=["truncated"],
    ),
    Case(
        "truncated-second-parallel-call",
        "truncated",
        "hybrid",
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS,
        thinking=False,
        truncate_after='"query": "caf',
        expected={
            "expected": {
                "content": None,
                "reasoning_content": None,
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}}],
            }
        },
        tags=["truncated", "parallel-calls", "unicode"],
        notes=(
            "The first call is complete; the second is cut mid-arguments. A correct parser keeps the complete call "
            "and drops the unterminated one without leaking its markup (the fix proposed in "
            "https://github.com/sgl-project/sglang/issues/30480: valid calls survive a truncated neighbour)."
        ),
    ),
    Case(
        "truncated-in-reasoning",
        "truncated",
        "hybrid",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris"})],
        reasoning="The user wants the current weather in Paris, so I will call the weather tool.",
        thinking=True,
        truncate_after="current weather",
        expected={
            "expected": {
                "content": None,
                "reasoning_content": "The user wants the current weather",
                "tool_calls": [],
            }
        },
        tags=["truncated", "reasoning"],
        notes="Cut inside <think> (no </think> yet): the partial text is reasoning, not content.",
    ),
    Case(
        "thinking2507-truncated-in-reasoning",
        "truncated",
        "thinking",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris"})],
        reasoning="The user wants the current weather in Paris, so I will call the weather tool.",
        truncate_after="current weather",
        expected={
            "expected": {
                "content": None,
                "reasoning_content": "The user wants the current weather",
                "tool_calls": [],
            }
        },
        tags=["truncated", "reasoning", "reasoning-prefilled"],
        notes="The prompt pre-fills <think>; the cut output has neither <think> nor </think> and is all reasoning.",
    ),
]


# --------------------------------------------------------------------------- records


def expected_from(case: Case) -> dict[str, Any]:
    return {
        "expected": {
            "content": case.content,
            "reasoning_content": case.reasoning,
            "tool_calls": [{"name": n, "arguments": a} for n, a in case.calls],
        }
    }


def base_record(
    fid: str, model: Model, provenance: dict[str, Any], offered: list[dict[str, Any]], raw: str, ids: list[int]
) -> dict[str, Any]:
    return {
        "id": f"{SLUG}/{fid}",
        "family": SLUG,
        "models": [model.repo, *model.siblings],
        "spec_version": "0.1",
        "provenance": provenance,
        "tools": offered,
        "raw_output": raw,
        "output_token_ids": ids,
        "tokenizer": {"repo": model.repo, "revision": model.revision, "mode": "hf"},
    }


def render_case(r: Renderer, case: Case) -> dict[str, Any]:
    m = r.model
    offered = tools(*case.tools)
    kw: dict[str, Any] = {}
    if case.thinking is not None:
        kw["enable_thinking"] = case.thinking
    history = [{"role": "user", "content": case.user}, *case.history]
    ids, gen_prompt = r.render(history, assistant_msg(case.calls, case.content, case.reasoning), offered, **kw)
    if case.truncate_after is not None:
        ids = r.truncate(ids, case.truncate_after)
    raw = r.decode(ids)
    rec = base_record(
        case.name,
        m,
        {
            "kind": "template_render",
            "source_url": f"https://huggingface.co/{m.repo}/blob/{m.revision}/tokenizer_config.json",
            "revision": m.revision,
            "license": "Apache-2.0",
            "generator": GENERATOR,
            "template_sha256": r.template_sha256,
        },
        offered,
        raw,
        ids,
    )
    rec["generation_prompt"] = gen_prompt
    if case.thinking is not None:
        rec["thinking"] = case.thinking
    rec.update(case.expected or expected_from(case))
    rec["tags"] = case.tags
    notes = [n for n in [case.notes] if n]
    if case.truncate_after is not None:
        notes.append(f"Token prefix of the full render, cut right after {case.truncate_after!r} appears.")
    notes.append("History render of the official template; Qwen3 generates this exact layout.")
    rec["notes"] = " ".join(notes)
    return rec


def imported_record(r: Renderer, fx: imported.Imported) -> dict[str, Any]:
    offered = [TOOLS[n] if isinstance(n, str) else n for n in fx.tools]
    rec = base_record(fx.name, r.model, fx.provenance, offered, fx.raw, r.encode(fx.raw))
    if fx.generation_prompt is not None:
        rec["generation_prompt"] = fx.generation_prompt
    if fx.thinking is not None:
        rec["thinking"] = fx.thinking
    rec.update(fx.expected)
    rec["tags"] = fx.tags
    rec["notes"] = (fx.notes + " " if fx.notes else "") + "output_token_ids: tokenizer.encode(raw_output)."
    return rec


def family_json(renderers: Mapping[str, Renderer]) -> dict[str, Any]:
    def ref(r: Renderer, gen_prompt: str) -> dict[str, Any]:
        return {
            "repo": r.model.repo,
            "revision": r.model.revision,
            "variant": r.model.variant,
            "default_generation_prompt": gen_prompt,
            "stop_tokens": sorted(r.tok.convert_ids_to_tokens(sorted(r.stop_ids))),
        }

    return {
        "slug": SLUG,
        "name": "Qwen3 (Hermes-style JSON tool calls)",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": [
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
        "reference_models": [
            ref(renderers["hybrid"], "<|im_start|>assistant\n"),
            ref(renderers["thinking"], "<|im_start|>assistant\n<think>\n"),
            ref(renderers["instruct"], "<|im_start|>assistant\n"),
        ],
        "format_notes": f"docs/formats/{SLUG}.md",
        "notes": (
            "Expected content/reasoning are the message fields the template was given; separator newlines the "
            "template adds around <think> and before <tool_call> are not part of them (soft-v1 covers engines that "
            "keep them). Generated by scripts/fixtures/qwen3-hermes/build.py."
        ),
    }


def build() -> dict[str, str]:
    renderers = {k: Renderer(m) for k, m in MODELS.items()}
    files: dict[str, list[dict[str, Any]]] = {}
    for case in CASES:
        files.setdefault(case.file, []).append(render_case(renderers[case.model], case))
    for fx in imported.FIXTURES:
        files.setdefault(fx.file, []).append(imported_record(renderers[fx.model], fx))
    for records in files.values():
        for rec in records:
            r = next(x for x in renderers.values() if x.model.repo == rec["tokenizer"]["repo"])
            if r.decode(rec["output_token_ids"]) != rec["raw_output"]:
                raise AssertionError(f"{rec['id']}: output_token_ids do not decode to raw_output")
    outputs = {f"{name}.jsonl": jsonl(records) for name, records in files.items()}
    outputs["family.json"] = json.dumps(family_json(renderers), indent=2, ensure_ascii=False) + "\n"
    return outputs


def emit(outputs: dict[str, str], check: bool) -> int:
    """Write ``outputs`` (file name -> text) to OUT_DIR, or with ``check`` compare them to what is on disk."""
    if check:
        on_disk = {p.name: p.read_text(encoding="utf-8") for p in OUT_DIR.glob("*") if p.suffix in (".jsonl", ".json")}
        stale = sorted(n for n in on_disk.keys() | outputs.keys() if on_disk.get(n) != outputs.get(n))
        for name in stale:
            print(f"out of date: {OUT_DIR.relative_to(REPO_ROOT) / name}")
        return 1 if stale else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jsonl"):
        old.unlink()
    for name, text in outputs.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    return 0


def jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--check", action="store_true", help="regenerate in memory and fail if fixtures/ differs")
    args = ap.parse_args(argv)
    outputs = build()
    n = sum(text.count("\n") for name, text in outputs.items() if name.endswith(".jsonl"))
    rc = emit(outputs, args.check)
    verb = "checked" if args.check else "wrote"
    print(
        f"{verb} {n} fixtures in {OUT_DIR.relative_to(REPO_ROOT)}" + (" (up to date)" if args.check and not rc else "")
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
