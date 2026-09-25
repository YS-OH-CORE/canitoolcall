"""Build the ``gemma4`` fixture corpus.

Provenance of every fixture written here:

* ``template_render``: an assistant message rendered through Google's OFFICIAL
  Gemma 4 ``chat_template.jinja`` with ``apply_chat_template(tokenize=True)``.
  The prompt ids are sliced off and the output is cut at the first stop id from
  ``generation_config.json`` (``<eos>``, ``<turn|>``, ``<|tool_response>``).
  ``raw_output`` is those ids decoded with ``skip_special_tokens=False``.
  Truncated fixtures are a token prefix of such a render (what ``max_tokens``
  produces).
* ``engine_test`` / ``bug_report``: see ``imported.py``.

One generation-vs-history difference is handled explicitly: with thinking OFF
the generation prompt pre-fills an empty thought channel
(``<|turn>model\\n<|channel>thought\\n<channel|>``) that a history render of the
same turn does not contain. The output is then the history render after
``<|turn>model\\n``, which is what the model generates after that pre-fill.

Only tokenizer/template files are downloaded (never weights), at a pinned
revision. Run from the repo root inside the transformers engine venv::

    .venvs/transformers/bin/python scripts/fixtures/gemma4/build.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imported  # noqa: E402

REPO_ROOT = HERE.parents[2]
SLUG = "gemma4"
OUT_DIR = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"

REPO = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SIBLINGS = ("google/gemma-4-26B-A4B-it", "google/gemma-4-12B-it")
"""Repos whose chat_template.jinja is byte-identical to the reference (checked 2026-09-25).
The E2B/E4B templates differ, so they are not listed."""
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class Renderer:
    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
        gen_cfg = json.loads(Path(hf_hub_download(REPO, "generation_config.json", revision=REVISION)).read_text())
        self.stop_ids: set[int] = set(gen_cfg["eos_token_id"])
        template = Path(hf_hub_download(REPO, "chat_template.jinja", revision=REVISION)).read_text(encoding="utf-8")
        self.template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
        self.empty_thought_ids = self.encode(EMPTY_THOUGHT)

    def _apply(self, messages: list[dict[str, Any]], tools: Sequence[Any], gen: bool, tokenize: bool, **kw: Any) -> Any:
        return self.tok.apply_chat_template(
            messages, tools=list(tools), add_generation_prompt=gen, tokenize=tokenize, **kw
        )

    def _ids(self, messages: list[dict[str, Any]], tools: Sequence[Any], gen: bool, **kw: Any) -> list[int]:
        out = self._apply(messages, tools, gen, True, **kw)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def render(
        self, history: list[dict[str, Any]], assistant: dict[str, Any], tools: Sequence[Any], **kw: Any
    ) -> tuple[list[int], str]:
        """Return ``(output_ids, generation_prompt)``."""
        prompt = self._ids(history, tools, True, **kw)
        full = self._ids([*history, assistant], tools, False, **kw)
        gen_prompt = str(self._apply(history, tools, True, False, **kw))[
            len(str(self._apply(history, tools, False, False, **kw))) :
        ]
        start = len(prompt)
        if full[:start] != prompt:
            # Thinking off: the prompt ends with an empty thought channel the history render lacks.
            n = len(self.empty_thought_ids)
            if not (gen_prompt.endswith(EMPTY_THOUGHT) and prompt[-n:] == self.empty_thought_ids):
                raise AssertionError("generation prompt is not a prefix of the rendered turn")
            start -= n
            if full[:start] != prompt[:start]:
                raise AssertionError("rendered turn diverges before the pre-filled thought channel")
        out = full[start:]
        cut = next(i for i, t in enumerate(out) if t in self.stop_ids)
        return out[:cut], gen_prompt

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.tok.decode(list(ids), skip_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        ids = list(self.tok.encode(text, add_special_tokens=False))
        if self.decode(ids) != text:
            raise AssertionError(f"lossy round trip for {text!r}")
        return ids

    def truncate(self, ids: list[int], after: str) -> list[int]:
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
        _tool("search", "Search the web.", {"query": {"type": "string"}, "filters": {"type": "object"}}, ["query"]),
        _tool("get_time", "Get the current UTC time. Takes no arguments.", {}),
        _tool(
            "create_event",
            "Create a calendar event.",
            {
                "title": {"type": "string"},
                "when": {"type": "object"},
                "attendees": {"type": "array", "items": {"type": "object"}},
                "reminders": {"type": "array"},
                "location": {"anyOf": [{"type": "object"}, {"type": "null"}], "nullable": True},
            },
            ["title", "when"],
        ),
        _tool("translate", "Translate text.", {"text": {"type": "string"}, "target": {"type": "string"}}, ["text"]),
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
                "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}], "nullable": True},
                "values": {"type": "array", "items": {"type": "number"}},
            },
            ["x", "y"],
        ),
        _tool(
            "set_labels",
            "Attach labels to an item.",
            {"item": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}}},
            ["item", "labels"],
        ),
        _tool(
            "record",
            "Store a record with arbitrary keys.",
            {
                "key with space": {"type": "number"},
                "s": {"type": "string"},
                "flag": {"type": "boolean"},
                "none": {"anyOf": [{"type": "string"}, {"type": "null"}], "nullable": True},
                "obj": {"type": "object"},
            },
        ),
        _tool(
            "set_fields",
            "Set fields whose names differ only by case.",
            {
                "b": {"type": "integer"},
                "A": {"type": "integer"},
                "a_c": {"type": "string"},
                "Zeta": {"type": "boolean"},
            },
        ),
        _tool(
            "apply_edits",
            "Apply text edits to a file.",
            {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                        "required": ["oldText", "newText"],
                    },
                },
            },
            ["path", "edits"],
        ),
    ]
}

_SUBMIT_VALUES: dict[str, Any] = {f"k{i:02d}": f"v{i:02d}" for i in range(45)}
_SUBMIT_VALUES["zz_members"] = ["a", "b"]
TOOLS["submit"] = _tool(
    "submit",
    "Submit all provided fields.",
    {
        **{k: {"type": "string"} for k in _SUBMIT_VALUES if k != "zz_members"},
        "zz_members": {"type": "array", "items": {"type": "string"}},
    },
    list(_SUBMIT_VALUES),
)


def tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


# --------------------------------------------------------------------------- cases

Call = tuple[str, dict[str, Any]]
STD_CALLS: list[Call] = [
    ("get_weather", {"city": "Zürich", "unit": "c"}),
    ("search", {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}),
]
LONG_CONTENT = "".join(
    f"{i:03d}: The quick brown fox jumps over the lazy dog; café, naïve, 東京.\n" for i in range(1, 61)
)


@dataclass
class Case:
    name: str
    file: str
    user: str
    tools: list[str]
    calls: list[Call] = field(default_factory=list)
    content: str | None = None
    reasoning: str | None = None
    thinking: bool = False
    tags: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    truncate_after: str | None = None
    expected: dict[str, Any] | None = None
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
    msgs = [assistant_msg(calls, None, reasoning)]
    msgs += [
        {"role": "tool", "tool_call_id": f"call_{i}", "name": calls[i][0], "content": r} for i, r in enumerate(results)
    ]
    return msgs


TRUNCATED = {
    "expected_error": {
        "reason": "Output cut by max_tokens inside the tool call: no complete call exists.",
        "accept": ["no_tool_calls", "content_passthrough", "exception"],
    }
}

CASES: list[Case] = [
    # ---- basic
    Case(
        "single-call",
        "basic",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        tags=["single-call"],
    ),
    Case(
        "single-call-thinking",
        "basic",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        reasoning="The user wants the current weather in Paris. I'll call get_weather.",
        thinking=True,
        tags=["single-call", "reasoning"],
    ),
    Case(
        "no-call",
        "basic",
        "Hi there!",
        ["get_weather"],
        content="Hello! How can I help you today?",
        tags=["no-call"],
    ),
    Case(
        "no-call-thinking",
        "basic",
        "Hi there!",
        ["get_weather"],
        content="Hello! How can I help you today?",
        reasoning="The user is just greeting me. No tool is needed.",
        thinking=True,
        tags=["no-call", "reasoning"],
    ),
    Case(
        "text-after-call",
        "basic",
        "Is it raining in London?",
        ["get_weather"],
        [("get_weather", {"city": "London"})],
        content="Let me check the current conditions in London.",
        tags=["single-call", "text-after-call"],
        notes="The official template renders an assistant message's text content AFTER its tool calls.",
    ),
    Case(
        "text-after-call-thinking",
        "basic",
        "Is it raining in London?",
        ["get_weather"],
        [("get_weather", {"city": "London"})],
        content="Let me check the current conditions in London.",
        reasoning="I need live weather data for London.",
        thinking=True,
        tags=["single-call", "text-after-call", "reasoning"],
    ),
    Case(
        "empty-arguments",
        "basic",
        "What time is it?",
        ["get_time"],
        [("get_time", {})],
        tags=["single-call", "empty-arguments"],
    ),
    Case(
        "empty-arguments-thinking",
        "basic",
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
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS,
        reasoning="I should call the tools.",
        thinking=True,
        tags=["parallel-calls", "reasoning", "unicode", "nested-json", "string-escapes"],
    ),
    Case(
        "parallel-three-calls",
        "parallel",
        "Weather in Paris, London and Tokyo?",
        ["get_weather"],
        [
            ("get_weather", {"city": "Paris"}),
            ("get_weather", {"city": "London"}),
            ("get_weather", {"city": "Tokyo", "unit": "c"}),
        ],
        tags=["parallel-calls"],
    ),
    Case(
        "parallel-mixed-with-empty",
        "parallel",
        "What time is it, and what's the weather in Oslo?",
        ["get_time", "get_weather"],
        [("get_time", {}), ("get_weather", {"city": "Oslo", "unit": "c"})],
        tags=["parallel-calls", "empty-arguments"],
    ),
    # ---- edge values
    Case(
        "nested-objects-and-arrays",
        "edge",
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
        tags=["single-call", "nested-json"],
        notes="Nested keys are unquoted; nested lists of objects and a null value.",
    ),
    Case(
        "edge-values-key-with-space",
        "edge",
        "Store the record.",
        ["record"],
        [
            (
                "record",
                {
                    "key with space": 1.5e-5,
                    "s": 'x,y} "q"',
                    "flag": False,
                    "none": None,
                    "obj": {"B": 1, "a": [1, "t"]},
                },
            )
        ],
        tags=[
            "single-call",
            "numeric-arguments",
            "string-escapes",
            "nested-json",
            "x-key-with-space",
            "x-dictsort-order",
        ],
        notes=(
            "Same input as the edge-value render in docs/formats/gemma4.md: a key containing spaces (unquoted), "
            "a Python-formatted float (1.5e-05), ',', '}' and '\"' inside a string, and case-insensitive key order."
        ),
    ),
    Case(
        "dictsort-case-insensitive-order",
        "edge",
        "Set the fields.",
        ["set_fields"],
        [("set_fields", {"b": 1, "A": 2, "a_c": "x", "Zeta": True})],
        tags=["single-call", "x-dictsort-order"],
        notes="Jinja dictsort orders keys case-insensitively (A, a_c, b, Zeta), not in insertion order.",
    ),
    Case(
        "marker-in-arguments",
        "edge",
        "Echo the markup back to me.",
        ["echo"],
        [("echo", {"text": "Wrap calls as <|tool_call>call:f{}<tool_call|> and close thoughts with <channel|>."})],
        tags=["single-call", "marker-in-arguments"],
        notes=(
            "The string contains the text of the call/channel markers, which the official tokenizer maps to their "
            'special-token ids. Inside a <|"|>-delimited string they are literal text.'
        ),
    ),
    Case(
        "unicode-emoji",
        "edge",
        "Translate this greeting to German.",
        ["translate"],
        [("translate", {"text": "こんにちは 🌸 مرحبا — naïve café 👩‍💻", "target": "de"})],
        tags=["single-call", "unicode"],
    ),
    Case(
        "numeric-arguments",
        "edge",
        "Compute it.",
        ["calculate"],
        [
            (
                "calculate",
                {"x": -3, "y": 2.5, "tolerance": 1e-05, "exact": True, "limit": None, "values": [0, -0.5, 1e21]},
            )
        ],
        tags=["single-call", "numeric-arguments", "x-python-float"],
        notes="Numbers are rendered with Python str(): 1e-05 and 1e+21 are not how json.dumps writes them everywhere.",
    ),
    Case(
        "string-with-newlines-and-backslashes",
        "edge",
        "Save a hello-world script.",
        ["write_file"],
        [
            (
                "write_file",
                {
                    "path": "C:\\Users\\me\\hello.py",
                    "content": 'def main():\n\tprint("hi {name}")\n\treturn {"ok": True}\n',
                },
            )
        ],
        tags=["single-call", "string-escapes"],
        notes='Strings are not escaped: raw newlines, tabs, backslashes, quotes and braces sit between <|"|> tokens.',
    ),
    Case(
        "json-text-in-string",
        "edge",
        "Echo this JSON text.",
        ["echo"],
        [("echo", {"text": '{"name": "x", "arguments": {"a": 1}}'})],
        tags=["single-call", "string-escapes", "x-json-in-string"],
    ),
    Case(
        "array-of-strings",
        "edge",
        "Label the ticket.",
        ["set_labels"],
        [("set_labels", {"item": "TICKET-42", "labels": ["bug", "p1", "needs triage", "a,b"]})],
        tags=["single-call", "x-array-of-strings"],
    ),
    Case(
        "long-arguments",
        "edge",
        "Write the sample file.",
        ["write_file"],
        [("write_file", {"path": "notes/sample.txt", "content": LONG_CONTENT})],
        tags=["single-call", "long-arguments", "unicode"],
    ),
    Case(
        "array-of-objects-with-braces",
        "edge",
        "Rename the function in hello.py.",
        ["apply_edits"],
        [
            (
                "apply_edits",
                {
                    "path": "hello.py",
                    "edits": [
                        {"oldText": "def hello() {", "newText": "def greet() {"},
                        {"oldText": 'print(f"{name}")', "newText": 'print(f"Hi {name}!")'},
                    ],
                },
            )
        ],
        tags=["single-call", "nested-json", "string-escapes", "regression"],
        notes=(
            "Array parameter whose string values contain '{' and '}' "
            "(cf. https://github.com/ggml-org/llama.cpp/issues/21384: the array came back as a JSON string)."
        ),
    ),
    Case(
        "many-strings-then-string-array",
        "edge",
        "Submit these values.",
        ["submit"],
        [("submit", _SUBMIT_VALUES)],
        tags=["single-call", "long-arguments", "regression"],
        notes=(
            "45 string values followed by a two-string array: the exact input of "
            "https://github.com/ollama/ollama/issues/18354 (string-placeholder collision dropped the call)."
        ),
    ),
    # ---- multi-turn (the next call continues the same model turn after <tool_response|>)
    Case(
        "multi-turn-final-answer",
        "multiturn",
        "Weather in Zürich?",
        ["get_weather"],
        content="It is 20 °C in Zürich right now.",
        history=tool_turn([("get_weather", {"city": "Zürich", "unit": "c"})], ['{"temp": 20}']),
        tags=["no-call", "multi-turn", "unicode"],
        notes="Thinking off: the model continues the same turn right after <tool_response|>.",
    ),
    Case(
        "multi-turn-second-call",
        "multiturn",
        "Weather in Zürich, then find a café there.",
        ["get_weather", "search"],
        [("search", {"query": "café Zürich", "filters": {"open_now": True}})],
        history=tool_turn([("get_weather", {"city": "Zürich", "unit": "c"})], ['{"temp": 20}']),
        tags=["single-call", "multi-turn", "unicode", "nested-json"],
    ),
    # ---- truncated
    Case(
        "truncated-mid-string",
        "truncated",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris", "unit": "c"})],
        truncate_after="Par",
        expected=TRUNCATED,
        tags=["truncated"],
    ),
    Case(
        "truncated-in-name",
        "truncated",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris"})],
        truncate_after="call:get",
        expected=TRUNCATED,
        tags=["truncated"],
    ),
    Case(
        "truncated-before-close",
        "truncated",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris"})],
        truncate_after='Paris<|"|>',
        expected=TRUNCATED,
        tags=["truncated"],
        notes="Cut before the closing '}' and <tool_call|>.",
    ),
    Case(
        "truncated-in-reasoning",
        "truncated",
        "What's the weather in Paris?",
        ["get_weather"],
        [("get_weather", {"city": "Paris"})],
        reasoning="The user wants the current weather in Paris, so I will call the weather tool.",
        thinking=True,
        truncate_after="current weather",
        expected={
            "expected": {"content": None, "reasoning_content": "The user wants the current weather", "tool_calls": []}
        },
        tags=["truncated", "reasoning"],
        notes="Cut inside the thought channel (no <channel|>): the partial text is reasoning (cf. vLLM #49717).",
    ),
    Case(
        "truncated-second-parallel-call",
        "truncated",
        "Weather in Zürich, and search for the best café?",
        ["get_weather", "search"],
        STD_CALLS,
        truncate_after="query:",
        expected={
            "expected": {
                "content": None,
                "reasoning_content": None,
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}}],
            }
        },
        tags=["truncated", "parallel-calls", "unicode"],
        notes="The first call is complete; the second is cut. A correct parser keeps the complete call only.",
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
    fid: str, provenance: dict[str, Any], offered: list[dict[str, Any]], raw: str, ids: list[int]
) -> dict[str, Any]:
    return {
        "id": f"{SLUG}/{fid}",
        "family": SLUG,
        "models": [REPO, *SIBLINGS],
        "spec_version": "0.1",
        "provenance": provenance,
        "tools": offered,
        "raw_output": raw,
        "output_token_ids": ids,
        "tokenizer": {"repo": REPO, "revision": REVISION, "mode": "hf"},
    }


def render_case(r: Renderer, case: Case) -> dict[str, Any]:
    offered = tools(*case.tools)
    history = [{"role": "user", "content": case.user}, *case.history]
    ids, gen_prompt = r.render(
        history, assistant_msg(case.calls, case.content, case.reasoning), offered, enable_thinking=case.thinking
    )
    if case.truncate_after is not None:
        ids = r.truncate(ids, case.truncate_after)
    rec = base_record(
        case.name,
        {
            "kind": "template_render",
            "source_url": f"https://huggingface.co/{REPO}/blob/{REVISION}/chat_template.jinja",
            "revision": REVISION,
            "license": "Apache-2.0",
            "generator": GENERATOR,
            "template_sha256": r.template_sha256,
        },
        offered,
        r.decode(ids),
        ids,
    )
    rec["generation_prompt"] = gen_prompt
    rec["thinking"] = case.thinking
    rec.update(case.expected or expected_from(case))
    rec["tags"] = case.tags
    notes = [n for n in [case.notes] if n]
    if case.truncate_after is not None:
        notes.append(f"Token prefix of the full render, cut right after {case.truncate_after!r} appears.")
    notes.append("History render of the official template; generation stops on <|tool_response>/<turn|>.")
    rec["notes"] = " ".join(notes)
    return rec


def imported_record(r: Renderer, fx: imported.Imported) -> dict[str, Any]:
    offered = [TOOLS[n] if isinstance(n, str) else n for n in fx.tools]
    rec = base_record(fx.name, fx.provenance, offered, fx.raw, r.encode(fx.raw))
    rec["generation_prompt"] = fx.generation_prompt
    rec["thinking"] = fx.thinking
    rec.update(fx.expected)
    rec["tags"] = fx.tags
    rec["notes"] = (fx.notes + " " if fx.notes else "") + "output_token_ids: tokenizer.encode(raw_output)."
    return rec


def family_json(r: Renderer) -> dict[str, Any]:
    return {
        "slug": SLUG,
        "name": "Gemma 4 (<|tool_call>call:NAME{...} object notation)",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": [
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            '<|"|>',
            "<|channel>",
            "<channel|>",
            "<|turn>",
            "<turn|>",
            "<|think|>",
        ],
        "reference_models": [
            {
                "repo": REPO,
                "revision": REVISION,
                "variant": "gemma4",
                "default_generation_prompt": "<|turn>model\n" + EMPTY_THOUGHT,
                "stop_tokens": [r.tok.convert_ids_to_tokens(i) for i in sorted(r.stop_ids)],
            }
        ],
        "format_notes": f"docs/formats/{SLUG}.md",
        "notes": (
            "default_generation_prompt is the thinking-off prompt (the template default). With enable_thinking=true "
            "the prompt is '<|turn>model\\n' and the model opens the thought channel itself; each fixture records its "
            "own generation_prompt and thinking flag. Generated by scripts/fixtures/gemma4/build.py."
        ),
    }


def build() -> dict[str, str]:
    r = Renderer()
    files: dict[str, list[dict[str, Any]]] = {}
    for case in CASES:
        files.setdefault(case.file, []).append(render_case(r, case))
    for fx in imported.FIXTURES:
        files.setdefault(fx.file, []).append(imported_record(r, fx))
    for records in files.values():
        for rec in records:
            if r.decode(rec["output_token_ids"]) != rec["raw_output"]:
                raise AssertionError(f"{rec['id']}: output_token_ids do not decode to raw_output")
    outputs = {f"{name}.jsonl": jsonl(records) for name, records in files.items()}
    outputs["family.json"] = json.dumps(family_json(r), indent=2, ensure_ascii=False) + "\n"
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
