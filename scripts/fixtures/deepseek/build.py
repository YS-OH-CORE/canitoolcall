# ruff: noqa: RUF001, RUF002  (DeepSeek markers use U+FF5C and U+2581 on purpose)
"""Build the ``deepseek`` fixture corpus (five incompatible sub-formats).

Sub-formats and their official renderers:

=========  ==========================  ===============================================
variant    reference model             official renderer
=========  ==========================  ===============================================
v3         DeepSeek-V3-0324            ``tokenizer_config.json`` Jinja chat template
v31        DeepSeek-V3.1               ``tokenizer_config.json`` Jinja chat template
v32        DeepSeek-V3.2               ``encoding/encoding_dsv32.py`` (DSML ``function_calls``)
v4         DeepSeek-V4-Flash           ``encoding/encoding_dsv4.py`` (DSML ``tool_calls``)
v41        DeepSeek-V4.1-Flash         ``encoding/encoding.py`` (DSML `` calls``)
=========  ==========================  ===============================================

From V3.2 on there is no Jinja template; DeepSeek ships a Python encoder, which
is imported from the pinned HF revision and run as-is. Each encoder also ships
``parse_message_from_completion_text``, DeepSeek's reference parser. Every
well-formed DSML render here is fed back through it and must reproduce the
fixture's ``expected`` exactly, so the expected values are checked against the
model vendor's own parser, not against any engine.

Rendering: the prompt (history ending with the user turn) and the full
conversation (history + the assistant turn) are rendered; the prompt must be a
prefix of the full render, in text and in token ids. The output is the
remainder, cut at the first ``<｜end▁of▁sentence｜>`` (the only stop id). Token
ids come from the official tokenizer at the same revision. Truncated fixtures
are a token prefix of such a render (what ``max_tokens`` produces).

Only tokenizer/template/encoder files are downloaded (never weights). Run from
the repo root inside the transformers engine venv::

    .venvs/transformers/bin/python scripts/fixtures/deepseek/build.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imported  # noqa: E402

REPO_ROOT = HERE.parents[2]
SLUG = "deepseek"
OUT_DIR = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"
EOS = "<｜end▁of▁sentence｜>"
SYSTEM = "You are a helpful assistant."


@dataclass(frozen=True)
class Variant:
    key: str
    repo: str
    revision: str
    source_file: str
    """Template or encoder file (repo-relative) that defines the format."""
    siblings: tuple[str, ...] = ()
    """Repos whose template/encoder is byte-identical (checked 2026-09-25)."""

    @property
    def is_encoder(self) -> bool:
        return self.source_file.endswith(".py")

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/blob/{self.revision}/{self.source_file}"


VARIANTS = {
    v.key: v
    for v in [
        Variant(
            "v3", "deepseek-ai/DeepSeek-V3-0324", "e9b33add76883f293d6bf61f6bd89b497e80e335", "tokenizer_config.json"
        ),
        Variant(
            "v31",
            "deepseek-ai/DeepSeek-V3.1",
            "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
            "tokenizer_config.json",
            ("deepseek-ai/DeepSeek-V3.1-Terminus",),
        ),
        Variant(
            "v32", "deepseek-ai/DeepSeek-V3.2", "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6", "encoding/encoding_dsv32.py"
        ),
        Variant(
            "v4",
            "deepseek-ai/DeepSeek-V4-Flash",
            "60d8d70770c6776ff598c94bb586a859a38244f1",
            "encoding/encoding_dsv4.py",
            ("deepseek-ai/DeepSeek-V4-Pro",),
        ),
        Variant(
            "v41", "deepseek-ai/DeepSeek-V4.1-Flash", "dba1be0a40aa45a94ad051997016db3960a90277", "encoding/encoding.py"
        ),
    ]
}
LABELS = {
    "v3": "deepseek-v3",
    "v31": "deepseek-v31",
    "v32": "deepseek-v32",
    "v4": "deepseek-v4",
    "v41": "deepseek-v41",
}


def _load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Renderer:
    def __init__(self, v: Variant) -> None:
        self.v = v
        self.tok = AutoTokenizer.from_pretrained(v.repo, revision=v.revision)
        self.eos_id = int(self.tok.convert_tokens_to_ids(EOS))
        src = hf_hub_download(v.repo, v.source_file, revision=v.revision)
        if v.is_encoder:
            self.encoder: ModuleType | None = _load_module(src, f"ds_encoder_{v.key}")
            self.source_sha256 = hashlib.sha256(Path(src).read_bytes()).hexdigest()
        else:
            self.encoder = None
            self.source_sha256 = hashlib.sha256(self.tok.chat_template.encode("utf-8")).hexdigest()

    # -- token helpers
    def decode(self, ids: Sequence[int]) -> str:
        return str(self.tok.decode(list(ids), skip_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        ids = list(self.tok.encode(text, add_special_tokens=False))
        if self.decode(ids) != text:
            raise AssertionError(f"{self.v.repo}: lossy round trip for {text!r}")
        return ids

    def truncate(self, ids: list[int], after: str) -> list[int]:
        for n in range(1, len(ids) + 1):
            if after in self.decode(ids[:n]):
                return ids[:n]
        raise AssertionError(f"{after!r} not found in render")

    # -- rendering
    def render(
        self, history: list[dict[str, Any]], assistant: dict[str, Any], tools: list[dict[str, Any]], thinking: bool
    ) -> tuple[list[int], str]:
        """Return ``(output_ids, generation_prompt)``."""
        if self.encoder is None:
            return self._render_jinja(history, assistant, tools, thinking)
        mode = "thinking" if thinking else "chat"
        msgs = [{"role": "system", "content": SYSTEM, "tools": tools}, *history]
        prompt = self.encoder.encode_messages(msgs, thinking_mode=mode)
        full = self.encoder.encode_messages([*msgs, assistant], thinking_mode=mode)
        if not full.startswith(prompt):
            raise AssertionError(f"{self.v.key}: prompt is not a prefix of the rendered conversation")
        p_ids, f_ids = self.encode(prompt), self.encode(full)
        if f_ids[: len(p_ids)] != p_ids:
            raise AssertionError(f"{self.v.key}: prompt token ids are not a prefix")
        out = f_ids[len(p_ids) :]
        out = out[: out.index(self.eos_id)]
        gen = next(
            g
            for g in ("<｜Assistant｜><think>", "<｜Assistant｜></think>", "\n\n<think>", "\n\n</think>")
            if prompt.endswith(g)
        )
        return out, gen

    def _render_jinja(
        self, history: list[dict[str, Any]], assistant: dict[str, Any], tools: list[dict[str, Any]], thinking: bool
    ) -> tuple[list[int], str]:
        kw: dict[str, Any] = {"thinking": thinking} if self.v.key == "v31" else {}

        def ids(msgs: list[dict[str, Any]], gen: bool) -> list[int]:
            out = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=True, **kw)
            return list(out["input_ids"] if hasattr(out, "keys") else out)

        def text(msgs: list[dict[str, Any]], gen: bool) -> str:
            return str(self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=False, **kw))

        prompt, full = ids(history, True), ids([*history, assistant], False)
        if full[: len(prompt)] != prompt:
            raise AssertionError(f"{self.v.key}: generation prompt is not a prefix of the rendered turn")
        out = full[len(prompt) :]
        out = out[: out.index(self.eos_id)]
        # The Jinja templates append the assistant header inside the user turn, so report the header itself.
        p_text = text(history, True)
        gen = next(g for g in ("<｜Assistant｜></think>", "<｜Assistant｜>", "") if p_text.endswith(g))
        return out, gen

    def oracle(self, raw: str, thinking: bool) -> dict[str, Any] | None:
        """DeepSeek's reference parser (encoders only), in fixture ``expected`` shape."""
        if self.encoder is None:
            return None
        msg = self.encoder.parse_message_from_completion_text(
            raw + EOS, thinking_mode="thinking" if thinking else "chat"
        )
        return {
            "content": msg["content"] or None,
            "reasoning_content": msg["reasoning_content"] or None,
            "tool_calls": [
                {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                for tc in msg["tool_calls"]
            ],
        }


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
                "limit": {"type": ["integer", "null"]},
                "values": {"type": "array", "items": {"type": "number"}},
                "code": {"type": "string"},
            },
            ["x", "y"],
        ),
        _tool(
            "create_event",
            "Create a calendar event.",
            {
                "title": {"type": "string"},
                "when": {"type": "object"},
                "attendees": {"type": "array", "items": {"type": "object"}},
            },
            ["title", "when"],
        ),
    ]
}


def tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


# --------------------------------------------------------------------------- cases

Call = tuple[str, dict[str, Any]]
STD_CALLS: list[Call] = [
    ("get_weather", {"city": "Zürich", "unit": "c"}),
    ("search", {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}),
]
NUMERIC_ARGS: dict[str, Any] = {
    "x": -3,
    "y": 2.5,
    "tolerance": 1e-05,
    "exact": True,
    "limit": None,
    "values": [0, -0.5, 1e21],
    "code": "007",
}
EVENT_ARGS: dict[str, Any] = {
    "title": "Design review",
    "when": {"start": "2026-10-01T09:00:00Z", "end": "2026-10-01T10:00:00Z"},
    "attendees": [{"name": "Ana", "optional": False}, {"name": "Bo", "optional": True, "roles": ["notes"]}],
}
CODE = 'if a < b and c > d:\n    print("x=\\"1\\"")  # tab\there\n'
LONG_CONTENT = "".join(f"{i:03d}: The quick brown fox jumps over the lazy dog; café, 東京.\n" for i in range(1, 41))
MARKER_TEXT = {
    "v3": 'Reply as ```json\n{"a": 1}\n``` or use <｜tool▁sep｜> tokens.',
    "v31": "Tokens like <｜tool▁call▁begin｜>f<｜tool▁sep｜>{} are markup.",
    "v32": 'Write <｜DSML｜invoke name="x"> to call a tool.',
    "v4": 'Write <｜DSML｜invoke name="x"> inside <｜DSML｜tool_calls> to call a tool.',
    "v41": 'Write <｜DSML｜ invoke name="x"> inside <｜DSML｜ calls> to call a tool.',
}


@dataclass
class Case:
    name: str
    file: str
    variant: str
    user: str
    tools: list[str]
    calls: list[Call] = field(default_factory=list)
    content: str | None = None
    reasoning: str | None = None
    thinking: bool = False
    tags: list[str] = field(default_factory=list)
    history: list[Call] = field(default_factory=list)
    """Earlier tool calls (one assistant turn) whose results precede the rendered turn."""
    truncate_after: str | None = None
    expected: dict[str, Any] | None = None
    notes: str | None = None


def cases_for(v: str) -> list[Case]:
    """The shared case matrix, instantiated for one sub-format."""
    dsml = v in ("v32", "v4", "v41")
    think = dsml  # V3/V3.1 templates cannot render reasoning with tool calls (see docs/formats/deepseek.md)
    f = v
    out = [
        Case(
            f"{v}-single-call",
            f,
            v,
            "What's the weather in Paris?",
            ["get_weather"],
            [("get_weather", {"city": "Paris", "unit": "c"})],
            tags=["single-call"],
        ),
        Case(
            f"{v}-parallel-two-calls",
            f,
            v,
            "Weather in Zürich, and search for the best café?",
            ["get_weather", "search"],
            STD_CALLS,
            tags=["parallel-calls", "unicode", "nested-json", "string-escapes"],
        ),
        Case(
            f"{v}-no-call",
            f,
            v,
            "Hi there!",
            ["get_weather"],
            content="Hello! How can I help you today?",
            tags=["no-call"],
        ),
        Case(
            f"{v}-text-before-call",
            f,
            v,
            "Is it raining in London?",
            ["get_weather"],
            [("get_weather", {"city": "London"})],
            content="Let me check the current conditions in London.",
            tags=["single-call", "text-before-call"],
        ),
        Case(
            f"{v}-empty-arguments",
            f,
            v,
            "What time is it?",
            ["get_time"],
            [("get_time", {})],
            tags=["single-call", "empty-arguments"],
        ),
        Case(
            f"{v}-numeric-arguments",
            f,
            v,
            "Compute it.",
            ["calculate"],
            [("calculate", NUMERIC_ARGS)],
            tags=["single-call", "numeric-arguments"],
            notes="'code' is the STRING \"007\"; numbers, booleans and null must keep their JSON types.",
        ),
        Case(
            f"{v}-unicode-emoji",
            f,
            v,
            "Translate this greeting to German.",
            ["translate"],
            [("translate", {"text": "こんにちは 🌸 مرحبا — naïve café 👩‍💻", "target": "de"})],
            tags=["single-call", "unicode"],
        ),
        Case(
            f"{v}-marker-in-arguments",
            f,
            v,
            "Echo the markup back to me.",
            ["echo"],
            [("echo", {"text": MARKER_TEXT[v]})],
            tags=["single-call", "marker-in-arguments"],
            notes="The argument contains format-marker text, which the official tokenizer maps to special-token ids.",
        ),
        Case(
            f"{v}-truncated-mid-arguments",
            f,
            v,
            "Weather in Zürich, and search for the best café?",
            ["get_weather", "search"],
            STD_CALLS[:1],
            truncate_after="Zür",
            expected={
                "expected_error": {
                    "reason": "Output cut by max_tokens inside the first call's arguments: no complete call exists.",
                    "accept": ["no_tool_calls", "content_passthrough", "exception"],
                }
            },
            tags=["truncated"],
        ),
    ]
    if v in ("v31", "v4", "v41"):
        out.append(
            Case(
                f"{v}-multi-turn-after-tool-result",
                f,
                v,
                "Weather in Zürich, then find a café there.",
                ["get_weather", "search"],
                [("search", {"query": "café Zürich"})],
                reasoning="It is 20C. Now the café search." if think else None,
                thinking=think,
                history=[("get_weather", {"city": "Zürich", "unit": "c"})],
                tags=["single-call", "multi-turn", "unicode"] + (["reasoning", "reasoning-prefilled"] if think else []),
            )
        )
    if dsml:
        out += [
            Case(
                f"{v}-reasoning-single-call",
                f,
                v,
                "What's the weather in Paris?",
                ["get_weather"],
                [("get_weather", {"city": "Paris", "unit": "c"})],
                reasoning="The user wants the current weather in Paris. I'll call get_weather.",
                thinking=True,
                tags=["single-call", "reasoning", "reasoning-prefilled"],
                notes="Thinking mode: the prompt ends with <think>, so the output holds only </think>.",
            ),
            Case(
                f"{v}-reasoning-parallel-text-before",
                f,
                v,
                "Weather in Zürich, and search for the best café?",
                ["get_weather", "search"],
                STD_CALLS,
                content="I'll look both up.",
                reasoning="Two independent lookups: weather and search.",
                thinking=True,
                tags=[
                    "parallel-calls",
                    "reasoning",
                    "reasoning-prefilled",
                    "text-before-call",
                    "unicode",
                    "nested-json",
                    "string-escapes",
                ],
            ),
            Case(
                f"{v}-reasoning-no-call",
                f,
                v,
                "What is 2 + 2?",
                ["get_weather"],
                content="2 + 2 = 4.",
                reasoning="Simple arithmetic, no tool needed.",
                thinking=True,
                tags=["no-call", "reasoning", "reasoning-prefilled"],
            ),
            Case(
                f"{v}-string-false-json",
                f,
                v,
                "Set up the design review.",
                ["create_event"],
                [("create_event", EVENT_ARGS)],
                tags=["single-call", "nested-json", "x-string-false-json"],
                notes='Objects and arrays are JSON inside string="false" parameters.',
            ),
            Case(
                f"{v}-unescaped-string-value",
                f,
                v,
                "Save the snippet.",
                ["write_file"],
                [("write_file", {"path": "a<b>.py", "content": CODE})],
                tags=["single-call", "string-escapes", "x-unescaped-string"],
                notes='string="true" values are raw: quotes, <, >, backslashes, tabs and newlines appear unescaped.',
            ),
            Case(
                f"{v}-truncated-in-reasoning",
                f,
                v,
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
                tags=["truncated", "reasoning", "reasoning-prefilled"],
                notes="Cut before </think>: the partial text is reasoning.",
            ),
            Case(
                f"{v}-truncated-second-parallel-call",
                f,
                v,
                "Weather in Zürich, and search for the best café?",
                ["get_weather", "search"],
                STD_CALLS,
                truncate_after='name="query"',
                expected={
                    "expected": {
                        "content": None,
                        "reasoning_content": None,
                        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}}],
                    }
                },
                tags=["truncated", "parallel-calls", "unicode"],
                notes="The first invoke is complete, the second is cut: a correct parser keeps only the first call.",
            ),
        ]
    if v == "v41":
        out += [
            Case(
                f"{v}-long-arguments",
                f,
                v,
                "Write the sample file.",
                ["write_file"],
                [("write_file", {"path": "notes/sample.txt", "content": LONG_CONTENT})],
                tags=["single-call", "long-arguments", "unicode"],
            ),
            Case(
                f"{v}-whitespace-string-value",
                f,
                v,
                "Echo it with the padding.",
                ["echo"],
                [("echo", {"text": "  padded\n"})],
                tags=["single-call", "string-escapes", "x-whitespace-in-string"],
                notes='Leading/trailing whitespace inside a string="true" value is part of the value.',
            ),
        ]
    if v in ("v3", "v31"):
        out.append(
            Case(
                f"{v}-nested-json",
                f,
                v,
                "Set up the design review.",
                ["create_event"],
                [("create_event", EVENT_ARGS)],
                tags=["single-call", "nested-json"],
            )
        )
    return out


CASES: list[Case] = [c for v in VARIANTS for c in cases_for(v)]


# --------------------------------------------------------------------------- records


def openai_calls(calls: Sequence[Call]) -> list[dict[str, Any]]:
    return [
        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
        for i, (n, a) in enumerate(calls)
    ]


def assistant_msg(case: Case) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": case.content}
    if case.reasoning is not None:
        msg["reasoning_content"] = case.reasoning
    if case.calls:
        msg["tool_calls"] = openai_calls(case.calls)
    if msg["content"] is None and not case.calls:
        msg["content"] = ""
    return msg


def history_msgs(case: Case) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": case.user}]
    if case.history:
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": openai_calls(case.history),
                "reasoning_content": "First, the weather." if case.thinking else None,
            }
        )
        msgs += [
            {"role": "tool", "tool_call_id": f"call_{i}", "content": '{"temp": 20}'} for i in range(len(case.history))
        ]
    return msgs


def expected_from(case: Case) -> dict[str, Any]:
    return {
        "content": case.content,
        "reasoning_content": case.reasoning,
        "tool_calls": [{"name": n, "arguments": a} for n, a in case.calls],
    }


def base_record(
    fid: str, v: Variant, provenance: dict[str, Any], offered: list[dict[str, Any]], raw: str, ids: list[int]
) -> dict[str, Any]:
    return {
        "id": f"{SLUG}/{fid}",
        "family": SLUG,
        "models": [v.repo, *v.siblings],
        "spec_version": "0.1",
        "provenance": provenance,
        "tools": offered,
        "raw_output": raw,
        "output_token_ids": ids,
        "tokenizer": {"repo": v.repo, "revision": v.revision, "mode": "hf"},
    }


def render_case(r: Renderer, case: Case) -> dict[str, Any]:
    v = r.v
    offered = tools(*case.tools)
    ids, gen_prompt = r.render(history_msgs(case), assistant_msg(case), offered, case.thinking)
    full_raw = r.decode(ids)
    notes = [n for n in [case.notes] if n]
    if case.expected is None:
        expected = expected_from(case)
        oracle = r.oracle(full_raw, case.thinking)
        if oracle is not None:
            if oracle != expected:
                raise AssertionError(f"{case.name}: DeepSeek reference parser disagrees:\n{oracle}\n!=\n{expected}")
            notes.append("expected == DeepSeek's reference parse_message_from_completion_text on this output.")
        exp: dict[str, Any] = {"expected": expected}
    else:
        exp = case.expected
    if case.truncate_after is not None:
        ids = r.truncate(ids, case.truncate_after)
        notes.append(f"Token prefix of the full render, cut right after {case.truncate_after!r} appears.")
    rec = base_record(
        case.name,
        v,
        {
            "kind": "template_render",
            "source_url": v.source_url,
            "revision": v.revision,
            "license": "MIT",
            "generator": GENERATOR,
            "template_sha256": r.source_sha256,
            "attribution": "Copyright (c) 2023 DeepSeek",
        },
        offered,
        r.decode(ids),
        ids,
    )
    rec["generation_prompt"] = gen_prompt
    rec["thinking"] = case.thinking
    rec.update(exp)
    rec["tags"] = [*case.tags, f"x-{LABELS[v.key]}"]
    renderer = "encoder" if v.is_encoder else "chat template"
    notes.append(f"Sub-format {LABELS[v.key]}; history render of the official {renderer}.")
    rec["notes"] = " ".join(notes)
    return rec


def imported_record(r: Renderer, fx: imported.Imported) -> dict[str, Any]:
    offered = [TOOLS[n] if isinstance(n, str) else n for n in fx.tools]
    rec = base_record(fx.name, r.v, fx.provenance, offered, fx.raw, r.encode(fx.raw))
    rec["generation_prompt"] = fx.generation_prompt
    rec["thinking"] = fx.thinking
    rec.update(fx.expected)
    rec["tags"] = [*fx.tags, f"x-{LABELS[r.v.key]}"]
    rec["notes"] = (fx.notes + " " if fx.notes else "") + "output_token_ids: tokenizer.encode(raw_output)."
    return rec


def family_json() -> dict[str, Any]:
    prompts = {
        "v3": "<｜Assistant｜>",
        "v31": "<｜Assistant｜></think>",
        "v32": "<｜Assistant｜></think>",
        "v4": "<｜Assistant｜></think>",
        "v41": "<｜Assistant｜></think>",
    }
    return {
        "slug": SLUG,
        "name": "DeepSeek (V3/R1, V3.1, V3.2 DSML, V4 DSML, V4.1 DSML)",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁sep｜>",
            "｜DSML｜",
            "<think>",
            "</think>",
            "<｜end▁of▁sentence｜>",
            "<｜Assistant｜>",
            "<｜User｜>",
        ],
        "reference_models": [
            {
                "repo": v.repo,
                "revision": v.revision,
                "variant": LABELS[v.key],
                "default_generation_prompt": prompts[v.key],
                "stop_tokens": [EOS],
            }
            for v in VARIANTS.values()
        ],
        "format_notes": f"docs/formats/{SLUG}.md",
        "notes": (
            "Five incompatible sub-formats; models[0] of each fixture selects the sub-format (its variant tag is "
            "x-deepseek-v3/-v31/-v32/-v4/-v41). V3.2+ have no Jinja template: DeepSeek's encoding/*.py is the "
            "official renderer, and its parse_message_from_completion_text confirms every well-formed expected value. "
            "default_generation_prompt is the non-thinking prompt; thinking fixtures record '<｜Assistant｜><think>'. "
            "Generated by scripts/fixtures/deepseek/build.py."
        ),
    }


def build() -> dict[str, str]:
    renderers = {k: Renderer(v) for k, v in VARIANTS.items()}
    files: dict[str, list[dict[str, Any]]] = {}
    for case in CASES:
        files.setdefault(case.file, []).append(render_case(renderers[case.variant], case))
    for fx in imported.FIXTURES:
        rec = imported_record(renderers[fx.variant], fx)
        oracle = renderers[fx.variant].oracle(fx.raw, fx.thinking) if fx.oracle else None
        if oracle is not None and oracle != rec.get("expected"):
            raise AssertionError(f"{fx.name}: DeepSeek reference parser disagrees: {oracle}")
        files.setdefault(fx.file, []).append(rec)
    for records in files.values():
        for rec in records:
            r = next(x for x in renderers.values() if x.v.repo == rec["tokenizer"]["repo"])
            if r.decode(rec["output_token_ids"]) != rec["raw_output"]:
                raise AssertionError(f"{rec['id']}: output_token_ids do not decode to raw_output")
    outputs = {f"{name}.jsonl": jsonl(records) for name, records in files.items()}
    outputs["family.json"] = json.dumps(family_json(), indent=2, ensure_ascii=False) + "\n"
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
