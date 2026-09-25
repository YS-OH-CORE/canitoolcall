# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["transformers==5.17.0", "jinja2>=3.1", "huggingface_hub>=1.0", "openai-harmony==0.0.8"]
# ///
"""Render gpt-oss (Harmony) fixtures through the OFFICIAL encoders.

Two renderers are used, both official:

* ``openai-harmony==0.0.8`` (OpenAI's reference Harmony encoder, the one gpt-oss was
  trained with and the one vLLM uses to synthesize parser test inputs in
  ``tests/parser/test_harmony.py::get_model_output_tokens``). We render
  ``[user, *assistant_messages]`` with ``auto_drop_analysis=False``, slice off
  ``render_conversation_for_completion([user], ASSISTANT)`` and cut at the first
  assistant stop token (``<|call|>`` / ``<|return|>``), exactly like a server does.
* the HF ``chat_template.jinja`` of openai/gpt-oss-20b (``apply_chat_template(tokenize=True)``).

Harmony history renders put the recipient in the ROLE header
(``<|start|>assistant to=functions.X<|channel|>commentary``). The model may also put it in
the channel header; those fixtures come from engine tests (see import_gpt_oss.py).

Run from the repo root::

    uv run --script scripts/fixtures/gpt-oss/render_gpt_oss.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    OUT_DIR,
    REF_REPO,
    REF_REVISION,
    STOP_IDS,
    check_ids,
    expected,
    hf_tokenizer,
    record,
    sha256_bytes,
    tools,
    write_jsonl,
)
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)

GENERATOR = "scripts/fixtures/gpt-oss/render_gpt_oss.py"
HARMONY_COMMIT = "ec7606df9e87e3d0a1fec9f50928c1e407f0c438"  # tag v0.0.8 of github.com/openai/harmony
HARMONY_SRC = f"https://github.com/openai/harmony/blob/{HARMONY_COMMIT}/src/encoding.rs"
HARMONY_RAW = f"https://raw.githubusercontent.com/openai/harmony/{HARMONY_COMMIT}/src/encoding.rs"
HF_TEMPLATE_URL = f"https://huggingface.co/{REF_REPO}/blob/{REF_REVISION}/chat_template.jinja"

END_ID = 200007  # <|end|>
MESSAGE_ID = 200008  # <|message|>

ALL_TOOLS = ("get_weather", "search", "get_time", "write_file", "set_alarm")

# ----------------------------------------------------------------------------- message DSL


@dataclass(frozen=True)
class Analysis:
    text: str


@dataclass(frozen=True)
class Preamble:
    """commentary-channel message WITHOUT a recipient: user-visible text before calls."""

    text: str


@dataclass(frozen=True)
class Final:
    text: str


@dataclass(frozen=True)
class Call:
    name: str
    args: dict[str, Any]
    content_type: str | None = "<|constrain|>json"
    compact: bool = False


Msg = Analysis | Preamble | Final | Call


@dataclass
class Case:
    name: str
    msgs: list[Msg]
    tags: list[str]
    notes: str | None = None
    tool_names: tuple[str, ...] = ALL_TOOLS
    # Keep only the first ``truncate`` output tokens (simulates max_tokens being hit).
    truncate: int | str | None = None
    truncate_expect: dict[str, Any] | None = None
    truncate_error: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _args_text(c: Call) -> str:
    if c.compact:
        return json.dumps(c.args, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(c.args, ensure_ascii=False)


def to_harmony(m: Msg) -> Message:
    if isinstance(m, Analysis):
        return Message.from_role_and_content(Role.ASSISTANT, m.text).with_channel("analysis")
    if isinstance(m, Preamble):
        return Message.from_role_and_content(Role.ASSISTANT, m.text).with_channel("commentary")
    if isinstance(m, Final):
        return Message.from_role_and_content(Role.ASSISTANT, m.text).with_channel("final")
    msg = (
        Message.from_role_and_content(Role.ASSISTANT, _args_text(m))
        .with_channel("commentary")
        .with_recipient(f"functions.{m.name}")
    )
    return msg if m.content_type is None else msg.with_content_type(m.content_type)


def expected_for(msgs: list[Msg]) -> dict[str, Any]:
    """The correct parse of a completed turn: analysis -> reasoning, preamble/final -> content,
    functions.* recipients -> tool calls. Everything after the first stop token is not generated."""
    reasoning = [m.text for m in msgs if isinstance(m, Analysis)]
    content = [m.text for m in msgs if isinstance(m, Preamble | Final)]
    calls = [(m.name, m.args) for m in msgs if isinstance(m, Call)]
    assert len(reasoning) <= 1 and len(content) <= 1, "ambiguous join; keep one of each"
    return expected(content[0] if content else None, reasoning[0] if reasoning else None, calls)


# ----------------------------------------------------------------------------- cases

LONG_CODE = "\n".join(
    [
        '"""Generated module used as a long tool argument."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    + [f'def step_{i:02d}(x: int) -> int:\n    """Return x plus {i}."""\n    return x + {i}\n' for i in range(60)]
)

WEATHER_REASONING = "The user asks for the weather in Paris. I should call get_weather."

CASES: list[Case] = [
    Case(
        "harmony-single-call",
        [Analysis(WEATHER_REASONING), Call("get_weather", {"city": "Paris", "unit": "c"})],
        ["single-call", "reasoning", "x-recipient-in-role", "x-constrain-token"],
    ),
    Case(
        "harmony-single-call-no-reasoning",
        [Call("get_weather", {"city": "Paris"})],
        ["single-call", "x-recipient-in-role", "x-constrain-token"],
        notes="The model skipped the analysis channel and started with the tool-call message.",
    ),
    Case(
        "harmony-compact-json",
        [Analysis("Need to use function get_weather."), Call("get_weather", {"city": "San Francisco"}, compact=True)],
        ["single-call", "reasoning", "x-recipient-in-role", "x-constrain-token"],
        notes="Arguments without whitespace, as in the Harmony spec's own examples.",
    ),
    Case(
        "harmony-unicode-emoji",
        [
            Analysis("用户想知道东京的天气。Call the tool 🌦️."),
            Call("search", {"query": "東京の天気 ☀️🌧️ — «prévisions» für Zürich", "filters": {"lang": "ja"}}),
        ],
        ["single-call", "reasoning", "unicode", "x-recipient-in-role"],
    ),
    Case(
        "harmony-nested-json",
        [
            Analysis("Search with structured filters."),
            Call(
                "search",
                {
                    "query": 'café "best"',
                    "filters": {
                        "tags": ["a", "b"],
                        "range": {"from": 1, "to": 3, "open": None},
                        "sort": [{"field": "rating", "desc": True}],
                        "exact": False,
                    },
                },
            ),
        ],
        ["single-call", "reasoning", "nested-json", "string-escapes", "x-recipient-in-role"],
    ),
    Case(
        "harmony-empty-arguments",
        [Analysis("Get the time."), Call("get_time", {})],
        ["single-call", "reasoning", "empty-arguments", "x-recipient-in-role"],
    ),
    Case(
        "harmony-marker-in-arguments",
        [
            Analysis("Write the Harmony cheat sheet to a file."),
            Call(
                "write_file",
                {
                    "path": "harmony.md",
                    "content": "A call ends with <|call|>; a turn with <|return|>.\n"
                    "Header: <|start|>assistant<|channel|>commentary to=functions.get_time "
                    "<|constrain|>json<|message|>{}<|end|>",
                },
            ),
        ],
        ["single-call", "reasoning", "marker-in-arguments", "x-recipient-in-role"],
        notes="The special-token strings inside the argument value are ORDINARY text tokens in "
        "output_token_ids (openai-harmony encodes message content with special tokens disallowed). "
        "A parser that works on text rather than token ids cannot tell them from real markers.",
    ),
    Case(
        "harmony-marker-in-reasoning",
        [
            Analysis("I must not write <|call|> or <|channel|>final myself; the tool call does it."),
            Call("get_time", {}),
        ],
        ["single-call", "reasoning", "empty-arguments", "x-marker-in-reasoning", "x-recipient-in-role"],
        notes="Marker strings inside the analysis text are ordinary text tokens.",
    ),
    Case(
        "harmony-preamble-then-call",
        [
            Analysis(WEATHER_REASONING),
            Preamble("Let me check the weather for you."),
            Call("get_weather", {"city": "Paris", "unit": "c"}),
        ],
        ["single-call", "reasoning", "text-before-call", "x-preamble-commentary", "x-recipient-in-role"],
        notes="A commentary message without a recipient is a user-visible preamble (Harmony spec), so it is content.",
    ),
    Case(
        "harmony-preamble-no-reasoning",
        [Preamble("Checking two sources — one moment."), Call("search", {"query": "vLLM release notes"})],
        ["single-call", "text-before-call", "x-preamble-commentary", "x-recipient-in-role"],
    ),
    Case(
        "harmony-no-call-final",
        [Analysis("A greeting. No tool is needed."), Final("Hello! How can I help you today?")],
        ["no-call", "reasoning"],
    ),
    Case(
        "harmony-final-only",
        [Final("Hi there 👋 — nothing to look up.")],
        ["no-call", "unicode"],
    ),
    Case(
        "harmony-final-json-lookalike",
        [
            Analysis("The user wants the literal JSON of a call, not a call."),
            Final('{"name": "get_weather", "arguments": {"city": "Paris"}}'),
        ],
        ["no-call", "reasoning", "x-json-in-content"],
        notes="Final-channel text that looks like a tool call must stay content.",
    ),
    Case(
        "harmony-string-escapes",
        [
            Analysis("Write a Windows path and a quoted line."),
            Call(
                "write_file",
                {"path": "C:\\Users\\me\\notes.txt", "content": 'line 1\n\t"quoted" and \\backslash\\\r\nend \u0007'},
            ),
        ],
        ["single-call", "reasoning", "string-escapes", "x-recipient-in-role"],
    ),
    Case(
        "harmony-numeric-arguments",
        [
            Analysis("Set the alarm."),
            Call("set_alarm", {"hour": 7, "minute": 30, "volume": 0.75, "repeat": True, "label": None}),
        ],
        ["single-call", "reasoning", "numeric-arguments", "x-recipient-in-role"],
    ),
    Case(
        "harmony-long-arguments",
        [Analysis("Write the module."), Call("write_file", {"path": "steps.py", "content": LONG_CODE})],
        ["single-call", "reasoning", "long-arguments", "x-recipient-in-role"],
    ),
    Case(
        "harmony-multiline-reasoning",
        [
            Analysis("Step 1: parse the request.\nStep 2: it needs live data.\n\nStep 3: call search."),
            Call("search", {"query": "weather Paris"}),
        ],
        ["single-call", "reasoning", "x-recipient-in-role"],
    ),
    Case(
        "harmony-content-type-json-text",
        [Analysis(WEATHER_REASONING), Call("get_weather", {"city": "Paris"}, content_type="json")],
        ["single-call", "reasoning", "x-recipient-in-role", "x-constrain-text"],
        notes="Content type written as plain ' json' (no <|constrain|> token), rendered by openai-harmony with "
        "content_type='json'. The HF chat template uses this spelling; vLLM's harmony parser accepts both.",
    ),
    Case(
        "harmony-no-content-type",
        [Analysis(WEATHER_REASONING), Call("get_weather", {"city": "Paris"}, content_type=None)],
        ["single-call", "reasoning", "x-recipient-in-role", "x-no-content-type"],
        notes="Tool call header without any content type, as in llama.cpp's gpt-oss tests.",
    ),
    Case(
        "harmony-truncated-in-arguments",
        [Analysis(WEATHER_REASONING), Call("get_weather", {"city": "Paris", "unit": "c"})],
        ["truncated", "reasoning", "x-recipient-in-role"],
        truncate=-4,
        truncate_error={
            "reason": "Output stopped (max_tokens) inside the tool-call JSON; there is no complete call.",
            "accept": ["no_tool_calls", "content_passthrough", "exception"],
        },
    ),
    Case(
        "harmony-truncated-in-reasoning",
        [Analysis("The user asks for the weather in Paris, so I will need to call get_weather.")],
        ["truncated", "reasoning", "no-call"],
        truncate=-6,
        notes="Output stopped (max_tokens) inside the analysis message: everything generated is reasoning.",
    ),
    Case(
        "harmony-truncated-in-header",
        [Analysis(WEATHER_REASONING), Call("get_weather", {"city": "Paris"})],
        ["truncated", "reasoning", "x-recipient-in-role"],
        truncate="header",
        notes="Output stopped inside the tool-call header, before <|message|>: the analysis is complete "
        "reasoning and there is no call yet.",
    ),
]


# ----------------------------------------------------------------------------- rendering


def harmony_render(enc: Any, msgs: list[Msg]) -> list[int]:
    cfg = RenderConversationConfig(auto_drop_analysis=False)
    user = Message.from_role_and_content(Role.USER, "What's the weather like?")
    prompt = enc.render_conversation_for_completion(Conversation.from_messages([user]), Role.ASSISTANT, config=cfg)
    full = enc.render_conversation(Conversation.from_messages([user, *map(to_harmony, msgs)]), config=cfg)
    assert full[: len(prompt)] == prompt, "prompt is not a prefix of the rendered conversation"
    out = list(full[len(prompt) :])
    cut = next((i for i, t in enumerate(out) if t in STOP_IDS), len(out))
    out = out[:cut]
    if isinstance(msgs[-1], Final) and out and out[-1] == END_ID:
        # Harmony spec: a generated final message ends with the stop token <|return|>, which the
        # history render replaces with <|end|>. Generation text ends before the stop token.
        out = out[:-1]
    return out


def official_parse(enc: Any, ids: list[int]) -> list[dict[str, Any]]:
    """Self-check with openai-harmony's own parser."""
    return [m.to_dict() for m in enc.parse_messages_from_completion_tokens(ids, Role.ASSISTANT)]


def harmony_records(tok: Any) -> list[dict[str, Any]]:
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    with urllib.request.urlopen(HARMONY_RAW) as r:
        encoder_sha = sha256_bytes(r.read())
    prov = {
        "kind": "template_render",
        "source_url": HARMONY_SRC,
        "revision": HARMONY_COMMIT,
        "license": "Apache-2.0",
        "generator": GENERATOR,
        "template_sha256": encoder_sha,
        "attribution": "Rendered with openai-harmony 0.0.8 (Copyright OpenAI, Apache-2.0).",
    }
    recs = []
    for c in CASES:
        ids = harmony_render(enc, c.msgs)
        exp: dict[str, Any] | None = expected_for(c.msgs)
        err = None
        if c.truncate == "header":
            # cut right before <|message|> of the (last) tool-call message
            ids = ids[: max(i for i, t in enumerate(ids) if t == MESSAGE_ID)]
            exp = expected(None, exp["reasoning_content"], [])  # type: ignore[index]
        elif isinstance(c.truncate, int):
            ids = ids[: c.truncate]
            if c.truncate_error is not None:
                exp, err = None, c.truncate_error
            else:
                parsed = official_parse_partial(enc, ids)
                exp = expected(None, parsed, [])
        else:
            parsed = official_parse(enc, ids)
            _assert_official_agrees(parsed, exp)
        raw = tok.decode(ids, skip_special_tokens=False)
        check_ids(tok, ids, raw)
        recs.append(
            record(
                name=c.name,
                provenance=prov,
                tools=tools(*c.tool_names),
                raw_output=raw,
                output_token_ids=ids,
                expected=exp,
                expected_error=err,
                tags=c.tags,
                notes=_notes("History render by openai-harmony 0.0.8 (recipient in the role header).", c.notes),
            )
        )
    return recs


def official_parse_partial(enc: Any, ids: list[int]) -> str:
    """Reasoning text of a truncated analysis message, via harmony's StreamableParser."""
    from openai_harmony import StreamableParser

    p = StreamableParser(enc, role=Role.ASSISTANT)
    for t in ids:
        p.process(t)
    assert p.current_channel == "analysis"
    return str(p.current_content)


def _assert_official_agrees(parsed: list[dict[str, Any]], exp: dict[str, Any] | None) -> None:
    assert exp is not None
    calls = [
        (m["recipient"].removeprefix("functions."), json.loads(m["content"][0]["text"]))
        for m in parsed
        if m.get("recipient")
    ]
    assert calls == [(c["name"], c["arguments"]) for c in exp["tool_calls"]], (calls, exp)
    analysis = [m["content"][0]["text"] for m in parsed if m.get("channel") == "analysis"]
    assert (analysis[0] if analysis else None) == exp["reasoning_content"]


def _notes(*parts: str | None) -> str:
    return " ".join(p for p in parts if p)


# ----------------------------------------------------------------------------- HF chat template

HF_CASES: list[dict[str, Any]] = [
    {
        "name": "hf-template-single-call",
        "thinking": WEATHER_REASONING,
        "calls": [("get_weather", {"city": "Paris", "unit": "c"})],
        "tags": ["single-call", "reasoning", "x-recipient-in-role", "x-constrain-text"],
    },
    {
        "name": "hf-template-unicode-nested",
        "thinking": "Suche nach dem besten Café in Zürich.",
        "calls": [("search", {"query": 'café "best" Zürich ☕', "filters": {"tags": ["a", "b"], "max": 3}})],
        "tags": ["single-call", "reasoning", "unicode", "nested-json", "string-escapes", "x-recipient-in-role"],
    },
    {
        "name": "hf-template-empty-arguments",
        "thinking": "Get the time.",
        "calls": [("get_time", {})],
        "tags": ["single-call", "reasoning", "empty-arguments", "x-recipient-in-role", "x-constrain-text"],
    },
    {
        "name": "hf-template-no-reasoning",
        "thinking": None,
        "calls": [("set_alarm", {"hour": 6, "minute": 45, "repeat": False})],
        "tags": ["single-call", "numeric-arguments", "x-recipient-in-role", "x-constrain-text"],
    },
    {
        "name": "hf-template-final-answer",
        "thinking": "Nothing to call.",
        "content": "It is sunny in Paris today.",
        "calls": [],
        "tags": ["no-call", "reasoning"],
    },
]


def hf_records(tok: Any) -> list[dict[str, Any]]:
    template = tok.chat_template
    prov = {
        "kind": "template_render",
        "source_url": HF_TEMPLATE_URL,
        "revision": REF_REVISION,
        "license": "Apache-2.0",
        "generator": GENERATOR,
        "template_sha256": sha256_bytes(template.encode("utf-8")),
    }
    offered = tools(*ALL_TOOLS)
    user = [{"role": "user", "content": "What's the weather like?"}]

    def ids_of(messages: list[dict[str, Any]], gen: bool) -> list[int]:
        out = tok.apply_chat_template(messages, tools=offered, add_generation_prompt=gen, tokenize=True)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    recs = []
    for c in HF_CASES:
        asst: dict[str, Any] = {"role": "assistant"}
        if c.get("thinking"):
            asst["thinking"] = c["thinking"]
        if c["calls"]:
            asst["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in c["calls"]]
        else:
            asst["content"] = c["content"]
        prompt = ids_of(user, True)
        full = ids_of([*user, asst], False)
        assert full[: len(prompt)] == prompt, "prompt is not a prefix of the rendered conversation"
        out = full[len(prompt) :]
        cut = next((i for i, t in enumerate(out) if t in STOP_IDS), len(out))
        out = out[:cut]
        raw = tok.decode(out, skip_special_tokens=False)
        recs.append(
            record(
                name=c["name"],
                provenance=prov,
                tools=offered,
                raw_output=raw,
                output_token_ids=out,
                expected=expected(c.get("content"), c.get("thinking"), c["calls"]),
                tags=c["tags"],
                notes="History render by the HF chat_template.jinja (recipient in the role header, content type "
                "written as plain ' json'). The template renders only tool_calls[0] of a message.",
            )
        )
    return recs


def main() -> None:
    tok = hf_tokenizer()
    recs = harmony_records(tok) + hf_records(tok)
    write_jsonl(OUT_DIR / "rendered.jsonl", recs)


if __name__ == "__main__":
    main()
