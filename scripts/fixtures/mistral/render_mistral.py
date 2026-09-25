# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mistral-common[sentencepiece]==1.12.0", "transformers==5.17.0", "huggingface_hub>=1.0"]
# ///
"""Render Mistral fixtures through mistral-common, the reference encoder for every Mistral tokenizer
version (v3 JSON array, v11, v13/v15 compact ``[TOOL_CALLS]name[ARGS]{json}``, ``[THINK]`` tokens).

For each case we encode ``[user, assistant]`` (finetuning validation mode, which allows a final
assistant turn) and ``[user]`` (test mode), check that the prompt ids are a prefix, slice them off
and cut at ``</s>``. ``raw_output`` is the slice decoded with special tokens kept.

Tool calls are rendered WITHOUT call ids. In v11 a call id would add ``[CALL_ID]id`` to the history
render, but mistral-common's generation grammar (``guidance/grammar_factory.py``
``_TOOL_CALL_GRAMMAR = "[TOOL_CALLS] SAFE_WS? name [ARGS] SAFE_WS? %json …"``) has no call id: the
model does not generate it. ``[CALL_ID]`` fixtures come from engine tests (import_mistral.py).
In v3 the id would add an ``"id"`` key inside the JSON array, which the v3 engine tests also omit.

A case lists extra models (``also``) that share the format; each is added to ``models`` only if
its own tokenizer renders byte-identical ids.

Run from the repo root::

    uv run --script scripts/fixtures/mistral/render_mistral.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    DEVSTRAL2,
    MAGISTRAL_2509,
    MEDIUM35,
    MINISTRAL3,
    MINISTRAL3_REASONING,
    MISTRAL7B_V03,
    OUT_DIR,
    SMALL4,
    SMALL32,
    Ref,
    decode,
    expected,
    mistral_tokenizer,
    record,
    tokenizer_file,
    tools,
    write_jsonl,
)

GENERATOR = "scripts/fixtures/mistral/render_mistral.py"
MC_VERSION = "1.12.0"
ALL = ("get_weather", "search", "get_time", "write_file", "set_alarm", "create_user")
USER = "Please help me with this task."


@dataclass(frozen=True)
class Variant:
    label: str
    ref: Ref
    also: tuple[Ref, ...] = ()
    thinking: bool | None = None


V13_THINK = Variant("v13-think", MAGISTRAL_2509, (MINISTRAL3_REASONING, SMALL4, MEDIUM35), thinking=True)
V13 = Variant("v13", MINISTRAL3, (DEVSTRAL2, MAGISTRAL_2509))
V11 = Variant("v11", SMALL32)
V3 = Variant("v3", MISTRAL7B_V03)


@dataclass
class Case:
    name: str
    variant: Variant
    calls: list[tuple[str, dict[str, Any]]]
    tags: list[str]
    reasoning: str | None = None
    content: str | None = None
    notes: str | None = None
    truncate_after: str | None = None
    truncate_expected: dict[str, Any] | None = None
    truncate_error: dict[str, Any] | None = None


LONG = "\n".join(f"line {i:03d}: the quick brown fox jumps over the lazy dog" for i in range(120))
R = "The user wants the weather in Paris. I should call get_weather."

CASES: list[Case] = [
    # ---------------------------------------------------------------- v13 + [THINK] (Magistral 2509 & co.)
    Case(
        "v13think-single-call",
        V13_THINK,
        [("get_weather", {"city": "Paris", "unit": "c"})],
        ["single-call", "reasoning", "x-v13-compact", "x-think-special-token"],
        reasoning=R,
    ),
    Case(
        "v13think-content-before-call",
        V13_THINK,
        [("get_weather", {"city": "Paris"})],
        ["single-call", "reasoning", "text-before-call", "x-v13-compact", "x-think-special-token"],
        reasoning=R,
        content="Let me check that for you.",
    ),
    Case(
        "v13think-parallel-calls",
        V13_THINK,
        [
            ("get_weather", {"city": "Zürich", "unit": "c"}),
            ("search", {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}),
        ],
        [
            "parallel-calls",
            "reasoning",
            "unicode",
            "nested-json",
            "string-escapes",
            "x-v13-compact",
            "x-think-special-token",
        ],
        reasoning="I should call the tools.",
    ),
    Case(
        "v13think-no-call",
        V13_THINK,
        [],
        ["no-call", "reasoning", "x-think-special-token"],
        reasoning="A greeting; no tool needed.",
        content="Hello! How can I help you today?",
    ),
    Case(
        "v13think-empty-arguments",
        V13_THINK,
        [("get_time", {})],
        ["single-call", "reasoning", "empty-arguments", "x-v13-compact", "x-think-special-token"],
        reasoning="Get the time.",
    ),
    Case(
        "v13think-marker-in-arguments",
        V13_THINK,
        [
            (
                "write_file",
                {
                    "path": "fmt.md",
                    "content": 'Calls look like [TOOL_CALLS]name[ARGS]{"a": 1}; thinking is [THINK]...[/THINK].',
                },
            )
        ],
        ["single-call", "reasoning", "marker-in-arguments", "x-v13-compact", "x-think-special-token"],
        reasoning="Document the format; [TOOL_CALLS] here is just text.",
        notes="The marker strings inside the reasoning and the argument value are ORDINARY text tokens in "
        "output_token_ids; only the control-token ids delimit the call. A parser working on detokenized text "
        "cannot tell them apart.",
    ),
    Case(
        "v13think-multiline-reasoning",
        V13_THINK,
        [("search", {"query": "weather Paris"})],
        ["single-call", "reasoning", "x-v13-compact", "x-think-special-token"],
        reasoning="Step 1: parse.\nStep 2: needs live data.\n\nStep 3: call search.",
    ),
    Case(
        "v13think-truncated-in-reasoning",
        V13_THINK,
        [("get_weather", {"city": "Paris"})],
        ["truncated", "reasoning", "no-call", "x-think-special-token"],
        reasoning="The user wants the weather in Paris, so I will need to call the weather tool.",
        truncate_after="so I will need",
        truncate_expected=expected(None, "The user wants the weather in Paris, so I will need", []),
        notes="Output stopped (max_tokens) inside [THINK]: everything generated is reasoning.",
    ),
    Case(
        "v13think-truncated-in-arguments",
        V13_THINK,
        [("write_file", {"path": "/tmp/build.sh", "content": "rm -rf /tmp/build && make all"})],
        ["truncated", "reasoning", "x-v13-compact", "x-think-special-token"],
        reasoning="Write the script.",
        truncate_after="rm -rf /",
        truncate_error={
            "reason": "Output stopped (max_tokens) inside the arguments JSON; the call is incomplete.",
            "accept": ["no_tool_calls", "content_passthrough", "exception"],
        },
    ),
    Case(
        "v13think-stop-at-open-marker",
        V13_THINK,
        [("get_weather", {"city": "Paris"})],
        ["truncated", "reasoning", "text-before-call", "x-stop-at-open-marker", "x-think-special-token"],
        reasoning=R,
        content="Let me check that for you.",
        truncate_after="[TOOL_CALLS]",
        truncate_expected=expected("Let me check that for you.", R, []),
        notes="Output stopped right after the [TOOL_CALLS] control token: reasoning and content are complete, "
        "there is no call yet, and the marker must not leak into content "
        "(https://github.com/sgl-project/sglang/issues/35565).",
    ),
    # ---------------------------------------------------------------- v13 instruct (Ministral 3, Devstral 2)
    Case("v13-single-call", V13, [("get_weather", {"city": "Paris", "unit": "c"})], ["single-call", "x-v13-compact"]),
    Case(
        "v13-parallel-calls",
        V13,
        [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Berlin"}), ("get_time", {})],
        ["parallel-calls", "empty-arguments", "x-v13-compact"],
    ),
    Case(
        "v13-content-before-call",
        V13,
        [("search", {"query": "vLLM release notes"})],
        ["single-call", "text-before-call", "x-v13-compact"],
        content="I'll search for that.",
    ),
    Case("v13-no-call", V13, [], ["no-call"], content="Paris is the capital of France. 🇫🇷"),
    Case(
        "v13-nested-json",
        V13,
        [
            (
                "search",
                {"query": "hotels", "filters": {"price": {"min": 50, "max": 120.5}, "tags": ["pool"], "open": None}},
            )
        ],
        ["single-call", "nested-json", "x-v13-compact"],
    ),
    Case(
        "v13-unicode-emoji",
        V13,
        [("search", {"query": "東京の天気 ☀️🌧️ — «prévisions» für Zürich"})],
        ["single-call", "unicode", "x-v13-compact"],
    ),
    Case(
        "v13-string-escapes",
        V13,
        [("write_file", {"path": "C:\\Users\\me\\a.txt", "content": 'line 1\n\t"quoted" \\backslash\\ }{ ]['})],
        ["single-call", "string-escapes", "x-v13-compact"],
        notes="Braces and brackets inside a JSON string must not end the arguments object.",
    ),
    Case(
        "v13-numeric-arguments",
        V13,
        [("set_alarm", {"hour": 7, "minute": 30, "volume": 0.75, "repeat": True, "label": None})],
        ["single-call", "numeric-arguments", "x-v13-compact"],
    ),
    Case(
        "v13-long-arguments",
        V13,
        [("write_file", {"path": "big.txt", "content": LONG})],
        ["single-call", "long-arguments", "x-v13-compact"],
    ),
    Case(
        "v13-param-named-name",
        V13,
        [("create_user", {"name": "bar", "arguments": "--force", "id": "u-1"})],
        ["single-call", "x-v13-compact", "x-param-named-name"],
        notes="Argument keys 'name', 'arguments' and 'id' collide with the keys of the v3 JSON-array format. "
        "Ollama dropped such calls for devstral-small-2: https://github.com/ollama/ollama/issues/16932.",
    ),
    Case(
        "v13-array-lookalike-argument",
        V13,
        [("write_file", {"path": "calls.json", "content": '[{"name": "get_time", "arguments": {}}]'})],
        ["single-call", "marker-in-arguments", "x-v13-compact"],
        notes="A string argument that looks like a v3 tool-call array must stay a string.",
    ),
    # ---------------------------------------------------------------- v11 (Mistral-Small-3.2)
    Case("v11-single-call", V11, [("get_weather", {"city": "Paris", "unit": "c"})], ["single-call", "x-v11-args"]),
    Case(
        "v11-parallel-calls",
        V11,
        [("get_weather", {"city": "Zürich"}), ("search", {"query": 'café "best"', "filters": {"max": 3}})],
        ["parallel-calls", "unicode", "nested-json", "string-escapes", "x-v11-args"],
    ),
    Case(
        "v11-content-before-call",
        V11,
        [("get_time", {})],
        ["single-call", "text-before-call", "empty-arguments", "x-v11-args"],
        content="Checking the clock.",
    ),
    Case("v11-no-call", V11, [], ["no-call"], content="Sure — here is a haiku about rain."),
    # ---------------------------------------------------------------- v3 (Mistral-7B-Instruct-v0.3)
    Case("v3-single-call", V3, [("get_weather", {"city": "Paris", "unit": "c"})], ["single-call", "x-v3-json-array"]),
    Case(
        "v3-parallel-calls",
        V3,
        [("get_weather", {"city": "Zürich"}), ("search", {"query": 'café "best"', "filters": {"tags": ["a"]}})],
        ["parallel-calls", "unicode", "nested-json", "string-escapes", "x-v3-json-array"],
    ),
    Case("v3-empty-arguments", V3, [("get_time", {})], ["single-call", "empty-arguments", "x-v3-json-array"]),
    Case("v3-no-call", V3, [], ["no-call"], content="The capital of France is Paris."),
]


def tekken_sha(ref: Ref) -> str:
    """sha256 of the tokenizer file that defines the format (tekken.json or tokenizer.model.v3)."""
    return hashlib.sha256(tokenizer_file(ref).read_bytes()).hexdigest()


def _request(messages: list[Any], offered: list[dict[str, Any]]) -> Any:
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.tool_calls import Function, Tool

    mtools = [
        Tool(
            function=Function(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters=t["function"].get("parameters", {}),
            )
        )
        for t in offered
    ]
    return ChatCompletionRequest(messages=messages, tools=mtools)


def render(ref: Ref, case: Case, offered: list[dict[str, Any]]) -> list[int]:
    from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
    from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
    from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
    from mistral_common.tokens.tokenizers.base import TokenizerVersion

    user = UserMessage(content=USER)
    content: Any = case.content
    if case.reasoning is not None:
        content = [ThinkChunk(thinking=case.reasoning), *([TextChunk(text=case.content)] if case.content else [])]
    # v13+ validation requires call ids, but v13+ never tokenizes them (asserted below). v3/v11 get none.
    raw = mistral_tokenizer(ref).instruct_tokenizer.tokenizer
    ids_needed = raw.version >= TokenizerVersion.v13
    calls = [
        ToolCall(
            id=f"call{i:05d}" if ids_needed else "null",  # "null" = no id (mistral-common convention)
            function=FunctionCall(name=n, arguments=json.dumps(a, ensure_ascii=False)),
        )
        for i, (n, a) in enumerate(case.calls)
    ]
    asst = AssistantMessage(content=content, tool_calls=calls or None)
    prompt = mistral_tokenizer(ref).encode_chat_completion(_request([user], offered)).tokens
    # Same path as MistralTokenizer.encode_chat_completion minus request validation, which rejects
    # parallel calls without ids ("Duplicate tool call id null") even though v3/v11 generations carry none.
    ft = mistral_tokenizer(ref, finetuning=True)
    instruct = ft._instruct_request_normalizer.from_chat_completion_request(_request([user, asst], offered))
    full = ft.instruct_tokenizer.encode_instruct(instruct).tokens
    assert full[: len(prompt)] == prompt, f"{case.name}: prompt is not a prefix"
    out = list(full[len(prompt) :])
    out = out[: out.index(raw.eos_id)] if raw.eos_id in out else out
    assert "[CALL_ID]" not in decode(ref, out), f"{case.name}: [CALL_ID] rendered"
    return out


def truncate(ref: Ref, ids: list[int], marker: str) -> list[int]:
    for n in range(1, len(ids) + 1):
        if marker in decode(ref, ids[:n]):
            return ids[:n]
    raise AssertionError(marker)


def main() -> None:
    recs = []
    for c in CASES:
        v = c.variant
        offered = tools(*ALL)
        ids = render(v.ref, c, offered)
        models = [v.ref.repo] + [o.repo for o in v.also if render(o, c, offered) == ids]
        exp: dict[str, Any] | None = expected(c.content, c.reasoning, c.calls)
        err = None
        if c.truncate_after is not None:
            ids = truncate(v.ref, ids, c.truncate_after)
            exp, err = c.truncate_expected, c.truncate_error
        version = mistral_tokenizer(v.ref).instruct_tokenizer.tokenizer.version.value
        recs.append(
            record(
                name=c.name,
                models=models,
                provenance={
                    "kind": "template_render",
                    "source_url": v.ref.tree_url,
                    "revision": v.ref.revision,
                    "license": "Apache-2.0",
                    "generator": GENERATOR,
                    "template_sha256": tekken_sha(v.ref),
                    "attribution": f"Rendered with mistral-common {MC_VERSION} (Apache-2.0).",
                },
                tools=offered,
                raw_output=decode(v.ref, ids),
                ref=v.ref,
                output_token_ids=ids,
                thinking=v.thinking,
                expected=exp,
                expected_error=err,
                tags=c.tags,
                notes=" ".join(
                    p
                    for p in (
                        f"Rendered with mistral-common {MC_VERSION} from {v.ref.repo} (tokenizer {version}); "
                        "template_sha256 is the sha256 of the tokenizer file that defines the format.",
                        c.notes,
                    )
                    if p
                ),
            )
        )
    write_jsonl(OUT_DIR / "rendered.jsonl", recs)


if __name__ == "__main__":
    main()
