# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["transformers==5.17.0", "jinja2>=3.1", "huggingface_hub>=1.0"]
# ///
"""Render qwen3-xml fixtures (Qwen3-Coder, Qwen3.5/3.6/3.8) through the OFFICIAL HF chat templates.

For each case we render ``[user, assistant(reasoning_content, content, tool_calls)]`` with
``apply_chat_template(tokenize=True)`` at a pinned revision, slice off the
``add_generation_prompt=True`` prompt ids, and cut at the first stop id from
``generation_config.json`` (``<|im_end|>`` / ``<|endoftext|>``). ``raw_output`` is the slice
decoded with ``skip_special_tokens=False``.

A case lists extra models (``also``) that emit the same format; each is added to ``models`` only
if its own template renders byte-identical output ids for the case (checked here).

Run from the repo root::

    uv run --script scripts/fixtures/qwen3-xml/render_qwen3_xml.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    CODER,
    CODER_480,
    CODER_NEXT,
    GEN_PROMPT_CODER,
    GEN_PROMPT_NO_THINK,
    GEN_PROMPT_THINK,
    OUT_DIR,
    QWEN35,
    QWEN36,
    QWEN38,
    STOP_TOKENS,
    Ref,
    expected,
    record,
    sha256_text,
    tokenizer,
    tools,
    write_jsonl,
)

GENERATOR = "scripts/fixtures/qwen3-xml/render_qwen3_xml.py"
ALL = ("get_weather", "search", "get_time", "write_file", "edit_file", "set_alarm", "lookup")
USER = "Please help me with this task."


@dataclass(frozen=True)
class Variant:
    ref: Ref
    also: tuple[Ref, ...]
    generation_prompt: str
    thinking: bool | None = None


V38 = Variant(QWEN38, (QWEN36,), GEN_PROMPT_THINK)
V38_NO_THINK = Variant(QWEN38, (QWEN36,), GEN_PROMPT_NO_THINK, thinking=False)
V35 = Variant(QWEN35, (), GEN_PROMPT_THINK)
VCODER = Variant(CODER, (CODER_480,), GEN_PROMPT_CODER)
VCODER_NEXT = Variant(CODER_NEXT, (), GEN_PROMPT_CODER)


@dataclass
class Case:
    name: str
    variant: Variant
    calls: list[tuple[str, dict[str, Any]]]
    tags: list[str]
    reasoning: str | None = None
    content: str | None = None
    # expected arguments when they differ from the rendered input (never the case today)
    notes: str | None = None
    tool_names: tuple[str, ...] = ALL
    # Truncation (max_tokens): keep the shortest id prefix whose text contains this substring.
    truncate_after: str | None = None
    truncate_expected: dict[str, Any] | None = None
    truncate_error: dict[str, Any] | None = None
    extra_tags: list[str] = field(default_factory=list)


LONG_CODE = "\n".join(
    ['"""Generated module used as a long tool argument."""', "", "from __future__ import annotations", ""]
    + [f"def step_{i:02d}(x: int) -> int:\n    return x + {i}\n" for i in range(80)]
)
PY_SNIPPET = 'def greet(name: str) -> str:\n    """Say hi."""\n    return f"Hello, {name}!"\n\n\nprint(greet("Zoë"))'
R_WEATHER = "The user wants the weather in Paris. I should call get_weather."

CASES: list[Case] = [
    # ------------------------------------------------------------------ Qwen3.8 / 3.6 (thinking on)
    Case(
        "q38-single-call",
        V38,
        [("get_weather", {"city": "Paris", "unit": "c"})],
        ["single-call", "reasoning", "reasoning-prefilled"],
        reasoning=R_WEATHER,
    ),
    Case(
        "q38-parallel-calls",
        V38,
        [
            ("get_weather", {"city": "Zürich", "unit": "c"}),
            ("search", {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}),
        ],
        ["parallel-calls", "reasoning", "reasoning-prefilled", "unicode", "nested-json", "string-escapes"],
        reasoning="I should call the tools.",
    ),
    Case(
        "q38-three-parallel-calls",
        V38,
        [
            ("get_weather", {"city": "Paris"}),
            ("get_weather", {"city": "Berlin"}),
            ("get_weather", {"city": "Tokyo", "unit": "f"}),
        ],
        ["parallel-calls", "reasoning", "reasoning-prefilled"],
        reasoning="Three cities, three calls.",
    ),
    Case(
        "q38-unicode-emoji",
        V38,
        [("search", {"query": "東京の天気 ☀️🌧️ — «prévisions» für Zürich 🇨🇭"})],
        ["single-call", "reasoning", "reasoning-prefilled", "unicode"],
        reasoning="用户想知道东京的天气。🌦️",
    ),
    Case(
        "q38-nested-json",
        V38,
        [
            (
                "search",
                {
                    "query": "hotels",
                    "filters": {"price": {"min": 50, "max": 120.5}, "tags": ["pool", "wifi"], "open": None},
                    "sites": ["a.example", "b.example"],
                },
            )
        ],
        ["single-call", "reasoning", "reasoning-prefilled", "nested-json"],
        reasoning="Search with structured filters.",
        notes="Object and array values are rendered with tojson; the parser must decode them as JSON.",
    ),
    Case(
        "q38-empty-arguments",
        V38,
        [("get_time", {})],
        ["single-call", "reasoning", "reasoning-prefilled", "empty-arguments"],
        reasoning="Get the time.",
    ),
    Case(
        "q38-marker-in-arguments",
        V38,
        [
            (
                "write_file",
                {
                    "path": "notes.md",
                    "content": "Close tags inline: </parameter> and </function> and <parameter=path> are text here.",
                },
            )
        ],
        ["single-call", "reasoning", "reasoning-prefilled", "marker-in-arguments"],
        reasoning="Write the note.",
        notes="The value contains XML-looking markers that are not at a line start. The structural delimiter is "
        "'\\n</parameter>\\n', so a parser that splits on a bare '</parameter>' cuts the value short.",
    ),
    Case(
        "q38-tool-call-token-in-arguments",
        V38,
        [("write_file", {"path": "fmt.md", "content": "Qwen wraps calls in <tool_call> ... </tool_call> blocks."})],
        ["single-call", "reasoning", "reasoning-prefilled", "marker-in-arguments"],
        reasoning="Document the format.",
        notes="'<tool_call>' and '</tool_call>' are added tokens, so the tokenizer maps them to their marker ids "
        "even inside the argument value (see output_token_ids). Only the XML structure tells them apart.",
    ),
    Case(
        "q38-text-before-call",
        V38,
        [("get_weather", {"city": "Paris"})],
        ["single-call", "reasoning", "reasoning-prefilled", "text-before-call"],
        reasoning=R_WEATHER,
        content="Let me check the weather for you.",
        notes="Qwen3.5+ templates join content and the first <tool_call> with '\\n\\n', which is markup.",
    ),
    Case(
        "q38-no-call",
        V38,
        [],
        ["no-call", "reasoning", "reasoning-prefilled"],
        reasoning="A simple greeting; no tool is needed.",
        content="Hello! How can I help you today?",
    ),
    Case(
        "q38-multiline-string",
        V38,
        [("write_file", {"path": "greet.py", "content": PY_SNIPPET})],
        ["single-call", "reasoning", "reasoning-prefilled", "string-escapes", "x-multiline-string"],
        reasoning="Write the script.",
    ),
    Case(
        "q38-whitespace-significant",
        V38,
        [
            (
                "edit_file",
                {
                    "path": "a.py",
                    "old_string": "    if x:\n        return 1\n",
                    "new_string": "  two  spaces  ",
                },
            )
        ],
        ["single-call", "reasoning", "reasoning-prefilled", "x-whitespace-significant-string"],
        reasoning="Replace the block exactly.",
        notes="Leading indentation and a trailing newline are part of the values. Exactly one '\\n' on each side "
        "of a value is markup (vLLM #48753: https://github.com/vllm-project/vllm/issues/48753).",
    ),
    Case(
        "q38-numeric-and-literal-arguments",
        V38,
        [("set_alarm", {"hour": 7, "minute": 30, "volume": 0.75, "repeat": True, "label": None})],
        ["single-call", "reasoning", "reasoning-prefilled", "numeric-arguments", "x-schema-coercion"],
        reasoning="Set the alarm.",
        notes="Scalars are raw text; integer/number/boolean/null come back only through the tool's JSON schema.",
    ),
    Case(
        "q38-numeric-looking-strings",
        V38,
        [("lookup", {"zip": "02139", "account_id": "123456789012345678901", "flag": "true", "payload": '{"a": 1}'})],
        ["single-call", "reasoning", "reasoning-prefilled", "x-schema-coercion"],
        reasoning="Look it up.",
        notes="Every parameter is typed 'string' in the schema, so '02139', 'true' and '{\"a\": 1}' must stay strings.",
    ),
    Case(
        "q38-string-escapes",
        V38,
        [("write_file", {"path": "C:\\tmp\\x.txt", "content": 'a "quote", a \\backslash, <tag> & ampersand\ttab'})],
        ["single-call", "reasoning", "reasoning-prefilled", "string-escapes"],
        reasoning="Write it.",
        notes="String values are inserted verbatim with no escaping.",
    ),
    Case(
        "q38-long-arguments",
        V38,
        [("write_file", {"path": "steps.py", "content": LONG_CODE})],
        ["single-call", "reasoning", "reasoning-prefilled", "long-arguments", "x-multiline-string"],
        reasoning="Write the module.",
    ),
    Case(
        "q38-empty-reasoning",
        V38,
        [("get_time", {})],
        ["single-call", "empty-arguments", "reasoning-prefilled"],
        notes="Thinking enabled but the reasoning is empty: the completion starts with '\\n</think>'.",
    ),
    Case(
        "q38-thinking-disabled",
        V38_NO_THINK,
        [("get_weather", {"city": "Paris", "unit": "c"})],
        ["single-call"],
        notes="enable_thinking=false: the generation prompt pre-fills '<think>\\n\\n</think>\\n\\n'.",
    ),
    Case(
        "q38-thinking-disabled-parallel",
        V38_NO_THINK,
        [("get_weather", {"city": "Oslo"}), ("get_time", {})],
        ["parallel-calls", "empty-arguments"],
    ),
    Case(
        "q38-thinking-disabled-no-call",
        V38_NO_THINK,
        [],
        ["no-call"],
        content="Paris is the capital of France.",
    ),
    Case(
        "q38-truncated-in-reasoning",
        V38,
        [("get_weather", {"city": "Paris"})],
        ["truncated", "reasoning", "reasoning-prefilled", "no-call"],
        reasoning="The user wants the weather in Paris, so I will need to call the weather tool.",
        truncate_after="so I will need",
        truncate_expected=expected(None, "The user wants the weather in Paris, so I will need", []),
        notes="Output stopped (max_tokens) inside the reasoning: all of it is reasoning.",
    ),
    Case(
        "q38-truncated-in-parameter",
        V38,
        [("write_file", {"path": "/tmp/build.sh", "content": "rm -rf /tmp/build && make all"})],
        ["truncated", "reasoning", "reasoning-prefilled"],
        reasoning="Write the script.",
        truncate_after="rm -rf /",
        truncate_error={
            "reason": "Output stopped (max_tokens) inside a parameter value, before </parameter> and </function>. "
            "The call is incomplete; a truncated 'rm -rf /tmp/build' must not become a complete 'rm -rf /' "
            "(vLLM #57699).",
            "accept": ["no_tool_calls", "content_passthrough", "exception"],
        },
    ),
    Case(
        "q38-truncated-at-tool-call-open",
        V38,
        [("get_weather", {"city": "Paris"})],
        ["truncated", "reasoning", "reasoning-prefilled", "x-stop-at-open-marker"],
        reasoning=R_WEATHER,
        truncate_after="<tool_call>",
        truncate_expected=expected(None, R_WEATHER, []),
        notes="Output stopped right after the <tool_call> opener. The reasoning is complete and must be kept; "
        "there is no call yet and the marker must not leak into content (SGLang #35565: "
        "https://github.com/sgl-project/sglang/issues/35565).",
    ),
    # ------------------------------------------------------------------ Qwen3.5 (Python literals)
    Case(
        "q35-python-literals",
        V35,
        [("set_alarm", {"hour": 6, "minute": 0, "repeat": True, "label": "gym"})],
        ["single-call", "reasoning", "reasoning-prefilled", "numeric-arguments", "x-python-literals"],
        reasoning="Set a repeating alarm.",
        notes="The Qwen3.5-9B template writes Python literals ('True'); with a boolean schema the value is true.",
    ),
    Case(
        "q35-single-call",
        V35,
        [("get_weather", {"city": "Zürich", "unit": "c"})],
        ["single-call", "reasoning", "reasoning-prefilled", "unicode"],
        reasoning="Weather lookup.",
    ),
    # ------------------------------------------------------------------ Qwen3-Coder (no reasoning)
    Case("coder-single-call", VCODER, [("get_weather", {"city": "Paris", "unit": "c"})], ["single-call"]),
    Case(
        "coder-parallel-calls",
        VCODER,
        [
            ("get_weather", {"city": "Zürich", "unit": "c"}),
            ("search", {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}),
        ],
        ["parallel-calls", "unicode", "nested-json", "string-escapes"],
    ),
    Case(
        "coder-text-before-call",
        VCODER,
        [("write_file", {"path": "hello.py", "content": PY_SNIPPET})],
        ["single-call", "text-before-call", "x-multiline-string"],
        content="I'll create the file now.",
        notes="The Qwen3-Coder template writes '\\n' + content + '\\n' and then '\\n<tool_call>', so the content is "
        "followed by '\\n\\n' of markup.",
    ),
    Case("coder-no-call", VCODER, [], ["no-call"], content="Use `git rebase -i HEAD~3` to squash the commits."),
    Case(
        "coder-python-literals",
        VCODER,
        [("set_alarm", {"hour": 9, "minute": 15, "volume": 0.5, "repeat": False})],
        ["single-call", "numeric-arguments", "x-python-literals", "x-schema-coercion"],
        notes="The Qwen3-Coder-30B template writes Python literals ('False'); with a boolean schema the value "
        "is false.",
    ),
    Case("coder-empty-arguments", VCODER, [("get_time", {})], ["single-call", "empty-arguments"]),
    Case(
        "coder-marker-in-arguments",
        VCODER,
        [
            (
                "write_file",
                {"path": "x.md", "content": "Close with </function> then </tool_call>; open with <function=x>."},
            )
        ],
        ["single-call", "marker-in-arguments"],
        notes="XML-looking markers inside a value, not at line starts. '</tool_call>' is an added token, so it "
        "appears as its marker id in output_token_ids.",
    ),
    # ------------------------------------------------------------------ Qwen3-Coder-Next (JSON literals)
    Case(
        "coder-next-json-literals",
        VCODER_NEXT,
        [("set_alarm", {"hour": 9, "minute": 15, "repeat": False, "label": None})],
        ["single-call", "numeric-arguments", "x-schema-coercion"],
        notes="The Qwen3-Coder-Next template writes JSON literals ('false', 'null').",
    ),
    Case(
        "coder-next-parallel-calls",
        VCODER_NEXT,
        [("get_weather", {"city": "Lyon"}), ("get_weather", {"city": "Nice"})],
        ["parallel-calls"],
    ),
]


def _ids(tok: Any, messages: list[dict[str, Any]], offered: list[dict[str, Any]], gen: bool, **kw: Any) -> list[int]:
    out = tok.apply_chat_template(messages, tools=offered, add_generation_prompt=gen, tokenize=True, **kw)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def render(ref: Ref, case: Case, offered: list[dict[str, Any]]) -> tuple[list[int], bool]:
    """Return (output ids, sliced_by_text). See module docstring."""
    tok = tokenizer(ref)
    kw: dict[str, Any] = {}
    if case.variant.thinking is not None:
        kw["enable_thinking"] = case.variant.thinking
    user = [{"role": "user", "content": USER}]
    asst: dict[str, Any] = {"role": "assistant", "content": case.content or ""}
    if case.reasoning is not None:
        asst["reasoning_content"] = case.reasoning
    if case.calls:
        asst["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in case.calls]
    prompt = _ids(tok, user, offered, True, **kw)
    full = _ids(tok, [*user, asst], offered, False, **kw)
    prompt_text = tok.decode(prompt, skip_special_tokens=False)
    assert prompt_text.endswith(case.variant.generation_prompt), case.name
    stop_ids = {tok.convert_tokens_to_ids(s) for s in STOP_TOKENS}
    if full[: len(prompt)] == prompt:
        out = full[len(prompt) :]
        cut = next((i for i, t in enumerate(out) if t in stop_ids), len(out))
        return out[:cut], False
    # BPE merged the prompt's trailing '\n' with the next '\n' (e.g. '<think>\n' + '\n</think>'). A model
    # generating after the prompt cannot un-emit the prompt's tokens, so slice the TEXT and re-encode the
    # remainder (lossless for this tokenizer; asserted in encode round trip below).
    full_text = tok.decode(full, skip_special_tokens=False)
    assert full_text.startswith(prompt_text), f"{case.name}: prompt text is not a prefix"
    rest = full_text[len(prompt_text) :]
    rest = min((rest.split(s)[0] for s in STOP_TOKENS), key=len)
    ids = list(tok.encode(rest, add_special_tokens=False))
    assert tok.decode(ids, skip_special_tokens=False) == rest
    return ids, True


def truncate(tok: Any, ids: list[int], marker: str) -> list[int]:
    for n in range(1, len(ids) + 1):
        if marker in tok.decode(ids[:n], skip_special_tokens=False):
            return ids[:n]
    raise AssertionError(f"marker {marker!r} not found")


def main() -> None:
    recs = []
    for c in CASES:
        v = c.variant
        offered = tools(*c.tool_names)
        tok = tokenizer(v.ref)
        ids, by_text = render(v.ref, c, offered)
        models = [v.ref.repo]
        for other in v.also:
            if render(other, c, offered)[0] == ids and tokenizer(other).decode(ids, skip_special_tokens=False) == (
                tok.decode(ids, skip_special_tokens=False)
            ):
                models.append(other.repo)
        exp: dict[str, Any] | None = expected(c.content, c.reasoning, c.calls)
        err = None
        if c.truncate_after is not None:
            ids = truncate(tok, ids, c.truncate_after)
            exp, err = c.truncate_expected, c.truncate_error
        raw = tok.decode(ids, skip_special_tokens=False)
        template = tok.chat_template
        recs.append(
            record(
                name=c.name,
                models=models,
                provenance={
                    "kind": "template_render",
                    "source_url": v.ref.template_url,
                    "revision": v.ref.revision,
                    "license": "Apache-2.0",
                    "generator": GENERATOR,
                    "template_sha256": sha256_text(template),
                },
                tools=offered,
                raw_output=raw,
                tokenizer_ref=v.ref,
                output_token_ids=ids,
                generation_prompt=v.generation_prompt,
                thinking=v.thinking,
                expected=exp,
                expected_error=err,
                tags=c.tags,
                notes=" ".join(
                    p
                    for p in (
                        f"History render of the official {v.ref.repo} chat template.",
                        "The prompt/completion boundary fell inside a merged BPE token, so the rendered text was "
                        "sliced after the generation prompt and the remainder re-encoded."
                        if by_text
                        else None,
                        c.notes,
                    )
                    if p
                ),
            )
        )
    write_jsonl(OUT_DIR / "rendered.jsonl", recs)


if __name__ == "__main__":
    main()
