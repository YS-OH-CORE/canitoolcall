"""Kimi fixtures copied from engine test suites and public bug reports.

Each item quotes the raw model output verbatim from its source (line-anchored at a
pinned commit) and states the parse a correct parser must produce. ``build.py``
adds token ids by encoding the raw text with the reference tokenizer (literal
markers map to their ids, as the model emits them) and checks the round trip.

Sources:
* vLLM v0.30.0 (Apache-2.0), commit ced6857afa0ea7b2e3f0846a62e1394e90f15607.
  The K2 tests use moonshotai/Kimi-K2-Instruct; its tiktoken vocabulary is
  byte-identical to Kimi-K2-Instruct-0905 (same tiktoken.model blob).
* public GitHub issues (quoted, NOASSERTION)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from _shared.core import Provenance, Record, Tool, tool

VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
VLLM_ATTR = "Copyright contributors to the vLLM project (Apache-2.0)"
GEN_PROMPT_NO_THINK = {"k3": '<|open|>message role="assistant"<|sep|><|open|>response<|sep|>'}

# Marker spellings used by the vLLM tests (tests/tool_parsers/test_kimi_k2_tool_parser.py L33-L45).
SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
TOOL_BEGIN = "<|tool_call_begin|>"
TOOL_END = "<|tool_call_end|>"
ARG_BEGIN = "<|tool_call_argument_begin|>"


def _tool(tool_id: str, args: str) -> str:
    """vLLM's helper, verbatim (note the space after the id)."""
    return f"{TOOL_BEGIN}{tool_id} {ARG_BEGIN}{args}{TOOL_END}"


def _wrap(*tool_strs: str) -> str:
    return SECTION_BEGIN + "".join(tool_strs) + SECTION_END


# K3 helpers, verbatim from tests/tool_use/test_kimi_k3_tool_parser.py L20-L25 and L109-L123.
OPEN, CLOSE, SEP = "<|open|>", "<|close|>", "<|sep|>"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_OPEN = f"{OPEN}response{SEP}"
RESPONSE_CLOSE = f"{CLOSE}response{SEP}"


def _arg(key: str, typ: str, value: str) -> str:
    return f'{OPEN}argument key="{key}" type="{typ}"{SEP}{value}{CLOSE}argument{SEP}'


def _call(tool_name: str, index: int, *args: str) -> str:
    return f'{OPEN}call tool="{tool_name}" index="{index}"{SEP}{"".join(args)}{CLOSE}call{SEP}'


def _tools(*calls: str) -> str:
    return f"{OPEN}tools{SEP}{''.join(calls)}{CLOSE}tools{SEP}"


def vllm(path: str, lines: str) -> Provenance:
    return Provenance(
        "engine_test",
        f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/{path}#{lines}",
        VLLM_SHA,
        "Apache-2.0",
        "scripts/fixtures/kimi/imported.py",
        attribution=VLLM_ATTR,
    )


def issue(url: str, revision: str, attribution: str) -> Provenance:
    return Provenance(
        "bug_report", url, revision, "NOASSERTION", "scripts/fixtures/kimi/imported.py", attribution=attribution
    )


@dataclass
class Item:
    name: str
    model_key: str
    raw: str
    tools: list[Tool]
    provenance: Provenance
    tags: list[str]
    expected: dict[str, Any] | None = None
    expected_error: dict[str, Any] | None = None
    thinking: bool | None = None
    notes: str | None = None

    def record(self, slug: str, repo: str, tokenizer: dict[str, str], ids: list[int]) -> Record:
        gen_prompt = GEN_PROMPT_NO_THINK[self.model_key] if self.thinking is False else None
        sub = "x-kimi-k3" if self.model_key == "k3" else "x-kimi-k2"
        return Record(
            id=f"{slug}/{self.model_key}-{self.name}",
            family=slug,
            models=[repo],
            provenance=self.provenance,
            tools=self.tools,
            raw_output=self.raw,
            output_token_ids=ids,
            tokenizer=tokenizer,
            generation_prompt=gen_prompt,
            thinking=self.thinking,
            expected=self.expected,
            expected_error=self.expected_error,
            tags=[*self.tags, sub],
            notes=self.notes,
        )


def exp(content: str | None, *calls: tuple[str, dict[str, Any]], reasoning: str | None = None) -> dict[str, Any]:
    return {
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
    }


def stand_in(tool_name: str, props: dict[str, str] | None = None) -> Tool:
    return tool(tool_name, None, {k: {"type": v} for k, v in (props or {}).items()})


STANDIN = "The vLLM test exercises the parser without a tools list; the tool schemas here are minimal stand-ins."
K2T = "tests/tool_parsers/test_kimi_k2_tool_parser.py"
K3T = "tests/tool_use/test_kimi_k3_tool_parser.py"
K3R = "tests/reasoning/test_kimi_k3_reasoning_parser.py"
# tests/tool_use/test_kimi_k3_tool_parser.py L58-L72: the request's only tool.
CALC = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}]

ITEMS: list[Item] = [
    Item(
        "vllm-single-call-content-before",
        "k2i",
        "I'll check. " + _wrap(_tool("functions.get_weather:0", '{"city": "Beijing"}')),
        [stand_in("get_weather", {"city": "string"})],
        vllm(K2T, "L59-L66"),
        ["single-call", "text-before-call", "x-id-encodes-name"],
        expected=exp("I'll check. ", ("get_weather", {"city": "Beijing"})),
        notes=STANDIN + " vLLM's test helper puts a space between the call id and <|tool_call_argument_begin|>.",
    ),
    Item(
        "vllm-three-tool-calls",
        "k2i",
        "Multiple tasks. "
        + _wrap(
            _tool("functions.get_weather:0", '{"city": "New York"}'),
            _tool("functions.get_news:1", '{"topic": "technology"}'),
            _tool("functions.send_email:2", '{"to": "user@example.com", "subject": "Daily Update"}'),
        ),
        [
            stand_in("get_weather", {"city": "string"}),
            stand_in("get_news", {"topic": "string"}),
            stand_in("send_email", {"to": "string", "subject": "string"}),
        ],
        vllm(K2T, "L78-L96"),
        ["parallel-calls", "text-before-call", "x-global-idx"],
        expected=exp(
            "Multiple tasks. ",
            ("get_weather", {"city": "New York"}),
            ("get_news", {"topic": "technology"}),
            ("send_email", {"to": "user@example.com", "subject": "Daily Update"}),
        ),
        notes=STANDIN,
    ),
    Item(
        "vllm-multiline-json",
        "k2i",
        "Formatted. " + _wrap(_tool("functions.process_data:0", '{\n  "name": "test",\n  "value": 123\n}')),
        [stand_in("process_data", {"name": "string", "value": "integer"})],
        vllm(K2T, "L107-L119"),
        ["single-call", "text-before-call", "numeric-arguments"],
        expected=exp("Formatted. ", ("process_data", {"name": "test", "value": 123})),
        notes=STANDIN,
    ),
    Item(
        "vllm-id-without-functions-prefix",
        "k2i",
        "No prefix. " + _wrap(_tool("get_weather:0", '{"city": "Tokyo"}')),
        [stand_in("get_weather", {"city": "string"})],
        vllm(K2T, "L120-L126"),
        ["single-call", "text-before-call", "x-non-kimi-call-id"],
        expected=exp("No prefix. ", ("get_weather", {"city": "Tokyo"})),
        notes=STANDIN + " The id lacks the 'functions.' prefix; the name is still recoverable from '{name}:{idx}'.",
    ),
    Item(
        "vllm-noise-between-markers",
        "k2i",
        "Reasoning. "
        + SECTION_BEGIN
        + " spurious noise "
        + TOOL_BEGIN
        + "functions.test:0 "
        + ARG_BEGIN
        + '{"k": "v"} '
        + TOOL_END
        + SECTION_END,
        [stand_in("test", {"k": "string"})],
        vllm(K2T, "L370-L386"),
        ["single-call", "text-before-call", "malformed"],
        expected=exp("Reasoning. ", ("test", {"k": "v"})),
        notes=STANDIN + " The deltas of the streaming test, concatenated. Text inside the section but outside a call "
        "is not content (vLLM asserts it does not leak).",
    ),
    Item(
        "vllm-empty-tool-section",
        "k2i",
        "Reasoning. " + SECTION_BEGIN + SECTION_END,
        [stand_in("test", {"k": "string"})],
        vllm(K2T, "L388-L393"),
        ["no-call", "x-empty-section"],
        expected=exp("Reasoning. "),
        notes=STANDIN + " An empty tool-calls section yields no calls and no marker text in content.",
    ),
    Item(
        "vllm-content-after-tool-section",
        "k2i",
        "Before. "
        + SECTION_BEGIN
        + TOOL_BEGIN
        + "functions.get_weather:0 "
        + ARG_BEGIN
        + '{"city": "Tokyo"} '
        + TOOL_END
        + SECTION_END
        + " After tools.",
        [stand_in("get_weather", {"city": "string"})],
        vllm(K2T, "L435-L458"),
        ["single-call", "text-before-call", "text-after-call", "x-policy"],
        expected=exp("Before.  After tools.", ("get_weather", {"city": "Tokyo"})),
        notes=STANDIN + " The deltas of the streaming test, concatenated. vLLM's test asserts the trailing text is "
        "DROPPED; this fixture expects it kept, because text outside the tool-calls section is ordinary content and "
        "dropping it loses model output. Content is the text before and after the section, concatenated verbatim "
        "(spec/README.md, 'Content around tool calls'). Tagged x-policy: the expected value encodes that spec rule, "
        "which neither the Kimi format nor the cited test fixes, so the matrix can show it apart from format "
        "conformance.",
    ),
    Item(
        "vllm-no-call-thinking-disabled",
        "k3",
        f"answer{RESPONSE_CLOSE}",
        CALC,
        vllm(K3T, "L219-L235"),
        ["no-call"],
        expected=exp("answer"),
        thinking=False,
        notes="thinking=false: the generation prompt opened the response channel, so the completion starts with "
        "content and closes the response.",
    ),
    Item(
        "vllm-call-thinking-disabled",
        "k3",
        RESPONSE_CLOSE + _tools(_call("calc", 1, _arg("x", "number", "1"))),
        CALC,
        vllm(K3T, "L238-L261"),
        ["single-call", "numeric-arguments", "x-typed-args"],
        expected=exp(None, ("calc", {"x": 1})),
        thinking=False,
    ),
    Item(
        "vllm-truncated-tools",
        "k3",
        f'{RESPONSE_CLOSE}{OPEN}tools{SEP}{OPEN}call tool="calc" index="1"',
        CALC,
        vllm(K3T, "L264-L283"),
        ["truncated"],
        expected_error={
            "reason": "The tools element is cut off inside the first call's open tag.",
            "accept": ["no_tool_calls"],
        },
        thinking=False,
        notes="vLLM asserts no reasoning, no content and no calls: the XTML must not leak into content.",
    ),
    Item(
        "vllm-reasoning-prefix-consumed",
        "k3",
        f"step{THINK_CLOSE}{RESPONSE_OPEN}answer",
        CALC,
        vllm(K3R, "L70-L80"),
        ["no-call", "reasoning", "x-think-no-open-tag"],
        expected=exp("answer", reasoning="step"),
        thinking=True,
        notes="The generation prompt opened the think channel; the response is not closed in this test string.",
    ),
    Item(
        "bug-truncated-reasoning-recorded",
        "k3",
        "The user asks to reason step by step about a math problem. I only need ",
        [],
        issue(
            "https://github.com/vllm-project/vllm/issues/57353",
            "issue opened 2026-09-17",
            "Quoted from the issue's repro output (Kimi K3, max_tokens=10, no tools).",
        ),
        ["truncated", "reasoning", "regression", "x-think-no-open-tag"],
        expected=exp(None, reasoning="The user asks to reason step by step about a math problem. I only need "),
        thinking=True,
        notes="A real K3 generation cut by max_tokens inside the think channel. vLLM's non-streaming path returned "
        "it as content; with thinking enabled the template opens the think channel, so it is reasoning.",
    ),
    Item(
        "bug-response-only-completion",
        "k3",
        f"{RESPONSE_OPEN}Hello world{RESPONSE_CLOSE}{CLOSE}message{SEP}",
        CALC,
        issue(
            "https://github.com/vllm-project/vllm/issues/57688",
            "issue opened 2026-09-19",
            "Shape A of the issue's probe, written there in shorthand as '◁response▷Hello world◁/response▷◁/message▷'.",
        ),
        ["no-call", "regression", "x-skip-think-channel"],
        expected=exp("Hello world"),
        thinking=True,
        notes="The issue abbreviates <|open|>X<|sep|> as ◁X▷ and <|close|>X<|sep|> as ◁/X▷; this raw_output spells "
        "the markers out. The model skipped the think channel and answered in the response channel, so everything "
        "is content (the non-streaming vLLM result the issue treats as correct).",
    ),
]
