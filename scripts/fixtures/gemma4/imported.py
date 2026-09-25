"""``gemma4`` fixtures imported from engine test suites and bug reports.

Each ``raw`` string is copied verbatim from the cited line (extracted with
``ast.literal_eval``; streamed chunk lists are joined in order). Where the test
used its own tool, a minimal schema matching the call is given here (the
upstream tests pass tools only to satisfy the parser API). Expected values are
the correct parse per the Gemma 4 format; each agrees with the upstream
assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VLLM_REV = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # v0.30.0
SGLANG_REV = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"  # v0.5.20
VLLM_TOOL = f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/tool_parsers/test_gemma4_tool_parser.py"
VLLM_REASONING = f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/reasoning/test_gemma4_reasoning_parser.py"
SGLANG_FCP = f"https://github.com/sgl-project/sglang/blob/{SGLANG_REV}/test/registered/unit/function_call/test_function_call_parser.py"
THINKING_OFF_PROMPT = "<|turn>model\n<|channel>thought\n<channel|>"
THINKING_ON_PROMPT = "<|turn>model\n"


def vllm(url: str, line: int) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{url}#L{line}",
        "revision": VLLM_REV,
        "license": "Apache-2.0",
        "attribution": "Copyright contributors to the vLLM project",
    }


def sglang(line: int) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{SGLANG_FCP}#L{line}",
        "revision": SGLANG_REV,
        "license": "Apache-2.0",
        "attribution": "Copyright SGLang Team",
    }


def ok(
    calls: list[tuple[str, dict[str, Any]]], content: str | None = None, reasoning: str | None = None
) -> dict[str, Any]:
    return {
        "expected": {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
        }
    }


def _tool(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": properties}}}


LOCATION = {"location": {"type": "string"}, "unit": {"type": "string"}}
WEATHER = _tool("get_weather", LOCATION)
GET_TIME_LOC = _tool("get_time", {"location": {"type": "string"}})
COMPLEX = _tool(
    "complex_function", {"nested": {"type": "object"}, "list": {"type": "array", "items": {"type": "string"}}}
)
SET_STATUS = _tool(
    "set_status", {"is_active": {"type": "boolean"}, "count": {"type": "integer"}, "score": {"type": "number"}}
)
HYPHEN = _tool("get-weather", LOCATION)
DOTTED = _tool("weather.get", LOCATION)
GET_STATUS = _tool("get_status", {})
WEATHER_NESTED = _tool(
    "get_weather", {"location": {"type": "string"}, "nested": {"type": "array"}, "details": {"type": "object"}}
)
CREATE_WORKFLOW = _tool("create_workflow", {"name": {"type": "string"}, "connections": {"type": "object"}})


@dataclass(frozen=True)
class Imported:
    name: str
    file: str
    provenance: dict[str, Any]
    tools: list[Any]
    raw: str
    expected: dict[str, Any]
    tags: list[str]
    notes: str | None = None
    generation_prompt: str = THINKING_OFF_PROMPT
    thinking: bool = False


FIXTURES: list[Imported] = [
    # ---- vLLM tests/tool_parsers/test_gemma4_tool_parser.py
    Imported(
        "vllm-no-tool-calls",
        "engine-tests",
        vllm(VLLM_TOOL, 288),
        [WEATHER],
        "Hello, how can I help you today?",
        ok([], "Hello, how can I help you today?"),
        ["no-call"],
    ),
    Imported(
        "vllm-single-call",
        "engine-tests",
        vllm(VLLM_TOOL, 296),
        [WEATHER],
        '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>',
        ok([("get_weather", {"location": "London"})]),
        ["single-call"],
    ),
    Imported(
        "vllm-multiple-arguments",
        "engine-tests",
        vllm(VLLM_TOOL, 308),
        [WEATHER],
        '<|tool_call>call:get_weather{location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>}<tool_call|>',
        ok([("get_weather", {"location": "San Francisco", "unit": "celsius"})]),
        ["single-call"],
    ),
    Imported(
        "vllm-text-before-call",
        "engine-tests",
        vllm(VLLM_TOOL, 323),
        [WEATHER],
        'Let me check the weather for you. <|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>',
        ok([("get_weather", {"location": "Paris"})], "Let me check the weather for you."),
        ["single-call", "text-before-call"],
        "The template renders text after calls, but models also emit text before them.",
    ),
    Imported(
        "vllm-two-calls",
        "engine-tests",
        vllm(VLLM_TOOL, 336),
        [WEATHER, GET_TIME_LOC],
        '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
        '<|tool_call>call:get_time{location:<|"|>London<|"|>}<tool_call|>',
        ok([("get_weather", {"location": "London"}), ("get_time", {"location": "London"})]),
        ["parallel-calls"],
    ),
    Imported(
        "vllm-nested-arguments",
        "engine-tests",
        vllm(VLLM_TOOL, 350),
        [COMPLEX],
        '<|tool_call>call:complex_function{nested:{inner:<|"|>value<|"|>},list:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>',
        ok([("complex_function", {"nested": {"inner": "value"}, "list": ["a", "b"]})]),
        ["single-call", "nested-json"],
    ),
    Imported(
        "vllm-number-and-boolean",
        "engine-tests",
        vllm(VLLM_TOOL, 365),
        [SET_STATUS],
        "<|tool_call>call:set_status{is_active:true,count:42,score:3.14}<tool_call|>",
        ok([("set_status", {"is_active": True, "count": 42, "score": 3.14})]),
        ["single-call", "numeric-arguments"],
    ),
    Imported(
        "vllm-hyphenated-function-name",
        "engine-tests",
        vllm(VLLM_TOOL, 392),
        [HYPHEN],
        '<|tool_call>call:get-weather{location:<|"|>London<|"|>}<tool_call|>',
        ok([("get-weather", {"location": "London"})]),
        ["single-call", "x-name-punctuation"],
        "OpenAI tool names may contain '-'; the template renders them verbatim after 'call:'.",
    ),
    Imported(
        "vllm-dotted-function-name",
        "engine-tests",
        vllm(VLLM_TOOL, 402),
        [DOTTED],
        '<|tool_call>call:weather.get{location:<|"|>London<|"|>}<tool_call|>',
        ok([("weather.get", {"location": "London"})]),
        ["single-call", "x-name-punctuation"],
    ),
    Imported(
        "vllm-no-arguments",
        "engine-tests",
        vllm(VLLM_TOOL, 412),
        [GET_STATUS],
        "<|tool_call>call:get_status{}<tool_call|>",
        ok([("get_status", {})]),
        ["single-call", "empty-arguments"],
    ),
    # ---- vLLM tests/reasoning/test_gemma4_reasoning_parser.py (thinking on)
    Imported(
        "vllm-reasoning-thought-prefix",
        "engine-tests",
        vllm(VLLM_REASONING, 106),
        [WEATHER],
        "<|channel>thought\nActual reasoning here<channel|>Final answer",
        ok([], "Final answer", "Actual reasoning here"),
        ["no-call", "reasoning"],
        generation_prompt=THINKING_ON_PROMPT,
        thinking=True,
    ),
    Imported(
        "vllm-reasoning-thought-prefix-multiline",
        "engine-tests",
        vllm(VLLM_REASONING, 118),
        [WEATHER],
        "<|channel>thought\nLine1\nLine2<channel|>Answer",
        ok([], "Answer", "Line1\nLine2"),
        ["no-call", "reasoning"],
        generation_prompt=THINKING_ON_PROMPT,
        thinking=True,
    ),
    # ---- SGLang TestGemma4Detector
    Imported(
        "sglang-text-before-call",
        "engine-tests",
        sglang(5535),
        [WEATHER],
        'Some text before <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>',
        ok([("get_weather", {"location": "Tokyo"})], "Some text before "),
        ["single-call", "text-before-call"],
    ),
    Imported(
        "sglang-text-around-call",
        "engine-tests",
        sglang(5546),
        [WEATHER],
        'Some text before <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|> after',
        ok([("get_weather", {"location": "Tokyo"})], "Some text before  after"),
        ["single-call", "text-before-call", "text-after-call"],
        "The streamed chunks of the test, joined. Content is all text outside the call.",
    ),
    Imported(
        "sglang-nested-array-with-spaces",
        "engine-tests",
        sglang(5579),
        [WEATHER_NESTED],
        '<|tool_call>call:get_weather{location:<|"|>New York<|"|>,nested:[1, 2, {inner:<|"|>val<|"|>}]}<tool_call|>',
        ok([("get_weather", {"location": "New York", "nested": [1, 2, {"inner": "val"}]})]),
        ["single-call", "nested-json"],
        "The streamed chunks of the test, joined. Array items are separated by ', ' (not the template's ',').",
    ),
    Imported(
        "sglang-nested-object",
        "engine-tests",
        sglang(5618),
        [WEATHER_NESTED],
        '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>,details:{temp:25,unit:<|"|>celsius<|"|>}}<tool_call|>',
        ok([("get_weather", {"location": "Tokyo", "details": {"temp": 25, "unit": "celsius"}})]),
        ["single-call", "nested-json", "numeric-arguments"],
    ),
    Imported(
        "vllm-malformed-no-brace",
        "malformed",
        vllm(VLLM_TOOL, 942),
        [WEATHER],
        "<|tool_call>call:bad_func no brace<tool_call|>",
        {
            "expected_error": {
                "reason": "The call has no '{...}' argument object and names no offered tool; it is not a valid call.",
                "accept": ["no_tool_calls", "content_passthrough"],
            }
        },
        ["malformed"],
        "vLLM's test only asserts that the recovered name stays bounded by <tool_call|> (it returns a call named "
        "'bad_func no brace'); per the format there is no valid call here.",
    ),
    # ---- bug reports
    Imported(
        "bug-ollama-18390-key-with-spaces",
        "bug-reports",
        {
            "kind": "bug_report",
            "source_url": "https://github.com/ollama/ollama/issues/18390",
            "revision": "issue body (2026-09-11)",
            "license": "NOASSERTION",
        },
        [CREATE_WORKFLOW],
        '<|tool_call>call:create_workflow{name:<|"|>Demo<|"|>, connections:{Basic LLM Chain:{main:[[{node:<|"|>X<|"|>, '
        'type:<|"|>main<|"|>, index:0}]]}}}<tool_call|>',
        ok(
            [
                (
                    "create_workflow",
                    {
                        "name": "Demo",
                        "connections": {"Basic LLM Chain": {"main": [[{"node": "X", "type": "main", "index": 0}]]}},
                    },
                )
            ]
        ),
        ["single-call", "nested-json", "regression", "x-key-with-space"],
        "The 'bare key with spaces' input of the issue's parser-level Go test. Ollama dropped the whole call.",
    ),
]
