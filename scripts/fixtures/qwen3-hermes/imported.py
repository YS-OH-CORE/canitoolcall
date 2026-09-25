"""``qwen3-hermes`` fixtures imported from engine test suites and bug reports.

Each ``raw`` string is copied verbatim from the cited line (extracted with
``ast.literal_eval``, so implicit string concatenation and line continuations
are resolved exactly as Python does). ``tools`` names refer to ``build.TOOLS``;
where the test used its own tool, the schema is reproduced here from the test.

The expected values are what a correct parser must return per the Qwen3 Hermes
format, which in every case below agrees with the upstream test's assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VLLM_REV = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # v0.30.0
SGLANG_REV = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"  # v0.5.20
VLLM_HERMES = f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/tool_parsers/test_hermes_tool_parser.py"
VLLM_QWEN3_REASONING = (
    f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/reasoning/test_qwen3_reasoning_parser.py"
)
SGLANG_HERMES = f"https://github.com/sgl-project/sglang/blob/{SGLANG_REV}/test/registered/unit/function_call/test_hermes_detector.py"
SGLANG_FCP = f"https://github.com/sgl-project/sglang/blob/{SGLANG_REV}/test/registered/unit/function_call/test_function_call_parser.py"
NO_THINK_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
"""Qwen3 hybrid generation prompt with enable_thinking=false: output without <think> only occurs here."""


def vllm(url: str, line: int) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{url}#L{line}",
        "revision": VLLM_REV,
        "license": "Apache-2.0",
        "attribution": "Copyright contributors to the vLLM project",
    }


def sglang(url: str, line: int) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{url}#L{line}",
        "revision": SGLANG_REV,
        "license": "Apache-2.0",
        "attribution": "Copyright SGLang Team",
    }


def bug(url: str, revision: str) -> dict[str, Any]:
    return {"kind": "bug_report", "source_url": url, "revision": revision, "license": "NOASSERTION"}


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


def err(reason: str, *accept: str) -> dict[str, Any]:
    return {"expected_error": {"reason": reason, "accept": list(accept)}}


def _tool(name: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "parameters": params}}


FINAL_ANSWER = _tool("final_answer", {"trigger": {"type": "boolean"}}, ["trigger"])
GET_CURRENT_TEMPERATURE = _tool(
    "get_current_temperature", {"location": {"type": "string"}, "unit": {"type": "string"}}, ["location"]
)
SEARCH_Q = _tool("search", {"q": {"type": "string"}}, ["q"])
F_X = _tool("f", {"x": {"type": "integer"}})
WEATHER_SG = _tool(
    "get_weather", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]
)
SEARCH_QUERY = _tool("search", {"query": {"type": "string"}}, ["query"])
CURRENT_WEATHER = _tool(
    "get_current_weather",
    {
        "city": {"type": "string"},
        "state": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    ["city", "state", "unit"],
)


@dataclass(frozen=True)
class Imported:
    name: str
    file: str
    model: str
    provenance: dict[str, Any]
    tools: list[Any]
    raw: str
    expected: dict[str, Any]
    tags: list[str]
    notes: str | None = None
    generation_prompt: str | None = NO_THINK_PROMPT
    thinking: bool | None = False


FIXTURES: list[Imported] = [
    # ---- vLLM tests/tool_parsers/test_hermes_tool_parser.py
    Imported(
        "vllm-plain-text-forwarded",
        "engine-tests",
        "hybrid",
        vllm(VLLM_HERMES, 51),
        ["get_weather"],
        "This is some prior text that has nothing to do with tool calling.",
        ok([], "This is some prior text that has nothing to do with tool calling."),
        ["no-call"],
    ),
    Imported(
        "vllm-bug-19056-boolean-argument",
        "engine-tests",
        "hybrid",
        vllm(VLLM_HERMES, 83),
        [FINAL_ANSWER],
        '<tool_call>\n{"name": "final_answer", "arguments": {"trigger": true}}\n</tool_call>',
        ok([("final_answer", {"trigger": True})]),
        ["single-call", "regression"],
        "Regression test for https://github.com/vllm-project/vllm/issues/19056.",
    ),
    Imported(
        "vllm-no-newlines-no-spaces",
        "engine-tests",
        "hybrid",
        vllm(VLLM_HERMES, 117),
        [GET_CURRENT_TEMPERATURE],
        '<tool_call>{"name": "get_current_temperature","arguments": {"location":"San Francisco, California, '
        'United States", "unit": "celsius"}}</tool_call>',
        ok([("get_current_temperature", {"location": "San Francisco, California, United States", "unit": "celsius"})]),
        ["single-call", "x-no-newlines"],
        "No newline inside the tags and irregular JSON spacing; the tags alone delimit the call.",
    ),
    Imported(
        "vllm-text-then-call-no-separator",
        "engine-tests",
        "hybrid",
        vllm(VLLM_HERMES, 227),
        ["get_weather"],
        'Sure, let me check the weather.<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>',
        ok([("get_weather", {"city": "NYC"})], "Sure, let me check the weather."),
        ["single-call", "text-before-call", "x-no-newlines"],
    ),
    Imported(
        "vllm-two-calls-no-separator",
        "engine-tests",
        "hybrid",
        vllm(VLLM_HERMES, 258),
        [SEARCH_Q],
        '<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call>'
        '<tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>',
        ok([("search", {"q": "cats"}), ("search", {"q": "dogs"})]),
        ["parallel-calls", "x-no-newlines"],
    ),
    Imported(
        "vllm-invalid-json-missing-brace",
        "malformed",
        "hybrid",
        vllm(VLLM_HERMES, 381),
        [FINAL_ANSWER],
        '<tool_call>\n{"name": "final_answer", "arguments": {"trigger": true}',
        err(
            "The call JSON is missing its closing brace and there is no </tool_call>; no valid call exists.",
            "no_tool_calls",
            "content_passthrough",
        ),
        ["malformed", "truncated"],
        "vLLM asserts tools_called is false for this output.",
    ),
    Imported(
        "vllm-content-and-call-single-chunk",
        "engine-tests",
        "hybrid",
        vllm(VLLM_HERMES, 397),
        [F_X],
        'Hi!<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>',
        ok([("f", {"x": 1})], "Hi!"),
        ["single-call", "text-before-call", "numeric-arguments", "x-no-newlines"],
    ),
    # ---- SGLang test_hermes_detector.py / test_function_call_parser.py (Qwen25Detector)
    Imported(
        "sglang-text-before-call-with-space",
        "engine-tests",
        "hybrid",
        sglang(SGLANG_HERMES, 86),
        [WEATHER_SG],
        "I will check the weather for you. "
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
        ok([("get_weather", {"city": "Tokyo"})], "I will check the weather for you."),
        ["single-call", "text-before-call", "x-no-newlines"],
        "Content is the text before the call; the separating space is whitespace (soft-v1).",
    ),
    Imported(
        "sglang-malformed-json-in-tags",
        "malformed",
        "hybrid",
        sglang(SGLANG_HERMES, 107),
        [WEATHER_SG],
        "<tool_call>not valid json</tool_call>",
        err(
            "The tool_call body is not JSON. SGLang documents returning the original text as content.",
            "no_tool_calls",
            "content_passthrough",
        ),
        ["malformed"],
    ),
    Imported(
        "sglang-qwen25-four-parallel-calls",
        "engine-tests",
        "hybrid",
        sglang(SGLANG_FCP, 5416),
        [CURRENT_WEATHER],
        '<tool_call>\n{"name": "get_current_weather", "arguments": '
        '{"city": "NYC", "state": "NY", "unit": "fahrenheit"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_current_weather", "arguments": '
        '{"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_current_weather", "arguments": '
        '{"city": "Minneapolis", "state": "MN", "unit": "fahrenheit"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_current_weather", "arguments": '
        '{"city": "Los Angeles", "state": "CA", "unit": "fahrenheit"}}\n</tool_call>',
        ok(
            [
                ("get_current_weather", {"city": "NYC", "state": "NY", "unit": "fahrenheit"}),
                ("get_current_weather", {"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}),
                ("get_current_weather", {"city": "Minneapolis", "state": "MN", "unit": "fahrenheit"}),
                ("get_current_weather", {"city": "Los Angeles", "state": "CA", "unit": "fahrenheit"}),
            ]
        ),
        ["parallel-calls"],
    ),
    # ---- vLLM tests/reasoning/test_qwen3_reasoning_parser.py
    Imported(
        "vllm-reasoning-multiline",
        "engine-tests",
        "hybrid",
        vllm(VLLM_QWEN3_REASONING, 108),
        ["get_weather"],
        "<think>This is a reasoning\nsection</think>This is the rest\nThat",
        ok([], "This is the rest\nThat", "This is a reasoning\nsection"),
        ["no-call", "reasoning"],
        generation_prompt="<|im_start|>assistant\n",
        thinking=True,
    ),
    Imported(
        "vllm-reasoning-without-start-token",
        "engine-tests",
        "thinking",
        vllm(VLLM_QWEN3_REASONING, 34),
        ["get_weather"],
        "This is a reasoning section</think>This is the rest",
        ok([], "This is the rest", "This is a reasoning section"),
        ["no-call", "reasoning", "reasoning-prefilled"],
        "The Thinking-2507 generation prompt ends with <think>, so the output holds only </think>.",
        generation_prompt="<|im_start|>assistant\n<think>\n",
        thinking=None,
    ),
    # ---- bug reports
    Imported(
        "bug-sglang-30480-truncated-mid-arguments",
        "bug-reports",
        "hybrid",
        bug("https://github.com/sgl-project/sglang/issues/30480", "issue body (2026-07)"),
        [_tool("get_weather", {"city": {"type": "string"}})],
        'I will check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "San Fr',
        err(
            "Cut by max_tokens mid-arguments. The issue reports non-streaming leaking the raw <tool_call> markup "
            "into content while streaming drops it; a correct parser returns no call and does not leak markup.",
            "no_tool_calls",
        ),
        ["truncated", "text-before-call", "regression"],
        'Quoted from the issue: input \'I will check.\\n<tool_call>\\n{"name": "get_weather", "arguments": '
        '{"city": "San Fr\'.',
    ),
    Imported(
        "bug-sglang-30480-truncated-at-opener",
        "bug-reports",
        "hybrid",
        bug("https://github.com/sgl-project/sglang/issues/30480", "issue body (2026-07)"),
        [_tool("get_weather", {"city": {"type": "string"}})],
        "I will check.\n<tool_call>",
        err(
            "Cut by max_tokens right at the <tool_call> opener. The issue reports the Qwen25 detector leaking the "
            "bare opener into non-streaming content.",
            "no_tool_calls",
        ),
        ["truncated", "text-before-call", "regression"],
        "Quoted from the issue: input 'I will check.\\n<tool_call>'.",
    ),
]
