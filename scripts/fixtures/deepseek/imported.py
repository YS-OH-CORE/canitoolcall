# ruff: noqa: RUF001  (DeepSeek markers use U+FF5C and U+2581 on purpose)
"""``deepseek`` fixtures imported from engine test suites.

Each ``raw`` string is copied verbatim from the cited line (extracted with
``ast.literal_eval``; where the test builds the string from module constants,
the construction is reproduced exactly and noted). Tool schemas are minimal
ones matching the calls (the upstream tests pass tools only to satisfy the
parser API). For DSML variants with ``oracle=True``, ``build.py`` also checks
``expected`` against DeepSeek's reference ``parse_message_from_completion_text``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VLLM_REV = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # v0.30.0
SGLANG_REV = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"  # v0.5.20
VLLM_V3 = f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/tool_parsers/test_deepseekv3_tool_parser.py"
VLLM_V31 = f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/tool_parsers/test_deepseekv31_tool_parser.py"
VLLM_V41 = f"https://github.com/vllm-project/vllm/blob/{VLLM_REV}/tests/parser/engine/test_deepseek_v41.py"
SGLANG_FCP = f"https://github.com/sgl-project/sglang/blob/{SGLANG_REV}/test/registered/unit/function_call/test_function_call_parser.py"
CHAT_V3 = "<｜Assistant｜>"
CHAT = "<｜Assistant｜></think>"
THINK = "<｜Assistant｜><think>"


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


def err(reason: str, *accept: str) -> dict[str, Any]:
    return {"expected_error": {"reason": reason, "accept": list(accept)}}


def _tool(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": properties}}}


WEATHER = _tool("get_weather", {"city": {"type": "string"}, "unit": {"type": "string"}, "count": {"type": "integer"}})
HOTELS = _tool("search_hotels", {"location": {"type": "string"}, "check_in": {"type": "string"}})
TEST_FUNCTION = _tool(
    "test_function",
    {
        "string_field": {"type": "string"},
        "int_field": {"type": "integer"},
        "float_field": {"type": "number"},
        "bool_field": {"type": "boolean"},
        "null_field": {"type": "null"},
        "array_field": {"type": "array"},
        "object_field": {"type": "object"},
        "empty_array": {"type": "array"},
        "empty_object": {"type": "object"},
    },
)
CURRENT_TIME = _tool("get_current_time", {})
SEND_MESSAGE = _tool(
    "send_message", {"text": {"type": "string"}, "path": {"type": "string"}, "newline": {"type": "string"}}
)
FOO = _tool("foo", {"x": {"type": "integer"}})
ADD = _tool("add", {"x": {"type": "number"}, "y": {"type": "number"}})
GET_DATE = _tool("get_date", {})

V41_CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
V41_EXPECTED_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_weather", {"city": "杭州", "count": 42}),
    ("add", {"x": 1.5, "y": 2.25}),
]


@dataclass(frozen=True)
class Imported:
    name: str
    file: str
    variant: str
    provenance: dict[str, Any]
    tools: list[Any]
    raw: str
    expected: dict[str, Any]
    tags: list[str]
    notes: str | None = None
    generation_prompt: str = CHAT
    thinking: bool = False
    oracle: bool = False


FIXTURES: list[Imported] = [
    # ---- vLLM tests/tool_parsers/test_deepseekv3_tool_parser.py (V3 / R1 format)
    Imported(
        "vllm-v3-no-tool-calls",
        "engine-tests",
        "v3",
        vllm(VLLM_V3, 25),
        [WEATHER],
        "How can I help you today? I can check weather for you.",
        ok([], "How can I help you today? I can check weather for you."),
        ["no-call"],
        generation_prompt=CHAT_V3,
    ),
    Imported(
        "vllm-v3-single-call",
        "engine-tests",
        "v3",
        vllm(VLLM_V3, 27),
        [WEATHER],
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Tokyo", '
        '"unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        ok([("get_weather", {"city": "Tokyo", "unit": "celsius"})]),
        ["single-call"],
        generation_prompt=CHAT_V3,
    ),
    Imported(
        "vllm-v3-parallel-no-newline",
        "engine-tests",
        "v3",
        vllm(VLLM_V3, 31),
        [WEATHER, HOTELS],
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Tokyo", '
        '"unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search_hotels\n```json\n'
        '{"location": "Tokyo", "check_in": "2025-01-15"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        ok(
            [
                ("get_weather", {"city": "Tokyo", "unit": "celsius"}),
                ("search_hotels", {"location": "Tokyo", "check_in": "2025-01-15"}),
            ]
        ),
        ["parallel-calls"],
        "Calls are adjacent here; the official template separates them with '\\n'.",
        generation_prompt=CHAT_V3,
    ),
    Imported(
        "vllm-v3-various-data-types",
        "engine-tests",
        "v3",
        vllm(VLLM_V3, 39),
        [TEST_FUNCTION],
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test_function\n```json\n{"string_field": '
        '"hello", "int_field": 42, "float_field": 3.14, "bool_field": true, "null_field": null, "array_field": '
        '["a", "b", "c"], "object_field": {"nested": "value"}, "empty_array": [], "empty_object": {}}\n```'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ok(
            [
                (
                    "test_function",
                    {
                        "string_field": "hello",
                        "int_field": 42,
                        "float_field": 3.14,
                        "bool_field": True,
                        "null_field": None,
                        "array_field": ["a", "b", "c"],
                        "object_field": {"nested": "value"},
                        "empty_array": [],
                        "empty_object": {},
                    },
                )
            ]
        ),
        ["single-call", "numeric-arguments", "nested-json"],
        generation_prompt=CHAT_V3,
    ),
    Imported(
        "vllm-v3-escaped-strings",
        "engine-tests",
        "v3",
        vllm(VLLM_V3, 61),
        [SEND_MESSAGE],
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>send_message\n```json\n{"text": "He said '
        '\\"hello\\"", "path": "C:\\\\Users\\\\file", "newline": "line1\\nline2"}\n```<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>",
        ok([("send_message", {"text": 'He said "hello"', "path": "C:\\Users\\file", "newline": "line1\nline2"})]),
        ["single-call", "string-escapes"],
        generation_prompt=CHAT_V3,
    ),
    Imported(
        "vllm-v3-malformed-missing-brace",
        "malformed",
        "v3",
        vllm(VLLM_V3, 68),
        [WEATHER],
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Tokyo"\n```'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        err(
            "The arguments JSON is missing its closing brace; no valid call exists.",
            "no_tool_calls",
            "content_passthrough",
        ),
        ["malformed"],
        "malformed_input_outputs[0] of the test.",
        generation_prompt=CHAT_V3,
    ),
    Imported(
        "vllm-v3-malformed-missing-call-tokens",
        "malformed",
        "v3",
        vllm(VLLM_V3, 68),
        [WEATHER],
        '<｜tool▁calls▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Tokyo"}\n```<｜tool▁calls▁end｜>',
        err(
            "<｜tool▁call▁begin｜>/<｜tool▁call▁end｜> are missing inside the calls section; "
            "no well-formed call exists.",
            "no_tool_calls",
            "content_passthrough",
        ),
        ["malformed"],
        "malformed_input_outputs[1] of the test.",
        generation_prompt=CHAT_V3,
    ),
    # ---- vLLM tests/tool_parsers/test_deepseekv31_tool_parser.py
    Imported(
        "vllm-v31-text-before-call",
        "engine-tests",
        "v31",
        vllm(VLLM_V31, 25),
        [FOO],
        'normal text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>",
        ok([("foo", {"x": 1})], "normal text"),
        ["single-call", "text-before-call"],
    ),
    # ---- vLLM tests/parser/engine/test_deepseek_v41.py (text = "Checking." + CALLS, with/without "Plan.</think>")
    Imported(
        "vllm-v41-text-before-parallel-calls",
        "engine-tests",
        "v41",
        vllm(VLLM_V41, 59),
        [WEATHER, ADD],
        "Checking." + V41_CALLS,
        ok(V41_EXPECTED_CALLS, "Checking."),
        ["parallel-calls", "text-before-call", "unicode", "numeric-arguments"],
        "text = 'Checking.' + CALLS (CALLS defined at line 18), thinking=False.",
        oracle=True,
    ),
    Imported(
        "vllm-v41-reasoning-text-before-parallel-calls",
        "engine-tests",
        "v41",
        vllm(VLLM_V41, 59),
        [WEATHER, ADD],
        "Plan.</think>Checking." + V41_CALLS,
        ok(V41_EXPECTED_CALLS, "Checking.", "Plan."),
        ["parallel-calls", "text-before-call", "reasoning", "reasoning-prefilled", "unicode", "numeric-arguments"],
        "text = 'Plan.</think>' + 'Checking.' + CALLS (CALLS defined at line 18), thinking=True.",
        generation_prompt=THINK,
        thinking=True,
        oracle=True,
    ),
    # ---- SGLang TestDeepSeekV32Detector
    Imported(
        "sglang-v32-text-before-empty-invoke",
        "engine-tests",
        "v32",
        sglang(1802),
        [GET_DATE],
        'Let me get the current date for you.\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_date">\n'
        "</｜DSML｜invoke>\n</｜DSML｜function_calls>",
        ok([("get_date", {})], "Let me get the current date for you."),
        ["single-call", "text-before-call", "empty-arguments"],
        "An invoke with no parameter lines (the encoder itself renders an empty line between the tags).",
        oracle=True,
    ),
    # ---- SGLang TestDeepSeekV4Detector
    Imported(
        "sglang-v4-self-closing-invoke",
        "engine-tests",
        "v4",
        sglang(2258),
        [_tool("submit", {})],
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="submit"/>\n</｜DSML｜tool_calls>',
        ok([("submit", {})]),
        ["single-call", "empty-arguments", "x-self-closing-invoke"],
        "SGLang's test states 'V4 emits <｜DSML｜invoke name=\"x\"/> for zero-arg tools'. The official V4 encoder "
        "renders an open/close pair instead, and DeepSeek's reference parser rejects this form; kept because the "
        "engine reports the model generating it.",
    ),
]
