"""Llama fixtures copied from engine test suites and public bug reports.

Each item quotes the raw model output verbatim from its source (line-anchored at a
pinned commit) and states the parse a correct parser must produce. ``build.py``
adds token ids by encoding the raw text with the (mirrored) reference tokenizer and
checks that the encoding round-trips.

Sources:
* vLLM v0.30.0 (Apache-2.0), commit ced6857afa0ea7b2e3f0846a62e1394e90f15607.
  The llama3_json tests load meta-llama/Llama-3.2-1B-Instruct, whose vocabulary
  is the Llama 3 one used for every Llama 3.x fixture here.
* SGLang v0.5.20 (Apache-2.0), commit 94602c9c2b7cbdb8efd5c52802dac6a1c180089e
* public GitHub issues (quoted, NOASSERTION)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from _core import Provenance, Record, Tool, tool

VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
SGLANG_SHA = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
GEN = "scripts/fixtures/llama/imported.py"


def vllm(path: str, lines: str) -> Provenance:
    return Provenance(
        "engine_test",
        f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/{path}#{lines}",
        VLLM_SHA,
        "Apache-2.0",
        GEN,
        attribution="Copyright contributors to the vLLM project (Apache-2.0)",
    )


def sglang(path: str, lines: str) -> Provenance:
    return Provenance(
        "engine_test",
        f"https://github.com/sgl-project/sglang/blob/{SGLANG_SHA}/{path}#{lines}",
        SGLANG_SHA,
        "Apache-2.0",
        GEN,
        attribution="Copyright SGLang contributors (Apache-2.0)",
    )


def issue(url: str, revision: str, attribution: str) -> Provenance:
    return Provenance("bug_report", url, revision, "NOASSERTION", GEN, attribution=attribution)


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
    notes: str | None = None

    def record(self, slug: str, repo: str, tokenizer: dict[str, str], ids: list[int]) -> Record:
        prefix = "l4" if self.model_key == "l4" else "l3"
        sub = "x-llama-pythonic" if self.model_key == "l4" else "x-llama-json"
        return Record(
            id=f"{slug}/{prefix}-{self.name}",
            family=slug,
            models=[repo],
            provenance=self.provenance,
            tools=self.tools,
            raw_output=self.raw,
            output_token_ids=ids,
            tokenizer=tokenizer,
            expected=self.expected,
            expected_error=self.expected_error,
            tags=[*self.tags, sub],
            notes=self.notes,
        )


def exp(content: str | None, *calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "content": content,
        "reasoning_content": None,
        "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
    }


def stand_in(tool_name: str, props: dict[str, Any] | None = None) -> Tool:
    return tool(tool_name, None, {k: v if isinstance(v, dict) else {"type": v} for k, v in (props or {}).items()})


STANDIN = "The engine test runs without a tools list; the tool schemas here are minimal stand-ins."
L3J = "tests/tool_parsers/test_llama3_json_tool_parser.py"
L4P = "tests/tool_parsers/test_llama4_pythonic_tool_parser.py"
SG32 = "test/registered/unit/function_call/test_llama32_detector.py"

# tests/tool_parsers/test_llama4_pythonic_tool_parser.py (L17-L63) as tools.
L4_TOOLS = [
    stand_in("get_weather", {"city": "string", "metric": "string"}),
    stand_in(
        "register_user",
        {
            "name": "string",
            "age": "integer",
            "address": "object",
            "role": {"type": ["string", "null"]},
            "passed_test": "boolean",
            "aliases": {"type": "array", "items": {"type": "string"}},
        },
    ),
]
# SGLang tests/.../test_llama32_detector.py L15-L51, verbatim.
SG_TOOLS = [
    tool(
        "get_weather",
        "Get weather information",
        {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        ["city"],
    ),
    tool("search", "Search the web", {"query": {"type": "string", "description": "Search query"}}, ["query"]),
]

ITEMS: list[Item] = [
    # ---- Llama 3.x JSON: vLLM llama3_json
    Item(
        "vllm-parameters-key",
        "l33",
        '{"name": "searchTool", "parameters": {"query": "test query", "limit": 10}}',
        [stand_in("searchTool", {"query": "string", "limit": "integer"})],
        vllm(L3J, "L42-L53"),
        ["single-call", "numeric-arguments", "x-parameters-key"],
        expected=exp(None, ("searchTool", {"query": "test query", "limit": 10})),
        notes=STANDIN,
    ),
    Item(
        "vllm-arguments-key",
        "l33",
        '{"name": "searchTool", "arguments": {"query": "test"}}',
        [stand_in("searchTool", {"query": "string"})],
        vllm(L3J, "L76-L84"),
        ["single-call", "x-arguments-key"],
        expected=exp(None, ("searchTool", {"query": "test"})),
        notes=STANDIN + " Llama's documented key is 'parameters'; models also emit 'arguments', which vLLM and SGLang "
        "accept.",
    ),
    Item(
        "vllm-deeply-nested",
        "l33",
        '{"name": "complexTool", "parameters": {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}}',
        [stand_in("complexTool", {"level1": "object"})],
        vllm(L3J, "L146-L167"),
        ["single-call", "nested-json", "x-parameters-key"],
        expected=exp(None, ("complexTool", {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}})),
        notes=STANDIN,
    ),
    Item(
        "vllm-brackets-in-strings",
        "l33",
        '{"name": "searchTool", "parameters": {'
        '"query": "test {value} [complex]",'
        '"nested": {"inner": "more {brackets}"}'
        "}}",
        [stand_in("searchTool", {"query": "string", "nested": "object"})],
        vllm(L3J, "L195-L214"),
        ["single-call", "nested-json", "string-escapes", "x-parameters-key"],
        expected=exp(None, ("searchTool", {"query": "test {value} [complex]", "nested": {"inner": "more {brackets}"}})),
        notes=STANDIN + " Braces and brackets inside strings must not confuse brace matching.",
    ),
    Item(
        "vllm-escaped-quotes",
        "l33",
        '{"name": "parserTool", "parameters": {"text": "He said \\"Hello {world}\\""}}',
        [stand_in("parserTool", {"text": "string"})],
        vllm(L3J, "L217-L231"),
        ["single-call", "string-escapes", "x-parameters-key"],
        expected=exp(None, ("parserTool", {"text": 'He said "Hello {world}"'})),
        notes=STANDIN,
    ),
    Item(
        "vllm-json-without-name-is-content",
        "l33",
        '{"parameters": {}}',
        [stand_in("searchTool", {"query": "string"})],
        vllm(L3J, "L234-L241"),
        ["no-call", "x-json-in-content-not-a-call"],
        expected=exp('{"parameters": {}}'),
        notes=STANDIN + " A JSON object without a 'name' is not a call; the text is returned unchanged as content.",
    ),
    # ---- Llama 3.x JSON: SGLang llama3 (Llama32Detector)
    Item(
        "sglang-python-tag-call",
        "l33",
        '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Beijing"}}',
        SG_TOOLS,
        sglang(SG32, "L70-L76"),
        ["single-call", "x-python-tag-prefix", "x-arguments-key"],
        expected=exp(None, ("get_weather", {"city": "Beijing"})),
    ),
    Item(
        "sglang-text-before-python-tag",
        "l33",
        'Let me check. <|python_tag|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}',
        SG_TOOLS,
        sglang(SG32, "L84-L88"),
        ["single-call", "text-before-call", "x-python-tag-prefix", "x-arguments-key", "x-text-plus-call"],
        expected=exp("Let me check. ", ("get_weather", {"city": "Tokyo"})),
        notes="SGLang asserts the normal text keeps its trailing space.",
    ),
    # ---- Llama 4 pythonic: vLLM llama4_pythonic
    Item(
        "vllm-pythonic-no-call",
        "l4",
        "How can I help you today?",
        L4_TOOLS,
        vllm(L4P, "L66-L78"),
        ["no-call"],
        expected=exp("How can I help you today?"),
    ),
    Item(
        "vllm-pythonic-more-types",
        "l4",
        "[register_user(name='Doe', age=9, address={'city': 'LA', 'state': 'CA'}, role=None, passed_test=True, "
        "aliases=['John', 'Johnny'])]",
        L4_TOOLS,
        vllm(L4P, "L22-L37"),
        ["single-call", "nested-json", "numeric-arguments", "x-pythonic-nested-list"],
        expected=exp(
            None,
            (
                "register_user",
                {
                    "name": "Doe",
                    "age": 9,
                    "address": {"city": "LA", "state": "CA"},
                    "role": None,
                    "passed_test": True,
                    "aliases": ["John", "Johnny"],
                },
            ),
        ),
        notes="Python literals: None/True map to null/true, a dict and a list literal to JSON.",
    ),
    Item(
        "vllm-pythonic-parameterless",
        "l4",
        "[get_weather()]",
        L4_TOOLS,
        vllm(L4P, "L39-L43"),
        ["single-call", "empty-arguments"],
        expected=exp(None, ("get_weather", {})),
    ),
    Item(
        "vllm-pythonic-escaped-strings",
        "l4",
        r"[get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')]",
        L4_TOOLS,
        vllm(L4P, "L54-L60"),
        ["single-call", "string-escapes", "x-pythonic-quote-escaping"],
        expected=exp(None, ("get_weather", {"city": "Martha's Vineyard", "metric": '"cool units"'})),
    ),
    Item(
        "vllm-pythonic-parallel-no-spaces",
        "l4",
        "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]",
        L4_TOOLS,
        vllm(L4P, "L163-L170"),
        ["parallel-calls", "x-pythonic-parallel"],
        expected=exp(
            None, ("get_weather", {"city": "LA", "metric": "C"}), ("register_user", {"name": "Doe", "age": 9})
        ),
    ),
    Item(
        "vllm-pythonic-python-start-end",
        "l4",
        "<|python_start|>[get_weather(city='LA', metric='C')]<|python_end|>",
        L4_TOOLS,
        vllm(L4P, "L61-L63"),
        ["single-call", "x-python-tag-prefix"],
        expected=exp(None, ("get_weather", {"city": "LA", "metric": "C"})),
        notes="Llama 4's <|python_start|>/<|python_end|> wrapper around the call list; neither marker is content.",
    ),
    # ---- bug reports
    Item(
        "bug-leading-json-object-in-content",
        "l33",
        '{"a": 1} is a dict',
        [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        issue(
            "https://github.com/sgl-project/sglang/issues/35562",
            "issue opened 2026-08-19",
            "Quoted from the issue's reproduction (Llama-3.2-1B-Instruct, tools=[get_weather]).",
        ),
        ["no-call", "regression", "x-json-in-content-not-a-call"],
        expected=exp('{"a": 1} is a dict'),
        notes="Not a call (no name/parameters). SGLang's llama3 parser deleted the leading JSON and returned "
        "'is a dict'. The issue's expectation: when no call is found the text must be unchanged.",
    ),
    Item(
        "bug-empty-json-object-in-content",
        "l33",
        "{}",
        [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        issue(
            "https://github.com/sgl-project/sglang/issues/35562",
            "issue opened 2026-08-19",
            "Quoted from the issue: input '{}' -> content '' (the message is gone).",
        ),
        ["no-call", "regression", "x-json-in-content-not-a-call"],
        expected=exp("{}"),
    ),
    Item(
        "bug-whole-call-single-delta",
        "l33",
        '{"name": "get_weather", "arguments": {"city": "Tokyo"}}',
        SG_TOOLS,
        issue(
            "https://github.com/vllm-project/vllm/issues/48294",
            "issue opened 2026-07-11",
            "Quoted from the issue's minimal reproduction for llama3_json.",
        ),
        ["single-call", "regression", "x-single-delta", "x-arguments-key"],
        expected=exp(None, ("get_weather", {"city": "Tokyo"})),
        notes="vLLM's llama3_json streaming path emitted nothing when this whole message arrived in one delta "
        "(the 'one' chunking strategy replays exactly that).",
    ),
    Item(
        "bug-leading-underscore-identifier",
        "l4",
        "[search(_limit=5)]",
        [stand_in("search", {"_limit": "integer"})],
        issue(
            "https://github.com/vllm-project/vllm/issues/56840",
            "issue opened 2026-09-14",
            "Quoted from the issue's OUTPUTS list.",
        ),
        ["single-call", "regression", "x-leading-underscore-identifier"],
        expected=exp(None, ("search", {"_limit": 5})),
        notes="Valid Python; JSON-Schema property names may start with '_'. vLLM's non-streaming pythonic regex "
        "rejected it and returned the text as content, while streaming produced the call.",
    ),
]
