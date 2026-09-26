"""GLM fixtures copied from engine test suites and public bug reports.

Each item quotes the raw model output verbatim from its source (line-anchored at a
pinned commit) and states the parse a correct parser must produce. ``build.py``
adds token ids by encoding the raw text with the reference tokenizer (special
tokens written literally map to their ids, as the model emits them) and checks
that the encoding round-trips.

Sources:
* vLLM v0.30.0 (Apache-2.0), commit ced6857afa0ea7b2e3f0846a62e1394e90f15607
* Ollama (MIT), commit 7af393188defd52d370464de0d2064649cab9b41
* public GitHub issues (quoted, NOASSERTION)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from _shared.core import Provenance, Record, Tool, tool

VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
OLLAMA_SHA = "7af393188defd52d370464de0d2064649cab9b41"
VLLM_ATTR = "Copyright contributors to the vLLM project (Apache-2.0)"
GEN_PROMPT_NO_THINK = {"glm45": "<|assistant|>\n<think></think>", "glm47": "<|assistant|></think>"}


def vllm(path: str, lines: str) -> Provenance:
    return Provenance(
        "engine_test",
        f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/{path}#{lines}",
        VLLM_SHA,
        "Apache-2.0",
        "scripts/fixtures/glm/imported.py",
        attribution=VLLM_ATTR,
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
    extra_models: list[str] = field(default_factory=list)

    def record(self, slug: str, repo: str, tokenizer: dict[str, str], ids: list[int]) -> Record:
        gen_prompt = GEN_PROMPT_NO_THINK[self.model_key] if self.thinking is False else None
        return Record(
            id=f"{slug}/{self.model_key}-{self.name}",
            family=slug,
            models=[repo, *self.extra_models],
            provenance=self.provenance,
            tools=self.tools,
            raw_output=self.raw,
            output_token_ids=ids,
            tokenizer=tokenizer,
            generation_prompt=gen_prompt,
            thinking=self.thinking,
            expected=self.expected,
            expected_error=self.expected_error,
            tags=[*self.tags, f"x-{self.model_key}"],
            notes=self.notes,
        )


def calls(*pairs: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": n, "arguments": a} for n, a in pairs]


# Tools exactly as declared in tests/tool_parsers/test_glm4_moe_tool_parser.py (L39-L71).
GLM45_TEST_TOOLS = [
    tool(
        "get_current_weather",
        None,
        {"city": {"type": "string"}, "state": {"type": "string"}, "unit": {"type": "string"}},
    ),
    tool(
        "calculate",
        None,
        {
            "operation": {"type": "string"},
            "a": {"type": "number"},
            "b": {"type": "number"},
            "enabled": {"type": "boolean"},
        },
    ),
    {"type": "function", "function": {"name": "get_time", "parameters": {}}},
]
# Tools exactly as declared in tests/tool_parsers/test_glm47_moe_tool_parser.py (L31-L49).
GLM47_TEST_TOOLS = [
    {"type": "function", "function": {"name": "get_current_date", "parameters": {}}},
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
            },
        },
    },
]

T45 = "tests/tool_parsers/test_glm4_moe_tool_parser.py"
T47 = "tests/tool_parsers/test_glm47_moe_tool_parser.py"
R47 = "tests/reasoning/test_glm4_moe_reasoning_parser.py"

ITEMS: list[Item] = [
    Item(
        "vllm-newline-format-content-before",
        "glm45",
        "I'll check it. <tool_call>get_current_weather\n"
        "<arg_key>city</arg_key>\n<arg_value>Dallas</arg_value>\n"
        "<arg_key>state</arg_key>\n<arg_value>TX</arg_value>\n"
        "<arg_key>unit</arg_key>\n<arg_value>fahrenheit</arg_value>\n"
        "</tool_call>",
        GLM45_TEST_TOOLS,
        vllm(T45, "L114-L137"),
        ["single-call", "text-before-call", "x-whitespace-variant-45"],
        expected={
            "content": "I'll check it.",
            "reasoning_content": None,
            "tool_calls": calls(("get_current_weather", {"city": "Dallas", "state": "TX", "unit": "fahrenheit"})),
        },
        thinking=False,
        notes="vLLM expects the separator space before <tool_call> to be dropped from content.",
    ),
    Item(
        "vllm-newline-format-two-calls",
        "glm45",
        "<tool_call>get_current_weather\n<arg_key>city</arg_key><arg_value>Dallas</arg_value>\n</tool_call>\n"
        "<tool_call>get_current_weather\n<arg_key>city</arg_key><arg_value>Orlando</arg_value>\n</tool_call>",
        GLM45_TEST_TOOLS,
        vllm(T45, "L140-L159"),
        ["parallel-calls", "x-whitespace-variant-45"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(
                ("get_current_weather", {"city": "Dallas"}), ("get_current_weather", {"city": "Orlando"})
            ),
        },
        thinking=False,
        notes="The newline between the two calls is a separator, not content.",
    ),
    Item(
        "vllm-coerces-schema-types",
        "glm45",
        "<tool_call>calculate\n<arg_key>operation</arg_key><arg_value>add</arg_value>\n"
        "<arg_key>a</arg_key><arg_value>42</arg_value>\n<arg_key>b</arg_key><arg_value>3.14</arg_value>\n"
        "<arg_key>enabled</arg_key><arg_value>true</arg_value>\n</tool_call>",
        GLM45_TEST_TOOLS,
        vllm(T45, "L162-L180"),
        ["single-call", "numeric-arguments", "x-schema-coercion"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(("calculate", {"operation": "add", "a": 42, "b": 3.14, "enabled": True})),
        },
        thinking=False,
    ),
    Item(
        "vllm-zero-arg-newline",
        "glm45",
        "<tool_call>get_time\n</tool_call>",
        GLM45_TEST_TOOLS,
        vllm(T45, "L183-L194"),
        ["single-call", "empty-arguments", "x-whitespace-variant-45"],
        expected={"content": None, "reasoning_content": None, "tool_calls": calls(("get_time", {}))},
        thinking=False,
    ),
    Item(
        "vllm-with-think-tags",
        "glm45",
        "<think>This is a reasoning section</think>This is the rest",
        GLM47_TEST_TOOLS,
        vllm(R47, "L22-L27"),
        ["no-call", "reasoning"],
        expected={"content": "This is the rest", "reasoning_content": "This is a reasoning section", "tool_calls": []},
        notes="From the glm45 reasoning-parser test (WITH_THINK). GLM-4.5/4.6 generate the <think> opener "
        "themselves (the generation prompt is just <|assistant|>), so this shape is a GLM-4.5 completion. The test "
        "runs against the GLM-4.7 tokenizer; the tag ids are the same (151350/151351).",
    ),
    Item(
        "ollama-content-after-call",
        "glm45",
        "<think>thinking</think><tool_call>test</tool_call>after tool",
        [{"type": "function", "function": {"name": "test", "parameters": {"type": "object", "properties": {}}}}],
        Provenance(
            "engine_test",
            f"https://github.com/ollama/ollama/blob/{OLLAMA_SHA}/model/parsers/glm46_test.go#L108-L119",
            OLLAMA_SHA,
            "MIT",
            "scripts/fixtures/glm/imported.py",
            attribution="Copyright (c) Ollama (MIT)",
        ),
        ["single-call", "empty-arguments", "reasoning", "text-after-call"],
        expected={
            "content": "after tool",
            "reasoning_content": "thinking",
            "tool_calls": calls(("test", {})),
        },
        notes="From Ollama's GLM-4.6 parser test ('tool call with content after'). GLM-4.5/4.6 open <think> "
        "themselves, so this is a complete completion after the <|assistant|> generation prompt. The tool schema "
        "is a stand-in; the test declares no tools.",
    ),
    Item(
        "vllm-whitespace-preserved-in-value",
        "glm47",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>",
        GLM47_TEST_TOOLS,
        vllm(T47, "L160-L164"),
        ["single-call", "x-significant-whitespace", "x-whitespace-variant-47"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(("get_weather", {"city": "  Beijing  "})),
        },
        thinking=False,
    ),
    Item(
        "vllm-content-before-no-space",
        "glm47",
        "Checking.<tool_call>get_current_date</tool_call>",
        GLM47_TEST_TOOLS,
        vllm(T47, "L166-L170"),
        ["single-call", "text-before-call", "empty-arguments", "x-whitespace-variant-47"],
        expected={"content": "Checking.", "reasoning_content": None, "tool_calls": calls(("get_current_date", {}))},
        thinking=False,
    ),
    Item(
        "vllm-whitespace-only-content",
        "glm47",
        "  \n  <tool_call>get_current_date</tool_call>",
        GLM47_TEST_TOOLS,
        vllm(T47, "L185-L188"),
        ["single-call", "empty-arguments", "x-whitespace-variant-47"],
        expected={"content": None, "reasoning_content": None, "tool_calls": calls(("get_current_date", {}))},
        thinking=False,
        notes="Whitespace-only text before the call is not content (vLLM asserts content is None).",
    ),
    Item(
        "vllm-args-with-newlines",
        "glm47",
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>",
        GLM47_TEST_TOOLS,
        vllm(T47, "L154-L158"),
        ["single-call", "x-whitespace-variant-45"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(("get_weather", {"city": "Beijing"})),
        },
        thinking=False,
        notes="GLM-4.5-style newlines inside a GLM-4.7 completion; the glm47 parser must accept both variants.",
    ),
    Item(
        "vllm-reasoning-without-open-tag",
        "glm47",
        "This is a reasoning section</think>This is the rest",
        GLM47_TEST_TOOLS,
        vllm(R47, "L50-L55"),
        ["no-call", "reasoning", "x-think-no-open-tag"],
        expected={"content": "This is the rest", "reasoning_content": "This is a reasoning section", "tool_calls": []},
        notes="GLM-4.7's generation prompt ends with <think>, so the completion contains only </think>.",
    ),
    Item(
        "bug-three-calls-same-tool",
        "glm47",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>\n"
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>B</arg_value></tool_call>\n"
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>C</arg_value></tool_call>",
        [
            tool("get_weather", None, {"city": {"type": "string"}}),
            tool("search", None, {"query": {"type": "string"}}),
        ],
        Provenance(
            "bug_report",
            "https://github.com/sgl-project/sglang/issues/33324",
            "issue opened 2026-08-03",
            "NOASSERTION",
            "scripts/fixtures/glm/imported.py",
            attribution="Quoted from the issue's concrete example (tools [get_weather, search]).",
        ),
        ["parallel-calls", "regression", "x-call-index-vs-tool-index"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": calls(
                ("get_weather", {"city": "A"}), ("get_weather", {"city": "B"}), ("get_weather", {"city": "C"})
            ),
        },
        thinking=False,
        notes="SGLang's non-streaming path indexed calls by tool position ([0,0,0]), so OpenAI clients merged them. "
        "Tool schemas are minimal stand-ins; the issue only names the tools.",
    ),
]
