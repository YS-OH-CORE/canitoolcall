# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mistral-common[sentencepiece]==1.12.0", "transformers==5.17.0", "huggingface_hub>=1.0"]
# ///
"""Import Mistral fixtures from engine test suites and public bug reports.

Every raw output is COPIED VERBATIM from the cited source (file + line range at a pinned commit, or
the issue URL). Token ids are built the way the source tests build them: control markers
([TOOL_CALLS], [ARGS], [CALL_ID], [THINK], [/THINK]) become their control-token ids and the text
between them is encoded with the model's tokenizer (vLLM ``encode_mistral_output`` /
``_encode_v13``). Pre-v11 (v3) outputs use the HF tokenizer of Mistral-7B-Instruct-v0.3, as vLLM's
``mistral_pre_v11_tokenizer`` fixture does. The decode round trip is asserted for every fixture.

The sources' requests do not always define tools; tool schemas are written to match the arguments.

Run from the repo root::

    uv run --script scripts/fixtures/mistral/import_mistral.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    MAGISTRAL_2509,
    MINISTRAL3_REASONING,
    MISTRAL7B_V03,
    OUT_DIR,
    SMALL32,
    Ref,
    encode_with_markers,
    expected,
    record,
    write_jsonl,
)

GENERATOR = "scripts/fixtures/mistral/import_mistral.py"
VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # vLLM v0.30.0
VLLM_TOOLS = f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/tests/parser/mistral/test_tool_calls.py"
VLLM_REASON = f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/tests/parser/mistral/test_reasoning.py"
VLLM_ATTR = "Copyright contributors to the vLLM project (Apache-2.0)."
LLAMA_SHA = "7fe450e19305b828c199d602c23a8337aaa1f03b"  # commit of llama.cpp tag v0.5.0 (tag object c13fcbf6)
LLAMA_TEST = f"https://github.com/ggml-org/llama.cpp/blob/{LLAMA_SHA}/tests/test-chat.cpp"
LLAMA_ATTR = "Copyright (c) 2023-2026 The ggml authors (MIT)."

DEVSTRAL_2507 = Ref("mistralai/Devstral-Small-2507", "bd165ab26cebbcc2eea2c4ecbfc07f3ac42b3c39")
MAGISTRAL_2506 = Ref("mistralai/Magistral-Small-2506", "ad2fce5b4113139e1877dcadfd826c9262ad8e8c")  # v11, text <think>


def tool(
    name: str, props: dict[str, Any], required: list[str] | None = None, desc: str | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {"type": "object", "properties": props}
    if required is not None:
        params["required"] = required
    fn: dict[str, Any] = {"name": name, "parameters": params}
    if desc:
        fn["description"] = desc
    return {"type": "function", "function": fn}


NUM = {"type": "number"}
STR = {"type": "string"}
T_ADD = tool("add", {"a": NUM, "b": NUM})
T_MULTIPLY = tool("multiply", {"a": NUM, "b": NUM})
T_ADD_THIS = tool("add_this_and_that", {"a": NUM, "b": NUM})
T_WEATHER = tool(
    "get_current_weather", {"city": STR, "state": STR, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}
)
T_AGE = tool("get_age", {"name": STR})
# vLLM tests/parser/mistral/test_reasoning.py _SAMPLE_TOOLS (L27-L40)
T_GET_WEATHER = tool("get_weather", {"city": STR}, ["city"], "Get the current weather in a city.")
# llama.cpp tests/test-chat.cpp special_function_tool (L425-L437), special_function_tool_with_optional_param (L439-L456)
T_SPECIAL = tool("special_function", {"arg1": {"type": "integer", "description": "The arg."}}, ["arg1"], "I'm special")
T_SPECIAL_OPT = tool(
    "special_function_with_opt",
    {
        "arg1": {"type": "integer", "description": "The arg."},
        "arg2": {"type": "integer", "description": "The optional arg."},
    },
    ["arg1"],
    "I'm special but have optional stuff",
)
SF = {"arg1": 1}
SFO = {"arg1": 1, "arg2": 2}
WEATHER_SF = {"city": "San Francisco", "state": "CA", "unit": "celsius"}


def vllm(url: str, line: str) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{url}#{line}",
        "revision": VLLM_SHA,
        "license": "Apache-2.0",
        "generator": GENERATOR,
        "attribution": VLLM_ATTR,
    }


def llama(line: str) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{LLAMA_TEST}#{line}",
        "revision": LLAMA_SHA,
        "license": "MIT",
        "generator": GENERATOR,
        "attribution": LLAMA_ATTR,
    }


CASES: list[dict[str, Any]] = [
    # ---------------------------------------------------------------- vLLM pre-v11 (JSON array)
    dict(
        name="vllm-v3-no-space",
        ref=MISTRAL7B_V03,
        prov=vllm(VLLM_TOOLS, "L359-L369"),
        text='[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5, "b": 4}}]',
        tools=[T_ADD],
        expected=expected(None, None, [("add", {"a": 3.5, "b": 4})]),
        tags=["single-call", "numeric-arguments", "x-v3-json-array"],
        notes="test_extract_tool_calls_pre_v11_tokenizer (single_tool_add).",
    ),
    dict(
        name="vllm-v3-arguments-before-name",
        ref=MISTRAL7B_V03,
        prov=vllm(VLLM_TOOLS, "L384-L397"),
        text='[TOOL_CALLS] [{"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, '
        '"name": "get_current_weather"}]',
        tools=[T_WEATHER],
        expected=expected(None, None, [("get_current_weather", WEATHER_SF)]),
        tags=["single-call", "x-v3-json-array", "x-key-order"],
        notes="test_extract_tool_calls_pre_v11_tokenizer (argument_before_name).",
    ),
    dict(
        name="vllm-v3-name-inside-arguments",
        ref=MISTRAL7B_V03,
        prov=vllm(VLLM_TOOLS, "L398-L413"),
        text='[TOOL_CALLS] [{"arguments":{"name": "John Doe"}, "name": "get_age"}]',
        tools=[T_AGE],
        expected=expected(None, None, [("get_age", {"name": "John Doe"})]),
        tags=["single-call", "x-v3-json-array", "x-param-named-name", "x-key-order"],
        notes="test_extract_tool_calls_pre_v11_tokenizer (argument_before_name_and_name_in_argument).",
    ),
    dict(
        name="vllm-v3-parallel",
        ref=MISTRAL7B_V03,
        prov=vllm(VLLM_TOOLS, "L414-L432"),
        text='[TOOL_CALLS] [{"name": "add", "arguments": {"a": 3.5, "b": 4}}, {"name": "get_current_weather", '
        '"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}]',
        tools=[T_ADD, T_WEATHER],
        expected=expected(None, None, [("add", {"a": 3.5, "b": 4}), ("get_current_weather", WEATHER_SF)]),
        tags=["parallel-calls", "numeric-arguments", "x-v3-json-array"],
        notes="test_extract_tool_calls_pre_v11_tokenizer (multiple_tools).",
    ),
    dict(
        name="vllm-v3-content-before-call",
        ref=MISTRAL7B_V03,
        prov=vllm(VLLM_TOOLS, "L433-L443"),
        text='Hello[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1, "b": 2}}]',
        tools=[T_ADD],
        expected=expected("Hello", None, [("add", {"a": 1, "b": 2})]),
        tags=["single-call", "text-before-call", "x-v3-json-array"],
        notes="test_extract_tool_calls_pre_v11_tokenizer (content_before_tool).",
    ),
    dict(
        name="vllm-v3-malformed-not-json",
        ref=MISTRAL7B_V03,
        prov=vllm(VLLM_TOOLS, "L508-L517"),
        text="[TOOL_CALLS] not json at all",
        tools=[T_ADD],
        expected_error={
            "reason": "[TOOL_CALLS] is followed by text that is not a JSON array, so there is no tool call. vLLM "
            "returns the text after the marker as content.",
            "accept": ["no_tool_calls", "content_passthrough"],
        },
        tags=["malformed", "no-call", "x-v3-json-array"],
        notes="test_extract_tool_calls_pre_v11_regex_fallback_fails.",
    ),
    # ---------------------------------------------------------------- vLLM v11 ([ARGS])
    dict(
        name="vllm-v11-single-call",
        ref=SMALL32,
        prov=vllm(VLLM_TOOLS, "L578-L590"),
        text='[TOOL_CALLS]add_this_and_that[ARGS]{"a": 3.5, "b": 4}',
        tools=[T_ADD_THIS],
        expected=expected(None, None, [("add_this_and_that", {"a": 3.5, "b": 4})]),
        tags=["single-call", "numeric-arguments", "x-v11-args"],
        notes="test_extract_tool_calls (single_tool_add_args).",
    ),
    dict(
        name="vllm-v11-parallel",
        ref=SMALL32,
        prov=vllm(VLLM_TOOLS, "L605-L620"),
        text='[TOOL_CALLS]add[ARGS]{"a": 3.5, "b": 4}[TOOL_CALLS]multiply[ARGS]{"a": 3, "b": 6}',
        tools=[T_ADD, T_MULTIPLY],
        expected=expected(None, None, [("add", {"a": 3.5, "b": 4}), ("multiply", {"a": 3, "b": 6})]),
        tags=["parallel-calls", "numeric-arguments", "x-v11-args"],
        notes="test_extract_tool_calls (multiple_tool_calls_args).",
    ),
    dict(
        name="vllm-v11-content-before-call",
        ref=SMALL32,
        prov=vllm(VLLM_TOOLS, "L621-L631"),
        text='hi[TOOL_CALLS]add[ARGS]{"a": 1, "b": 2}',
        tools=[T_ADD],
        expected=expected("hi", None, [("add", {"a": 1, "b": 2})]),
        tags=["single-call", "text-before-call", "x-v11-args", "x-content-then-calls-single-delta"],
        notes="test_extract_tool_calls (content_before_tool_args).",
    ),
    # ---------------------------------------------------------------- vLLM v13 reasoning
    dict(
        name="vllm-v13-think-no-end",
        ref=MAGISTRAL_2509,
        thinking=True,
        prov=vllm(VLLM_REASON, "L156-L163"),
        text="[THINK]r",
        tools=[T_GET_WEATHER],
        expected=expected(None, "r", []),
        tags=["no-call", "reasoning", "truncated", "x-think-special-token"],
        notes="test_mistral_reasoning_v13 (v13_think_no_end): [THINK] opened but never closed.",
    ),
    dict(
        name="vllm-v11-think-text",
        ref=MAGISTRAL_2506,
        thinking=True,
        prov=vllm(VLLM_REASON, "L286-L301"),
        text="<think>r</think>c",
        tools=[T_GET_WEATHER],
        expected=expected("c", "r", []),
        tags=["no-call", "reasoning", "x-think-text"],
        notes="test_reasoning_v11_plain_text_think: v11 Magistral (2506) reasons inside plain-text <think> tags, "
        "which are ordinary tokens, not control tokens.",
    ),
    # ---------------------------------------------------------------- llama.cpp Ministral-3-14B-Reasoning-2512
    dict(
        name="llamacpp-ministral3-reasoning-call",
        ref=MINISTRAL3_REASONING,
        thinking=True,
        prov=llama("L2615-L2622"),
        text='[THINK]I\'m\nthinking[/THINK][TOOL_CALLS]special_function[ARGS]{"arg1":1}',
        tools=[T_SPECIAL],
        expected=expected(None, "I'm\nthinking", [("special_function", SF)]),
        tags=["single-call", "reasoning", "numeric-arguments", "x-v13-compact", "x-think-special-token"],
    ),
    dict(
        name="llamacpp-ministral3-parallel",
        ref=MINISTRAL3_REASONING,
        thinking=True,
        prov=llama("L2624-L2636"),
        text='[TOOL_CALLS]special_function[ARGS]{"arg1": 1}'
        '[TOOL_CALLS]special_function_with_opt[ARGS]{"arg1": 1, "arg2": 2}',
        tools=[T_SPECIAL, T_SPECIAL_OPT],
        expected=expected(None, None, [("special_function", SF), ("special_function_with_opt", SFO)]),
        tags=["parallel-calls", "numeric-arguments", "x-v13-compact"],
    ),
    dict(
        name="llamacpp-ministral3-marker-in-reasoning",
        ref=MINISTRAL3_REASONING,
        thinking=True,
        prov=llama("L2649-L2661"),
        text='[THINK]Let me think about [TOOL_CALLS]special_function[ARGS]{"arg1":1} and more[/THINK]'
        '[TOOL_CALLS]special_function[ARGS]{"arg1": 1}',
        tools=[T_SPECIAL],
        expected=expected(
            None, 'Let me think about [TOOL_CALLS]special_function[ARGS]{"arg1":1} and more', [("special_function", SF)]
        ),
        tags=["single-call", "reasoning", "x-marker-in-reasoning", "x-v13-compact", "x-think-special-token"],
        notes="'fake tool call marker in reasoning'. Here the markers inside [THINK] are control-token ids "
        "(as a tokenizer that parses special tokens produces them): everything between [THINK] and [/THINK] "
        "is reasoning, and only the call after [/THINK] is a tool call.",
    ),
    # ---------------------------------------------------------------- llama.cpp Mistral-Small-3.2 ([CALL_ID])
    dict(
        name="llamacpp-v11-call-id",
        ref=SMALL32,
        prov=llama("L6244-L6248"),
        text='[TOOL_CALLS]special_function[CALL_ID]123456789[ARGS]{"arg1": 1}',
        tools=[T_SPECIAL],
        expected=expected(None, None, [("special_function", SF)]),
        tags=["single-call", "numeric-arguments", "x-v11-call-id"],
        notes="v11 FUNC_BRACKET_TAG format with a [CALL_ID]. mistral-common's generation grammar has no "
        "[CALL_ID], but the v11 tokenizer defines it and llama.cpp parses it. The id is not part of the "
        "expected result; the name must not absorb it.",
    ),
    dict(
        name="llamacpp-v11-call-id-parallel",
        ref=SMALL32,
        prov=llama("L6250-L6261"),
        text='[TOOL_CALLS]special_function[CALL_ID]000000001[ARGS]{"arg1": 1}'
        '[TOOL_CALLS]special_function_with_opt[CALL_ID]000000002[ARGS]{"arg1": 1, "arg2": 2}',
        tools=[T_SPECIAL, T_SPECIAL_OPT],
        expected=expected(None, None, [("special_function", SF), ("special_function_with_opt", SFO)]),
        tags=["parallel-calls", "numeric-arguments", "x-v11-call-id"],
    ),
    # ---------------------------------------------------------------- llama.cpp Devstral-Small-2507
    dict(
        name="llamacpp-devstral-content-before-call",
        ref=DEVSTRAL_2507,
        prov=llama("L6280-L6284"),
        text='Hello, world!\nWhat\'s up?[TOOL_CALLS]special_function[ARGS]{"arg1": 1}',
        tools=[T_SPECIAL],
        expected=expected("Hello, world!\nWhat's up?", None, [("special_function", SF)]),
        tags=["single-call", "text-before-call", "numeric-arguments", "x-v11-args"],
    ),
    # ---------------------------------------------------------------- bug reports
    dict(
        name="bug-devstral-glob-braces-in-arguments",
        ref=DEVSTRAL_2507,
        prov={
            "kind": "bug_report",
            "source_url": "https://github.com/ollama/ollama/issues/11470",
            "revision": "issue-11470",
            "license": "NOASSERTION",
            "generator": GENERATOR,
        },
        text='[TOOL_CALLS]file_search[ARGS]{"query":"/{.github/copilot-instructions.md,AGENT.md,AGENTS.md,CLAUDE.md,'
        '.cursorrules,.windsurfrules,.clinerules,.cursor/rules/,.windsurf/rules/,.clinerules/,README.md}"}',
        tools=[tool("file_search", {"query": STR}, ["query"])],
        expected=expected(
            None,
            None,
            [
                (
                    "file_search",
                    {
                        "query": "/{.github/copilot-instructions.md,AGENT.md,AGENTS.md,CLAUDE.md,.cursorrules,"
                        ".windsurfrules,.clinerules,.cursor/rules/,.windsurf/rules/,.clinerules/,README.md}"
                    },
                )
            ],
        ),
        tags=["single-call", "regression", "x-v11-args"],
        notes="A Devstral generation quoted in the issue (the reporter used Devstral via Ollama; the exact model "
        "revision is not stated). Braces inside the JSON string must not confuse argument extraction.",
    ),
]


def main() -> None:
    recs = []
    for c in CASES:
        ref: Ref = c["ref"]
        ids = encode_with_markers(ref, c["text"])
        recs.append(
            record(
                name=c["name"],
                models=[ref.repo],
                provenance=c["prov"],
                tools=c["tools"],
                raw_output=c["text"],
                ref=ref,
                output_token_ids=ids,
                thinking=c.get("thinking"),
                expected=c.get("expected"),
                expected_error=c.get("expected_error"),
                tags=c["tags"],
                notes=c.get("notes"),
            )
        )
    write_jsonl(OUT_DIR / "imported.jsonl", recs)


if __name__ == "__main__":
    main()
