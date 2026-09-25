# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["transformers==5.17.0", "jinja2>=3.1", "huggingface_hub>=1.0"]
# ///
"""Import qwen3-xml fixtures from engine test suites and public bug reports.

Every raw output below is COPIED VERBATIM from the cited source (file + line range at a pinned
commit, or the issue URL). Token ids come from the reference model's HF tokenizer; the round trip
is asserted to be lossless (Qwen's markers are added tokens matched in text).

Expected contents omit the newline(s) between text and a following ``<tool_call>``: the chat
templates emit them as markup (Qwen3.5+: content + '\\n\\n'; Qwen3-Coder: '\\n' + content + '\\n'
then '\\n<tool_call>'). Engines that keep them get soft_pass under soft-v1.

Run from the repo root::

    uv run --script scripts/fixtures/qwen3-xml/import_qwen3_xml.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    CODER,
    CODER_480,
    GEN_PROMPT_CODER,
    GEN_PROMPT_NO_THINK,
    GEN_PROMPT_THINK,
    OUT_DIR,
    QWEN35,
    QWEN38,
    Ref,
    encode_checked,
    expected,
    record,
    write_jsonl,
)

GENERATOR = "scripts/fixtures/qwen3-xml/import_qwen3_xml.py"

LLAMA_SHA = "7fe450e19305b828c199d602c23a8337aaa1f03b"  # commit of llama.cpp tag v0.5.0 (tag object c13fcbf6)
LLAMA_TEST = f"https://github.com/ggml-org/llama.cpp/blob/{LLAMA_SHA}/tests/test-chat.cpp"
LLAMA_ATTR = "Copyright (c) 2023-2026 The ggml authors (MIT)."
VLLM_SHA = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # vLLM v0.30.0
VLLM_TEST = f"https://github.com/vllm-project/vllm/blob/{VLLM_SHA}/tests/tool_parsers/test_qwen3coder_tool_parser.py"
VLLM_ATTR = "Copyright contributors to the vLLM project (Apache-2.0)."


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


def s(desc: str | None = None) -> dict[str, Any]:
    return {"type": "string", **({"description": desc} if desc else {})}


def i(desc: str) -> dict[str, Any]:
    return {"type": "integer", "description": desc}


# llama.cpp tests/test-chat.cpp tool definitions (line ranges at v0.5.0).
L_SPECIAL = tool("special_function", {"arg1": i("The arg.")}, ["arg1"], "I'm special")  # L425-L437
L_SPECIAL_OPT = tool(  # L439-L456
    "special_function_with_opt",
    {"arg1": i("The arg."), "arg2": i("The optional arg.")},
    ["arg1"],
    "I'm special but have optional stuff",
)
L_EMPTY = tool("empty_args", {}, None, "A tool that takes no arguments")  # L458-L465
L_PYTHON = tool("python", {"code": s("Python code to execute.")}, ["code"], "an ipython interpreter")  # L481-L494
L_HTML = tool("html", {"markup": s("HTML markup to validate.")}, ["markup"], "an html validator")  # L496-L509
L_TODO = tool(  # L541-L553
    "todo_list",
    {"todos": {"type": "array", "description": "List of TODO list items"}},
    ["todos"],
    "Create or update the todo list",
)
L_EDIT = tool(  # L556-L577
    "edit",
    {
        "filename": s("Path of file to edit"),
        "oldString": s("String to replace"),
        "newString": s("New (replacement) value"),
    },
    ["filename", "oldString", "newString"],
    "Edit file",
)
L_TERMINAL = tool("run_in_terminal", {"command": s("Shell command to run")}, ["command"], "Run a shell command.")

# vLLM tests/tool_parsers/test_qwen3coder_tool_parser.py (L42-L51, L53-L60, L499-L550).
V_WEATHER = tool(
    "get_current_weather",
    {
        "city": s("The city name"),
        "state": s("The state code"),
        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
    },
    ["city", "state"],
    "Get the current weather",
)
V_AREA = tool(
    "calculate_area",
    {"shape": {"type": "string"}, "dimensions": {"type": "object"}, "precision": {"type": "integer"}},
    None,
    "Calculate area of a shape",
)
V_ANYOF = tool(
    "test_anyof",
    {
        "anyof_int": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 5},
        "anyof_str": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "anyof_array": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "anyof_obj": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        "type_as_array": {"type": ["integer", "null"]},
        "multi_non_null": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
    },
)


def llama(line: str) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{LLAMA_TEST}#{line}",
        "revision": LLAMA_SHA,
        "license": "MIT",
        "generator": GENERATOR,
        "attribution": LLAMA_ATTR,
    }


def vllm(line: str) -> dict[str, Any]:
    return {
        "kind": "engine_test",
        "source_url": f"{VLLM_TEST}#{line}",
        "revision": VLLM_SHA,
        "license": "Apache-2.0",
        "generator": GENERATOR,
        "attribution": VLLM_ATTR,
    }


def bug(url: str, rev: str) -> dict[str, Any]:
    return {"kind": "bug_report", "source_url": url, "revision": rev, "license": "NOASSERTION", "generator": GENERATOR}


CALL_SPECIAL_1 = (
    "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>"
)
TERMINAL_PWD = (
    "<tool_call>\n<function=run_in_terminal>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
)
V_WEATHER_CALL = (
    "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n</parameter>\n"
    "<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>"
)
DALLAS = {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}

Q35_MODELS = [QWEN35.repo]
CODER_MODELS = [CODER.repo, CODER_480.repo]

CASES: list[dict[str, Any]] = [
    # ------------------------------------------------------------ llama.cpp: Qwen3.5-4B.jinja tests
    dict(
        name="llamacpp-q35-reasoning-then-call",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_THINK,
        prov=llama("L2164-L2174"),
        text="I'm\nthinking\n</think>\n\n" + CALL_SPECIAL_1,
        tools=[L_SPECIAL],
        expected=expected(None, "I'm\nthinking", [("special_function", {"arg1": 1})]),
        tags=["single-call", "reasoning", "reasoning-prefilled", "numeric-arguments"],
    ),
    dict(
        name="llamacpp-q35-parallel-no-think",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_NO_THINK,
        thinking=False,
        prov=llama("L2176-L2198"),
        text=CALL_SPECIAL_1 + "\n<tool_call>\n<function=special_function_with_opt>\n<parameter=arg1>\n1\n</parameter>\n"
        "<parameter=arg2>\n2\n</parameter>\n</function>\n</tool_call>",
        tools=[L_SPECIAL, L_SPECIAL_OPT],
        expected=expected(
            None, None, [("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})]
        ),
        tags=["parallel-calls", "numeric-arguments"],
    ),
    dict(
        name="llamacpp-q35-python-multiline",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_NO_THINK,
        thinking=False,
        prov=llama("L2200-L2219"),
        text='<tool_call>\n<function=python>\n<parameter=code>\ndef hello():\n    print("Hello, world!")\n\nhello()\n'
        "</parameter>\n</function>\n</tool_call>",
        tools=[L_PYTHON],
        expected=expected(None, None, [("python", {"code": 'def hello():\n    print("Hello, world!")\n\nhello()'})]),
        tags=["single-call", "string-escapes", "x-multiline-string"],
    ),
    dict(
        name="llamacpp-q35-value-trailing-newline",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_NO_THINK,
        thinking=False,
        prov=llama("L2245-L2273"),
        text="<tool_call>\n<function=edit>\n<parameter=filename>\nfoo.c\n</parameter>\n<parameter=oldString>\n#iclunde\n"
        "</parameter>\n<parameter=newString>\n#include\n\n</parameter>\n</function>\n</tool_call>",
        tools=[L_EDIT],
        expected=expected(
            None, None, [("edit", {"filename": "foo.c", "oldString": "#iclunde", "newString": "#include\n"})]
        ),
        tags=["single-call", "x-whitespace-significant-string"],
        notes="A value that ends in a newline renders as '#include\\n\\n</parameter>'; exactly one '\\n' is markup.",
    ),
    dict(
        name="llamacpp-q35-code-leading-indent",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_NO_THINK,
        thinking=False,
        prov=llama("L2276-L2293"),
        text='<tool_call>\n<function=python>\n<parameter=code>\n    print("Hello, world!")\n</parameter>\n'
        "</function>\n</tool_call>",
        tools=[L_PYTHON],
        expected=expected(None, None, [("python", {"code": '    print("Hello, world!")'})]),
        tags=["single-call", "x-whitespace-significant-string"],
    ),
    dict(
        name="llamacpp-q35-call-inside-think",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_THINK,
        prov=llama("L2323-L2340"),
        text="Need to inspect the current directory.\n" + TERMINAL_PWD,
        tools=[L_TERMINAL],
        expected=expected(None, "Need to inspect the current directory.", [("run_in_terminal", {"command": "pwd"})]),
        tags=["single-call", "reasoning", "reasoning-prefilled", "malformed", "x-call-inside-think"],
        notes="The model starts a tool call without closing the prefilled <think> block. llama.cpp's test "
        "('a tool call ends the prefilled thinking block, with or without a closing </think>') treats "
        "<tool_call> as ending the reasoning. This is the most-reported failure for the family: "
        "https://github.com/ggml-org/llama.cpp/issues/20837, https://github.com/vllm-project/vllm/issues/39056.",
    ),
    dict(
        name="llamacpp-q35-empty-args",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_NO_THINK,
        thinking=False,
        prov=llama("L2342-L2352"),
        text="<tool_call>\n<function=empty_args>\n</function>\n</tool_call>",
        tools=[L_EMPTY],
        expected=expected(None, None, [("empty_args", {})]),
        tags=["single-call", "empty-arguments"],
    ),
    dict(
        name="llamacpp-q35-reasoning-content-call",
        ref=QWEN35,
        models=Q35_MODELS,
        gen=GEN_PROMPT_THINK,
        prov=llama("L2459-L2480"),
        text="I should inspect the directory.\n</think>\n\nLet me inspect it now.\n" + TERMINAL_PWD,
        tools=[L_TERMINAL],
        expected=expected(
            "Let me inspect it now.", "I should inspect the directory.", [("run_in_terminal", {"command": "pwd"})]
        ),
        tags=["single-call", "reasoning", "reasoning-prefilled", "text-before-call"],
    ),
    # ------------------------------------------------------------ llama.cpp: Qwen3-Coder.jinja tests
    dict(
        name="llamacpp-coder-bare-function",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=llama("L3559-L3569"),
        text="<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>",
        tools=[L_SPECIAL],
        expected=expected(None, None, [("special_function", {"arg1": 1})]),
        tags=["single-call", "malformed", "numeric-arguments", "x-bare-function"],
        notes="'Some models skip the opening <tool_call> and go straight to <function=>' (also Ollama #17353).",
    ),
    dict(
        name="llamacpp-coder-bare-function-parallel",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=llama("L3586-L3612"),
        text="<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
        "<function=special_function_with_opt>\n<parameter=arg1>\n1\n</parameter>\n<parameter=arg2>\n2\n</parameter>\n"
        "</function>\n</tool_call>",
        tools=[L_SPECIAL, L_SPECIAL_OPT],
        expected=expected(
            None, None, [("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})]
        ),
        tags=["parallel-calls", "malformed", "numeric-arguments", "x-bare-function"],
    ),
    dict(
        name="llamacpp-coder-unicode-cjk",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=llama("L3664-L3680"),
        text="<tool_call>\n<function=python>\n<parameter=code>\n格\n</parameter>\n</function>\n</tool_call>",
        tools=[L_PYTHON],
        expected=expected(None, None, [("python", {"code": "格"})]),
        tags=["single-call", "unicode"],
    ),
    dict(
        name="llamacpp-coder-html-value",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=llama("L3682-L3702"),
        text="<tool_call>\n<function=html>\n<parameter=markup>\n<html>\n <head>\n  <title>Hello!</title>\n </head>\n"
        "</html>\n</parameter>\n</function>\n</tool_call>",
        tools=[L_HTML],
        expected=expected(
            None, None, [("html", {"markup": "<html>\n <head>\n  <title>Hello!</title>\n </head>\n</html>"})]
        ),
        tags=["single-call", "marker-in-arguments", "x-multiline-string"],
    ),
    dict(
        name="llamacpp-coder-array-of-objects",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=llama("L3704-L3720"),
        text="<tool_call>\n<function=todo_list>\n<parameter=todos>\n"
        '[{"item": "Check stuff", "selected": false}, {"item": "Prepare stuff", "selected": true}]\n'
        "</parameter>\n</function>\n</tool_call>",
        tools=[L_TODO],
        expected=expected(
            None,
            None,
            [
                (
                    "todo_list",
                    {
                        "todos": [
                            {"item": "Check stuff", "selected": False},
                            {"item": "Prepare stuff", "selected": True},
                        ]
                    },
                )
            ],
        ),
        tags=["single-call", "nested-json"],
    ),
    # ------------------------------------------------ vLLM tests/tool_parsers/test_qwen3coder_tool_parser.py
    dict(
        name="vllm-coder-text-before-call-no-newline",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=vllm("L270-L295"),
        text="Sure! Let me check the weather for you." + V_WEATHER_CALL,
        tools=[V_WEATHER, V_AREA],
        expected=expected("Sure! Let me check the weather for you.", None, [("get_current_weather", DALLAS)]),
        tags=["single-call", "text-before-call"],
        notes="single_tool_with_content: the content runs straight into <tool_call> with no newline.",
    ),
    dict(
        name="vllm-coder-anyof-coercion",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=vllm("L499-L596"),
        text="<tool_call>\n<function=test_anyof>\n<parameter=anyof_int>\n5\n</parameter>\n<parameter=anyof_str>\nhello\n"
        '</parameter>\n<parameter=anyof_array>\n["a", "b", "c"]\n</parameter>\n<parameter=anyof_obj>\n'
        '{"key": "value"}\n</parameter>\n<parameter=type_as_array>\n42\n</parameter>\n<parameter=multi_non_null>\n'
        "some text\n</parameter>\n</function>\n</tool_call>",
        tools=[V_ANYOF],
        expected=expected(
            None,
            None,
            [
                (
                    "test_anyof",
                    {
                        "anyof_int": 5,
                        "anyof_str": "hello",
                        "anyof_array": ["a", "b", "c"],
                        "anyof_obj": {"key": "value"},
                        "type_as_array": 42,
                        "multi_non_null": "some text",
                    },
                )
            ],
        ),
        tags=["single-call", "nested-json", "numeric-arguments", "x-schema-coercion"],
        notes="test_extract_tool_calls_anyof_type_conversion: nullable anyOf / type-array schemas (Pydantic v2).",
    ),
    dict(
        name="vllm-coder-missing-close-parameter",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=vllm("L932-L971"),
        text="Let me check the weather for you:\n<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nDallas\n"
        "<parameter=state>\nTX\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>",
        tools=[V_WEATHER, V_AREA],
        expected=expected("Let me check the weather for you:", None, [("get_current_weather", DALLAS)]),
        tags=["single-call", "text-before-call", "malformed", "x-missing-close-param"],
        notes="test_extract_tool_calls_missing_closing_parameter_tag: the next <parameter= ends the unclosed value.",
    ),
    # ------------------------------------------------------------ bug reports
    dict(
        name="bug-missing-close-parameter-before-function",
        ref=QWEN38,
        models=[QWEN38.repo],
        gen=GEN_PROMPT_NO_THINK,
        thinking=False,
        prov=bug("https://github.com/vllm-project/vllm/issues/57699", "issue-57699"),
        text="<tool_call>\n<function=ThinQ_Connect>\n<parameter=body>\n"
        '{"airConJobMode": "AIR_CLEAN", "windStrength": "HIGH", "monitoringEnabled": true}\n</function>\n</tool_call>',
        tools=[tool("ThinQ_Connect", {"body": {"type": "object"}}, ["body"])],
        expected=expected(
            None,
            None,
            [
                (
                    "ThinQ_Connect",
                    {"body": {"airConJobMode": "AIR_CLEAN", "windStrength": "HIGH", "monitoringEnabled": True}},
                )
            ],
        ),
        tags=["single-call", "malformed", "nested-json", "regression", "x-missing-close-param"],
        notes="Raw text from the issue (Qwen3.8-27B, BFCL live_simple_40-17-0, temperature 0). The last parameter is "
        "closed by </function> without </parameter>; the call is complete, so the value must be kept (vLLM "
        "returns {} non-streaming). The issue's tool schema is not shown; 'body' is typed as an object here. "
        "The thinking mode of the request is not stated; the fixture assumes the no-think prompt.",
    ),
    dict(
        name="bug-coder-missing-tool-call-opener",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=bug("https://github.com/ollama/ollama/issues/18530", "issue-18530"),
        text="I will first check for all Markdown files in the current directory and its subdirectories, and then list "
        "them using the glob tool.\n\n<function=glob>\n<parameter=pattern>\n**/*.md\n</parameter>\n</function>\n"
        "</tool_call>",
        tools=[
            tool("glob", {"pattern": {"type": "string"}}, ["pattern"], "Find files by glob pattern"),
        ],
        expected=expected(
            "I will first check for all Markdown files in the current directory and its subdirectories, and then "
            "list them using the glob tool.",
            None,
            [("glob", {"pattern": "**/*.md"})],
        ),
        tags=["single-call", "text-before-call", "malformed", "regression", "x-bare-function"],
        notes="A real qwen3-coder:30b generation captured by the reporter with raw: true (temperature 0, seed 1): "
        "the model omitted the <tool_call> opener after a prose preamble.",
    ),
    dict(
        name="bug-coder-text-after-call",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=bug("https://github.com/sgl-project/sglang/issues/40739", "issue-40739"),
        text="before<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>after",
        tools=[tool("get_weather", {"city": {"type": "string"}}, ["city"])],
        expected=expected("beforeafter", None, [("get_weather", {"city": "Paris"})]),
        tags=["single-call", "text-before-call", "text-after-call", "regression", "x-no-newlines"],
        notes="The issue's input string (no newlines inside the call). Its expected behaviour: streaming and "
        "non-streaming both return content 'beforeafter' and the call.",
    ),
    dict(
        name="bug-coder-number-outside-int64",
        ref=CODER,
        models=CODER_MODELS,
        gen=GEN_PROMPT_CODER,
        prov=bug("https://github.com/ollama/ollama/issues/18421", "issue-18421"),
        text="<tool_call><function=calculate>\n<parameter=x>\n1e20\n</parameter>\n</function></tool_call>",
        tools=[tool("calculate", {"x": {"type": "number"}})],
        expected=expected(None, None, [("calculate", {"x": 1e20})]),
        tags=["single-call", "numeric-arguments", "regression"],
        notes="The issue's parser input: a number outside the int64 range must be preserved (expected "
        "100000000000000000000, not 9223372036854775807).",
    ),
]


def main() -> None:
    recs = []
    for c in CASES:
        ref: Ref = c["ref"]
        ids = encode_checked(ref, c["text"])
        recs.append(
            record(
                name=c["name"],
                models=c["models"],
                provenance=c["prov"],
                tools=c["tools"],
                raw_output=c["text"],
                tokenizer_ref=ref,
                output_token_ids=ids,
                generation_prompt=c["gen"],
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
