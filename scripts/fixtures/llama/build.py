# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "transformers==5.17.0",
#     "tokenizers==0.23.2",
#     "tiktoken==0.12.0",
#     "jinja2==3.1.6",
#     "huggingface-hub==1.33.0",
#     "jsonschema==4.25.1",
# ]
# ///
"""Build the ``llama`` fixture corpus (Llama 3.x JSON calls, Llama 4 pythonic calls).

The meta-llama repos are gated, so this script never touches them. It uses
documented, verified mirrors instead:

* template: ``unsloth/Llama-3.3-70B-Instruct`` ``chat_template.jinja``, checked here to
  be byte-identical to llama.cpp's verbatim copy of the HF template
  (``models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja``);
* tokenizer: the same mirror, whose ``tokenizer.json`` sha256 equals the gated
  meta-llama/Llama-3.3-70B-Instruct LFS pointer (6b9e4e7f...); Llama 4 likewise
  (unsloth/Llama-4-Scout-17B-16E-Instruct, 172c9eb4...).

Provenance of every fixture written here:

* ``template_render``: Llama 3.x JSON tool calls rendered through the official
  template (via the mirror) with ``apply_chat_template(tokenize=True)``. The template
  only renders one call per message and drops ``content`` next to a call, so text
  around calls comes from engine tests instead. There is no template render for
  Llama 4: the mirror's Llama 4 template differs from Meta's (different git blob),
  so Llama 4 fixtures come from Meta's recorded examples and engine tests only.
* ``recorded``: model outputs quoted from Meta's prompt-format docs.
* ``engine_test`` / ``bug_report``: see ``imported.py``.

Run from the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/llama/build.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imported  # noqa: E402
from _core import (  # noqa: E402
    REPO_ROOT,
    Message,
    Provenance,
    Record,
    Rendered,
    Tool,
    assistant,
    call,
    cut_before,
    encode_raw,
    expected_from,
    render,
    sha256_file,
    tool,
    validate_records,
    write_jsonl,
)
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SLUG = "llama"
OUT = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"

LLAMACPP_SHA = "a25c9865fe03c954c93fd755b5d79ae86ba99750"
LLAMACPP_TEMPLATE = (
    f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMACPP_SHA}/models/templates/"
    "meta-llama-Llama-3.3-70B-Instruct.jinja"
)


@dataclass(frozen=True)
class Model:
    key: str
    repo: str
    """The (gated) meta-llama repo: fixtures list it in ``models``."""
    revision: str
    mirror: str
    mirror_revision: str
    license: str
    siblings: tuple[tuple[str, str, str], ...] = ()
    """(meta repo, mirror repo, mirror revision) checked to render identically."""


L33 = Model(
    "l33",
    "meta-llama/Llama-3.3-70B-Instruct",
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    "unsloth/Llama-3.3-70B-Instruct",
    "99cd0d2c829e92a67c844f9144c2509632e5c87f",
    "LicenseRef-llama3.3-community",
    siblings=(
        (
            "meta-llama/Llama-3.1-8B-Instruct",
            "unsloth/Llama-3.1-8B-Instruct",
            "4699cc75b550f9c6f3173fb80f4703b62d946aa5",
        ),
        (
            "meta-llama/Llama-3.2-3B-Instruct",
            "unsloth/Llama-3.2-3B-Instruct",
            "006f5dcd1393c3add266de40994ba96225e9689d",
        ),
    ),
)
L4 = Model(
    "l4",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
    "afd8e498c87bda51c7ea8ec68ea2f7c066e6340b",
    "LicenseRef-llama4-community",
)


@dataclass
class Loaded:
    model: Model
    tok: Any
    stop_ids: list[int]
    template_sha256: str
    siblings: list[tuple[str, Any, list[int]]] = field(default_factory=list)

    @property
    def tokenizer_pin(self) -> dict[str, str]:
        return {"repo": self.model.mirror, "revision": self.model.mirror_revision, "mode": "hf"}

    def template_url(self) -> str:
        return f"https://huggingface.co/{self.model.mirror}/blob/{self.model.mirror_revision}/chat_template.jinja"


def stop_ids(repo: str, revision: str) -> list[int]:
    gen = json.loads(Path(hf_hub_download(repo, "generation_config.json", revision=revision)).read_text())
    stops = gen["eos_token_id"]
    return [int(s) for s in (stops if isinstance(stops, list) else [stops])]


def load(model: Model) -> Loaded:
    tok = AutoTokenizer.from_pretrained(model.mirror, revision=model.mirror_revision)
    sha = sha256_file(hf_hub_download(model.mirror, "chat_template.jinja", revision=model.mirror_revision))
    sibs = [
        (meta, AutoTokenizer.from_pretrained(mirror, revision=rev), stop_ids(mirror, rev))
        for meta, mirror, rev in model.siblings
    ]
    return Loaded(model, tok, stop_ids(model.mirror, model.mirror_revision), sha, sibs)


def check_llamacpp_copy(ld: Loaded) -> None:
    """The mirror's template must be byte-identical to llama.cpp's copy of the HF template."""
    with urllib.request.urlopen(LLAMACPP_TEMPLATE, timeout=60) as resp:  # read-only GET
        theirs = hashlib.sha256(resp.read()).hexdigest()
    if theirs != ld.template_sha256:
        raise AssertionError(f"mirror template {ld.template_sha256} != llama.cpp copy {theirs}")


# --------------------------------------------------------------------------- tools

T_WEATHER = tool(
    "get_weather",
    "Get the current weather for a city.",
    {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    ["city"],
)
T_SEARCH = tool("search_web", "Search the web.", {"query": {"type": "string"}}, ["query"])
T_TIME = tool("get_time", "Get the current UTC time.")
T_WRITE = tool(
    "write_file",
    "Write a text file.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
T_EVENT = tool(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "duration_minutes": {"type": "integer"},
        "all_day": {"type": "boolean"},
        "reminder_minutes": {"type": ["integer", "null"]},
    },
    ["title"],
)
T_SETTINGS = tool(
    "update_settings",
    "Update user settings.",
    {"settings": {"type": "object"}, "dry_run": {"type": "boolean"}},
    ["settings"],
)
T_CONVERT = tool(
    "convert_units",
    "Convert a value between units.",
    {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
    ["value", "from_unit", "to_unit"],
)
TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_EVENT, T_SETTINGS, T_CONVERT]
LONG_TEXT = "\n".join(f"{i:03d}: The quick brown fox jumps over the lazy dog." for i in range(60))


@dataclass
class Case:
    name: str
    tags: list[str]
    msg: Message
    context: list[Message] = field(default_factory=list)
    notes: str | None = None
    user: str = "Please help."


CASES: list[Case] = [
    Case(
        "single-call",
        ["single-call", "x-parameters-key"],
        assistant(calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})]),
        user="What's the weather in Paris?",
    ),
    Case(
        "nested-json",
        ["single-call", "nested-json", "numeric-arguments", "x-parameters-key"],
        assistant(
            calls=[
                call(
                    "update_settings",
                    {
                        "settings": {
                            "theme": {"mode": "dark", "accent": [255, 128, 0]},
                            "notifications": {"email": True, "sms": False},
                        },
                        "dry_run": False,
                    },
                )
            ]
        ),
    ),
    Case(
        "unicode-emoji",
        ["single-call", "unicode", "x-parameters-key"],
        assistant(calls=[call("search_web", {"query": "Zürich café ☕ 東京の天気 🌸"})]),
        notes="The template uses tojson with ensure_ascii=False, so non-ASCII text is literal.",
    ),
    Case(
        "empty-arguments",
        ["single-call", "empty-arguments", "x-parameters-key"],
        assistant(calls=[call("get_time", {})]),
    ),
    Case(
        "marker-in-arguments",
        ["single-call", "marker-in-arguments", "x-parameters-key"],
        assistant(
            calls=[
                call(
                    "write_file",
                    {
                        "path": "docs/llama.md",
                        "content": "Tool calls may start with <|python_tag|>; headers look like "
                        "<|start_header_id|>ipython<|end_header_id|>.",
                    },
                )
            ]
        ),
        notes="The JSON string contains <|python_tag|> and header tokens, tokenized as the special tokens. A parser "
        "that splits on <|python_tag|> must not cut inside the string.",
    ),
    Case(
        "string-escapes",
        ["single-call", "string-escapes", "x-parameters-key"],
        assistant(
            calls=[
                call(
                    "write_file",
                    {"path": "C:\\Users\\ana\\notes.txt", "content": 'line 1\n\tline 2 "quoted" {braces} \\ end'},
                )
            ]
        ),
    ),
    Case(
        "typed-values",
        ["single-call", "numeric-arguments", "x-parameters-key"],
        assistant(
            calls=[
                call(
                    "create_event",
                    {
                        "title": "Standup",
                        "attendees": ["ana@example.com", "bo@example.com"],
                        "duration_minutes": 15,
                        "all_day": False,
                        "reminder_minutes": None,
                    },
                )
            ]
        ),
    ),
    Case(
        "long-arguments",
        ["single-call", "long-arguments", "x-parameters-key"],
        assistant(calls=[call("write_file", {"path": "fox.txt", "content": LONG_TEXT})]),
    ),
    Case(
        "no-call",
        ["no-call"],
        assistant(content="Paris is the capital of France."),
        user="What is the capital of France?",
    ),
    Case(
        "json-answer-not-a-call",
        ["no-call", "x-json-in-content-not-a-call"],
        assistant(content='{"capital": "Paris", "country": "France"}'),
        user="Answer as JSON: what is the capital of France?",
        notes="A plain JSON answer: it has no name/parameters keys, so it is content, not a call "
        "(cf. https://github.com/sgl-project/sglang/issues/35562).",
    ),
    Case(
        "multi-turn-second-call",
        ["single-call", "multi-turn", "x-parameters-key"],
        assistant(calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})]),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(calls=[call("get_weather", {"city": "Rome"})]),
            {"role": "ipython", "content": '{"temp_c": 21.5}'},
        ],
    ),
]


def render_case(ld: Loaded, case: Case, tok: Any | None = None, stops: list[int] | None = None) -> Rendered:
    context = case.context or [{"role": "user", "content": case.user}]
    return render(tok or ld.tok, context, case.msg, tools=TOOLS, stop_ids=stops or ld.stop_ids)


def default_gen_prompt(tok: Any) -> str:
    probe = [{"role": "user", "content": "x"}]
    full = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    bare = tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    return str(full)[len(str(bare)) :]


def models_for(ld: Loaded) -> list[str]:
    out = [ld.model.repo]
    for meta, stok, sstops in ld.siblings:
        if all(render_case(ld, c).raw_output == render_case(ld, c, stok, sstops).raw_output for c in CASES):
            out.append(meta)
        else:
            print(f"  sibling {meta} renders differently; not listed")
    return out


def build_renders(ld: Loaded) -> tuple[list[dict[str, Any]], dict[str, Rendered]]:
    models = models_for(ld)
    prov = Provenance(
        "template_render",
        ld.template_url(),
        ld.model.mirror_revision,
        ld.model.license,
        GENERATOR,
        ld.template_sha256,
        attribution="Official Llama 3.3 template via the unsloth mirror (byte-identical to llama.cpp's copy at "
        f"{LLAMACPP_SHA}); meta-llama repos are gated.",
    )
    records, renders = [], {}
    for case in CASES:
        r = render_case(ld, case)
        renders[case.name] = r
        records.append(
            Record(
                id=f"{SLUG}/l3-{case.name}",
                family=SLUG,
                models=models,
                provenance=prov,
                tools=TOOLS,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=ld.tokenizer_pin,
                expected=expected_from(case.msg),
                tags=[*case.tags, "x-llama-json"],
                notes=case.notes,
            ).to_dict()
        )
    return records, renders


# --------------------------------------------------------------------------- recorded (Meta docs)

META_SHA = "0e0b8c519242d5833d8c11bffc1232b77ad7f301"
META_NOTE = (
    "Quoted from the 'Model Response Format' block of Meta's prompt-format doc; these blocks are model outputs "
    "(the doc's base-model examples stop mid-sentence at the generation limit). The stop token that ends the block "
    "is cut, as spec/README.md requires."
)


def meta_doc(path: str, lines: str, license_id: str) -> Provenance:
    return Provenance(
        "recorded",
        f"https://github.com/meta-llama/llama-models/blob/{META_SHA}/models/{path}#{lines}",
        META_SHA,
        license_id,
        GENERATOR,
        attribution="Copyright (c) Meta Platforms, Inc. and affiliates. Short example quoted with attribution.",
    )


T_TRENDING = tool(
    "trending_songs",
    "Returns the trending songs on a Music site",
    {
        "n": {"description": "The number of songs to return"},
        "genre": {"description": "The genre of the songs to return"},
    },
    ["n"],
)
T_WEATHER_META = tool(
    "get_weather",
    "Get weather info for places",
    {
        "city": {"type": "string", "description": "The name of the city to get the weather for"},
        "metric": {
            "type": "string",
            "description": "The metric for weather. Options are: celsius, fahrenheit",
            "default": "celsius",
        },
    },
    ["city"],
)
T_USER_INFO = tool(
    "get_user_info",
    "Retrieve details for a specific user by their unique identifier.",
    {"user_id": {"type": "integer"}, "special": {"type": "string", "default": "none"}},
    ["user_id"],
)

TRENDING_RAW = (
    '<|python_tag|>{\n    "type": "function",\n    "name": "trending_songs",\n    "parameters": {\n'
    '        "n": "10",\n        "genre": "all"\n    }\n}'
)
PARALLEL_RAW = '[get_weather(city="San Francisco"), get_weather(city="Seattle")]'
USER_INFO_RAW = "[get_user_info(user_id=7890, special='black')]"


def verify_quote(path: str, raw: str, stop: str) -> None:
    """The quoted output (plus the stop token the doc shows) must appear verbatim in Meta's doc."""
    url = f"https://raw.githubusercontent.com/meta-llama/llama-models/{META_SHA}/models/{path}"
    with urllib.request.urlopen(url, timeout=60) as resp:  # read-only GET
        doc = resp.read().decode("utf-8")
    if f"```\n{raw}{stop}\n```" not in doc:
        raise AssertionError(f"{raw!r} is not quoted verbatim in {url}")


def build_recorded(l33: Loaded, l4: Loaded) -> list[dict[str, Any]]:
    verify_quote("llama3_3/prompt_format.md", TRENDING_RAW, "<|eom_id|>")
    verify_quote("llama4/prompt_format.md", PARALLEL_RAW, "<|eot|>")
    verify_quote("llama4/prompt_format.md", USER_INFO_RAW, "<|eot|>")
    out = []

    def rec(ld: Loaded, name: str, raw: str, prov: Provenance, tools: list[Tool], tags: list[str], **kw: Any) -> None:
        out.append(
            Record(
                id=f"{SLUG}/{name}",
                family=SLUG,
                models=[ld.model.repo],
                provenance=prov,
                tools=tools,
                raw_output=raw,
                output_token_ids=encode_raw(ld.tok, raw),
                tokenizer=ld.tokenizer_pin,
                tags=tags,
                **kw,
            ).to_dict()
        )

    rec(
        l33,
        "l3-meta-python-tag-type-function",
        TRENDING_RAW,
        meta_doc("llama3_3/prompt_format.md", "L403-L412", "LicenseRef-llama3.3-community"),
        [T_TRENDING],
        ["single-call", "x-llama-json", "x-python-tag-prefix", "x-extra-type-key", "x-eom-terminator"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": [{"name": "trending_songs", "arguments": {"n": "10", "genre": "all"}}],
        },
        notes=META_NOTE + ' The model prefixed <|python_tag|>, added a "type": "function" key and ended with '
        "<|eom_id|> (Environment: ipython). The doc declares the tool with an (invalid) list of properties typed "
        "'object'; the schema here keeps the names and descriptions without types.",
    )
    rec(
        l4,
        "l4-meta-pythonic-parallel",
        PARALLEL_RAW,
        meta_doc("llama4/prompt_format.md", "L204-L207", "LicenseRef-llama4-community"),
        [T_WEATHER_META],
        ["parallel-calls", "x-llama-pythonic", "x-pythonic-parallel"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "San Francisco"}},
                {"name": "get_weather", "arguments": {"city": "Seattle"}},
            ],
        },
        notes=META_NOTE,
    )
    rec(
        l4,
        "l4-meta-pythonic-int-and-string",
        USER_INFO_RAW,
        meta_doc("llama4/prompt_format.md", "L259-L262", "LicenseRef-llama4-community"),
        [T_USER_INFO],
        ["single-call", "numeric-arguments", "x-llama-pythonic"],
        expected={
            "content": None,
            "reasoning_content": None,
            "tool_calls": [{"name": "get_user_info", "arguments": {"user_id": 7890, "special": "black"}}],
        },
        notes=META_NOTE + " Single quotes are Python string syntax, not part of the value.",
    )
    # Truncated: the parallel list without its closing bracket (the failure mode of vLLM #30722, where the model
    # stopped before completing the list).
    ids = encode_raw(l4.tok, PARALLEL_RAW)
    cut_ids = ids[:-1]
    raw = l4.tok.decode(cut_ids, skip_special_tokens=False)
    assert PARALLEL_RAW.startswith(raw) and not raw.endswith("]"), raw
    out.append(
        Record(
            id=f"{SLUG}/l4-truncated-missing-close-bracket",
            family=SLUG,
            models=[l4.model.repo],
            provenance=meta_doc("llama4/prompt_format.md", "L204-L207", "LicenseRef-llama4-community"),
            tools=[T_WEATHER_META],
            raw_output=raw,
            output_token_ids=cut_ids,
            tokenizer=l4.tokenizer_pin,
            expected_error={
                "reason": "The pythonic call list is cut before its final string, call and list close.",
                "accept": ["no_tool_calls", "content_passthrough", "exception"],
            },
            tags=["truncated", "malformed", "x-llama-pythonic"],
            notes="Token prefix of llama/l4-meta-pythonic-parallel without its last token (which carries the closing "
            "quote, parenthesis and bracket): the 'missing bracket' failure mode reported in "
            "https://github.com/vllm-project/vllm/issues/30722.",
        ).to_dict()
    )
    return out


def build_truncated(l33: Loaded, renders: dict[str, Rendered]) -> list[dict[str, Any]]:
    r = renders["typed-values"]
    raw, ids = cut_before(r, l33.tok, '"all_day"')
    return [
        Record(
            id=f"{SLUG}/l3-truncated-inside-parameters",
            family=SLUG,
            models=[l33.model.repo],
            provenance=Provenance(
                "template_render",
                l33.template_url(),
                l33.model.mirror_revision,
                l33.model.license,
                GENERATOR,
                l33.template_sha256,
            ),
            tools=TOOLS,
            raw_output=raw,
            output_token_ids=ids,
            tokenizer=l33.tokenizer_pin,
            expected_error={
                "reason": "max_tokens hit inside the parameters object; the JSON never closes.",
                "accept": ["no_tool_calls", "content_passthrough", "exception"],
            },
            tags=["truncated", "x-llama-json"],
            notes="Token prefix of llama/l3-typed-values.",
        ).to_dict()
    ]


MARKERS = [
    "<|python_tag|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|eot|>",
    "<|eom|>",
    "<|header_start|>",
    "<|header_end|>",
]


def family_json(lds: dict[str, Loaded]) -> dict[str, Any]:
    refs = []
    for key, variant in (("l33", "llama3-json"), ("l4", "llama4-pythonic")):
        ld = lds[key]
        refs.append(
            {
                "repo": ld.model.repo,
                "revision": ld.model.revision,
                "variant": variant,
                "default_generation_prompt": default_gen_prompt(ld.tok),
                "stop_tokens": [ld.tok.convert_ids_to_tokens(i) for i in ld.stop_ids],
                "gated": True,
                "tokenizer_mode": "hf",
            }
        )
    return {
        "slug": SLUG,
        "name": "Llama (3.1, 3.2, 3.3, 4)",
        "spec_version": "0.1",
        "has_reasoning": False,
        "markers": MARKERS,
        "reference_models": refs,
        "format_notes": "docs/formats/llama.md",
        "notes": 'Two syntaxes selected by models[0]: meta-llama/Llama-3* -> JSON ({"name", "parameters"}, '
        "optional <|python_tag|>; vLLM llama3_json, SGLang llama3), meta-llama/Llama-4* -> pythonic "
        "([f(a=1), g()]; vLLM llama4_pythonic, SGLang pythonic). Tags x-llama-json / x-llama-pythonic say which. "
        "All meta-llama repos are gated; every fixture's tokenizer pin names the ungated mirror used instead: "
        "unsloth/Llama-3.3-70B-Instruct@99cd0d2c829e92a67c844f9144c2509632e5c87f (tokenizer.json sha256 equal to "
        "meta-llama/Llama-3.3-70B-Instruct's; chat_template.jinja byte-identical to llama.cpp's copy of the HF "
        "template) and unsloth/Llama-4-Scout-17B-16E-Instruct@afd8e498c87bda51c7ea8ec68ea2f7c066e6340b "
        "(tokenizer.json sha256 equal to meta-llama/Llama-4-Scout-17B-16E-Instruct's; its chat template is NOT "
        "Meta's, so do not use it to build Llama 4 prompts).",
    }


def main() -> None:
    lds = {"l33": load(L33), "l4": load(L4)}
    check_llamacpp_copy(lds["l33"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(family_json(lds), ensure_ascii=False, indent=2) + "\n")
    recs, renders = build_renders(lds["l33"])
    write_jsonl(OUT / "l3-render.jsonl", recs)
    write_jsonl(OUT / "recorded.jsonl", build_recorded(lds["l33"], lds["l4"]))
    write_jsonl(OUT / "truncated.jsonl", build_truncated(lds["l33"], renders))

    imported_recs = []
    for item in imported.ITEMS:
        ld = lds[item.model_key]
        ids = encode_raw(ld.tok, item.raw)
        imported_recs.append(item.record(SLUG, ld.model.repo, ld.tokenizer_pin, ids).to_dict())
    write_jsonl(OUT / "imported.jsonl", imported_recs)

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} llama fixtures")


if __name__ == "__main__":
    main()
