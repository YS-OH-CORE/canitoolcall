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
"""Build the ``kimi`` fixture corpus (Kimi-K2.x section tokens and Kimi-K3 XTML).

Provenance of every fixture written here:

* ``template_render``:
  - K2.x: an assistant message rendered through the OFFICIAL ``chat_template.jinja``
    (Kimi-K2.6 with thinking, Kimi-K2-Instruct-0905 without) at a pinned revision;
  - K3: rendered by Moonshot's OFFICIAL encoder ``encoding_k3.py``, which the K3
    tokenizer's ``apply_chat_template(tokenize=True)`` calls; structural markers
    are encoded as control tokens and text as ordinary tokens, exactly as the
    encoder specifies.
  Both go through ``_core.render`` (prompt ids must prefix the full render, output
  cut at ``<|im_end|>`` / ``<|end_of_msg|>``). Truncated fixtures are token prefixes
  of renders, which is what ``max_tokens`` produces.
* ``engine_test`` / ``bug_report``: raw strings copied from vLLM tests or public
  issues, see ``imported.py``.

Tool-call ids follow Moonshot's guidance for K2 (``functions.{name}:{idx}`` with a
global counter, docs/tool_call_guidance.md in Kimi-K2-Instruct-0905).

Run from the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/kimi/build.py
"""

from __future__ import annotations

import json
import os
import sys
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

SLUG = "kimi"
OUT = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    key: str
    repo: str
    revision: str
    license: str
    template_file: str
    """The file that defines the format (hashed into template_sha256)."""
    siblings: tuple[tuple[str, str], ...] = ()
    thinking_kwargs: dict[str, Any] | None = None
    no_thinking_kwargs: dict[str, Any] | None = None
    thinking_field: bool = True
    """Whether fixtures record ``thinking`` (False for models without a thinking mode)."""


K26 = Model(
    "k26",
    "moonshotai/Kimi-K2.6",
    "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
    "LicenseRef-modified-mit",
    "chat_template.jinja",
    siblings=(
        ("moonshotai/Kimi-K2.5", "4d01dfe0332d63057c186e0b262165819efb6611"),
        ("moonshotai/Kimi-K2-Thinking", "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"),
        ("moonshotai/Kimi-K2.7-Code", "74797c9c62378b951a1f6fcf5c4631024e9b8bef"),
    ),
    # preserve_thinking keeps reasoning on the rendered (last) assistant turn even when it has no tool calls.
    thinking_kwargs={"preserve_thinking": True},
    no_thinking_kwargs={"thinking": False, "preserve_thinking": True},
)
K2I = Model(
    "k2i",
    "moonshotai/Kimi-K2-Instruct-0905",
    "ac6c49f04883bd0a0598b790693a72061c676629",
    "LicenseRef-modified-mit",
    "chat_template.jinja",
    siblings=(("moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50"),),
    thinking_kwargs={},
    thinking_field=False,
)
K3 = Model(
    "k3",
    "moonshotai/Kimi-K3",
    "f831ab66814297da540d832a5235f8e904f29d06",
    "LicenseRef-kimi-k3",
    "encoding_k3.py",
    thinking_kwargs={"thinking": True},
    no_thinking_kwargs={"thinking": False},
)
MODELS = (K26, K2I, K3)


@dataclass
class Loaded:
    model: Model
    tok: Any
    stop_ids: list[int]
    template_sha256: str
    siblings: list[tuple[str, Any]]

    @property
    def tokenizer_pin(self) -> dict[str, str]:
        return {"repo": self.model.repo, "revision": self.model.revision, "mode": "hf"}

    def template_url(self) -> str:
        return f"https://huggingface.co/{self.model.repo}/blob/{self.model.revision}/{self.model.template_file}"

    def encode(self, text: str) -> list[int]:
        # tokenization_kimi.TikTokenTokenizer.encode maps literal <|...|> markers to their ids.
        return [int(i) for i in self.tok.encode(text)]


def load(model: Model) -> Loaded:
    tok = AutoTokenizer.from_pretrained(model.repo, revision=model.revision, trust_remote_code=True)
    gen = json.loads(Path(hf_hub_download(model.repo, "generation_config.json", revision=model.revision)).read_text())
    stops = gen["eos_token_id"]
    stops = stops if isinstance(stops, list) else [stops]
    sha = sha256_file(hf_hub_download(model.repo, model.template_file, revision=model.revision))
    sibs = [
        (repo, AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=True))
        for repo, rev in model.siblings
    ]
    return Loaded(model, tok, [int(s) for s in stops], sha, sibs)


# --------------------------------------------------------------------------- tools

T_WEATHER = tool(
    "get_weather",
    "Get the current weather for a city.",
    {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    ["city"],
)
T_SEARCH = tool(
    "search_web",
    "Search the web.",
    {
        "query": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "site": {"type": "string"},
                "max_results": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    ["query"],
)
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
        "location": {"type": "object"},
    },
    ["title"],
)
T_CONVERT = tool(
    "convert_units",
    "Convert a value between units.",
    {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
    ["value", "from_unit", "to_unit"],
)
T_QA = tool(
    "save_answer",
    "Store an answer under odd-looking keys.",
    {"q&a": {"type": "string"}, 'say "hi"': {"type": "string"}},
)
TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_EVENT, T_CONVERT]
TOOLS_K3: list[Tool] = [*TOOLS, T_QA]

LONG_TEXT = "\n".join(f"{i:03d}: The quick brown fox jumps over the lazy dog." for i in range(60))


# --------------------------------------------------------------------------- cases


@dataclass
class Case:
    name: str
    tags: list[str]
    msg: Message
    variants: tuple[str, ...]
    thinking: bool = True
    context: list[Message] = field(default_factory=list)
    notes: str | None = None
    user: str = "Please help."


def ids_k2(msg: Message, start: int = 0) -> Message:
    """Give each call Moonshot's id ``functions.{name}:{idx}`` (global counter from ``start``)."""
    for i, tc in enumerate(msg.get("tool_calls") or []):
        tc["id"] = f"functions.{tc['function']['name']}:{start + i}"
    return msg


def tool_results(msg: Message) -> list[Message]:
    return [
        {"role": "tool", "tool_call_id": tc.get("id", f"call_{i}"), "name": tc["function"]["name"], "content": "ok"}
        for i, tc in enumerate(msg.get("tool_calls") or [])
    ]


K2 = ("k26", "k2i")
ALL = ("k26", "k2i", "k3")
CASES: list[Case] = [
    Case(
        "single-call",
        ["single-call", "reasoning", "x-id-encodes-name"],
        assistant(
            reasoning="The user wants the weather in Paris in Celsius.",
            calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})],
        ),
        ALL,
        user="What's the weather in Paris?",
    ),
    Case(
        "parallel-same-tool",
        ["parallel-calls", "reasoning", "x-global-idx"],
        assistant(
            reasoning="Two cities, two lookups.",
            calls=[call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Tokyo"})],
        ),
        ("k26", "k3"),
    ),
    Case(
        "parallel-three-tools",
        ["parallel-calls", "nested-json", "reasoning", "x-global-idx"],
        assistant(
            reasoning="Weather, time and a search.",
            calls=[
                call("get_weather", {"city": "Zürich"}),
                call("get_time", {}),
                call("search_web", {"query": "cafés", "filters": {"site": "example.com", "max_results": 3}}),
            ],
        ),
        ("k26",),
    ),
    Case(
        "content-before-call",
        ["single-call", "text-before-call", "reasoning"],
        assistant(
            content="Let me check that for you.",
            reasoning="Live data needed.",
            calls=[call("get_weather", {"city": "Berlin"})],
        ),
        ALL,
    ),
    Case(
        "no-call",
        ["no-call", "reasoning"],
        assistant(content="Paris is the capital of France.", reasoning="A simple fact; no tool needed."),
        ALL,
        user="What is the capital of France?",
    ),
    Case(
        "nested-json",
        ["single-call", "nested-json", "numeric-arguments", "reasoning"],
        assistant(
            reasoning="Create the event with a structured location.",
            calls=[
                call(
                    "create_event",
                    {
                        "title": "Offsite",
                        "attendees": ["ana@example.com", "bo@example.com"],
                        "location": {"name": "HQ", "geo": {"lat": 47.3769, "lon": 8.5417}, "floors": [1, 2]},
                    },
                )
            ],
        ),
        ("k26", "k3"),
    ),
    Case(
        "typed-values",
        ["single-call", "numeric-arguments", "reasoning", "x-typed-args"],
        assistant(
            reasoning="Standup with no reminder.",
            calls=[
                call(
                    "create_event",
                    {"title": "Standup", "duration_minutes": 15, "all_day": False, "reminder_minutes": None},
                )
            ],
        ),
        ("k3",),
        notes="Kimi-K3 renders each argument with an XTML type attribute (string/number/boolean/null).",
    ),
    Case(
        "unicode-emoji",
        ["single-call", "unicode", "reasoning"],
        assistant(
            reasoning="Search with the exact wording.",
            calls=[call("search_web", {"query": "Zürich café ☕ 東京の天気 🌸"})],
        ),
        ("k26",),
    ),
    Case(
        "empty-arguments",
        ["single-call", "empty-arguments", "reasoning"],
        assistant(reasoning="Just get the time.", calls=[call("get_time", {})]),
        ("k26", "k3"),
    ),
    Case(
        "string-escapes",
        ["single-call", "string-escapes", "reasoning"],
        assistant(
            reasoning="Save the note verbatim.",
            calls=[
                call(
                    "write_file",
                    {"path": "C:\\Users\\ana\\notes.txt", "content": 'line 1\n\tline 2 "quoted" \\ end'},
                )
            ],
        ),
        ("k26", "k3"),
        notes="K2 carries JSON (escaped); K3 carries string values raw.",
    ),
    Case(
        "marker-in-arguments",
        ["single-call", "marker-in-arguments", "reasoning"],
        assistant(
            reasoning="Document the tool-call markers.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "docs/kimi.md",
                        "content": "A section opens with <|tool_calls_section_begin|> and each call with "
                        "<|tool_call_begin|>; </think> ends reasoning.",
                    },
                )
            ],
        ),
        ("k26",),
        notes="The JSON string contains opening markers and </think>; the official template tokenizes them as "
        "marker tokens. None of them ends the argument JSON, which is closed by <|tool_call_end|>.",
    ),
    Case(
        "marker-terminator-in-string",
        ["single-call", "marker-in-arguments", "reasoning", "x-marker-terminator-in-string"],
        assistant(
            reasoning="Write the literal end marker into the file.",
            calls=[call("write_file", {"path": "end.txt", "content": "stop at <|tool_call_end|> please"})],
        ),
        ("k26",),
        notes="The JSON string contains <|tool_call_end|>, tokenized as the marker token by the template. Only a "
        "JSON-aware parser recovers the full argument; the call's real end is the second <|tool_call_end|>.",
    ),
    Case(
        "control-marker-text-in-value",
        ["single-call", "marker-in-arguments", "reasoning", "x-control-token-vs-text"],
        assistant(
            reasoning="Write the literal XTML marker text.",
            calls=[call("write_file", {"path": "x.txt", "content": "<|close|>argument<|sep|> is how args end"})],
        ),
        ("k3",),
        notes="encoding_k3 encodes argument text with allow_special=False, so the value's <|close|>/<|sep|> are "
        "ORDINARY tokens while the real markers are control tokens. The text is identical; only output_token_ids "
        "tell them apart.",
    ),
    Case(
        "xtml-attr-escaping",
        ["single-call", "reasoning", "x-xtml-attr-escaping"],
        assistant(
            reasoning="Keys need escaping.",
            calls=[call("save_answer", {"q&a": "fish & chips", 'say "hi"': 'she said "hi"'})],
        ),
        ("k3",),
        notes='Attribute values escape & and " (&amp; &quot;); argument bodies stay raw.',
    ),
    Case(
        "long-arguments",
        ["single-call", "long-arguments", "reasoning"],
        assistant(
            reasoning="Write the long file.", calls=[call("write_file", {"path": "fox.txt", "content": LONG_TEXT})]
        ),
        ("k3",),
    ),
    Case(
        "empty-think",
        ["single-call", "x-empty-think"],
        assistant(calls=[call("get_weather", {"city": "Rome"})]),
        ("k26", "k3"),
        notes="Thinking enabled but the reasoning is empty: the completion starts with the think close marker.",
    ),
    Case(
        "thinking-disabled-call",
        ["single-call"],
        assistant(calls=[call("get_weather", {"city": "Madrid", "unit": "celsius"})]),
        ("k26", "k3"),
        thinking=False,
        notes="thinking=false: the generation prompt pre-fills the empty reasoning (K2) / opens the response (K3).",
    ),
    Case(
        "thinking-disabled-no-call",
        ["no-call"],
        assistant(content="Hello! How can I help?"),
        ("k3",),
        thinking=False,
        user="Hi",
    ),
    Case(
        "multi-turn-second-call",
        ["single-call", "multi-turn", "reasoning", "x-global-idx"],
        assistant(
            reasoning="Now convert it.",
            calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})],
        ),
        ("k26", "k3"),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(
                reasoning="Get the weather first.",
                calls=[call("get_weather", {"city": "Rome"}, "functions.get_weather:0")],
            ),
            {
                "role": "tool",
                "tool_call_id": "functions.get_weather:0",
                "name": "get_weather",
                "content": '{"c": 21.5}',
            },
        ],
        notes="K2's idx is a global counter, so the second turn's call is functions.convert_units:1. K3's index "
        "restarts at 1 in every message.",
    ),
]


# --------------------------------------------------------------------------- build


def template_kwargs(model: Model, thinking: bool) -> dict[str, Any]:
    kw = model.thinking_kwargs if thinking else model.no_thinking_kwargs
    assert kw is not None, f"{model.key} has no thinking={thinking} mode"
    return dict(kw)


def prepared(ld: Loaded, case: Case) -> tuple[list[Message], Message]:
    context = json.loads(json.dumps(case.context)) or [{"role": "user", "content": case.user}]
    msg = json.loads(json.dumps(case.msg))
    if ld.model.key == "k2i":
        msg.pop("reasoning_content", None)  # Kimi-K2-Instruct has no thinking
        for m in context:
            m.pop("reasoning_content", None)
    if ld.model.key in K2:
        prior = sum(len(m.get("tool_calls") or []) for m in context if m.get("role") == "assistant")
        ids_k2(msg, prior)
    else:
        for i, tc in enumerate(msg.get("tool_calls") or []):
            tc["id"] = f"call_{i}"
    return context, msg


def render_case(ld: Loaded, case: Case, tok: Any | None = None) -> tuple[Message, Rendered]:
    context, msg = prepared(ld, case)
    tools = TOOLS_K3 if ld.model.key == "k3" else TOOLS
    r = render(
        tok or ld.tok,
        context,
        msg,
        tools=tools,
        stop_ids=ld.stop_ids,
        followups=tool_results(msg),
        template_kwargs=template_kwargs(ld.model, case.thinking),
    )
    return msg, r


def default_gen_prompt(ld: Loaded, thinking: bool = True) -> str:
    probe = [{"role": "user", "content": "x"}]
    kw = template_kwargs(ld.model, thinking)
    full = ld.tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False, **kw)
    bare = ld.tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False, **kw)
    return str(full)[len(str(bare)) :]


def models_for(ld: Loaded, cases: list[Case]) -> list[str]:
    out = [ld.model.repo]
    for repo, stok in ld.siblings:
        try:
            same = all(render_case(ld, c)[1].raw_output == render_case(ld, c, stok)[1].raw_output for c in cases)
        except Exception as exc:  # a sibling template may reject a kwarg or a message shape
            print(f"  sibling {repo}: {type(exc).__name__}: {exc}")
            same = False
        if same:
            out.append(repo)
        else:
            print(f"  sibling {repo} renders differently; not listed")
    return out


def tag_for(ld: Loaded) -> list[str]:
    return ["x-kimi-k3"] if ld.model.key == "k3" else ["x-kimi-k2"]


def build_renders(ld: Loaded) -> tuple[list[dict[str, Any]], dict[str, tuple[Message, Rendered]]]:
    cases = [c for c in CASES if ld.model.key in c.variants]
    models = models_for(ld, cases)
    default_gp = default_gen_prompt(ld)
    prov = Provenance(
        "template_render", ld.template_url(), ld.model.revision, ld.model.license, GENERATOR, ld.template_sha256
    )
    records: list[dict[str, Any]] = []
    renders: dict[str, tuple[Message, Rendered]] = {}
    for case in cases:
        msg, r = render_case(ld, case)
        renders[case.name] = (msg, r)
        tags = list(case.tags)
        if ld.model.key == "k2i":
            tags = [t for t in tags if t != "reasoning"]
        if ld.model.key == "k3":  # K2-only concepts: the name lives in the id, idx is global
            tags = [t for t in tags if t not in ("x-id-encodes-name", "x-global-idx")]
        if case.thinking and ld.model.key != "k2i" and "x-empty-think" not in tags:
            tags.append("x-think-no-open-tag")
        records.append(
            Record(
                id=f"{SLUG}/{ld.model.key}-{case.name}",
                family=SLUG,
                models=models,
                provenance=prov,
                tools=TOOLS_K3 if ld.model.key == "k3" else TOOLS,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=ld.tokenizer_pin,
                generation_prompt=r.generation_prompt if r.generation_prompt != default_gp else None,
                thinking=case.thinking if ld.model.thinking_field else None,
                expected=expected_from(msg),
                tags=[*tags, *tag_for(ld)],
                notes=case.notes,
            ).to_dict()
        )
    return records, renders


def by_name(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


def build_truncated(
    lds: dict[str, Loaded], renders: dict[str, dict[str, tuple[Message, Rendered]]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    accept = ["no_tool_calls", "content_passthrough", "exception"]

    def rec(ld: Loaded, name: str, case: Case, raw: str, ids: list[int], **kw: Any) -> None:
        r = renders[ld.model.key][case.name][1]
        out.append(
            Record(
                id=f"{SLUG}/{ld.model.key}-{name}",
                family=SLUG,
                models=[ld.model.repo],
                provenance=Provenance(
                    "template_render",
                    ld.template_url(),
                    ld.model.revision,
                    ld.model.license,
                    GENERATOR,
                    ld.template_sha256,
                ),
                tools=TOOLS_K3 if ld.model.key == "k3" else TOOLS,
                raw_output=raw,
                output_token_ids=ids,
                tokenizer=ld.tokenizer_pin,
                generation_prompt=r.generation_prompt if not case.thinking else None,
                thinking=case.thinking,
                **kw,
            ).to_dict()
        )

    k26, k3 = lds["k26"], lds["k3"]
    nothink = by_name("thinking-disabled-call")
    raw, ids = cut_before(renders["k26"][nothink.name][1], k26.tok, '"celsius"')
    rec(
        k26,
        "truncated-inside-arguments",
        nothink,
        raw,
        ids,
        expected_error={"reason": "max_tokens hit inside the argument JSON; the call never closes.", "accept": accept},
        tags=["truncated", "x-kimi-k2"],
        notes="Token prefix of kimi/k26-thinking-disabled-call.",
    )
    raw, ids = cut_before(renders["k3"][nothink.name][1], k3.tok, "Madrid")
    rec(
        k3,
        "truncated-inside-argument",
        nothink,
        raw,
        ids,
        expected_error={"reason": "max_tokens hit inside the first <argument> element.", "accept": accept},
        tags=["truncated", "x-kimi-k3"],
        notes="Token prefix of kimi/k3-thinking-disabled-call.",
    )
    single = by_name("single-call")
    for key, marker in (("k26", " in Celsius"), ("k3", " in Celsius")):
        ld = lds[key]
        raw, ids = cut_before(renders[key][single.name][1], ld.tok, marker)
        rec(
            ld,
            "truncated-in-reasoning",
            single,
            raw,
            ids,
            expected={"content": None, "reasoning_content": raw, "tool_calls": []},
            tags=["truncated", "reasoning", "x-think-no-open-tag", *tag_for(ld)],
            notes=f"Token prefix of kimi/{key}-single-call. The generation prompt opened the think channel, so the "
            "unterminated text is reasoning (see https://github.com/vllm-project/vllm/issues/57353 for K3).",
        )
    return out


MARKERS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
    "<think>",
    "</think>",
    "<|im_end|>",
    "<|im_assistant|>",
    "<|im_system|>",
    "<|im_middle|>",
    "<|open|>",
    "<|close|>",
    "<|sep|>",
    "<|end_of_msg|>",
]
VARIANT = {"k26": "kimi-k2-thinking", "k2i": "kimi-k2-instruct", "k3": "kimi-k3-xtml"}


def family_json(lds: dict[str, Loaded]) -> dict[str, Any]:
    refs = []
    for key, ld in lds.items():
        refs.append(
            {
                "repo": ld.model.repo,
                "revision": ld.model.revision,
                "variant": VARIANT[key],
                "default_generation_prompt": default_gen_prompt(ld),
                "stop_tokens": [ld.tok.convert_ids_to_tokens(i) for i in ld.stop_ids],
                "tokenizer_mode": "hf",
            }
        )
    return {
        "slug": SLUG,
        "name": "Kimi (K2.x section tokens, K3 XTML)",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": MARKERS,
        "reference_models": refs,
        "format_notes": "docs/formats/kimi.md",
        "notes": "Two unrelated sub-formats selected by models[0]: Kimi-K2* -> kimi_k2 (section tokens, the tool "
        "name lives in the call id functions.{name}:{idx}); Kimi-K3 -> kimi_k3 (XTML control tokens). Tags x-kimi-k2 "
        "/ x-kimi-k3 name the sub-format. K3 re-uses ids 163586-163588 for different tokens than K2, so token ids "
        "only make sense with the fixture's own tokenizer pin. Kimi tokenizers are tiktoken-based "
        "(trust_remote_code=True).",
    }


def main() -> None:
    lds = {m.key: load(m) for m in MODELS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(family_json(lds), ensure_ascii=False, indent=2) + "\n")
    renders: dict[str, dict[str, tuple[Message, Rendered]]] = {}
    for key, ld in lds.items():
        recs, renders[key] = build_renders(ld)
        write_jsonl(OUT / f"{key}-render.jsonl", recs)
    write_jsonl(OUT / "truncated.jsonl", build_truncated(lds, renders))

    imported_recs = []
    for item in imported.ITEMS:
        ld = lds[item.model_key]
        ids = encode_raw(ld.tok, item.raw, ld.encode)
        imported_recs.append(item.record(SLUG, ld.model.repo, ld.tokenizer_pin, ids).to_dict())
    write_jsonl(OUT / "imported.jsonl", imported_recs)

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} kimi fixtures")


if __name__ == "__main__":
    main()
