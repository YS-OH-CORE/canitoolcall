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
"""Build the ``glm`` fixture corpus (GLM-4.5/4.6, GLM-4.7, GLM-5.3).

Provenance of every fixture written here:

* ``template_render``: an assistant message rendered through the OFFICIAL chat
  template of the reference model (``chat_template.jinja`` at a pinned revision)
  with ``apply_chat_template(tokenize=True)``; see ``_core.render``. Truncated
  fixtures are a token prefix of such a render, which is what ``max_tokens``
  produces.
* ``engine_test`` / ``bug_report``: raw strings copied from vLLM tests or public
  issues, see ``imported.py``.

Run from the repo root (deterministic; re-running leaves ``git diff`` clean)::

    uv run --script scripts/fixtures/glm/build.py
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

SLUG = "glm"
OUT = REPO_ROOT / "fixtures" / SLUG
GENERATOR = f"scripts/fixtures/{SLUG}/build.py"


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    key: str
    repo: str
    revision: str
    license: str
    siblings: tuple[tuple[str, str], ...] = ()
    """(repo, revision) whose rendered output is checked to be identical, then listed in ``models``."""
    thinking_kwargs: dict[str, Any] = field(default_factory=dict)
    no_thinking_kwargs: dict[str, Any] | None = None


GLM45 = Model(
    "glm45",
    "zai-org/GLM-4.5",
    "cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
    "MIT",
    siblings=(
        ("zai-org/GLM-4.5-Air", "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"),
        ("zai-org/GLM-4.6", "be72194883d968d7923a07e2f61681ea9a2826d1"),
    ),
    no_thinking_kwargs={"enable_thinking": False},
)
GLM47 = Model(
    "glm47",
    "zai-org/GLM-4.7",
    "602d01efcdd332c5238ca4bcede555defbe83eb7",
    "MIT",
    siblings=(("zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67"),),
    no_thinking_kwargs={"enable_thinking": False},
)
GLM53 = Model(
    "glm53",
    "zai-org/GLM-5.3",
    "aca966e4e02791568aa6a4ced368624b3d897f42",
    "LicenseRef-glm-5.3",
    no_thinking_kwargs=None,  # GLM-5.3 always opens <think>; effort is set via the system prompt
)
MODELS = (GLM45, GLM47, GLM53)


@dataclass
class Loaded:
    model: Model
    tok: Any
    stop_ids: list[int]
    template_sha256: str
    siblings: list[tuple[str, Any, list[int]]]

    @property
    def tokenizer_pin(self) -> dict[str, str]:
        return {"repo": self.model.repo, "revision": self.model.revision, "mode": "hf"}

    def template_url(self) -> str:
        return f"https://huggingface.co/{self.model.repo}/blob/{self.model.revision}/chat_template.jinja"


def stop_ids(repo: str, revision: str) -> list[int]:
    gen = json.loads(Path(hf_hub_download(repo, "generation_config.json", revision=revision)).read_text())
    stops = gen["eos_token_id"]
    return [int(s) for s in (stops if isinstance(stops, list) else [stops])]


def load(model: Model) -> Loaded:
    tok = AutoTokenizer.from_pretrained(model.repo, revision=model.revision)
    sha = sha256_file(hf_hub_download(model.repo, "chat_template.jinja", revision=model.revision))
    sibs = [
        (repo, AutoTokenizer.from_pretrained(repo, revision=rev), stop_ids(repo, rev)) for repo, rev in model.siblings
    ]
    return Loaded(model, tok, stop_ids(model.repo, model.revision), sha, sibs)


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
T_TIMER = tool(
    "set_timer",
    "Start a countdown timer.",
    {"seconds": {"type": "integer"}, "label": {"type": "string"}},
    ["seconds"],
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
T_ORDER = tool(
    "lookup_order",
    "Look up an order by its id.",
    {"order_id": {"type": "string"}, "reference": {"type": "string"}},
    ["order_id"],
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
TOOLS: list[Tool] = [T_WEATHER, T_SEARCH, T_TIME, T_WRITE, T_TIMER, T_EVENT, T_ORDER, T_SETTINGS, T_CONVERT]

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


def tool_results(msg: Message) -> list[Message]:
    return [{"role": "tool", "content": '{"ok": true}'} for _ in msg.get("tool_calls") or []]


ALL = ("glm45", "glm47", "glm53")
CASES: list[Case] = [
    Case(
        "single-call",
        ["single-call", "reasoning"],
        assistant(
            reasoning="The user wants the current weather in Paris, in Celsius.",
            calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})],
        ),
        ALL,
        user="What's the weather in Paris?",
    ),
    Case(
        "parallel-same-tool",
        ["parallel-calls", "reasoning"],
        assistant(
            reasoning="Two cities, so two independent weather lookups.",
            calls=[call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Tokyo"})],
        ),
        ALL,
        user="Weather in Paris and Tokyo?",
    ),
    Case(
        "parallel-mixed-nested",
        ["parallel-calls", "nested-json", "reasoning"],
        assistant(
            reasoning="Check the weather, then search for cafés.",
            calls=[
                call("get_weather", {"city": "Zürich", "unit": "celsius"}),
                call(
                    "search_web",
                    {
                        "query": "best cafés",
                        "filters": {"site": "example.com", "max_results": 5, "tags": ["coffee", "wifi"]},
                    },
                ),
            ],
        ),
        ("glm45",),
    ),
    Case(
        "content-before-call",
        ["single-call", "text-before-call", "reasoning"],
        assistant(
            content="I'll look that up for you.",
            reasoning="Need live data, so call the tool.",
            calls=[call("get_weather", {"city": "Berlin"})],
        ),
        ("glm45", "glm47"),
        notes="The template separates content from the call (GLM-4.5 with a newline); the correct content is the "
        "message text without that separator.",
    ),
    Case(
        "no-call",
        ["no-call", "reasoning"],
        assistant(content="Paris is the capital of France.", reasoning="A simple fact; no tool is needed."),
        ALL,
        user="What is the capital of France?",
    ),
    Case(
        "nested-object",
        ["single-call", "nested-json", "numeric-arguments", "reasoning"],
        assistant(
            reasoning="Apply the dark theme and disable SMS.",
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
            ],
        ),
        ("glm45", "glm47"),
    ),
    Case(
        "unicode-emoji",
        ["single-call", "unicode", "reasoning"],
        assistant(
            reasoning="Search with the user's exact wording.",
            calls=[call("search_web", {"query": "Zürich café ☕ 東京の天気 🌸"})],
        ),
        ("glm45", "glm47"),
    ),
    Case(
        "empty-arguments",
        ["single-call", "empty-arguments", "reasoning"],
        assistant(reasoning="Just fetch the time.", calls=[call("get_time", {})]),
        ("glm45", "glm47"),
    ),
    Case(
        "marker-in-arguments",
        ["single-call", "marker-in-arguments", "reasoning"],
        assistant(
            reasoning="Write the format note to the docs file.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "docs/format.md",
                        "content": "Wrap each call in <tool_call> and </tool_call>; keys go in <arg_key>.",
                    },
                )
            ],
        ),
        ("glm45", "glm47"),
        notes="The string value contains <tool_call>, </tool_call> and <arg_key> (tokenized as the marker tokens). "
        "It does not contain </arg_value>, so the value is still delimited unambiguously.",
    ),
    Case(
        "typed-values",
        ["single-call", "numeric-arguments", "x-typed-values", "reasoning"],
        assistant(
            reasoning="Create the standup with a null reminder.",
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
            ],
        ),
        ("glm47",),
        notes="Non-string values are rendered with tojson (15, false, null, a JSON array); strings are raw.",
    ),
    Case(
        "number-looking-string",
        ["single-call", "regression", "x-number-looking-string", "reasoning"],
        assistant(
            reasoning="Look up the order id exactly as given.",
            calls=[call("lookup_order", {"order_id": "123_456", "reference": "0042"})],
        ),
        ("glm47",),
        notes="String values are raw, so 123_456 and 0042 must stay strings (the schema says string). "
        "SGLang's GLM detectors turned '123_456' into 123456 via ast.literal_eval: "
        "https://github.com/sgl-project/sglang/issues/30644",
    ),
    Case(
        "schema-coercion",
        ["single-call", "numeric-arguments", "x-schema-coercion", "reasoning"],
        assistant(
            reasoning="Three second timer labelled 3.",
            calls=[call("set_timer", {"seconds": 3, "label": "3"})],
        ),
        ALL,
        notes="Both values render as the bare text 3; only the tool schema tells the integer from the string.",
    ),
    Case(
        "string-escapes",
        ["single-call", "string-escapes", "reasoning"],
        assistant(
            reasoning="Save the note verbatim.",
            calls=[
                call(
                    "write_file",
                    {
                        "path": "C:\\Users\\ana\\notes.txt",
                        "content": 'line 1\n\tline 2 with "quotes" and a backslash \\ end',
                    },
                )
            ],
        ),
        ("glm47",),
        notes="String values are written raw (real newline, tab, quotes and backslashes), not JSON-escaped.",
    ),
    Case(
        "significant-whitespace",
        ["single-call", "x-significant-whitespace", "reasoning"],
        assistant(
            reasoning="Keep the spacing exactly.",
            calls=[call("write_file", {"path": "pad.txt", "content": "  two  spaces\n"})],
        ),
        ("glm47",),
        notes="Values have no wrapping newlines in GLM, so leading/trailing whitespace inside <arg_value> is data.",
    ),
    Case(
        "long-arguments",
        ["single-call", "long-arguments", "reasoning"],
        assistant(
            reasoning="Write the long file.",
            calls=[call("write_file", {"path": "fox.txt", "content": LONG_TEXT})],
        ),
        ("glm47",),
    ),
    Case(
        "json-looking-string",
        ["single-call", "x-json-looking-string", "reasoning"],
        assistant(
            reasoning="Search for the literal JSON text.",
            calls=[call("search_web", {"query": '{"a": 1}'})],
        ),
        ("glm47",),
        notes="The string parameter holds text that is valid JSON; the schema says string, so it must stay "
        'the string "{\\"a\\": 1}".',
    ),
    Case(
        "thinking-disabled",
        ["single-call"],
        assistant(calls=[call("get_weather", {"city": "Berlin", "unit": "celsius"})]),
        ("glm45", "glm47"),
        thinking=False,
        notes="enable_thinking=false: the generation prompt pre-fills the (empty) reasoning.",
    ),
    Case(
        "call-order-vs-tool-order",
        ["parallel-calls", "reasoning", "x-call-index-vs-tool-index"],
        assistant(
            reasoning="Weather, then search, then weather again.",
            calls=[
                call("get_weather", {"city": "Oslo"}),
                call("search_web", {"query": "Oslo museums"}),
                call("get_weather", {"city": "Bergen"}),
            ],
        ),
        ("glm45", "glm47"),
        notes="The same tool is called twice around another one; call indices must be 0,1,2, not the tools-list "
        "position (https://github.com/sgl-project/sglang/issues/33324).",
    ),
    Case(
        "multi-turn-second-call",
        ["single-call", "multi-turn", "reasoning"],
        assistant(
            reasoning="Now convert the temperature.",
            calls=[call("convert_units", {"value": 21.5, "from_unit": "celsius", "to_unit": "fahrenheit"})],
        ),
        ("glm47",),
        context=[
            {"role": "user", "content": "Weather in Rome, in Fahrenheit please."},
            assistant(reasoning="Get the weather first.", calls=[call("get_weather", {"city": "Rome"})]),
            {"role": "tool", "content": '{"temp_c": 21.5}'},
        ],
    ),
]


# --------------------------------------------------------------------------- build


def template_kwargs(model: Model, thinking: bool) -> dict[str, Any]:
    if thinking:
        return dict(model.thinking_kwargs)
    assert model.no_thinking_kwargs is not None
    return dict(model.no_thinking_kwargs)


def render_case(ld: Loaded, case: Case, tok: Any | None = None, stops: list[int] | None = None) -> Rendered:
    context = case.context or [{"role": "user", "content": case.user}]
    return render(
        tok or ld.tok,
        context,
        case.msg,
        tools=TOOLS,
        stop_ids=stops or ld.stop_ids,
        followups=tool_results(case.msg),
        template_kwargs=template_kwargs(ld.model, case.thinking),
        require_stop=bool(case.msg.get("tool_calls")),
    )


def models_for(ld: Loaded, cases: list[Case]) -> list[str]:
    """The reference repo plus every sibling whose render is identical for all cases."""
    out = [ld.model.repo]
    for repo, stok, sstops in ld.siblings:
        if all(render_case(ld, c).raw_output == render_case(ld, c, stok, sstops).raw_output for c in cases):
            out.append(repo)
        else:
            print(f"  sibling {repo} renders differently; not listed")
    return out


def default_gen_prompt(ld: Loaded) -> str:
    probe = [{"role": "user", "content": "x"}]
    full = ld.tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    bare = ld.tok.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    return str(full)[len(str(bare)) :]


def build_renders(ld: Loaded) -> tuple[list[dict[str, Any]], dict[str, Rendered]]:
    cases = [c for c in CASES if ld.model.key in c.variants]
    models = models_for(ld, cases)
    default_gp = default_gen_prompt(ld)
    prov = Provenance(
        "template_render", ld.template_url(), ld.model.revision, ld.model.license, GENERATOR, ld.template_sha256
    )
    records: list[dict[str, Any]] = []
    renders: dict[str, Rendered] = {}
    for case in cases:
        r = render_case(ld, case)
        renders[case.name] = r
        tags = [*case.tags, f"x-{ld.model.key}"]
        records.append(
            Record(
                id=f"{SLUG}/{ld.model.key}-{case.name}",
                family=SLUG,
                models=models,
                provenance=prov,
                tools=TOOLS,
                raw_output=r.raw_output,
                output_token_ids=r.output_token_ids,
                tokenizer=ld.tokenizer_pin,
                generation_prompt=r.generation_prompt if r.generation_prompt != default_gp else None,
                thinking=case.thinking,
                expected=expected_from(case.msg),
                tags=tags,
                notes=case.notes,
            ).to_dict()
        )
    return records, renders


def by_name(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


def build_truncated(lds: dict[str, Loaded], renders: dict[str, dict[str, Rendered]]) -> list[dict[str, Any]]:
    """Token-prefix cuts of renders: what ``max_tokens`` produces."""
    out: list[dict[str, Any]] = []
    accept = ["no_tool_calls", "content_passthrough", "exception"]

    def rec(ld: Loaded, name: str, case: Case, raw: str, ids: list[int], **kw: Any) -> None:
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
                tools=TOOLS,
                raw_output=raw,
                output_token_ids=ids,
                tokenizer=ld.tokenizer_pin,
                generation_prompt=renders[ld.model.key][case.name].generation_prompt if not case.thinking else None,
                thinking=case.thinking,
                **kw,
            ).to_dict()
        )

    g47, g45 = lds["glm47"], lds["glm45"]
    nothink = by_name("thinking-disabled")
    r = renders["glm47"][nothink.name]
    raw, ids = cut_before(r, g47.tok, "celsius")
    rec(
        g47,
        "truncated-inside-arg-value",
        nothink,
        raw,
        ids,
        expected_error={
            "reason": "max_tokens hit inside the second <arg_value>; the call never closes.",
            "accept": accept,
        },
        tags=["truncated", "x-glm47"],
        notes="Token prefix of glm/glm47-thinking-disabled.",
    )
    raw, ids = cut_before(r, g47.tok, "</tool_call>")
    rec(
        g47,
        "truncated-before-close-call",
        nothink,
        raw,
        ids,
        expected_error={
            "reason": "max_tokens hit after the last </arg_value>; </tool_call> is missing.",
            "accept": accept,
        },
        tags=["truncated", "x-glm47"],
        notes="Token prefix of glm/glm47-thinking-disabled. A parser should not emit a call that never closed.",
    )
    r = renders["glm45"][nothink.name]
    raw, ids = cut_before(r, g45.tok, "<arg_value>", occurrence=2)
    rec(
        g45,
        "truncated-after-arg-key",
        nothink,
        raw,
        ids,
        expected_error={"reason": "max_tokens hit between two arguments of an open call.", "accept": accept},
        tags=["truncated", "x-glm45"],
        notes="Token prefix of glm/glm45-thinking-disabled.",
    )
    # Truncated inside the reasoning: well defined (everything is reasoning, no call).
    single = by_name("single-call")
    r = renders["glm47"][single.name]
    raw, ids = cut_before(r, g47.tok, " in Celsius")
    rec(
        g47,
        "truncated-in-reasoning",
        single,
        raw,
        ids,
        expected={"content": None, "reasoning_content": raw, "tool_calls": []},
        tags=["truncated", "reasoning", "x-glm47", "x-think-no-open-tag"],
        notes="Token prefix of glm/glm47-single-call. The generation prompt opened <think>, so the unterminated "
        "text is reasoning, not content.",
    )
    return out


def build_missing_close_arg(g47: Loaded, renders: dict[str, Rendered]) -> dict[str, Any]:
    """vLLM #57826: the model omits the last </arg_value> but closes the call."""
    case = by_name("thinking-disabled")
    r = renders[case.name]
    close_id = g47.tok.convert_tokens_to_ids("</arg_value>")
    positions = [i for i, t in enumerate(r.output_token_ids) if t == close_id]
    ids = r.output_token_ids[: positions[-1]] + r.output_token_ids[positions[-1] + 1 :]
    raw = g47.tok.decode(ids, skip_special_tokens=False)
    return Record(
        id=f"{SLUG}/glm47-missing-last-close-arg-value",
        family=SLUG,
        models=[g47.model.repo],
        provenance=Provenance(
            "bug_report",
            "https://github.com/vllm-project/vllm/issues/57826",
            "issue opened 2026-09-20",
            "NOASSERTION",
            GENERATOR,
            attribution='Issue text: "a tool call that omits the last closing arg_value tag yields non-streaming '
            'arguments without the trailing parameter."',
        ),
        tools=TOOLS,
        raw_output=raw,
        output_token_ids=ids,
        tokenizer=g47.tokenizer_pin,
        generation_prompt=r.generation_prompt,
        thinking=False,
        expected=expected_from(case.msg),
        tags=["single-call", "malformed", "regression", "x-glm47", "x-missing-close-arg", "x-derived"],
        notes="raw_output is NOT quoted from the issue (it gives no GLM string): it is derived from "
        "glm/glm47-thinking-disabled by deleting the last </arg_value> token, the malformation "
        "#57826 describes. </tool_call> still closes the call, so the trailing value is recoverable; the issue (and "
        "the qwen3 fix it mirrors, vLLM #57707) expects the parser to keep it.",
    ).to_dict()


MARKERS = [
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|endoftext|>",
]
VARIANT = {"glm45": "glm45-newline", "glm47": "glm47-compact", "glm53": "glm53-compact"}


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
        "name": "GLM (4.5, 4.6, 4.7, 5.x)",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": MARKERS,
        "reference_models": refs,
        "format_notes": "docs/formats/glm.md",
        "notes": "Two whitespace variants of one XML-ish format: GLM-4.5/4.6 put newlines between the name, keys and "
        "values; GLM-4.7 and 5.x are compact. String values are raw (no escaping); other values are JSON, so a parser "
        "needs the tool schema. The variant is selected by models[0] (zai-org/GLM-4.5 -> glm45, zai-org/GLM-4.7 and "
        "zai-org/GLM-5.3 -> glm47 parsers). Tags x-glm45/x-glm47/x-glm53 name the reference template.",
    }


def main() -> None:
    lds = {m.key: load(m) for m in MODELS}
    (OUT / "family.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(family_json(lds), ensure_ascii=False, indent=2) + "\n")
    renders: dict[str, dict[str, Rendered]] = {}
    for key, ld in lds.items():
        recs, renders[key] = build_renders(ld)
        write_jsonl(OUT / f"{key}-render.jsonl", recs)
    truncated = build_truncated(lds, renders)
    truncated.append(build_missing_close_arg(lds["glm47"], renders["glm47"]))
    write_jsonl(OUT / "truncated-malformed.jsonl", truncated)

    imported_recs: list[dict[str, Any]] = []
    for item in imported.ITEMS:
        ld = lds[item.model_key]
        ids = encode_raw(ld.tok, item.raw)
        imported_recs.append(item.record(SLUG, ld.model.repo, ld.tokenizer_pin, ids).to_dict())
    write_jsonl(OUT / "imported.jsonl", imported_recs)

    everything = [json.loads(line) for p in sorted(OUT.glob("*.jsonl")) for line in p.read_text().splitlines()]
    validate_records(everything)
    print(f"validated {len(everything)} glm fixtures")


if __name__ == "__main__":
    main()
