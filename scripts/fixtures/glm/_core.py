"""Shared helpers for the fixture generators of fixtures group 3 (kimi, glm, llama).

This file is kept byte-identical in ``scripts/fixtures/{kimi,glm,llama}/_core.py`` so
that each family directory stays self-contained (one builder owns each directory).

The generators run as PEP 723 scripts (``uv run --script``) in a throwaway env with
pinned ``transformers``/``tokenizers``/``tiktoken``. Only tokenizer and template
files are downloaded, each at a pinned revision; model weights are never fetched.

The rendering recipe follows spec/README.md ("Token ids come first"):

1. render ``context`` with ``add_generation_prompt=True`` and ``tokenize=True``
   (the prompt ids);
2. render ``context + [assistant] + followups`` with ``add_generation_prompt=False``
   and ``tokenize=True`` (the full ids); the prompt ids must be a prefix;
3. the output ids are the rest, cut at the first stop id from
   ``generation_config.json``;
4. ``raw_output`` is the output ids decoded with ``skip_special_tokens=False``, and
   is cross-checked against the text render.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPEC_VERSION = "0.1"
REPO_ROOT = Path(__file__).resolve().parents[3]

Message = dict[str, Any]
Tool = dict[str, Any]


# --------------------------------------------------------------------------- tools


def tool(
    name: str, description: str | None, properties: Mapping[str, Any] | None = None, required: Sequence[str] = ()
) -> Tool:
    """An OpenAI-format function tool (``description=None`` omits the field)."""
    params: dict[str, Any] = {"type": "object", "properties": dict(properties or {})}
    if required:
        params["required"] = list(required)
    fn: dict[str, Any] = {"name": name}
    if description is not None:
        fn["description"] = description
    fn["parameters"] = params
    return {"type": "function", "function": fn}


def call(name: str, arguments: Mapping[str, Any], call_id: str | None = None) -> dict[str, Any]:
    """An OpenAI-format assistant tool call with a *dict* of arguments."""
    tc: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": dict(arguments)}}
    if call_id is not None:
        tc["id"] = call_id
    return tc


def assistant(
    content: str | None = None, reasoning: str | None = None, calls: Sequence[dict[str, Any]] = ()
) -> Message:
    msg: Message = {"role": "assistant", "content": content if content is not None else ""}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = list(calls)
    return msg


def expected_from(msg: Message) -> dict[str, Any]:
    """The correct parse of a rendered assistant message: the message itself."""
    return {
        "content": msg.get("content") or None,
        "reasoning_content": msg.get("reasoning_content") or None,
        "tool_calls": [
            {"name": tc["function"]["name"], "arguments": dict(tc["function"]["arguments"])}
            for tc in msg.get("tool_calls") or []
        ],
    }


# --------------------------------------------------------------------------- hashing


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- rendering


def _ids(out: Any) -> list[int]:
    """``apply_chat_template(tokenize=True)`` returns a list or a BatchEncoding."""
    if isinstance(out, Mapping) or hasattr(out, "keys"):
        out = out["input_ids"]
    ids = list(out)
    if ids and isinstance(ids[0], list):  # batched
        ids = ids[0]
    return [int(i) for i in ids]


@dataclass
class Rendered:
    raw_output: str
    output_token_ids: list[int]
    generation_prompt: str
    stopped: bool
    """True when a stop id terminated the output (False: the render simply ended)."""


def render(
    tok: Any,
    context: Sequence[Message],
    msg: Message,
    *,
    tools: Sequence[Tool] | None,
    stop_ids: Iterable[int],
    followups: Sequence[Message] = (),
    template_kwargs: Mapping[str, Any] | None = None,
    require_stop: bool = True,
) -> Rendered:
    """Render one assistant generation through the tokenizer's official chat template."""
    kw = dict(template_kwargs or {})
    stops = set(stop_ids)
    ctx = list(context)
    full_msgs = [*ctx, msg, *followups]

    prompt_ids = _ids(
        tok.apply_chat_template(ctx, tools=tools or None, add_generation_prompt=True, tokenize=True, **kw)
    )
    full_ids = _ids(
        tok.apply_chat_template(full_msgs, tools=tools or None, add_generation_prompt=False, tokenize=True, **kw)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError("prompt ids are not a prefix of the full render (history render != generation)")
    rest = full_ids[len(prompt_ids) :]
    cut = next((i for i, t in enumerate(rest) if t in stops), None)
    if cut is None and require_stop:
        raise AssertionError("no stop id found after the assistant message")
    out_ids = rest if cut is None else rest[:cut]

    # Text-level cross-check: the decoded ids must equal the text render.
    prompt_text = tok.apply_chat_template(ctx, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
    bare_text = tok.apply_chat_template(ctx, tools=tools or None, add_generation_prompt=False, tokenize=False, **kw)
    full_text = tok.apply_chat_template(
        full_msgs, tools=tools or None, add_generation_prompt=False, tokenize=False, **kw
    )
    assert isinstance(prompt_text, str) and isinstance(full_text, str) and isinstance(bare_text, str)
    if not full_text.startswith(prompt_text):
        raise AssertionError("prompt text is not a prefix of the full text render")
    raw = tok.decode(out_ids, skip_special_tokens=False)
    if not full_text[len(prompt_text) :].startswith(raw):
        raise AssertionError(f"decoded ids disagree with the text render: {raw!r}")
    if not prompt_text.startswith(bare_text):
        raise AssertionError("cannot isolate the generation prompt")
    return Rendered(raw, out_ids, prompt_text[len(bare_text) :], cut is not None)


def encode_raw(tok: Any, raw: str, encode: Callable[[str], list[int]] | None = None) -> list[int]:
    """Token ids of a copied raw output (engine tests, bug reports, recordings).

    Special tokens written literally in ``raw`` map to their ids, which is what the
    model emits. The decode must round-trip exactly, otherwise ids are omitted.
    """
    ids = encode(raw) if encode is not None else tok.encode(raw, add_special_tokens=False)
    ids = [int(i) for i in ids]
    back = tok.decode(ids, skip_special_tokens=False)
    if back != raw:
        raise AssertionError(f"encode/decode does not round-trip: {raw!r} -> {back!r}")
    return ids


# --------------------------------------------------------------------------- records


@dataclass
class Provenance:
    kind: str
    source_url: str
    revision: str
    license: str
    generator: str | None = None
    template_sha256: str | None = None
    attribution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class Record:
    id: str
    family: str
    models: list[str]
    provenance: Provenance
    tools: list[Tool]
    raw_output: str
    tags: list[str]
    expected: dict[str, Any] | None = None
    expected_error: dict[str, Any] | None = None
    output_token_ids: list[int] | None = None
    tokenizer: dict[str, str] | None = None
    generation_prompt: str | None = None
    thinking: bool | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "models": self.models,
            "spec_version": SPEC_VERSION,
            "provenance": self.provenance.to_dict(),
            "tools": self.tools,
            "raw_output": self.raw_output,
        }
        if self.output_token_ids is not None:
            out["output_token_ids"] = self.output_token_ids
        if self.tokenizer is not None:
            out["tokenizer"] = self.tokenizer
        if self.generation_prompt is not None:
            out["generation_prompt"] = self.generation_prompt
        if self.thinking is not None:
            out["thinking"] = self.thinking
        if self.expected is not None:
            out["expected"] = self.expected
        if self.expected_error is not None:
            out["expected_error"] = self.expected_error
        out["tags"] = self.tags
        if self.notes is not None:
            out["notes"] = self.notes
        return out


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    """Validate against spec/fixture.schema.json, check id uniqueness and tool names."""
    import jsonschema

    schema = json.loads((REPO_ROOT / "spec" / "fixture.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    seen: set[str] = set()
    for rec in records:
        errors = sorted(validator.iter_errors(rec), key=lambda e: list(e.path))
        if errors:
            raise AssertionError(f"{rec['id']}: {errors[0].message} at {list(errors[0].path)}")
        if rec["id"] in seen:
            raise AssertionError(f"duplicate id {rec['id']}")
        seen.add(rec["id"])
        offered = {t["function"]["name"] for t in rec["tools"]}
        for tc in (rec.get("expected") or {}).get("tool_calls", []):
            if tc["name"] not in offered:
                raise AssertionError(f"{rec['id']}: expected call {tc['name']!r} is not an offered tool")


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(records):3d} fixtures -> {path.relative_to(REPO_ROOT)}")


def truncate(rendered: Rendered, tok: Any, keep: int) -> tuple[str, list[int]]:
    """A ``max_tokens`` cut: the first ``keep`` output ids of a render."""
    ids = rendered.output_token_ids[:keep]
    return tok.decode(ids, skip_special_tokens=False), ids


def cut_before(rendered: Rendered, tok: Any, marker: str, occurrence: int = 1) -> tuple[str, list[int]]:
    """Cut a render at the last token boundary at or before the n-th ``marker``.

    Used for truncated fixtures: the result is a token prefix of the render, which is
    exactly what ``max_tokens`` produces.
    """
    raw = rendered.raw_output
    pos = -1
    for _ in range(occurrence):
        pos = raw.index(marker, pos + 1)
    ids = rendered.output_token_ids
    best = 0
    for i in range(len(ids) + 1):
        text = tok.decode(ids[:i], skip_special_tokens=False)
        if len(text) > pos:
            break
        if raw.startswith(text):
            best = i
    ids = ids[:best]
    return tok.decode(ids, skip_special_tokens=False), ids
