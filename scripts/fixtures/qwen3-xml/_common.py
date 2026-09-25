"""Shared helpers for the qwen3-xml fixture generators (render_qwen3_xml.py, import_qwen3_xml.py)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SLUG = "qwen3-xml"
OUT_DIR = REPO_ROOT / "fixtures" / SLUG


@dataclass(frozen=True)
class Ref:
    """A model repo pinned to a revision."""

    repo: str
    revision: str

    @property
    def template_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/blob/{self.revision}/chat_template.jinja"

    @property
    def tokenizer_pin(self) -> dict[str, str]:
        return {"repo": self.repo, "revision": self.revision, "mode": "hf"}


QWEN38 = Ref("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
QWEN36 = Ref("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0")
QWEN35 = Ref("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
CODER = Ref("Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
CODER_480 = Ref("Qwen/Qwen3-Coder-480B-A35B-Instruct", "9d90cf8fca1bf7b7acca42d3fc9ae694a2194069")
CODER_NEXT = Ref("Qwen/Qwen3-Coder-Next", "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb")

# generation_config.json eos_token_id: <|im_end|>, <|endoftext|> (same strings for both vocabularies).
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
GEN_PROMPT_THINK = "<|im_start|>assistant\n<think>\n"
GEN_PROMPT_NO_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
GEN_PROMPT_CODER = "<|im_start|>assistant\n"


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": _fn(
        "get_weather",
        "Get the current weather for a city.",
        {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        ["city"],
    ),
    "search": _fn(
        "search",
        "Search the web.",
        {
            "query": {"type": "string"},
            "filters": {"type": "object"},
            "sites": {"type": "array", "items": {"type": "string"}},
        },
        ["query"],
    ),
    "get_time": _fn("get_time", "Get the current UTC time.", {}, []),
    "write_file": _fn(
        "write_file",
        "Write text to a file.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    "edit_file": _fn(
        "edit_file",
        "Replace an exact string in a file.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"],
    ),
    "set_alarm": _fn(
        "set_alarm",
        "Set an alarm.",
        {
            "hour": {"type": "integer"},
            "minute": {"type": "integer"},
            "volume": {"type": "number"},
            "repeat": {"type": "boolean"},
            "label": {"type": ["string", "null"]},
        },
        ["hour", "minute"],
    ),
    "lookup": _fn(
        "lookup",
        "Look up a record by its identifiers.",
        {
            "zip": {"type": "string"},
            "account_id": {"type": "string"},
            "flag": {"type": "string"},
            "payload": {"type": "string"},
        },
        ["zip"],
    ),
}


def tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@cache
def tokenizer(ref: Ref) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)


def expected(
    content: str | None = None, reasoning: str | None = None, calls: list[tuple[str, dict[str, Any]]] | None = None
) -> dict[str, Any]:
    return {
        "content": content or None,
        "reasoning_content": reasoning or None,
        "tool_calls": [{"name": n, "arguments": a} for n, a in (calls or [])],
    }


def record(
    *,
    name: str,
    models: list[str],
    provenance: dict[str, Any],
    tools: list[dict[str, Any]],
    raw_output: str,
    tags: list[str],
    tokenizer_ref: Ref | None = None,
    output_token_ids: list[int] | None = None,
    generation_prompt: str | None = None,
    thinking: bool | None = None,
    expected: dict[str, Any] | None = None,
    expected_error: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    assert (expected is None) != (expected_error is None), name
    rec: dict[str, Any] = {
        "id": f"{SLUG}/{name}",
        "family": SLUG,
        "models": models,
        "spec_version": "0.1",
        "provenance": provenance,
        "tools": tools,
        "raw_output": raw_output,
    }
    if output_token_ids is not None:
        assert tokenizer_ref is not None
        rec["output_token_ids"] = output_token_ids
        rec["tokenizer"] = tokenizer_ref.tokenizer_pin
    if generation_prompt is not None:
        rec["generation_prompt"] = generation_prompt
    if thinking is not None:
        rec["thinking"] = thinking
    if expected is not None:
        rec["expected"] = expected
    if expected_error is not None:
        rec["expected_error"] = expected_error
    rec["tags"] = tags
    if notes:
        rec["notes"] = notes
    return rec


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate fixture ids"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} fixtures to {path.relative_to(REPO_ROOT)}")


def encode_checked(ref: Ref, text: str) -> list[int]:
    """Tokenize raw text with the reference HF tokenizer and require a lossless round trip.

    Qwen's <tool_call>/<think> markers are added tokens that the HF tokenizer matches in text,
    so re-encoding is exact for this family (verified here for every fixture).
    """
    tok = tokenizer(ref)
    ids = list(tok.encode(text, add_special_tokens=False))
    assert tok.decode(ids, skip_special_tokens=False) == text, text
    return ids
