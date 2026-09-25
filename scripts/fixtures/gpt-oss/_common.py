"""Shared helpers for the gpt-oss fixture generators (render_gpt_oss.py, import_gpt_oss.py).

Nothing here invents format text: callers render through openai-harmony / the HF chat
template, or copy raw outputs verbatim from engine tests and bug reports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SLUG = "gpt-oss"
OUT_DIR = REPO_ROOT / "fixtures" / SLUG

REF_REPO = "openai/gpt-oss-20b"
REF_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MODELS = [REF_REPO, "openai/gpt-oss-120b"]
TOKENIZER = {"repo": REF_REPO, "revision": REF_REVISION, "mode": "hf"}

# Stop ids from gpt-oss-20b generation_config.json (eos_token_id) that end an assistant turn.
CALL_ID = 200012  # <|call|>
RETURN_ID = 200002  # <|return|>
STOP_IDS = (CALL_ID, RETURN_ID)


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
        {"query": {"type": "string"}, "filters": {"type": "object"}},
        ["query"],
    ),
    "get_time": _fn("get_time", "Get the current UTC time.", {}, []),
    "write_file": _fn(
        "write_file",
        "Write text to a file.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
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
}


def tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(
    *,
    name: str,
    provenance: dict[str, Any],
    tools: list[dict[str, Any]],
    raw_output: str,
    tags: list[str],
    expected: dict[str, Any] | None = None,
    expected_error: dict[str, Any] | None = None,
    output_token_ids: list[int] | None = None,
    notes: str | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Build one fixture record in the canonical key order."""
    assert (expected is None) != (expected_error is None), name
    rec: dict[str, Any] = {
        "id": f"{SLUG}/{name}",
        "family": SLUG,
        "models": models or MODELS,
        "spec_version": "0.1",
        "provenance": provenance,
        "tools": tools,
        "raw_output": raw_output,
    }
    if output_token_ids is not None:
        rec["output_token_ids"] = output_token_ids
        rec["tokenizer"] = TOKENIZER
    if expected is not None:
        rec["expected"] = expected
    if expected_error is not None:
        rec["expected_error"] = expected_error
    rec["tags"] = tags
    if notes:
        rec["notes"] = notes
    return rec


def expected(
    content: str | None = None, reasoning: str | None = None, calls: list[tuple[str, dict[str, Any]]] | None = None
) -> dict[str, Any]:
    return {
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [{"name": n, "arguments": a} for n, a in (calls or [])],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate fixture ids"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} fixtures to {path.relative_to(REPO_ROOT)}")


def hf_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REF_REPO, revision=REF_REVISION)


def check_ids(tok: Any, ids: list[int], text: str) -> None:
    """The ids must decode (HF tokenizer, skip_special_tokens=False) to exactly raw_output."""
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == text, f"id/text mismatch:\n{decoded!r}\n{text!r}"
