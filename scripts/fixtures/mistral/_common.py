"""Shared helpers for the mistral fixture generators (render_mistral.py, import_mistral.py).

The Mistral format is defined by the mistral-common tokenizer version, not by a Jinja template, so
mistral-common (the reference encoder) renders and tokenizes everything here. Control tokens such
as [TOOL_CALLS] are only ever produced as control-token ids: mistral-common never maps the TEXT
"[TOOL_CALLS]" to the control token, which is why output_token_ids are mandatory for this family.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SLUG = "mistral"
OUT_DIR = REPO_ROOT / "fixtures" / SLUG


@dataclass(frozen=True)
class Ref:
    repo: str
    revision: str
    mode: str = "mistral"  # tokenizer mode recorded in the fixture pin

    @property
    def tree_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/tree/{self.revision}"

    @property
    def pin(self) -> dict[str, str]:
        return {"repo": self.repo, "revision": self.revision, "mode": self.mode}


# v13+ with [THINK]/[/THINK] control tokens
MAGISTRAL_2509 = Ref("mistralai/Magistral-Small-2509", "a31cc96ab10cf19bc42c628fedf1e359e0853c49")
MINISTRAL3_REASONING = Ref("mistralai/Ministral-3-14B-Reasoning-2512", "51f9210f3cd20f3452a80d5819d15dc61cc50630")
SMALL4 = Ref("mistralai/Mistral-Small-4-119B-2603", "a11f36bebf709121056b1dbcc943d1c6afbe494d")
MEDIUM35 = Ref("mistralai/Mistral-Medium-3.5-128B", "22b2b868a15677cfa6061277ed2f653d1349a9ab")
# v13+ instruct (no reasoning)
MINISTRAL3 = Ref("mistralai/Ministral-3-14B-Instruct-2512", "29439f81c2be264d8d393273f99e7db9c0961120")
DEVSTRAL2 = Ref("mistralai/Devstral-Small-2-24B-Instruct-2512", "55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128")
# v11: [TOOL_CALLS]name[CALL_ID]id[ARGS]{json} in history; the generation grammar has no [CALL_ID]
SMALL32 = Ref("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "95a6d26c4bfb886c58daf9d3f7332c857cb27b43")
# v3 (SentencePiece): [TOOL_CALLS] [{"name": ..., "arguments": {...}}]. The repo ships tokenizer.json with
# the same ids; fixtures pin mode "hf" because the HF tokenizer decodes SentencePiece spaces as text.
MISTRAL7B_V03 = Ref("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", mode="hf")

GEN_PROMPT = "[/INST]"
CONTROL_MARKERS = ("[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "[THINK]", "[/THINK]")
EOS = "</s>"


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
    "create_user": _fn(
        "create_user",
        "Create a user account.",
        {"name": {"type": "string"}, "arguments": {"type": "string"}, "id": {"type": "string"}},
        ["name"],
    ),
}


def tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


@cache
def mistral_tokenizer(ref: Ref, finetuning: bool = False) -> Any:
    """mistral-common tokenizer. ``finetuning=True`` allows a conversation that ends with an assistant
    turn (the serving validator requires the last message to be user/tool)."""
    from mistral_common.protocol.instruct.validator import ValidationMode
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    mode = ValidationMode.finetuning if finetuning else ValidationMode.test
    return MistralTokenizer.from_file(str(tokenizer_file(ref)), mode=mode)


@cache
def tokenizer_file(ref: Ref) -> Path:
    """The file that defines the format: tekken.json, or tokenizer.model.v3 for SentencePiece v3.

    Downloaded by name at the pinned revision (no repo listing, which keeps HF API calls low)."""
    from huggingface_hub import hf_hub_download

    fname = "tokenizer.model.v3" if ref.repo == "mistralai/Mistral-7B-Instruct-v0.3" else "tekken.json"
    return Path(hf_hub_download(ref.repo, fname, revision=ref.revision))


@cache
def hf_tokenizer(ref: Ref) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)


def decode(ref: Ref, ids: list[int]) -> str:
    """raw_output: ids decoded with special tokens kept, by the tokenizer the pin names."""
    if ref.mode == "hf":
        return str(hf_tokenizer(ref).decode(ids, skip_special_tokens=False))
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

    raw = mistral_tokenizer(ref).instruct_tokenizer.tokenizer
    return str(raw.decode(ids, special_token_policy=SpecialTokenPolicy.KEEP))


def encode_with_markers(ref: Ref, text: str) -> list[int]:
    """Token ids for an engine-test string: control markers become their control-token ids and the text
    between them is encoded with the base tokenizer. This mirrors vLLM's ``encode_mistral_output`` /
    ``_encode_v13`` test helpers (real generation emits the markers as single control tokens)."""
    if ref.mode == "hf":
        tok = hf_tokenizer(ref)
        ids = list(tok.encode(text, add_special_tokens=False))
    else:
        raw = mistral_tokenizer(ref).instruct_tokenizer.tokenizer
        ids = []
        for part in re.split("(" + "|".join(map(re.escape, CONTROL_MARKERS)) + ")", text):
            if part in CONTROL_MARKERS:
                ids.append(raw.get_special_token(part))
            elif part:
                ids.extend(raw.encode(part, bos=False, eos=False))
    assert decode(ref, ids) == text, (decode(ref, ids), text)
    return ids


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
    ref: Ref,
    output_token_ids: list[int],
    tags: list[str],
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
        "output_token_ids": output_token_ids,
        "tokenizer": ref.pin,
        "generation_prompt": GEN_PROMPT,
    }
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
