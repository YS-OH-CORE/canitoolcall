"""Test-only corpora derived from the rendered Qwen3 sample (data/fixtures).

Nothing here is typed by hand: every raw output is a slice of the sample's
template-rendered ``raw_output``, and every expected parse is a subset of the
sample's expected parse. These records exist only in temporary directories
created by the tests and are never part of the fixture corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canitoolcall.fixtures import read_jsonl

HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "data" / "fixtures" / "qwen3-hermes"
SAMPLE_ID = "qwen3-hermes/sample-parallel-reasoning"

TRUNCATED_ID = "qwen3-hermes/derived-truncated"
SINGLE_ID = "qwen3-hermes/derived-single-call"
OTHER_ID = "other/derived-unsupported"


def sample_record() -> dict[str, Any]:
    ((_, rec),) = list(read_jsonl(SAMPLE_DIR / "sample.jsonl"))
    return rec


def sample_family() -> dict[str, Any]:
    return json.loads((SAMPLE_DIR / "family.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _derived(rec: dict[str, Any], new_id: str, raw: str, note: str) -> dict[str, Any]:
    out = {k: v for k, v in rec.items() if k not in ("output_token_ids", "tokenizer", "expected")}
    out["id"] = new_id
    out["family"] = new_id.split("/", 1)[0]
    out["raw_output"] = raw
    out["provenance"] = {**rec["provenance"], "generator": "tests/core/corpus.py"}
    out["notes"] = f"TEST-ONLY, derived from {SAMPLE_ID}: {note}"
    return out


def truncated_record() -> dict[str, Any]:
    """The sample cut inside the first call's arguments (graceful failure expected)."""
    rec = sample_record()
    raw = rec["raw_output"]
    out = _derived(rec, TRUNCATED_ID, raw[: raw.index('"unit"')], "cut inside the first call's arguments")
    out["expected_error"] = {"reason": "generation truncated inside the first call", "accept": ["no_tool_calls"]}
    out["tags"] = ["truncated", "reasoning"]
    return out


def single_call_record() -> dict[str, Any]:
    """The sample cut after the first call (a complete single-call output)."""
    rec = sample_record()
    raw = rec["raw_output"]
    cut = raw.index("</tool_call>") + len("</tool_call>")
    out = _derived(rec, SINGLE_ID, raw[:cut], "cut after the first complete call")
    exp = rec["expected"]
    out["expected"] = {**exp, "tool_calls": exp["tool_calls"][:1]}
    out["tags"] = ["single-call", "reasoning", "unicode"]
    return out


def other_family_record() -> dict[str, Any]:
    """The sample relabelled into a family the reference adapter does not support."""
    rec = sample_record()
    out = _derived(rec, OTHER_ID, rec["raw_output"], "relabelled to exercise 'unsupported'")
    out["expected"] = rec["expected"]
    return out


def _write(dirpath: Path, family: dict[str, Any], records: list[dict[str, Any]], name: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "family.json").write_text(json.dumps(family, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (dirpath / name).open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_corpus(root: Path, *, other_family: bool = True) -> Path:
    """Write sample + derived fixtures under ``root``; returns ``root``."""
    fam = sample_family()
    _write(root / "qwen3-hermes", fam, [sample_record()], "sample.jsonl")
    with (root / "qwen3-hermes" / "derived.jsonl").open("w", encoding="utf-8") as fh:
        for r in (single_call_record(), truncated_record()):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if other_family:
        _write(
            root / "other",
            {**fam, "slug": "other", "name": "Test-only relabelled family"},
            [other_family_record()],
            "x.jsonl",
        )
    return root
