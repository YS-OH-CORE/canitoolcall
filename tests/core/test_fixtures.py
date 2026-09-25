from __future__ import annotations

import json
from pathlib import Path

import pytest

from canitoolcall.fixtures import (
    Fixture,
    fixtures_digest,
    load_families,
    load_fixtures,
    read_jsonl,
    validate,
)


def test_sample_is_valid(sample_fixtures_dir: Path) -> None:
    assert validate([sample_fixtures_dir]) == []


def test_load_sample(sample_fixtures_dir: Path) -> None:
    (fx,) = load_fixtures([sample_fixtures_dir])
    assert fx.id == "qwen3-hermes/sample-parallel-reasoning"
    assert fx.reference_model == "Qwen/Qwen3-0.6B"
    assert fx.provenance.kind == "template_render"
    assert fx.expected is not None
    assert [c.name for c in fx.expected.tool_calls] == ["get_weather", "search"]
    assert fx.output_token_ids and fx.tokenizer is not None
    assert fx.line == 1 and fx.source is not None


def test_roundtrip_is_lossless(sample_fixtures_dir: Path) -> None:
    path = sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"
    ((_, record),) = list(read_jsonl(path))
    assert Fixture.from_dict(record).to_dict() == record


def test_filters(sample_fixtures_dir: Path) -> None:
    assert load_fixtures([sample_fixtures_dir], families=["glm"]) == []
    assert len(load_fixtures([sample_fixtures_dir], tags=["parallel-calls", "x-none"])) == 1
    assert load_fixtures([sample_fixtures_dir], tags=["truncated"]) == []


def test_families(sample_fixtures_dir: Path) -> None:
    fams = load_families(sample_fixtures_dir)
    fam = fams["qwen3-hermes"]
    ref = fam.reference_for("Qwen/Qwen3-0.6B")
    assert ref is not None and ref.stop_tokens == ("<|im_end|>",)
    assert fam.reference_for("nope/nope") is None


def test_digest_is_order_independent(sample_fixtures_dir: Path) -> None:
    fxs = load_fixtures([sample_fixtures_dir])
    assert fixtures_digest(fxs) == fixtures_digest(list(reversed(fxs)))
    assert len(fixtures_digest(fxs)) == 64


def _copy_family(tmp_path: Path, sample_fixtures_dir: Path, mutate) -> Path:  # type: ignore[no-untyped-def]
    fam_dir = tmp_path / "qwen3-hermes"
    fam_dir.mkdir()
    (fam_dir / "family.json").write_bytes((sample_fixtures_dir / "qwen3-hermes" / "family.json").read_bytes())
    ((_, rec),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    mutate(rec)
    (fam_dir / "x.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda r: r.pop("provenance"), "provenance"),
        (lambda r: r["provenance"].pop("generator"), "generator"),
        (lambda r: r.update(id="glm/x"), "must start with"),
        (lambda r: r.pop("tokenizer"), "tokenizer pin"),
        (lambda r: r["expected"]["tool_calls"].append({"name": "nope", "arguments": {}}), "not an offered tool"),
        (lambda r: r.update(expected_error={"reason": "x", "accept": ["exception"]}), "expected"),
        (lambda r: r.update(tags=["not-a-tag"]), "tags"),
        (lambda r: r["provenance"].update(source_url="file:///etc/passwd"), "source_url"),
    ],
)
def test_validate_rejects(tmp_path: Path, sample_fixtures_dir: Path, mutate, needle: str) -> None:  # type: ignore[no-untyped-def]
    root = _copy_family(tmp_path, sample_fixtures_dir, mutate)
    issues = validate([root])
    assert issues, "expected a validation issue"
    assert any(needle in str(i) for i in issues), [str(i) for i in issues]


def test_validate_duplicate_ids(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    root = _copy_family(tmp_path, sample_fixtures_dir, lambda r: None)
    (root / "qwen3-hermes" / "y.jsonl").write_bytes((root / "qwen3-hermes" / "x.jsonl").read_bytes())
    assert any("duplicate id" in str(i) for i in validate([root]))


def test_validate_missing_family(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    root = _copy_family(tmp_path, sample_fixtures_dir, lambda r: None)
    (root / "qwen3-hermes" / "family.json").unlink()
    assert any("missing family.json" in str(i) for i in validate([root]))
