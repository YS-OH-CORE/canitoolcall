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
        (lambda r: r["tools"][0]["function"].update(parameters={"type": "no-such-type"}), "not a valid JSON Schema"),
        (lambda r: r.update(models=["Qwen/Qwen3-8B"]), "is not a reference_model"),
        (lambda r: r.update(raw_output=r["raw_output"] + "<|im_end|>"), "before the stop token"),
        (lambda r: r.update(spec_version="1.0"), "spec_version"),
        # A fixture must not point adapters at an arbitrary Hub repo (code execution via remote code).
        (lambda r: r["tokenizer"].update(repo="attacker/looks-like-qwen"), "nor its mirror"),
        (lambda r: r["tokenizer"].update(revision="main"), "does not match"),
        (lambda r: r["tokenizer"].update(revision="0" * 40), "nor its mirror"),
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


def test_validate_broken_family_json(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    root = _copy_family(tmp_path, sample_fixtures_dir, lambda r: None)
    (root / "qwen3-hermes" / "family.json").write_text("{not json", encoding="utf-8")
    assert any("invalid JSON" in str(i) for i in validate([root]))


def test_validate_family_schema_errors(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    root = _copy_family(tmp_path, sample_fixtures_dir, lambda r: None)
    fam_path = root / "qwen3-hermes" / "family.json"
    fam = json.loads(fam_path.read_text(encoding="utf-8"))
    del fam["markers"]
    fam_path.write_text(json.dumps(fam), encoding="utf-8")
    issues = [str(i) for i in validate([root])]
    assert any("markers" in i for i in issues), issues


def test_validate_invalid_jsonl(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    root = _copy_family(tmp_path, sample_fixtures_dir, lambda r: None)
    (root / "qwen3-hermes" / "bad.jsonl").write_text("{oops\n", encoding="utf-8")
    assert any("invalid JSON" in str(i) for i in validate([root]))


def test_load_reports_malformed_records(tmp_path: Path) -> None:
    d = tmp_path / "fam"
    d.mkdir()
    (d / "x.jsonl").write_text('{"id": "fam/x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"x\.jsonl:1: malformed fixture"):
        load_fixtures([tmp_path])
    (d / "x.jsonl").write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"x\.jsonl:1: invalid JSON"):
        load_fixtures([tmp_path])


def test_validate_rejects_branch_revisions_in_family(tmp_path: Path, sample_fixtures_dir: Path) -> None:
    root = _copy_family(tmp_path, sample_fixtures_dir, lambda r: None)
    fam_file = root / "qwen3-hermes" / "family.json"
    fam = json.loads(fam_file.read_text(encoding="utf-8"))
    fam["reference_models"][0]["revision"] = "main"
    fam_file.write_text(json.dumps(fam), encoding="utf-8")
    assert any("revision" in str(i) and "does not match" in str(i) for i in validate([root]))


def test_remote_code_only_for_reviewed_pins() -> None:
    from canitoolcall.adapters.base import REMOTE_CODE_ALLOWLIST, trusts_remote_code

    repo, rev = next(iter(REMOTE_CODE_ALLOWLIST))
    assert trusts_remote_code(repo, rev)
    assert not trusts_remote_code(repo, "main")
    assert not trusts_remote_code("attacker/looks-like-kimi", rev)
    assert not trusts_remote_code(repo, None)


def test_missing_relative_imports(tmp_path: Path) -> None:
    """Sibling modules an auto_map module imports relatively must be fetched too.

    Kimi-K2.6 ``tokenization_kimi.py`` imports ``.tool_declaration_ts`` and Kimi-K3's
    imports ``.encoding_k3`` (inside ``try:``); transformers loads both, so a prefetch
    that skips them breaks offline replays.
    """
    from canitoolcall.adapters.base import missing_relative_imports

    (tmp_path / "tokenization_kimi.py").write_text(
        "import os\n"
        "try:\n    from .encoding_k3 import build_chat_segments\n"
        "except ImportError:\n    from encoding_k3 import build_chat_segments\n"
        "from .tool_declaration_ts import encode\n"
    )
    (tmp_path / "configuration_kimi_k25.py").write_text("from .configuration_deepseek import DeepseekV3Config\n")
    (tmp_path / "configuration_deepseek.py").write_text("from transformers import PretrainedConfig\n")
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding" / "encode.py").write_text("from .pkg.util import x\nimport json\n")
    assert missing_relative_imports(tmp_path) == ["encoding/pkg/util.py", "encoding_k3.py", "tool_declaration_ts.py"]
    (tmp_path / "encoding_k3.py").write_text("from __future__ import annotations\n")
    assert missing_relative_imports(tmp_path) == ["encoding/pkg/util.py", "tool_declaration_ts.py"]
