from __future__ import annotations

from pathlib import Path

import pytest

from canitoolcall.cli import EXIT_ERROR, EXIT_FAILURES, EXIT_OK, main


def test_validate_ok(sample_fixtures_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(sample_fixtures_dir)]) == EXIT_OK
    assert "1 file(s) checked, 0 issue(s)" in capsys.readouterr().out


def test_validate_bad(tmp_path: Path) -> None:
    d = tmp_path / "fam"
    d.mkdir()
    (d / "x.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    assert main(["validate", str(tmp_path)]) == EXIT_FAILURES


def test_engines_lists_all(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["engines"]) == EXIT_OK
    out = capsys.readouterr().out
    for engine in ("vllm", "sglang", "llamacpp", "ollama", "transformers"):
        assert engine in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("canitoolcall ")


def test_stub_commands_fail_cleanly(tmp_path: Path) -> None:
    # Until the matrix builder lands, the command reports "not implemented" instead of crashing.
    assert main(["matrix", "--results", str(tmp_path), "--out", str(tmp_path / "o")]) in (EXIT_OK, EXIT_ERROR)
