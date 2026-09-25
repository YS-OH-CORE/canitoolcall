"""Core tests for the adapter registry and the worker protocol (fake adapter)."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from canitoolcall.adapters import ENGINES, adapter_class, engine_python
from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.adapters.worker import serve
from canitoolcall.fixtures import load_families, read_jsonl
from canitoolcall.results import ParseResult


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_registry_imports_without_engine(engine: str) -> None:
    cls = adapter_class(engine)
    assert cls.name == engine
    assert cls.pinned_version


def test_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        adapter_class("tgi")


def test_engine_python_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CANITOOLCALL_VLLM_PYTHON", "/opt/py")
    assert engine_python("vllm", tmp_path) == Path("/opt/py")


class EchoAdapter(Adapter):
    """Test double: 'parses' by echoing the decoded text as content."""

    name: ClassVar[str] = "echo"
    pinned_version: ClassVar[str] = "0"

    def version(self) -> str:
        return "0"

    def supports(self, family: str, model: str) -> Support:
        return Support(family == "qwen3-hermes", None if family == "qwen3-hermes" else "no parser")

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        return {"model": raw.model}

    def units(self, raw: ReplayInput) -> list[int]:
        assert raw.token_ids is not None
        return list(raw.token_ids)

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        return ParseResult(content=raw.text)

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        assert [i for c in chunks for i in c] == list(raw.token_ids or ())
        return ParseResult(content=f"{len(chunks)} chunks")


def test_worker_protocol(sample_fixtures_dir: Path) -> None:
    ((_, rec),) = list(read_jsonl(sample_fixtures_dir / "qwen3-hermes" / "sample.jsonl"))
    fam = load_families(sample_fixtures_dir)["qwen3-hermes"].to_dict()
    other = {**rec, "id": "glm/x", "family": "glm"}
    msgs = [
        {"op": "hello"},
        {"op": "replay", "fixture": rec, "family": fam, "strategies": ["one", "token", "char:0"]},
        {"op": "replay", "fixture": other, "family": None, "strategies": ["one"]},
        {"op": "bogus"},
        {"op": "shutdown"},
        {"op": "hello"},  # never reached
    ]
    out = io.StringIO()
    assert serve(EchoAdapter(), io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n\nnot json\n"), out) == 0
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(replies) == 5
    hello, ok, unsup, bogus, bye = replies
    assert hello["engine"] == "echo" and hello["protocol"] == 1
    assert ok["supported"] and ok["nonstream"]["content"] == rec["raw_output"]
    assert set(ok["streams"]) == {"one", "token"}  # char:* is skipped by the worker
    assert ok["streams"]["one"]["content"] == "1 chunks"
    assert ok["parser_config"] == {"model": "Qwen/Qwen3-0.6B"}
    assert unsup == {"ok": True, "fixture_id": "glm/x", "supported": False, "reason": "no parser"}
    assert not bogus["ok"]
    assert bye == {"ok": True}


def test_replay_input_uses_family_defaults(sample_fixtures_dir: Path) -> None:
    from canitoolcall.fixtures import load_fixtures

    (fx,) = load_fixtures([sample_fixtures_dir])
    fam = load_families(sample_fixtures_dir)["qwen3-hermes"]
    raw = ReplayInput.from_fixture(fx, fam)
    assert raw.generation_prompt == "<|im_start|>assistant\n"
    assert raw.stop_tokens == ("<|im_end|>",)
    assert raw.token_ids == fx.output_token_ids


@pytest.mark.engine("vllm")
def test_worker_starts_in_engine_env(engine_python_for: Path) -> None:
    """The stdlib-only core imports inside the engine's own interpreter."""
    import subprocess

    from canitoolcall.fixtures import repo_root

    root = repo_root()
    assert root is not None
    proc = subprocess.run(
        [str(engine_python_for), "-c", "import canitoolcall.adapters.worker, canitoolcall.chunking"],
        env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
