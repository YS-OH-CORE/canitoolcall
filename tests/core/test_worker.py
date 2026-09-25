"""Worker protocol tests: strategy realization, skip reasons, process-level behaviour."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Collection, Sequence
from typing import Any, ClassVar

from core_corpus import SAMPLE_ID, sample_family, sample_record
from reference_adapter import MARKERS, ReferenceAdapter, decode

from canitoolcall.adapters import adapter_class, is_adapter_spec
from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.adapters.worker import handle, replay
from canitoolcall.chunking import ChunkStrategy, parse_strategies
from canitoolcall.fixtures import Family, Fixture
from canitoolcall.results import ParseResult
from canitoolcall.runner import worker_env


class RecordingAdapter(Adapter):
    """Records the chunk groups it is given; exposes special ids 1 and 2."""

    name: ClassVar[str] = "recording"
    pinned_version: ClassVar[str] = "0"

    def __init__(self, special: Collection[int] | None = (1, 2)) -> None:
        self.special = special
        self.calls: list[list[list[int]]] = []

    def version(self) -> str:
        return "0"

    def supports(self, family: str, model: str) -> Support:
        return Support(True)

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        return {}

    def units(self, raw: ReplayInput) -> list[int]:
        return [10, 1, 11, 12, 2, 13]

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        return self.special

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        return ParseResult(content="x")

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        self.calls.append([list(c) for c in chunks])
        return ParseResult(content="x")


def _fixture() -> tuple[Fixture, Family]:
    return Fixture.from_dict(sample_record()), Family.from_dict(sample_family())


def test_special_strategy_groups_at_special_ids() -> None:
    fx, fam = _fixture()
    a = RecordingAdapter()
    reply = replay(a, fx, fam, parse_strategies(["special", "one", "special"]))
    assert list(reply["streams"]) == ["special", "one"]  # duplicates run once
    assert a.calls == [[[10], [1], [11, 12], [2], [13]], [[10, 1, 11, 12, 2, 13]]]
    assert reply["skipped"] == {}


def test_skip_reasons() -> None:
    fx, fam = _fixture()
    a = RecordingAdapter(special=None)
    reply = replay(a, fx, fam, parse_strategies(["one", "special", "char:0"]))
    assert list(reply["streams"]) == ["one"]
    assert reply["skipped"] == {
        "special": "adapter does not expose the engine's special token ids",
        "char:0": "synthetic character split: adapter has no text-delta streaming path",
    }


def test_char_strategy_uses_text_path() -> None:
    fx, fam = _fixture()
    reply = replay(ReferenceAdapter(), fx, fam, [ChunkStrategy("char", 0), ChunkStrategy("char", 5)])
    assert set(reply["streams"]) == {"char:0", "char:5"}
    assert reply["streams"]["char:0"] == reply["nonstream"]


def test_reference_adapter_toy_tokenizer_roundtrip() -> None:
    fx, _ = _fixture()
    a = ReferenceAdapter()
    raw = ReplayInput.from_fixture(fx)
    ids = a.units(raw)
    assert decode(ids) == fx.raw_output
    special = a.special_token_ids(raw)
    assert special is not None
    assert sum(i in special for i in ids) == sum(fx.raw_output.count(m) for m in MARKERS) == 6


def test_handle_replay_message() -> None:
    fx, fam = _fixture()
    reply = handle(
        RecordingAdapter(), {"op": "replay", "fixture": fx.to_dict(), "family": fam.to_dict(), "strategies": ["token"]}
    )
    assert reply["ok"] and reply["fixture_id"] == SAMPLE_ID
    assert list(reply["streams"]) == ["token"]


def test_hello_reports_pins() -> None:
    reply = handle(ReferenceAdapter(), {"op": "hello"})
    assert reply["pinned_version"] == "1.0"
    assert reply["python"] == ".".join(map(str, sys.version_info[:3]))


def test_adapter_specs() -> None:
    assert is_adapter_spec("reference_adapter:ReferenceAdapter")
    assert is_adapter_spec("a.b.c:D")
    for bad in ("vllm", "a:", ":B", "a b:C", "a:B.c", "a:B:C"):
        assert not is_adapter_spec(bad), bad
    assert adapter_class("reference_adapter:ReferenceAdapter") is ReferenceAdapter


def test_adapter_spec_must_be_an_adapter() -> None:
    import pytest

    with pytest.raises(TypeError, match="not an Adapter"):
        adapter_class("core_corpus:Path")


def test_worker_process_keeps_protocol_channel_clean(reference_env: dict[str, str]) -> None:
    """Engine output on sys.stdout or straight to fd 1 must end up on stderr."""
    fx, fam = _fixture()
    msgs = [
        {"op": "hello"},
        {"op": "replay", "fixture": fx.to_dict(), "family": fam.to_dict(), "strategies": ["one", "token"]},
        {"op": "shutdown"},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "canitoolcall.adapters.worker", "reference_adapter:ReferenceAdapter"],
        input="".join(json.dumps(m, ensure_ascii=False) + "\n" for m in msgs),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**worker_env(reference_env), "REFERENCE_ADAPTER_BUG": "noisy"},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]  # every line is protocol JSON
    assert [r["ok"] for r in replies] == [True, True, True]
    assert replies[1]["nonstream"]["reasoning_content"] == "I should call the tools."
    assert "Zürich" in replies[1]["nonstream"]["tool_calls"][0]["arguments_raw"]  # UTF-8 end to end
    assert "engine chatter on sys.stdout" in proc.stderr
    assert "engine chatter on fd 1" in proc.stderr


def test_worker_usage_error() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "canitoolcall.adapters.worker"],
        capture_output=True,
        text=True,
        env=worker_env(),
        timeout=60,
    )
    assert proc.returncode == 2
    assert "usage" in proc.stderr


def test_unavailable_engine_is_one_line_not_a_traceback() -> None:
    from canitoolcall.adapters.base import AdapterUnavailable

    class Missing(ReferenceAdapter):
        def version(self) -> str:
            raise AdapterUnavailable("No module named 'vllm'")

    reply = handle(Missing(), {"op": "hello"})
    assert reply == {"ok": False, "unavailable": True, "error": "No module named 'vllm'"}


def test_hello_reports_tokens_per_step() -> None:
    assert handle(ReferenceAdapter(), {"op": "hello"})["tokens_per_step"] == "many"
