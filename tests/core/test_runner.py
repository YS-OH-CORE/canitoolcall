"""End-to-end runner tests: real worker subprocesses running the reference adapter.

The worker interpreter is the dev interpreter (``sys.executable``); the
reference adapter (tests/core/reference_adapter.py) is loaded through its
``module:Class`` spec, so these tests exercise the same process boundary,
protocol, restart logic and checks that engine runs use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from core_corpus import OTHER_ID, SAMPLE_ID, SINGLE_ID, TRUNCATED_ID, write_corpus

from canitoolcall.chunking import DEFAULT_STRATEGIES, ChunkStrategy, parse_strategies
from canitoolcall.fixtures import load_fixtures, spec_dir
from canitoolcall.results import RunResults, Status
from canitoolcall.runner import (
    FixtureValidationError,
    RunConfig,
    WorkerClient,
    WorkerError,
    core_pythonpath,
    evaluate,
    family_for,
    run,
    run_and_write,
    select_fixtures,
    worker_env,
)

PY = Path(sys.executable)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "fixtures")


def config(corpus: Path, spec: str, env: dict[str, str], **kw: object) -> RunConfig:
    return RunConfig(engine=spec, fixtures=(corpus,), python=PY, env=env, **kw)  # type: ignore[arg-type]


def schema_validate(results: RunResults) -> None:
    import jsonschema

    schema = json.loads((spec_dir() / "results.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(results.to_dict())


# --------------------------------------------------------------------------- full runs


def test_run_reference_adapter(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    results = run(config(corpus, reference_adapter_spec, reference_env))
    schema_validate(results)

    assert results.engine.name == "reference"
    assert results.engine.version == "1.0"
    assert results.engine.details == {"parser": "reference-hermes", "pinned_version": "1.0", "tokens_per_step": "many"}
    assert results.run.strategies == tuple(s.id for s in DEFAULT_STRATEGIES)
    assert results.run.normalization == "soft-v1"
    assert results.run.python == ".".join(map(str, sys.version_info[:3]))
    assert results.run.fixtures_digest and len(results.run.fixtures_digest) == 64
    assert results.run.started_at <= results.run.finished_at

    cases = {c.fixture_id: c for c in results.cases}
    assert [c.fixture_id for c in results.cases] == [OTHER_ID, SINGLE_ID, TRUNCATED_ID, SAMPLE_ID]
    assert cases[OTHER_ID].status is Status.UNSUPPORTED
    assert cases[OTHER_ID].reason == "reference adapter implements only qwen3-hermes"
    assert cases[OTHER_ID].checks == ()
    for fid in (SAMPLE_ID, SINGLE_ID, TRUNCATED_ID):
        case = cases[fid]
        assert case.status is Status.PASS, [c for c in case.checks if c.status is not Status.PASS]
        assert case.parser_config == {"parser": "reference-hermes", "starts_in_reasoning": False}
        assert case.observed is None  # include_observed="failures"
        assert case.skipped_strategies == {}
        strategies = {c.strategy for c in case.checks}
        assert strategies >= {"nonstream", *(s.id for s in DEFAULT_STRATEGIES)}

    sample_checks = {c.check for c in cases[SAMPLE_ID].checks}
    assert sample_checks == {
        "expected_match",
        "stream_equals_nonstream",
        "split_invariance",
        "no_leakage",
        "arguments_json",
        "arguments_schema",
        "parallel_order",
    }
    assert {c.check for c in cases[TRUNCATED_ID].checks} == {
        "expected_error",
        "stream_equals_nonstream",
        "split_invariance",
        "no_leakage",
    }
    assert results.summary()["totals"] == {"pass": 3, "soft_pass": 0, "fail": 0, "error": 0, "unsupported": 1}


def test_run_and_write(
    tmp_path: Path, corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    out = tmp_path / "results"
    path = run_and_write(config(corpus, reference_adapter_spec, reference_env, out_dir=out, include_observed="all"))
    assert path == out / "reference-1.0.json"
    loaded = RunResults.load(path)
    assert all(c.observed is not None for c in loaded.cases if c.status is not Status.UNSUPPORTED)
    assert [p.name for p in out.iterdir()] == ["reference-1.0.json"]  # no temp files left behind


def test_parallel_jobs_give_identical_cases(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    serial = run(config(corpus, reference_adapter_spec, reference_env, include_observed="all"))
    parallel = run(config(corpus, reference_adapter_spec, reference_env, include_observed="all", jobs=3))
    assert parallel.cases == serial.cases
    assert parallel.run.fixtures_digest == serial.run.fixtures_digest


def test_filters(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    by_family = run(config(corpus, reference_adapter_spec, reference_env, families=("other",)))
    assert [c.fixture_id for c in by_family.cases] == [OTHER_ID]
    by_tag = run(config(corpus, reference_adapter_spec, reference_env, tags=("truncated",)))
    assert [c.fixture_id for c in by_tag.cases] == [TRUNCATED_ID]
    by_id = run(config(corpus, reference_adapter_spec, reference_env, ids=("qwen3-hermes/derived-*",)))
    assert [c.fixture_id for c in by_id.cases] == [SINGLE_ID, TRUNCATED_ID]


def test_nothing_selected(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="no fixtures selected"):
        run(config(corpus, reference_adapter_spec, reference_env, families=("nope",)))


def test_invalid_fixtures_are_rejected(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    bad = corpus / "qwen3-hermes" / "derived.jsonl"
    lines = bad.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["tags"] = ["not-a-tag"]
    bad.write_text("\n".join([json.dumps(rec), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(FixtureValidationError) as info:
        run(config(corpus, reference_adapter_spec, reference_env))
    assert "tags" in str(info.value)
    # validate=False runs anyway (the record still loads).
    assert run(config(corpus, reference_adapter_spec, reference_env, validate=False)).cases


# --------------------------------------------------------------------------- strategies


def test_special_skipped_when_adapter_has_no_special_ids(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": f"no-special@{SAMPLE_ID}"}
    results = run(config(corpus, reference_adapter_spec, env, families=("qwen3-hermes",)))
    cases = {c.fixture_id: c for c in results.cases}
    assert cases[SAMPLE_ID].skipped_strategies == {"special": "adapter does not expose the engine's special token ids"}
    assert "special" not in {c.strategy for c in cases[SAMPLE_ID].checks}
    assert cases[SAMPLE_ID].status is Status.PASS
    assert cases[SINGLE_ID].skipped_strategies == {}
    schema_validate(results)


def test_synthetic_char_strategy_never_counts(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    strategies = parse_strategies(["one", "token", "char:0", "char:3"])
    results = run(config(corpus, reference_adapter_spec, reference_env, strategies=strategies, ids=(SAMPLE_ID,)))
    (case,) = results.cases
    assert case.status is Status.PASS
    assert {"char:0", "char:3"} <= {c.strategy for c in case.checks}
    assert results.run.strategies == ("one", "token", "char:0", "char:3")


# --------------------------------------------------------------------------- injected engine faults


def test_leak_is_caught(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": f"leak@{SAMPLE_ID}"}
    results = run(config(corpus, reference_adapter_spec, env, ids=(SAMPLE_ID,)))
    (case,) = results.cases
    assert case.status is Status.FAIL
    failed = {c.check for c in case.checks if c.status is Status.FAIL}
    assert failed == {"expected_match", "no_leakage"}  # consistent across strategies
    leak_rows = [c for c in case.checks if c.check == "no_leakage"]
    assert all(c.detail == "reasoning_content contains '</think>'" for c in leak_rows)
    assert case.observed is not None  # kept for failures
    assert case.observed.nonstream.reasoning_content == "I should call the tools.\n</think>"


def test_whole_output_in_one_delta_bug_is_caught(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": "drop-one"}
    results = run(config(corpus, reference_adapter_spec, env, ids=(SAMPLE_ID,)))
    (case,) = results.cases
    assert case.status is Status.FAIL
    failing = {(c.check, c.strategy) for c in case.checks if c.status is Status.FAIL}
    assert failing == {
        ("expected_match", "one"),
        ("stream_equals_nonstream", "one"),
        ("split_invariance", "*"),
        ("parallel_order", "one"),
    }


def test_engine_exception_is_an_outcome(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": "raise"}
    results = run(config(corpus, reference_adapter_spec, env, families=("qwen3-hermes",)))
    cases = {c.fixture_id: c for c in results.cases}
    assert cases[SAMPLE_ID].status is Status.FAIL  # expected a parse, got an exception
    assert cases[SAMPLE_ID].harness_error is None
    # The truncated fixture only accepts no_tool_calls, so an exception fails it too.
    assert cases[TRUNCATED_ID].status is Status.FAIL
    # Streams raise the same way as non-streaming: no stream-vs-nonstream failure.
    assert all(c.status is Status.PASS for c in cases[SAMPLE_ID].checks if c.check == "stream_equals_nonstream")


def test_harness_failure_is_error_and_run_continues(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": f"harness@{SINGLE_ID}"}
    results = run(config(corpus, reference_adapter_spec, env))
    cases = {c.fixture_id: c for c in results.cases}
    assert cases[SINGLE_ID].status is Status.ERROR
    assert "injected harness failure" in (cases[SINGLE_ID].harness_error or "")
    assert cases[SAMPLE_ID].status is Status.PASS
    assert cases[TRUNCATED_ID].status is Status.PASS
    schema_validate(results)


def test_engine_stdout_noise_does_not_corrupt_protocol(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": "noisy"}
    results = run(config(corpus, reference_adapter_spec, env, families=("qwen3-hermes",)))
    assert all(c.status is Status.PASS for c in results.cases)


def test_crash_restarts_worker(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": f"crash@{SINGLE_ID}"}
    results = run(config(corpus, reference_adapter_spec, env))
    cases = {c.fixture_id: c for c in results.cases}
    assert cases[SINGLE_ID].status is Status.ERROR
    assert "exited unexpectedly" in (cases[SINGLE_ID].harness_error or "")
    assert "exit code 3" in (cases[SINGLE_ID].harness_error or "")
    # Fixtures after the crash ran in a fresh worker.
    assert cases[TRUNCATED_ID].status is Status.PASS
    assert cases[SAMPLE_ID].status is Status.PASS


def test_hang_times_out_and_restarts(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": f"hang@{SINGLE_ID}"}
    results = run(config(corpus, reference_adapter_spec, env, timeout_s=2.0, families=("qwen3-hermes",)))
    cases = {c.fixture_id: c for c in results.cases}
    assert cases[SINGLE_ID].status is Status.ERROR
    assert "timed out after 2s" in (cases[SINGLE_ID].harness_error or "")
    assert cases[TRUNCATED_ID].status is Status.PASS
    assert cases[SAMPLE_ID].status is Status.PASS


# --------------------------------------------------------------------------- WorkerClient


def test_worker_client_hello_and_replay(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    fixtures = load_fixtures([corpus], families=["qwen3-hermes"], tags=["parallel-calls"])
    (fx,) = fixtures
    fam = family_for(fx)
    assert fam is not None and fam.slug == "qwen3-hermes"
    with WorkerClient(reference_adapter_spec, PY, reference_env, timeout_s=30) as w:
        hello = w.hello()
        assert hello["engine"] == "reference" and hello["protocol"] == 1
        assert hello["pinned_version"] == "1.0"
        reply = w.replay(fx, fam, [ChunkStrategy("one"), ChunkStrategy("special"), ChunkStrategy("char", 0)])
        assert set(reply["streams"]) == {"one", "special", "char:0"}
        assert reply["skipped"] == {}
        case = evaluate(fx, fam, reply)
        assert case.status is Status.PASS
    assert not w.alive


def test_worker_client_missing_interpreter(tmp_path: Path) -> None:
    w = WorkerClient("vllm", tmp_path / "nope" / "python")
    with pytest.raises(WorkerError, match="interpreter not found"):
        w.hello()


def test_worker_client_unknown_engine(reference_env: dict[str, str]) -> None:
    with (
        WorkerClient("no_such_module:Adapter", PY, reference_env, timeout_s=30) as w,
        pytest.raises(WorkerError) as info,
    ):
        w.hello()
    assert "ModuleNotFoundError" in str(info.value) or "exited" in str(info.value)


def test_worker_client_reports_ok_false(reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    with WorkerClient(reference_adapter_spec, PY, reference_env, timeout_s=30) as w:
        with pytest.raises(WorkerError, match="unknown op") as info:
            w.request({"op": "bogus"})
        assert not info.value.fatal
        assert w.alive  # a protocol-level error does not kill the worker
        assert w.hello()["engine"] == "reference"


def test_run_fails_cleanly_when_hello_fails(corpus: Path, reference_env: dict[str, str]) -> None:
    with pytest.raises(WorkerError):
        run(config(corpus, "no_such_module:Adapter", reference_env, startup_timeout_s=30))


# --------------------------------------------------------------------------- evaluate / helpers


def test_evaluate_malformed_reply(corpus: Path) -> None:
    (fx,) = load_fixtures([corpus], tags=["parallel-calls"], families=["qwen3-hermes"])
    with pytest.raises(WorkerError, match="malformed"):
        evaluate(fx, None, {"ok": True, "supported": True, "streams": {}})


def test_evaluate_include_observed_modes(corpus: Path) -> None:
    (fx,) = load_fixtures([corpus], tags=["parallel-calls"], families=["qwen3-hermes"])
    empty = {"content": None, "reasoning_content": None, "tool_calls": [], "exception": None}
    reply = {"ok": True, "supported": True, "nonstream": empty, "streams": {"one": empty}, "parser_config": {}}
    failing = evaluate(fx, None, reply, "failures")
    assert failing.status is Status.FAIL and failing.observed is not None
    assert evaluate(fx, None, reply, "none").observed is None
    assert evaluate(fx, None, reply, "all").observed is not None


def test_select_fixtures_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        select_fixtures(RunConfig(engine="vllm", fixtures=(tmp_path / "missing",)))


def test_family_for_without_source(corpus: Path) -> None:
    (fx,) = load_fixtures([corpus], families=["other"])
    assert family_for(fx) is not None
    from dataclasses import replace

    assert family_for(replace(fx, source=None)) is None


def test_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
    monkeypatch.setenv("VIRTUAL_ENV", "/main/venv")
    env = worker_env({"PYTHONPATH": "/extra", "HF_HUB_OFFLINE": "1"})
    assert env["PYTHONPATH"].split(":") == [str(core_pythonpath()), "/extra"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "VIRTUAL_ENV" not in env
    assert (core_pythonpath() / "canitoolcall" / "__init__.py").is_file()


def test_duplicate_strategies_run_once(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]
) -> None:
    strategies = parse_strategies(["one", "token", "one"])
    results = run(config(corpus, reference_adapter_spec, reference_env, strategies=strategies, ids=(SAMPLE_ID,)))
    assert results.run.strategies == ("one", "token")
    rows = [c for c in results.cases[0].checks if c.check == "stream_equals_nonstream"]
    assert [c.strategy for c in rows] == ["one", "token"]


def test_bad_config(corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="jobs"):
        run(config(corpus, reference_adapter_spec, reference_env, jobs=0))
    with pytest.raises(ValueError, match="strategy"):
        run(config(corpus, reference_adapter_spec, reference_env, strategies=()))


def test_failures_only_under_multi_token_chunks_do_not_count_for_one_token_engines(
    corpus: Path, reference_env: dict[str, str]
) -> None:
    """Ollama-like engine (one event per token): a stream that breaks only when a delta holds the
    whole output is reported, but the case does not fail."""
    env = {**reference_env, "REFERENCE_ADAPTER_BUG": "drop-one"}
    results = run(config(corpus, "reference_adapter:OneTokenReferenceAdapter", env, ids=(SAMPLE_ID,)))
    schema_validate(results)
    assert results.engine.details["tokens_per_step"] == "one"
    assert set(results.run.synthetic_strategies) >= {"one", "special", "rand:1:8"}
    assert "token" not in results.run.synthetic_strategies
    (case,) = results.cases
    assert case.status is Status.PASS
    assert any(c.strategy == "one" and c.status is Status.FAIL for c in case.checks)  # still reported
    # Loading the results back gives the same per-run synthetic set (the matrix uses it).
    assert RunResults.from_dict(results.to_dict()).run.synthetic_strategies == results.run.synthetic_strategies


def test_checks_crashing_on_a_reply_is_a_harness_error(
    corpus: Path, reference_adapter_spec: str, reference_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import canitoolcall.checks

    def boom(*args: object) -> list[object]:
        raise TypeError("unexpected engine output")

    monkeypatch.setattr(canitoolcall.checks, "run_checks", boom)
    results = run(config(corpus, reference_adapter_spec, reference_env, ids=(SAMPLE_ID,)))
    (case,) = results.cases
    assert case.status is Status.ERROR and "unexpected engine output" in (case.harness_error or "")
