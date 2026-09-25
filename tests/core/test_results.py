from __future__ import annotations

import json
from pathlib import Path

from canitoolcall.results import (
    CaseResult,
    CheckResult,
    EngineInfo,
    Observation,
    ParsedToolCall,
    ParseResult,
    RunInfo,
    RunResults,
    Status,
    worst_status,
)


def test_worst_status_order() -> None:
    assert worst_status([]) is Status.PASS
    assert worst_status([Status.PASS, Status.SOFT_PASS]) is Status.SOFT_PASS
    assert worst_status([Status.SOFT_PASS, Status.ERROR]) is Status.ERROR
    assert worst_status([Status.ERROR, Status.FAIL, Status.PASS]) is Status.FAIL


def test_parsed_tool_call_arguments_are_strict() -> None:
    assert ParsedToolCall("f", '{"a": 1}').arguments() == {"a": 1}
    try:
        ParsedToolCall("f", "").arguments()
    except json.JSONDecodeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty arguments must not decode")


def _run() -> RunResults:
    pr = ParseResult(content=None, reasoning_content="r", tool_calls=(ParsedToolCall("f", "{}"),))
    obs = Observation(nonstream=pr, streams={"one": pr, "token": ParseResult(exception="ValueError: x")})
    cases = (
        CaseResult("fam/a", "fam", Status.PASS, (CheckResult("expected_match", "nonstream", Status.PASS),)),
        CaseResult(
            "fam/b",
            "fam",
            Status.FAIL,
            (CheckResult("stream_equals_nonstream", "token", Status.FAIL, "exception"),),
            parser_config={"tool_parser": "hermes"},
            observed=obs,
        ),
        CaseResult("other/c", "other", Status.UNSUPPORTED, reason="no parser for this model"),
        CaseResult(
            "fam/d",
            "fam",
            Status.ERROR,
            harness_error="Traceback ...",
            skipped_strategies={"special": "adapter does not expose the engine's special token ids"},
        ),
    )
    return RunResults(
        canitoolcall_version="0.1.0.dev0",
        engine=EngineInfo("vllm", "0.30.0", None, {"torch": "2.14.0"}),
        run=RunInfo(
            "2026-09-25T00:00:00Z",
            "2026-09-25T00:00:01Z",
            "darwin-arm64",
            "3.12.13",
            "0" * 64,
            ("one", "token"),
            "soft-v1",
        ),
        cases=cases,
    )


def test_summary_counts() -> None:
    s = _run().summary()
    assert s["totals"] == {"pass": 1, "soft_pass": 0, "fail": 1, "error": 1, "unsupported": 1}
    assert s["by_family"]["other"]["unsupported"] == 1


def test_write_load_roundtrip(tmp_path: Path) -> None:
    run = _run()
    path = run.write(tmp_path / run.default_filename())
    assert path.name == "vllm-0.30.0.json"
    assert RunResults.load(path) == run


def test_results_match_schema(tmp_path: Path) -> None:
    import jsonschema

    from canitoolcall.fixtures import spec_dir

    schema = json.loads((spec_dir() / "results.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(_run().to_dict())


def test_optional_case_fields_are_omitted_when_unset() -> None:
    d = CaseResult("fam/a", "fam", Status.PASS).to_dict()
    assert set(d) == {"fixture_id", "family", "status", "checks"}
    d = _run().to_dict()["cases"][2]
    assert d["reason"] == "no parser for this model"


def test_default_filename_is_filesystem_safe() -> None:
    run = _run()
    odd = RunResults(run.canitoolcall_version, EngineInfo("llamacpp", "b1234/a25c9865 dirty"), run.run, ())
    assert odd.default_filename() == "llamacpp-b1234_a25c9865_dirty.json"


def test_write_is_atomic_and_overwrites(tmp_path: Path) -> None:
    run = _run()
    path = tmp_path / "deep" / "out.json"
    run.write(path)
    run.write(path)
    assert [p.name for p in path.parent.iterdir()] == ["out.json"]
    assert RunResults.load(path) == run


def test_load_rejects_unknown_major(tmp_path: Path) -> None:
    import pytest

    d = _run().to_dict()
    d["schema_version"] = "9.0"
    path = tmp_path / "x.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        RunResults.load(path)


def test_results_file_is_readable_by_others(tmp_path: Path) -> None:
    import os

    mask = os.umask(0o022)
    try:
        path = _run().write(tmp_path / "out.json")
    finally:
        os.umask(mask)
    assert path.stat().st_mode & 0o777 == 0o644


def test_stream_accumulator_concatenates_like_openai_python() -> None:
    from canitoolcall.results import StreamAccumulator

    acc = StreamAccumulator()
    acc.add_openai_delta({"reasoning_content": "a", "content": None})
    acc.add_openai_delta({"reasoning": "b", "content": "c"})
    acc.add_openai_delta({"tool_calls": [{"index": 0, "function": {"name": "get", "arguments": '{"x"'}}]})
    acc.add_openai_delta({"tool_calls": [{"index": 0, "function": {"name": "_weather", "arguments": ": 1}"}}]})
    acc.add_tool_call(0, "_weather")  # a duplicated name delta shows up doubled, as clients see it
    acc.append_tool_call("g", "{}")
    assert acc.result() == ParseResult(
        content="c",
        reasoning_content="ab",
        tool_calls=(ParsedToolCall("get_weather_weather", '{"x": 1}'), ParsedToolCall("g", "{}")),
    )


def test_parse_result_from_dict_validates_types() -> None:
    import pytest

    with pytest.raises(TypeError, match="content"):
        ParseResult.from_dict({"content": 5})
    with pytest.raises(TypeError, match="name"):
        ParseResult.from_dict({"tool_calls": [{"name": None, "arguments_raw": "{}"}]})
    kept = ParseResult.from_dict({"tool_calls": [{"name": "f", "arguments_raw": None}]})
    assert kept.tool_calls[0].arguments_raw is None  # an engine outcome, judged by the checks
