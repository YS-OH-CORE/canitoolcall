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
        CaseResult("other/c", "other", Status.UNSUPPORTED),
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
    assert s["totals"] == {"pass": 1, "soft_pass": 0, "fail": 1, "error": 0, "unsupported": 1}
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
