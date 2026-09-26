"""Probe tests against the local mock OpenAI-compatible server (tests/probe/conftest.py)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

import jsonschema
import pytest

from canitoolcall import cli
from canitoolcall.probe import (
    BUILTIN_SCENARIOS,
    EQUIVALENCE_CHECK,
    LEAK_MARKERS,
    REASONING_DELIMITERS,
    TOOL_CALL_MARKERS,
    ProbeExpectation,
    ProbeOutcome,
    ProbeReport,
    Scenario,
    _redact,
    chat,
    check_base_url,
    compare_modes,
    evaluate,
    evaluate_with_warnings,
    insecure_key_transport,
    iter_sse_data,
    probe,
    safe_url,
)
from canitoolcall.results import ParsedToolCall, ParseResult

SECRET = "sk-test-DO-NOT-LEAK-0123456789"


def outcome(report: ProbeReport, sid: str, *, stream: bool | None = None, equiv: bool = False) -> ProbeOutcome:
    for o in report.outcomes:
        if o.scenario_id != sid:
            continue
        if equiv and o.check == EQUIVALENCE_CHECK:
            return o
        if not equiv and o.check != EQUIVALENCE_CHECK and o.stream == stream:
            return o
    raise KeyError((sid, stream, equiv))


# --------------------------------------------------------------------------- scenario set


def test_builtin_scenarios_are_well_formed() -> None:
    ids = [s.id for s in BUILTIN_SCENARIOS]
    assert len(ids) == len(set(ids))
    assert 8 <= len(ids) <= 12
    for sc in BUILTIN_SCENARIOS:
        offered = {t["function"]["name"]: t["function"]["parameters"] for t in sc.tools}
        for params in offered.values():
            jsonschema.Draft202012Validator.check_schema(params)
        assert set(sc.expect.tool_names) <= set(offered), sc.id
        for tool, keys in sc.expect.required_argument_keys.items():
            assert set(keys) <= set(offered[tool]["properties"]), sc.id
        body = sc.request_body("m", stream=True)
        assert body["stream"] is True and body["temperature"] == 0
        json.dumps(body)


def test_scenarios_cover_the_required_behaviours() -> None:
    ids = {s.id for s in BUILTIN_SCENARIOS}
    for required in ("single-call", "parallel-calls", "no-call", "nested-args", "unicode-args", "reasoning-then-call"):
        assert required in ids


# --------------------------------------------------------------------------- end to end


def test_conforming_server_passes_everything(mock_openai: Any) -> None:
    report = probe(mock_openai.base_url, "mock-model", timeout_s=10)
    n = len(BUILTIN_SCENARIOS)
    assert len(report.outcomes) == 3 * n
    bad = [(o.scenario_id, o.stream, o.check, o.detail) for o in report.outcomes if o.status != "pass"]
    assert bad == []
    assert all(not o.warnings for o in report.outcomes)
    assert len(mock_openai.requests) == 2 * n
    assert sorted(b["stream"] for _, b in mock_openai.requests) == [False] * n + [True] * n
    assert {b["model"] for _, b in mock_openai.requests} == {"mock-model"}
    # report order: scenario order, then non-stream, stream, equivalence
    assert [o.scenario_id for o in report.outcomes[:3]] == [BUILTIN_SCENARIOS[0].id] * 3
    assert [(o.stream, o.check) for o in report.outcomes[:3]] == [
        (False, "scenario"),
        (True, "scenario"),
        (True, EQUIVALENCE_CHECK),
    ]


def test_stream_accumulation_matches_nonstream(mock_openai: Any) -> None:
    sc = next(s for s in BUILTIN_SCENARIOS if s.id == "unicode-args")
    ns, _ = chat(mock_openai.base_url, "m", sc, stream=False)
    st, _ = chat(mock_openai.base_url, "m", sc, stream=True)
    assert ns == st
    assert json.loads(st.tool_calls[0].arguments_raw)["body"] == "Café ☕ in 東京 — naïve résumé 🚀"


def test_reasoning_is_reported_under_either_field_name(mock_openai: Any) -> None:
    mock_openai.override("reasoning-then-call", reasoning_key="reasoning")
    report = probe(mock_openai.base_url, "m", scenarios=[s for s in BUILTIN_SCENARIOS if s.id == "reasoning-then-call"])
    for stream in (False, True):
        o = outcome(report, "reasoning-then-call", stream=stream)
        assert o.status == "pass"
        assert o.observed is not None and o.observed.reasoning_content == "9.9 = 9.90 and 9.90 > 9.11."
        assert "reasoning_content: yes" in (o.detail or "")


def test_no_stream_mode_skips_equivalence(mock_openai: Any) -> None:
    report = probe(mock_openai.base_url, "m", stream_modes=(False,), scenarios=BUILTIN_SCENARIOS[:2])
    assert [o.stream for o in report.outcomes] == [False, False]
    assert all(o.check == "scenario" for o in report.outcomes)


def test_concurrency_does_not_change_the_report(mock_openai: Any) -> None:
    a = probe(mock_openai.base_url, "m", concurrency=1)
    b = probe(mock_openai.base_url, "m", concurrency=8)
    key = [(o.scenario_id, o.stream, o.check, o.status, o.observed) for o in a.outcomes]
    assert key == [(o.scenario_id, o.stream, o.check, o.status, o.observed) for o in b.outcomes]


# --------------------------------------------------------------------------- failure diagnostics


def test_marker_leak_in_stream_only(mock_openai: Any) -> None:
    leaked = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    mock_openai.override("single-call", stream=True, content=leaked, tool_calls=())
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    assert outcome(report, "single-call", stream=False).status == "pass"
    st = outcome(report, "single-call", stream=True)
    assert st.status == "fail"
    assert "marker '<tool_call>' leaked into content" in (st.detail or "")
    assert "expected at least 1 tool call" in (st.detail or "")
    eq = outcome(report, "single-call", equiv=True)
    assert eq.status == "fail" and "tool_calls" in (eq.detail or "")


def test_think_tags_in_content_fail(mock_openai: Any) -> None:
    mock_openai.override(
        "reasoning-then-call", reasoning=None, content="<think>\n9.9 is larger\n</think>\n\n", fragment=2
    )
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "reasoning-then-call"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    for stream in (False, True):
        o = outcome(report, "reasoning-then-call", stream=stream)
        assert o.status == "fail"
        assert "'<think>' leaked into content" in (o.detail or "")
        assert "'</think>' leaked into content" in (o.detail or "")


# Real qwen3:4b behaviour on Ollama: the model drafts its calls inside its
# thinking, then the server returns both calls correctly.
DRAFTED_REASONING = (
    "The user wants Paris and Tokyo. So the tool_call XMLs would be:\n"
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
)


def test_markers_drafted_in_reasoning_pass_with_a_warning(mock_openai: Any, capsys: pytest.CaptureFixture[str]) -> None:
    mock_openai.override("parallel-calls", reasoning=DRAFTED_REASONING)
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "parallel-calls"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    for stream in (False, True):
        o = outcome(report, "parallel-calls", stream=stream)
        assert o.status == "pass", o.detail
        assert o.observed is not None and [c.name for c in o.observed.tool_calls] == ["get_weather"] * 2
        assert any("reasoning mentions tool-call markers" in w and "'<tool_call>'" in w for w in o.warnings)
        assert any("not a parser leak" in w for w in o.warnings)
    assert outcome(report, "parallel-calls", equiv=True).status == "pass"
    text = report.render_text()
    row = next(line for line in text.splitlines() if line.startswith("parallel-calls"))
    assert row.split() == ["parallel-calls", "pass*", "pass*", "pass"]
    assert "2 with warnings" in text
    assert "parallel-calls [stream]: reasoning mentions tool-call markers" in text
    assert "problems:" not in text
    # warnings are visible but do not change the exit code
    rc = cli.main(["probe", "--base-url", mock_openai.base_url, "--model", "m"])
    printed = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "pass*" in printed and "reasoning mentions tool-call markers" in printed


def test_markers_in_content_still_fail_when_reasoning_also_has_them(mock_openai: Any) -> None:
    mock_openai.override("single-call", reasoning=DRAFTED_REASONING, content="</tool_call>")
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    for stream in (False, True):
        o = outcome(report, "single-call", stream=stream)
        assert o.status == "fail"
        assert "marker '</tool_call>' leaked into content" in (o.detail or "")
        assert "leaked into reasoning_content" not in (o.detail or "")
        # the calls are right, so no "call inside the reasoning" hint
        assert "hint:" not in (o.detail or "")
        assert any("reasoning mentions tool-call markers" in w for w in o.warnings)


def test_missing_calls_with_markers_only_in_reasoning_fail_with_a_hint(mock_openai: Any) -> None:
    mock_openai.override("parallel-calls", reasoning=DRAFTED_REASONING, tool_calls=())
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "parallel-calls"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    for stream in (False, True):
        o = outcome(report, "parallel-calls", stream=stream)
        assert o.status == "fail"
        detail = o.detail or ""
        assert detail.startswith("expected at least 2 tool call(s)")
        assert "hint: reasoning_content contains '</tool_call>', '<tool_call>'" in detail
        assert "leaked into reasoning_content" not in detail


def test_evaluate_reasoning_markers_are_warnings_elsewhere_failures() -> None:
    sc = _scenario(tool_names=("get_weather",))
    good = _call("get_weather", '{"city": "Paris"}')
    status, detail, warnings = evaluate_with_warnings(
        sc, ParseResult(reasoning_content="<tool_call>draft</tool_call>", tool_calls=(good,))
    )
    assert status == "pass" and "reasoning_content: yes" in (detail or "")
    assert len(warnings) == 1 and "'</tool_call>', '<tool_call>'" in warnings[0]
    assert evaluate(sc, ParseResult(reasoning_content="<tool_call>", tool_calls=(good,)))[0] == "pass"
    status, _, warnings = evaluate_with_warnings(sc, ParseResult(tool_calls=(good,)))
    assert status == "pass" and warnings == ()
    leaked_arg = _call("get_weather", '{"city": "<tool_call>Paris"}')
    status, detail, _ = evaluate_with_warnings(
        sc, ParseResult(reasoning_content="<tool_call>", tool_calls=(leaked_arg,))
    )
    assert status == "fail" and "leaked into tool arguments" in (detail or "")


def test_think_tag_left_in_reasoning_fails(mock_openai: Any) -> None:
    # e.g. sglang#35083: the glm45 reasoning parser left "\n<think>" in reasoning_content
    mock_openai.override("reasoning-then-call", reasoning="\n<think>9.9 = 9.90 and 9.90 > 9.11.")
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "reasoning-then-call"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    for stream in (False, True):
        o = outcome(report, "reasoning-then-call", stream=stream)
        assert o.status == "fail"
        assert o.observed is not None and [c.name for c in o.observed.tool_calls] == ["record_answer"]
        assert (
            "reasoning delimiter '<think>' leaked into reasoning_content (the reasoning parser did not strip it)"
            in (o.detail or "")
        )
        assert o.warnings == ()


def test_harmony_channel_token_in_reasoning_fails(mock_openai: Any) -> None:
    reasoning = "Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary"
    mock_openai.override("single-call", reasoning=reasoning)
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    for stream in (False, True):
        o = outcome(report, "single-call", stream=stream)
        assert o.status == "fail"
        detail = o.detail or ""
        for m in ("<|channel|>", "<|start|>", "<|end|>"):
            assert f"reasoning delimiter {m!r} leaked into reasoning_content" in detail
    assert "single-call [stream] fail: reasoning delimiter" in report.render_text()


def test_evaluate_splits_reasoning_markers_by_group() -> None:
    sc = _scenario(tool_names=("get_weather",))
    good = _call("get_weather", '{"city": "Paris"}')
    # a drafted call plus a stray delimiter: the delimiter fails, the draft only warns
    status, detail, warnings = evaluate_with_warnings(
        sc, ParseResult(reasoning_content="</think><tool_call>{}</tool_call>", tool_calls=(good,))
    )
    assert status == "fail"
    assert "reasoning delimiter '</think>' leaked into reasoning_content" in (detail or "")
    assert "tool_call" not in (detail or "")
    assert len(warnings) == 1 and "'</tool_call>', '<tool_call>'" in warnings[0] and "'</think>'" not in warnings[0]
    for m in REASONING_DELIMITERS:
        assert evaluate(sc, ParseResult(reasoning_content=f"x{m}y", tool_calls=(good,)))[0] == "fail", m
    for m in TOOL_CALL_MARKERS:
        status, _, warnings = evaluate_with_warnings(sc, ParseResult(reasoning_content=f"x{m}y", tool_calls=(good,)))
        assert status == "pass" and len(warnings) == 1, m


def test_parallel_call_dropped_in_stream(mock_openai: Any) -> None:
    mock_openai.override("parallel-calls", stream=True, tool_calls=(("get_weather", '{"city": "Paris"}'),))
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "parallel-calls"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    assert outcome(report, "parallel-calls", stream=False).status == "pass"
    st = outcome(report, "parallel-calls", stream=True)
    assert st.status == "fail" and "expected at least 2 tool call(s)" in (st.detail or "")
    assert outcome(report, "parallel-calls", equiv=True).status == "fail"


def test_empty_string_arguments_fail(mock_openai: Any) -> None:
    mock_openai.override("empty-args", tool_calls=(("get_server_time", ""),))
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "empty-args"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    for stream in (False, True):
        o = outcome(report, "empty-args", stream=stream)
        assert o.status == "fail" and "not valid JSON" in (o.detail or "")


def test_arguments_as_object_is_a_protocol_failure(mock_openai: Any) -> None:
    mock_openai.override("single-call", stream=False, tool_calls=(("get_weather", {"city": "Paris"}),))
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1], stream_modes=(False,))
    o = outcome(report, "single-call", stream=False)
    assert o.status == "fail"
    assert "arguments is dict, not a JSON string" in (o.detail or "")


def test_missing_index_in_stream_deltas_fails(mock_openai: Any) -> None:
    mock_openai.override("parallel-calls", stream=True, omit_index=True)
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "parallel-calls"]
    report = probe(mock_openai.base_url, "m", scenarios=sc, stream_modes=(True,))
    o = outcome(report, "parallel-calls", stream=True)
    assert o.status == "fail" and "without an integer 'index'" in (o.detail or "")
    # ids still let the accumulator separate the two calls
    assert o.observed is not None and len(o.observed.tool_calls) == 2


def test_repeated_name_and_missing_id_are_warnings(mock_openai: Any) -> None:
    mock_openai.override("single-call", repeat_name=True, omit_id=True)
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    st = outcome(report, "single-call", stream=True)
    assert st.status == "pass"
    assert any("name repeated" in w for w in st.warnings)
    assert any("no id" in w for w in st.warnings)
    ns = outcome(report, "single-call", stream=False)
    assert ns.status == "pass" and any("no id" in w for w in ns.warnings)
    assert "pass*" in report.render_text()


def test_wrong_finish_reason_is_a_warning(mock_openai: Any) -> None:
    mock_openai.override("single-call", finish_reason="stop")
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    for stream in (False, True):
        o = outcome(report, "single-call", stream=stream)
        assert o.status == "pass" and any("finish_reason is 'stop'" in w for w in o.warnings)


def test_ignored_stream_flag_fails(mock_openai: Any) -> None:
    mock_openai.override("single-call", ignore_stream=True)
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    assert outcome(report, "single-call", stream=False).status == "pass"
    st = outcome(report, "single-call", stream=True)
    assert st.status == "fail" and "stream=true was ignored" in (st.detail or "")


def test_error_event_in_stream_is_an_error(mock_openai: Any) -> None:
    mock_openai.override("single-call", stream=True, error_event=True)
    report = probe(mock_openai.base_url, "m", scenarios=BUILTIN_SCENARIOS[:1])
    st = outcome(report, "single-call", stream=True)
    assert st.status == "error" and "boom" in (st.detail or "")
    assert outcome(report, "single-call", equiv=True).status == "skip"


def test_http_error_is_reported_as_error(mock_openai: Any) -> None:
    mock_openai.override("no-call", status=500)
    sc = [s for s in BUILTIN_SCENARIOS if s.id == "no-call"]
    report = probe(mock_openai.base_url, "m", scenarios=sc)
    for stream in (False, True):
        o = outcome(report, "no-call", stream=stream)
        assert o.status == "error" and "HTTP 500" in (o.detail or "") and o.observed is None
    assert outcome(report, "no-call", equiv=True).status == "skip"


def test_unreachable_endpoint_is_an_error() -> None:
    report = probe("http://127.0.0.1:9/v1", "m", scenarios=BUILTIN_SCENARIOS[:1], timeout_s=2)
    assert {o.status for o in report.outcomes} == {"error", "skip"}


# --------------------------------------------------------------------------- API key hygiene


def test_api_key_is_sent_but_never_reported(mock_openai: Any) -> None:
    report = probe(mock_openai.base_url, "m", api_key=SECRET, scenarios=BUILTIN_SCENARIOS[:2])
    assert {h.get("Authorization") for h, _ in mock_openai.requests} == {f"Bearer {SECRET}"}
    assert SECRET not in json.dumps(report.to_dict(), ensure_ascii=False)
    assert SECRET not in report.render_text()


def test_api_key_echoed_in_error_is_redacted(mock_openai: Any) -> None:
    body = json.dumps({"error": {"message": f"Incorrect API key provided: {SECRET}"}})
    mock_openai.override("single-call", status=401, error_body=body)
    report = probe(mock_openai.base_url, "m", api_key=SECRET, scenarios=BUILTIN_SCENARIOS[:1])
    o = outcome(report, "single-call", stream=False)
    assert o.status == "error" and "HTTP 401" in (o.detail or "") and "***" in (o.detail or "")
    assert SECRET not in json.dumps(report.to_dict()) and SECRET not in report.render_text()


def test_safe_url_drops_credentials_and_query() -> None:
    assert safe_url("https://user:pw@api.example.com:8443/v1?key=abc#x") == "https://api.example.com:8443/v1"
    assert safe_url("http://localhost:8000/v1/") == "http://localhost:8000/v1/"


# --------------------------------------------------------------------------- report + CLI


def test_report_json_and_text(mock_openai: Any) -> None:
    mock_openai.override("no-call", stream=True, content="<|im_end|>42")
    report = probe(mock_openai.base_url, "m")
    d = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert d["model"] == "m" and d["base_url"] == mock_openai.base_url
    assert d["summary"] == {"pass": 3 * len(BUILTIN_SCENARIOS) - 1, "fail": 1, "error": 0, "skip": 0}
    row = next(o for o in d["outcomes"] if o["scenario_id"] == "no-call" and o["stream"] and o["check"] == "scenario")
    assert row["status"] == "fail" and row["observed"]["content"] == "<|im_end|>42"
    text = report.render_text()
    header = next(line for line in text.splitlines() if line.startswith("scenario"))
    assert header.split() == ["scenario", "non-stream", "stream", "stream=non-stream"]
    no_call = next(line for line in text.splitlines() if line.startswith("no-call"))
    # the equivalence row is structural: content is present in both modes
    assert no_call.split() == ["no-call", "pass", "fail", "pass"]
    assert "no-call [stream] fail: marker '<|im_end|>' leaked into content" in text


def test_cli_probe_writes_table_and_json(
    mock_openai: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANITOOLCALL_TEST_KEY", SECRET)
    out = tmp_path / "probe.json"
    argv = ["probe", "--base-url", mock_openai.base_url, "--model", "m", "--api-key-env", "CANITOOLCALL_TEST_KEY"]
    rc = cli.main([*argv, "--json", str(out)])
    printed = capsys.readouterr().out
    assert rc == cli.EXIT_OK
    assert "single-call" in printed and "summary:" in printed and SECRET not in printed
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["pass"] == 3 * len(BUILTIN_SCENARIOS)
    assert SECRET not in out.read_text(encoding="utf-8")
    assert {h.get("Authorization") for h, _ in mock_openai.requests} == {f"Bearer {SECRET}"}

    # `--json -`: JSON alone on stdout, the table on stderr.
    assert cli.main([*argv, "--json", "-", "--concurrency", "1"]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert json.loads(captured.out)["summary"]["pass"] == 3 * len(BUILTIN_SCENARIOS)
    assert "summary:" in captured.err and SECRET not in captured.out + captured.err

    mock_openai.override("single-call", tool_calls=())
    assert cli.main([*argv, "--no-stream"]) == cli.EXIT_FAILURES


# --------------------------------------------------------------------------- units


def test_iter_sse_data_handles_comments_crlf_and_multiline() -> None:
    lines = [b": ping\r\n", b'data: {"a":\r\n', b"data: 1}\r\n", b"\r\n", b"event: x\n", b"data:[DONE]\n"]
    assert list(iter_sse_data(lines)) == ['{"a":\n1}', "[DONE]"]


def _scenario(**expect: Any) -> Scenario:
    weather = next(s for s in BUILTIN_SCENARIOS if s.id == "single-call").tools
    return Scenario("t", "t", ({"role": "user", "content": "x"},), weather, ProbeExpectation(**expect))


def _call(name: str, args: str) -> ParsedToolCall:
    return ParsedToolCall(name, args)


def test_evaluate_schema_and_tool_name() -> None:
    sc = _scenario(tool_names=("get_weather",))
    status, detail = evaluate(sc, ParseResult(tool_calls=(_call("get_weather", '{"city": 3, "unit": "k"}'),)))
    assert status == "fail" and "$.city" in (detail or "") and "$.unit" in (detail or "")
    status, detail = evaluate(sc, ParseResult(tool_calls=(_call("get_wether", '{"city": "Paris"}'),)))
    assert status == "fail" and "not an offered tool" in (detail or "")
    status, detail = evaluate(sc, ParseResult(tool_calls=(_call("get_weather", '["Paris"]'),)))
    assert status == "fail" and "not a JSON object" in (detail or "")


def test_evaluate_leaks_in_argument_values_and_replacement_chars() -> None:
    sc = _scenario(tool_names=("get_weather",))
    status, detail = evaluate(sc, ParseResult(tool_calls=(_call("get_weather", '{"city": "Paris</tool_call>"}'),)))
    assert status == "fail" and "leaked into tool arguments" in (detail or "")
    status, detail = evaluate(sc, ParseResult(tool_calls=(_call("get_weather", '{"city": "Z�rich"}'),)))
    assert status == "fail" and "U+FFFD" in (detail or "")


def test_evaluate_expectations() -> None:
    no_call = _scenario(expect_content=True)
    assert evaluate(no_call, ParseResult(content="42"))[0] == "pass"
    assert evaluate(no_call, ParseResult(content="  "))[0] == "fail"
    assert evaluate(no_call, ParseResult(content="42", tool_calls=(_call("get_weather", "{}"),)))[0] == "fail"
    order = _scenario(tool_names=("get_weather", "get_weather"), min_calls=2)
    one = _call("get_weather", '{"city": "a"}')
    assert evaluate(order, ParseResult(tool_calls=(one, one)))[0] == "pass"
    assert evaluate(order, ParseResult(tool_calls=(one,)))[0] == "fail"
    keeps = _scenario(tool_names=("get_weather",), argument_contains=("東京",))
    assert evaluate(keeps, ParseResult(tool_calls=(_call("get_weather", '{"city": "東京"}'),)))[0] == "pass"
    assert evaluate(keeps, ParseResult(tool_calls=(_call("get_weather", '{"city": "Tokyo"}'),)))[0] == "fail"
    reasoning = _scenario(expect_reasoning=True)
    assert evaluate(reasoning, ParseResult(content="x"))[0] == "fail"
    assert evaluate(_scenario(), ParseResult(exception="ValueError: x"))[0] == "fail"


def test_compare_modes_is_structural() -> None:
    a = ParseResult(content="It is sunny", tool_calls=(_call("get_weather", '{"city": "Paris"}'),))
    b = ParseResult(content="Sunny!", tool_calls=(_call("get_weather", '{"city": "paris"}'),))
    assert compare_modes(a, b) == ("pass", None)
    c = ParseResult(content="It is sunny", tool_calls=(_call("get_weather", '{"town": "Paris"}'),))
    status, detail = compare_modes(a, c)
    assert status == "fail" and "tool_calls" in (detail or "")


def test_leak_markers_are_unique_and_non_trivial() -> None:
    assert len(LEAK_MARKERS) == len(set(LEAK_MARKERS))
    assert all(len(m) >= 3 for m in LEAK_MARKERS)
    assert not set(TOOL_CALL_MARKERS) & set(REASONING_DELIMITERS)
    assert set(LEAK_MARKERS) == set(TOOL_CALL_MARKERS) | set(REASONING_DELIMITERS)
    assert {"<think>", "</think>", "<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|return|>"} <= set(
        REASONING_DELIMITERS
    )


# --------------------------------------------------------------------------- transport safety


class _Recorder(BaseHTTPRequestHandler):
    seen: ClassVar[list[str | None]] = []

    def do_POST(self) -> None:
        type(self).seen.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"choices": []}')

    def log_message(self, *args: Any) -> None:
        pass


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_redirects_are_not_followed_so_the_key_stays_put() -> None:
    other, other_url = _serve(_Recorder)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"{other_url.replace('127.0.0.1', 'localhost')}/v1/chat/completions?k=1")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    first, first_url = _serve(Redirect)
    try:
        report = probe(f"{first_url}/v1", "m", api_key=SECRET, scenarios=BUILTIN_SCENARIOS[:1], stream_modes=(False,))
        o = outcome(report, BUILTIN_SCENARIOS[0].id, stream=False)
        assert o.status == "error" and "redirect" in (o.detail or "") and "not followed" in (o.detail or "")
        assert "k=1" not in (o.detail or "")  # the Location is shown without its query
        assert _Recorder.seen == []  # the second server never got a request, let alone the key
    finally:
        first.shutdown()
        other.shutdown()


def test_redact_skips_tiny_keys_and_catches_url_encoding() -> None:
    assert _redact("not known", "k") == "not known"
    key = "sk-a/b+c=0123456789"
    assert _redact(f"x {key} y {quote(key, safe='')}", key) == "x *** y ***"


@pytest.mark.parametrize(
    "url",
    ["notaurl", "ftp://host/v1", "http://user:pw@host/v1", "http:///v1", "http://host/v1?key=1", "http://h:99999/"],
)
def test_check_base_url_rejects(url: str) -> None:
    with pytest.raises(ValueError):
        check_base_url(url)


def test_insecure_key_transport() -> None:
    assert insecure_key_transport("http://10.0.0.5:8000/v1")
    assert not insecure_key_transport("http://localhost:8000/v1")
    assert not insecure_key_transport("http://127.0.0.1:8000/v1")
    assert not insecure_key_transport("http://[::1]:8000/v1")
    assert not insecure_key_transport("https://api.example.com/v1")


def test_cli_probe_usage_errors_exit_2(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["probe", "--base-url", "notaurl", "--model", "x"])
    assert e.value.code == cli.EXIT_ERROR
    # A key would cross the network in cleartext: refused unless --allow-insecure.
    monkeypatch.setenv("CANITOOLCALL_API_KEY", SECRET)
    assert cli.main(["probe", "--base-url", "http://10.255.255.1:8000/v1", "--model", "x"]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "refusing" in err and SECRET not in err
    # An unreachable server is one line and exit 2, not 16 identical errors.
    monkeypatch.delenv("CANITOOLCALL_API_KEY")
    assert cli.main(["probe", "--base-url", "http://127.0.0.1:9/v1", "--model", "x", "--timeout", "2"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "cannot reach http://127.0.0.1:9/v1" in err


def test_cli_probe_does_not_read_openai_api_key_by_default(
    mock_openai: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.delenv("CANITOOLCALL_API_KEY", raising=False)
    cli.main(["probe", "--base-url", mock_openai.base_url, "--model", "m", "--no-stream"])
    assert {h.get("Authorization") for h, _ in mock_openai.requests} == {None}
