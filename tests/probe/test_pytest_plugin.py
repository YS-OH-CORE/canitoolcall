"""Tests for the pytest plugin (entry point ``pytest11: canitoolcall``).

Inner pytest sessions run through ``pytester``; the plugin is loaded there by
its installed entry point, exactly as in an engine repo. The corpus is the
core sample (tests/core/data/fixtures: one rendered Qwen3 fixture).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from canitoolcall.chunking import DEFAULT_STRATEGIES
from canitoolcall.fixtures import Fixture, load_fixtures
from canitoolcall.pytest_plugin import assert_conforms, assert_equivalent, expected_result, family_for
from canitoolcall.results import ParsedToolCall, ParseResult

pytest_plugins = ["pytester"]

SAMPLE_ID = "qwen3-hermes/sample-parallel-reasoning"


@pytest.fixture
def sample(sample_fixtures_dir: Path) -> Fixture:
    (fx,) = load_fixtures([sample_fixtures_dir])
    return fx


def _corpus_args(root: Path) -> list[str]:
    return ["-p", "no:cacheprovider", f"--canitoolcall-fixtures={root}"]


# --------------------------------------------------------------------------- inert + options


def test_plugin_is_registered_and_inert(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_plain():\n    assert 1 + 1 == 2\n")
    result = pytester.runpytest("-v", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*test_plain PASSED*"])
    assert "canitoolcall" in "".join(pytester.runpytest("--help").stdout.lines)


def test_help_lists_options(pytester: pytest.Pytester) -> None:
    out = "\n".join(pytester.runpytest("--help").stdout.lines)
    for opt in ("--canitoolcall-fixtures", "--canitoolcall-family", "--canitoolcall-tag", "--canitoolcall-strategy"):
        assert opt in out


# --------------------------------------------------------------------------- parametrization


def test_parametrizes_by_fixture_id(pytester: pytest.Pytester, sample_fixtures_dir: Path) -> None:
    pytester.makepyfile(
        """
        def test_ids(canitoolcall_fixture):
            assert canitoolcall_fixture.family == "qwen3-hermes"
        """
    )
    result = pytester.runpytest("-v", *_corpus_args(sample_fixtures_dir))
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines([f"*test_ids[[]{SAMPLE_ID}[]] PASSED*"])


def test_family_and_tag_filters(pytester: pytest.Pytester, sample_fixtures_dir: Path) -> None:
    pytester.makepyfile("def test_x(canitoolcall_fixture):\n    pass\n")
    args = _corpus_args(sample_fixtures_dir)
    pytester.runpytest(*args, "--canitoolcall-family", "qwen3-hermes").assert_outcomes(passed=1)
    pytester.runpytest(*args, "--canitoolcall-family", "glm").assert_outcomes(skipped=1)
    pytester.runpytest(*args, "--canitoolcall-tag", "unicode").assert_outcomes(passed=1)
    pytester.runpytest(*args, "--canitoolcall-tag", "truncated").assert_outcomes(skipped=1)


def test_strategy_parametrization(pytester: pytest.Pytester, sample_fixtures_dir: Path) -> None:
    pytester.makepyfile(
        """
        from canitoolcall.chunking import split

        def test_stream(canitoolcall_fixture, canitoolcall_strategy):
            ids = canitoolcall_fixture.output_token_ids
            groups = split(ids, canitoolcall_strategy, special={ids[0]})
            assert [i for g in groups for i in g] == list(ids)
        """
    )
    args = _corpus_args(sample_fixtures_dir)
    result = pytester.runpytest("-v", *args)
    result.assert_outcomes(passed=len(DEFAULT_STRATEGIES))
    result.stdout.fnmatch_lines([f"*test_stream[[]{SAMPLE_ID}-rand:3:8[]] PASSED*"])
    only = pytester.runpytest(*args, "--canitoolcall-strategy", "one", "--canitoolcall-strategy", "rand:7:2")
    only.assert_outcomes(passed=2)


def test_family_fixture(pytester: pytest.Pytester, sample_fixtures_dir: Path) -> None:
    pytester.makepyfile(
        """
        def test_family(canitoolcall_family):
            assert canitoolcall_family.slug == "qwen3-hermes"
            assert "<tool_call>" in canitoolcall_family.markers
        """
    )
    pytester.runpytest(*_corpus_args(sample_fixtures_dir)).assert_outcomes(passed=1)


def test_usage_errors(pytester: pytest.Pytester, tmp_path: Path) -> None:
    pytester.makepyfile("def test_x(canitoolcall_fixture, canitoolcall_strategy):\n    pass\n")
    missing = pytester.runpytest(f"--canitoolcall-fixtures={tmp_path / 'nope'}")
    assert missing.ret == pytest.ExitCode.USAGE_ERROR
    missing.stderr.fnmatch_lines(["*--canitoolcall-fixtures*does not exist*"])
    bad = pytester.runpytest(f"--canitoolcall-fixtures={tmp_path}", "--canitoolcall-strategy", "bogus")
    assert bad.ret == pytest.ExitCode.USAGE_ERROR
    bad.stderr.fnmatch_lines(["*unknown chunking strategy*"])


def test_engine_style_suite_end_to_end(pytester: pytest.Pytester, sample_fixtures_dir: Path, tmp_path: Path) -> None:
    """A toy 'engine' whose parser is right non-streaming and drops reasoning when streaming."""
    root = tmp_path / "corpus"
    shutil.copytree(sample_fixtures_dir, root)
    pytester.makepyfile(
        """
        from dataclasses import replace
        from canitoolcall.chunking import split
        from canitoolcall.pytest_plugin import assert_conforms, assert_equivalent, expected_result

        def engine_parse(fx):
            return expected_result(fx)

        def engine_parse_stream(fx, groups):
            return replace(expected_result(fx), reasoning_content=None) if len(groups) > 1 else expected_result(fx)

        def test_conformance(canitoolcall_fixture):
            assert_conforms(canitoolcall_fixture, engine_parse(canitoolcall_fixture))

        def test_streaming(canitoolcall_fixture, canitoolcall_strategy):
            groups = split(canitoolcall_fixture.output_token_ids, canitoolcall_strategy, special=set())
            streamed = engine_parse_stream(canitoolcall_fixture, groups)
            assert_equivalent(engine_parse(canitoolcall_fixture), streamed)
        """
    )
    result = pytester.runpytest(
        "-v", *_corpus_args(root), "--canitoolcall-strategy=one", "--canitoolcall-strategy=token"
    )
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines([f"*test_streaming[[]{SAMPLE_ID}-token[]] FAILED*", "*parses differ (fail)*"])


# --------------------------------------------------------------------------- assert_conforms


def test_expected_result_round_trips(sample: Fixture) -> None:
    exp = expected_result(sample)
    assert [c.name for c in exp.tool_calls] == ["get_weather", "search"]
    assert json.loads(exp.tool_calls[0].arguments_raw) == {"city": "Zürich", "unit": "c"}
    assert family_for(sample) is not None


def test_assert_conforms_accepts_expected_and_dict_form(sample: Fixture) -> None:
    exp = expected_result(sample)
    assert_conforms(sample, exp)
    assert_conforms(sample, exp.to_dict())
    # key order and whitespace in arguments do not matter
    reordered = replace(
        exp,
        tool_calls=(ParsedToolCall("get_weather", '{ "unit":"c","city":"Zürich" }'), exp.tool_calls[1]),
    )
    assert_conforms(sample, reordered)


def test_assert_conforms_wrong_argument_shows_diff(sample: Fixture) -> None:
    exp = expected_result(sample)
    wrong = replace(
        exp, tool_calls=(ParsedToolCall("get_weather", '{"city": "Zurich", "unit": "c"}'), exp.tool_calls[1])
    )
    with pytest.raises(AssertionError) as info:
        assert_conforms(sample, wrong)
    msg = str(info.value)
    assert f"{SAMPLE_ID} does not conform" in msg
    assert "expected_match: fail" in msg
    assert '-        "city": "Zürich",' in msg and '+        "city": "Zurich",' in msg
    assert "sample.jsonl:1" in msg


def test_assert_conforms_soft_whitespace(sample: Fixture) -> None:
    padded = replace(expected_result(sample), reasoning_content="\nI should call the tools.\n")
    with pytest.raises(AssertionError, match="soft_pass"):
        assert_conforms(sample, padded)
    assert_conforms(sample, padded, soft=True)


def test_assert_conforms_catches_leakage_and_bad_json(sample: Fixture) -> None:
    exp = expected_result(sample)
    leaked = replace(exp, content="</tool_call>")
    with pytest.raises(AssertionError, match="no_leakage: fail"):
        assert_conforms(sample, leaked)
    broken = replace(
        exp, tool_calls=(ParsedToolCall("get_weather", '{"city": "Zürich", "unit": "c"'), exp.tool_calls[1])
    )
    with pytest.raises(AssertionError) as info:
        assert_conforms(sample, broken)
    assert "arguments_json: fail" in str(info.value) and "<invalid JSON>" in str(info.value)
    with pytest.raises(AssertionError, match="parallel_order: fail"):
        assert_conforms(sample, replace(exp, tool_calls=exp.tool_calls[::-1]))


def test_assert_conforms_expected_error(sample: Fixture) -> None:
    from canitoolcall.fixtures import ExpectedError

    truncated = replace(
        sample,
        expected=None,
        expected_error=ExpectedError("truncated inside a tool call", ("no_tool_calls", "exception")),
    )
    assert_conforms(truncated, ParseResult(exception="ValueError: unterminated"))
    assert_conforms(truncated, ParseResult(content=None))
    with pytest.raises(AssertionError, match="expected_error: fail"):
        assert_conforms(truncated, expected_result(sample))


def test_assert_equivalent() -> None:
    a = ParseResult(content="hi", tool_calls=(ParsedToolCall("f", '{"a": 1}'),))
    assert_equivalent(a, ParseResult(content="hi", tool_calls=(ParsedToolCall("f", '{ "a" : 1 }'),)))
    with pytest.raises(AssertionError, match="soft_pass"):
        assert_equivalent(a, replace(a, content="hi\n"))
    assert_equivalent(a, replace(a, content="hi\n"), soft=True)
    with pytest.raises(AssertionError, match=r"parses differ \(fail\)"):
        assert_equivalent(a, replace(a, tool_calls=(ParsedToolCall("f", '{"a": "1"}'),)))
