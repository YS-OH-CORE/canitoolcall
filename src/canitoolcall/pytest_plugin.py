"""pytest plugin (entry point ``pytest11: canitoolcall``) for engines that
vendor the suite into their own CI.

The plugin is **inert unless requested**: it only adds command-line options
and a few fixtures, and it parametrizes only test functions that ask for the
``canitoolcall_fixture`` (or ``canitoolcall_strategy``) argument. Installing
canitoolcall never changes an unrelated test run.

Engine-side usage (``pip install canitoolcall``; the fixture corpus ships in
the wheel)::

    # tests/test_canitoolcall.py in an engine repo
    from canitoolcall.chunking import split
    from canitoolcall.pytest_plugin import assert_conforms, assert_equivalent
    from canitoolcall.results import ParseResult

    def test_conformance(canitoolcall_fixture):
        result = my_engine_parse(canitoolcall_fixture)        # -> ParseResult (or its dict form)
        assert_conforms(canitoolcall_fixture, result)

    def test_streaming(canitoolcall_fixture, canitoolcall_strategy):
        ids = canitoolcall_fixture.output_token_ids or my_tokenize(canitoolcall_fixture.raw_output)
        # seeded, identical on every machine; `special` (the engine's special-token ids)
        # is only needed by the `special` strategy, which splits at special-token boundaries
        groups = split(ids, canitoolcall_strategy, special=MY_SPECIAL_TOKEN_IDS)
        streamed = my_engine_parse_stream(canitoolcall_fixture, groups)
        assert_conforms(canitoolcall_fixture, streamed, soft=True)
        assert_equivalent(my_engine_parse(canitoolcall_fixture), streamed)

    $ pytest tests/test_canitoolcall.py --canitoolcall-family qwen3-hermes --canitoolcall-family glm

Each generated test id is the fixture id (and the strategy id), e.g.
``test_conformance[qwen3-hermes/parallel-two-calls-unicode]``, so a failure
names the exact fixture to replay. ``my_engine_parse`` is the engine's own
glue: build a request with ``fixture.tools``, run the tool/reasoning parser on
``fixture.raw_output`` and return a :class:`~canitoolcall.results.ParseResult`
(an exception raised by the parser should be returned as
``ParseResult(exception="Type: msg")``, see ``expected_error`` fixtures).

Options:

``--canitoolcall-fixtures PATH``  fixtures root or file (default: bundled corpus / ``$CANITOOLCALL_FIXTURES``)
``--canitoolcall-family SLUG``    restrict to a family (repeatable)
``--canitoolcall-tag TAG``        restrict to fixtures carrying a tag (repeatable)
``--canitoolcall-strategy ID``    chunking strategies for ``canitoolcall_strategy`` (repeatable;
                                  default: ``canitoolcall.chunking.DEFAULT_STRATEGIES``)

Fixtures provided:

``canitoolcall_fixture``   the :class:`~canitoolcall.fixtures.Fixture` (parametrized)
``canitoolcall_strategy``  a :class:`~canitoolcall.chunking.ChunkStrategy` (parametrized)
``canitoolcall_family``    the fixture's :class:`~canitoolcall.fixtures.Family` (``None`` if no family.json)
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from canitoolcall.chunking import DEFAULT_STRATEGIES, ChunkStrategy, parse_strategies
from canitoolcall.fixtures import Family, Fixture, load_fixtures
from canitoolcall.results import CheckResult, Observation, ParseResult, Status

FIXTURE_ARG = "canitoolcall_fixture"
STRATEGY_ARG = "canitoolcall_strategy"

_FIXTURES_KEY = pytest.StashKey[list[Fixture]]()
_STRATEGIES_KEY = pytest.StashKey[tuple[ChunkStrategy, ...]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("canitoolcall", "CanIToolCall conformance fixtures")
    group.addoption(
        "--canitoolcall-fixtures",
        action="store",
        default=None,
        metavar="PATH",
        help="fixtures root directory (default: the bundled corpus)",
    )
    group.addoption(
        "--canitoolcall-family",
        action="append",
        default=[],
        metavar="SLUG",
        help="only fixtures of this family (repeatable)",
    )
    group.addoption(
        "--canitoolcall-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="only fixtures carrying this tag (repeatable)",
    )
    group.addoption(
        "--canitoolcall-strategy",
        action="append",
        default=[],
        metavar="ID",
        help="chunking strategy for canitoolcall_strategy, e.g. one, token, rand:1:8 (repeatable)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Validate the canitoolcall options up front (a no-op when none are given)."""
    root = config.getoption("canitoolcall_fixtures", None)
    if root is not None and not Path(root).exists():
        raise pytest.UsageError(f"--canitoolcall-fixtures: {root} does not exist")
    try:
        config.stash[_STRATEGIES_KEY] = parse_strategies(config.getoption("canitoolcall_strategy", None))
    except ValueError as e:
        raise pytest.UsageError(f"--canitoolcall-strategy: {e}") from None


def _selected_fixtures(config: pytest.Config) -> list[Fixture]:
    """Load (once per session) the fixtures selected by the command-line options."""
    cached = config.stash.get(_FIXTURES_KEY, None)
    if cached is not None:
        return cached
    root = config.getoption("canitoolcall_fixtures")
    fixtures = load_fixtures(
        [Path(root)] if root is not None else None,
        families=config.getoption("canitoolcall_family") or None,
        tags=config.getoption("canitoolcall_tag") or None,
    )
    config.stash[_FIXTURES_KEY] = fixtures
    return fixtures


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize tests that request ``canitoolcall_fixture`` (ids = fixture ids)
    and/or ``canitoolcall_strategy`` (ids = strategy ids). Other tests are untouched."""
    if FIXTURE_ARG in metafunc.fixturenames:
        fixtures = _selected_fixtures(metafunc.config)
        metafunc.parametrize(FIXTURE_ARG, fixtures, ids=[f.id for f in fixtures])
    if STRATEGY_ARG in metafunc.fixturenames:
        strategies = metafunc.config.stash.get(_STRATEGIES_KEY, DEFAULT_STRATEGIES)
        metafunc.parametrize(STRATEGY_ARG, strategies, ids=[s.id for s in strategies])


@cache
def _load_family_file(path: Path) -> Family:
    return Family.from_dict(json.loads(path.read_text(encoding="utf-8")))


def family_for(fixture: Fixture) -> Family | None:
    """The ``family.json`` next to the file ``fixture`` was loaded from, if any."""
    if fixture.source is None:
        return None
    path = fixture.source.parent / "family.json"
    return _load_family_file(path) if path.is_file() else None


@pytest.fixture
def canitoolcall_family(canitoolcall_fixture: Fixture) -> Family | None:
    """Family metadata (markers, reference models) for the current ``canitoolcall_fixture``."""
    return family_for(canitoolcall_fixture)


# --------------------------------------------------------------------------- assertions


def expected_result(fixture: Fixture) -> ParseResult:
    """``fixture.expected`` as a :class:`ParseResult` (see :func:`canitoolcall.checks.expected_as_result`)."""
    from canitoolcall import checks

    if fixture.expected is None:
        raise ValueError(f"{fixture.id} has expected_error, not expected")
    return checks.expected_as_result(fixture.expected)


def _as_result(result: ParseResult | Mapping[str, Any]) -> ParseResult:
    return result if isinstance(result, ParseResult) else ParseResult.from_dict(result)


def _display(result: ParseResult) -> str:
    calls: list[dict[str, Any]] = []
    for tc in result.tool_calls:
        try:
            args: Any = tc.arguments()
        except json.JSONDecodeError:
            args = {"<invalid JSON>": tc.arguments_raw}
        calls.append({"name": tc.name, "arguments": args})
    shown = {
        "content": result.content,
        "reasoning_content": result.reasoning_content,
        "tool_calls": calls,
        "exception": result.exception,
    }
    return json.dumps(shown, indent=2, ensure_ascii=False, sort_keys=True)


def _diff(expected: str, observed: str) -> str:
    lines = difflib.unified_diff(expected.splitlines(), observed.splitlines(), "expected", "observed", lineterm="", n=3)
    return "\n".join(lines)


def _failing(rows: list[CheckResult], soft: bool) -> list[CheckResult]:
    bad = {Status.FAIL, Status.ERROR} | (set() if soft else {Status.SOFT_PASS})
    return [r for r in rows if r.status in bad]


def assert_conforms(
    fixture: Fixture,
    result: ParseResult | Mapping[str, Any],
    *,
    soft: bool = False,
    family: Family | None = None,
) -> None:
    """Assert that one engine parse conforms to ``fixture``.

    Runs the per-parse checks of :mod:`canitoolcall.checks`: the expected
    parse via :func:`~canitoolcall.checks.compare` (or ``expected_error``),
    no marker leakage, valid JSON-object arguments that match the tool schema,
    and call count/order. ``soft=True`` accepts soft-v1 whitespace differences
    (``soft_pass``). ``family`` defaults to the ``family.json`` next to the
    fixture file (it supplies the leakage markers). Raises AssertionError with
    the failing checks and a readable diff.
    """
    from canitoolcall import checks

    parsed = _as_result(result)
    fam = family if family is not None else family_for(fixture)
    obs = Observation(nonstream=parsed, streams={})
    rows: list[CheckResult] = []
    if fixture.expected is not None:
        want = expected_result(fixture)
        status = checks.compare(want, parsed)
        detail = None if status is Status.PASS else checks.describe_difference(want, parsed)
        rows.append(CheckResult("expected_match", checks.NONSTREAM, status, detail))
    else:
        rows += checks.check_expected_error(fixture, fam, obs)
    for fn in (
        checks.check_no_leakage,
        checks.check_arguments_json,
        checks.check_arguments_schema,
        checks.check_parallel_order,
    ):
        rows += fn(fixture, fam, obs)

    bad = _failing(rows, soft)
    if not bad:
        return
    lines = [f"{fixture.id} does not conform:"]
    lines += [f"  {r.check}: {r.status.value}" + (f" ({r.detail})" if r.detail else "") for r in bad]
    if fixture.expected is not None:
        lines += ["", _diff(_display(expected_result(fixture)), _display(parsed))]
    else:
        err = fixture.expected_error
        assert err is not None
        lines += ["", f"expected_error: {err.reason}; accept {list(err.accept)}", "observed:", _display(parsed)]
    if fixture.source is not None:
        lines += ["", f"fixture: {fixture.source}:{fixture.line}"]
    raise AssertionError("\n".join(lines))


def assert_equivalent(
    a: ParseResult | Mapping[str, Any], b: ParseResult | Mapping[str, Any], *, soft: bool = False
) -> None:
    """Assert two parses are equal (e.g. streaming vs non-streaming) under
    :func:`~canitoolcall.checks.compare`; ``soft=True`` accepts soft-v1 differences."""
    from canitoolcall import checks

    ra, rb = _as_result(a), _as_result(b)
    status = checks.compare(ra, rb)
    if status is Status.PASS or (soft and status is Status.SOFT_PASS):
        return
    raise AssertionError(f"parses differ ({status.value}):\n" + _diff(_display(ra), _display(rb)))
