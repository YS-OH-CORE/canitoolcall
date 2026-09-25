"""pytest plugin (entry point ``pytest11: canitoolcall``) for engines that
vendor the suite into their own CI.

The plugin is **inert unless enabled**: it only adds options, and generates
tests only for test functions that request the ``canitoolcall_fixture``
argument. Installing canitoolcall must never change an unrelated test run.

Intended engine-side usage::

    # tests/test_canitoolcall.py in an engine repo
    from canitoolcall.pytest_plugin import assert_conforms

    def test_conformance(canitoolcall_fixture):
        result = my_engine_parse(canitoolcall_fixture)   # -> ParseResult
        assert_conforms(canitoolcall_fixture, result)

    $ pytest --canitoolcall-family qwen3-hermes --canitoolcall-family glm

Options:

``--canitoolcall-fixtures PATH``  fixtures root (default: bundled corpus)
``--canitoolcall-family SLUG``    restrict to a family (repeatable)
``--canitoolcall-tag TAG``        restrict to fixtures with a tag (repeatable)
"""

from __future__ import annotations

from typing import Any

import pytest

from canitoolcall.fixtures import Fixture
from canitoolcall.results import ParseResult

FIXTURE_ARG = "canitoolcall_fixture"


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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize tests that request ``canitoolcall_fixture`` (ids = fixture ids)."""
    if FIXTURE_ARG not in metafunc.fixturenames:
        return
    raise NotImplementedError("canitoolcall pytest plugin: parametrization not implemented yet")


def assert_conforms(fixture: Fixture, result: ParseResult, *, soft: bool = False, **kwargs: Any) -> None:
    """Assert that one engine parse conforms to ``fixture`` (expected match or
    expected_error, no leakage, valid JSON arguments). ``soft=True`` accepts
    soft-v1 whitespace differences. Raises AssertionError with a readable diff.
    """
    raise NotImplementedError
