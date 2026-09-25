"""Shared pytest configuration for the canitoolcall test suite (core-owned).

Engine tests: mark a test ``@pytest.mark.engine("vllm")`` and use the
``engine_python`` fixture. The test is skipped unless that engine's isolated
interpreter exists (``$CANITOOLCALL_<ENGINE>_PYTHON`` or ``.venvs/<engine>``),
so ``uv run pytest`` stays green on a fresh checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from canitoolcall.adapters import PYTHON_ENV, engine_python
from canitoolcall.fixtures import repo_root

DATA = Path(__file__).parent / "core" / "data"


def _engine_interpreter(engine: str) -> Path | None:
    """The engine's isolated interpreter, or None when it is not set up."""
    py = engine_python(engine, repo_root())
    explicit = os.environ.get(PYTHON_ENV.format(ENGINE=engine.upper()))
    if not explicit and py == Path(sys.executable):
        return None
    return py if py.exists() else None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        marker = item.get_closest_marker("engine")
        if marker is None:
            continue
        engine = marker.args[0]
        if _engine_interpreter(engine) is None:
            item.add_marker(pytest.mark.skip(reason=f"{engine} env not set up (run scripts/engines/{engine}.sh)"))


@pytest.fixture
def engine_python_for(request: pytest.FixtureRequest) -> Path:
    """Interpreter for the engine named by the test's ``engine`` marker."""
    marker = request.node.get_closest_marker("engine")
    assert marker is not None, "engine_python_for requires @pytest.mark.engine(<name>)"
    py = _engine_interpreter(marker.args[0])
    assert py is not None
    return py


@pytest.fixture
def sample_fixtures_dir() -> Path:
    """Fixtures root holding the rendered Qwen3 sample (see core/data/make_sample.py)."""
    return DATA / "fixtures"
