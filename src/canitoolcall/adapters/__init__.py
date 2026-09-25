"""Engine adapters and their registry.

Each engine has exactly one module here (``vllm.py``, ``sglang.py``,
``llamacpp.py``, ``ollama.py``, ``transformers.py``) defining one
:class:`~canitoolcall.adapters.base.Adapter` subclass. Adapter modules must NOT
import their engine at module import time; import it inside methods so the
registry can be listed from the main dev env.

Adapters execute inside an isolated per-engine interpreter (see
:func:`engine_python`), driven over JSON lines by
:mod:`canitoolcall.adapters.worker`.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from canitoolcall.adapters.base import Adapter, AdapterUnavailable, ReplayInput, Support

ENGINES: dict[str, str] = {
    "vllm": "canitoolcall.adapters.vllm:VllmAdapter",
    "sglang": "canitoolcall.adapters.sglang:SglangAdapter",
    "llamacpp": "canitoolcall.adapters.llamacpp:LlamaCppAdapter",
    "ollama": "canitoolcall.adapters.ollama:OllamaAdapter",
    "transformers": "canitoolcall.adapters.transformers:TransformersAdapter",
}
"""Engine name -> ``module:Class``. Core-owned; adapter builders never edit it."""

PYTHON_ENV = "CANITOOLCALL_{ENGINE}_PYTHON"
"""Per-engine interpreter override, e.g. ``CANITOOLCALL_VLLM_PYTHON``."""


def adapter_class(engine: str) -> type[Adapter]:
    """Import and return the adapter class for ``engine`` (does not import the engine)."""
    try:
        target = ENGINES[engine]
    except KeyError:
        raise ValueError(f"unknown engine {engine!r}; known: {', '.join(ENGINES)}") from None
    mod_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        raise TypeError(f"{target} is not an Adapter subclass")
    return cls


def load_adapter(engine: str) -> Adapter:
    """Instantiate the adapter for ``engine`` in the *current* interpreter."""
    return adapter_class(engine)()


def engine_python(engine: str, repo_root: Path | None = None) -> Path:
    """Interpreter that has ``engine`` installed.

    Resolution order: ``$CANITOOLCALL_<ENGINE>_PYTHON``, then
    ``<repo>/.venvs/<engine>/bin/python`` (built by ``scripts/engines/<engine>.sh``),
    then the current interpreter.
    """
    env = os.environ.get(PYTHON_ENV.format(ENGINE=engine.upper()))
    if env:
        return Path(env)
    if repo_root is not None:
        venv_py = repo_root / ".venvs" / engine / "bin" / "python"
        if venv_py.exists():
            return venv_py
    return Path(sys.executable)


__all__ = [
    "ENGINES",
    "Adapter",
    "AdapterUnavailable",
    "ReplayInput",
    "Support",
    "adapter_class",
    "engine_python",
    "load_adapter",
]
