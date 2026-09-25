"""Ollama adapter: replays fixtures through Ollama's real built-in parsers
(``model/parsers``) and the legacy template-driven ``tools`` parser, via a
small compiled Go harness.

Owner: Ollama adapter builder (see docs/DESIGN.md). Harness source lives in
``harnesses/ollama/`` and is built by ``scripts/engines/ollama.sh`` against
Ollama pinned at ``pinned_version`` into ``.engines/ollama/`` (gitignored).

Harness contract (JSON lines on stdin/stdout, one process per run), mirroring
``server/routes.go``:

* built-in: ``p := parsers.ParserForName(name)``; ``p.Init(tools, nil, think)``
  once; ``p.Add(chunk, done)`` per chunk; report ``p.PreservedTokens()``
* legacy (families with no built-in parser, e.g. Kimi-K2, GLM-4.5, Llama):
  ``tools.NewParser(tmpl, tools).Add(s)`` with the model's Ollama template
* the runner renders only preserved tokens as text; use the same vocab-only
  GGUF detokenization as the llama.cpp adapter for exact fidelity

Built-in parser names at the pin include: qwen3, qwen3-thinking, qwen3.5,
qwen3-coder, harmony, deepseek3, glm-4.7, gemma4, gemma4-no-thinking, ministral,
cohere, olmo3, lfm2, nemotron-3-nano, functiongemma.

Reference implementation: ``.spikes/ollama/`` (local, gitignored).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.results import ParseResult

# Candidate built-in parser per family (None -> legacy template parser).
PARSERS: dict[str, dict[str, str | None]] = {
    "qwen3-hermes": {"Qwen3-*-Thinking-2507": "qwen3-thinking", "*": "qwen3"},
    "qwen3-xml": {"Qwen3-Coder": "qwen3-coder", "Qwen3.5": "qwen3.5", "*": "qwen3.5"},
    "gpt-oss": {"*": "harmony"},
    "deepseek": {"DeepSeek-V3.1": "deepseek3"},
    "kimi": {"*": None},
    "glm": {"GLM-4.7": "glm-4.7", "GLM-4.5": None},
    "llama": {"*": None},
    "mistral": {"Ministral": "ministral", "*": None},
    "gemma4": {"*": "gemma4"},
}


class OllamaAdapter(Adapter):
    name: ClassVar[str] = "ollama"
    pinned_version: ClassVar[str] = "7af393188defd52d370464de0d2064649cab9b41"

    def version(self) -> str:
        raise NotImplementedError

    def supports(self, family: str, model: str) -> Support:
        raise NotImplementedError

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        raise NotImplementedError

    def units(self, raw: ReplayInput) -> list[int]:
        raise NotImplementedError

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        raise NotImplementedError

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        raise NotImplementedError
