"""SGLang adapter: replays fixtures through SGLang's real function-call and
reasoning parsers.

Owner: SGLang adapter builder (see docs/DESIGN.md). Runs in ``.venvs/sglang``
(built by ``scripts/engines/sglang.sh``: SGLang 0.5.20 cp312 manylinux wheel
unzipped and added via ``.pth``).

SGLang parsers are text-only. Mirror ``serving_chat.py``:

* tools present -> ``skip_special_tokens=False`` when detokenizing id groups
* ``ReasoningParser(model_type, stream_reasoning=True, force_reasoning=...)`` runs
  FIRST: ``parse_non_stream`` / ``parse_stream_chunk`` / ``parse_stream_end``
* then ``FunctionCallParser(tools=[Tool], tool_call_parser=...)`` on the normal
  text: ``has_tool_call`` + ``parse_non_stream`` / ``parse_stream_chunk`` /
  ``parse_stream_end`` at finish
* stop-token trimming must match the engine (e.g. the gpt-oss detector needs
  ``<|call|>`` in the text)

Reference implementation: ``.spikes/sglang/spike_sglang.py`` (local, gitignored).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.results import ParseResult

# Candidate (tool_call_parser, reasoning model_type) per family; verify against
# ToolCallParserEnum / DetectorMap in SGLang 0.5.20 and refine per model.
PARSERS: dict[str, dict[str, tuple[str, str | None]]] = {
    "qwen3-hermes": {"*": ("qwen25", "qwen3")},
    "qwen3-xml": {"*": ("qwen3_coder", "qwen3")},
    "gpt-oss": {"*": ("gpt-oss", "gpt-oss")},
    "deepseek": {
        "DeepSeek-V3.1": ("deepseekv31", "deepseek-v3"),
        "DeepSeek-V3.2": ("deepseekv32", "deepseek-v3"),
        "DeepSeek-V4": ("deepseekv4", "deepseek-v4"),
        "DeepSeek-R1": ("deepseekv3", "deepseek-r1"),
    },
    "kimi": {"Kimi-K2": ("kimi_k2", "kimi_k2"), "Kimi-K3": ("kimi_k3", "kimi_k3")},
    "glm": {"GLM-4.5": ("glm45", "glm45"), "GLM-4.7": ("glm47", "glm45")},
    "llama": {"Llama-3": ("llama3", None), "Llama-4": ("pythonic", None)},
    "mistral": {"*": ("mistral", "mistral")},
    "gemma4": {"*": ("gemma4", "gemma4")},
}


class SglangAdapter(Adapter):
    name: ClassVar[str] = "sglang"
    pinned_version: ClassVar[str] = "0.5.20"

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
