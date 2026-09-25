"""vLLM adapter: replays fixtures through vLLM's real tool/reasoning parsers.

Owner: vLLM adapter builder (see docs/DESIGN.md). Runs in ``.venvs/vllm``
(built by ``scripts/engines/vllm.sh``: vLLM 0.30.0 manylinux wheel unzipped and
added via ``.pth``, CPU torch, no model weights).

Entry points (as used by ``vllm/entrypoints/openai/chat_completion/serving.py``):

* ``ParserManager.get_parser(tool_parser_name=..., reasoning_parser_name=...,
  enable_auto_tools=True, is_harmony=...)`` -> ``p = cls(tokenizer, request.tools)``
* ``request = p.adjust_request(request)`` FIRST (may force skip_special_tokens=False)
* non-streaming: ``p.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)``
* streaming: ``p.parse_delta(delta_text, delta_ids, request, prompt_token_ids=prompt_ids, finished=...)``
* text from ``vllm.v1.engine.detokenizer.IncrementalDetokenizer`` (never ``tok.decode``)
* tokenizer: ``vllm.tokenizers.get_tokenizer(model, tokenizer_mode='auto'|'mistral')``

Reference implementation: ``.spikes/vllm/spike_vllm.py`` (local, gitignored).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.results import ParseResult

# Candidate (tool_parser, reasoning_parser) per family from the spike and
# docs/formats/. Builders MUST verify each against vLLM 0.30.0 and refine per
# model (e.g. DeepSeek versions) before relying on it.
PARSERS: dict[str, dict[str, tuple[str, str | None]]] = {
    "qwen3-hermes": {"*": ("hermes", "qwen3")},
    "qwen3-xml": {"*": ("qwen3_coder", "qwen3")},
    "gpt-oss": {"*": ("openai", "openai_gptoss")},
    "deepseek": {
        "DeepSeek-V3.1": ("deepseek_v31", "deepseek_v3"),
        "DeepSeek-V3.2": ("deepseek_v32", "deepseek_v3"),
        "DeepSeek-V4": ("deepseek_v4", "deepseek_v4"),
        "DeepSeek-V4.1": ("deepseek_v41", "deepseek_v41"),
        "DeepSeek-R1": ("deepseek_v3", "deepseek_r1"),
    },
    "kimi": {"Kimi-K2": ("kimi_k2", "kimi_k2"), "Kimi-K3": ("kimi_k3", "kimi_k3")},
    "glm": {"GLM-4.5": ("glm45", "glm45"), "GLM-4.7": ("glm47", "glm47")},
    "llama": {"Llama-3": ("llama3_json", None), "Llama-4": ("llama4_pythonic", None)},
    "mistral": {"*": ("mistral", "mistral")},
    "gemma4": {"*": ("gemma4", "gemma4")},
}


class VllmAdapter(Adapter):
    name: ClassVar[str] = "vllm"
    pinned_version: ClassVar[str] = "0.30.0"

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
