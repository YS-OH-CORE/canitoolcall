"""llama.cpp adapter: replays fixtures through llama.cpp's real chat parser
(``common/chat.h``) via a small compiled C++ harness.

Owner: llama.cpp adapter builder (see docs/DESIGN.md). Harness source lives in
``harnesses/llamacpp/`` and is built by ``scripts/engines/llamacpp.sh`` against
llama.cpp pinned at ``pinned_version`` into ``.engines/llamacpp/`` (gitignored).
No model weights: the parser is derived from the Jinja chat template alone.

Harness contract (JSON lines on stdin/stdout, one process per run, not per fixture):

* ``common_chat_templates_init(vocab_only_model, template, bos, eos)`` +
  ``common_chat_templates_apply(inputs{messages, tools, reasoning_format,
  enable_thinking})`` -> ``common_chat_params{parser, preserved_tokens, ...}``
* non-streaming: ``common_chat_parse(text, is_partial=false, pp)``
* streaming exactly like llama-server: re-parse the accumulated text with
  ``is_partial=true`` after each chunk, accumulate ``compute_diffs``; trim a
  chunk end inside a UTF-8 sequence with ``validate_utf8``
* EXACT detokenization: load a vocab-only GGUF (``convert_hf_to_gguf.py
  --vocab-only``) and render ids with ``common_token_to_piece(vocab, id,
  special = id in preserved_tokens)``. Never approximate GGUF CONTROL tokens
  with HF ``special`` flags (wrong for Gemma 4's ``<|"|>``).
* pin the template source (GGUF-embedded, HF at revision, or llama.cpp's
  ``models/templates/`` copy) and ``enable_thinking`` in ``parser_config``.

Reference implementation: ``.spikes/llamacpp/`` (local, gitignored).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.results import ParseResult


class LlamaCppAdapter(Adapter):
    name: ClassVar[str] = "llamacpp"
    pinned_version: ClassVar[str] = "a25c9865fe03c954c93fd755b5d79ae86ba99750"

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
