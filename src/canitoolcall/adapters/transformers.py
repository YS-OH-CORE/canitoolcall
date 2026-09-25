"""HF transformers adapter (stretch): ``tokenizer.parse_response`` and the
streaming ``ResponseParser``, driven by the ``response_template`` a model repo
ships in ``tokenizer_config.json``.

Owner: transformers adapter builder (see docs/DESIGN.md). Runs in
``.venvs/transformers`` (transformers>=5.17, jinja2, jmespath; no torch).

Entry points (transformers 5.17):

* ``tok = AutoTokenizer.from_pretrained(repo, revision=...)``; ``tok.response_template``
* non-streaming: ``tok.parse_response(text_or_ids, prefix=prompt_text, tools=tools)``
* streaming: ``ResponseParser(template, prefix=..., tools=...)``; ``.feed(chunk)``
  per text delta; ``msg, events = parser.finalize()``
* the legacy ``response_schema`` key is silently dropped in 5.x; do not use it

Coverage rule: only report a family as supported when the model repo itself
ships a ``response_template`` at the pinned revision (as of 2026-09-25 only
``google/gemma-4-*``). Templates copied from transformers' tests are NOT what
users get from the Hub and must not be used for matrix results.

Reference implementation: ``.spikes/transformers/spike_transformers.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.results import ParseResult


class TransformersAdapter(Adapter):
    name: ClassVar[str] = "transformers"
    pinned_version: ClassVar[str] = "5.17.0"

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
