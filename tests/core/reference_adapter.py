"""A pure-Python *reference* adapter, for testing the runner without any engine.

This is a TEST DOUBLE, not an engine: it is never registered in ``ENGINES`` and
its results never appear in the matrix. It lets the core tests drive the real
worker process, the real runner and the real checks end to end.

It implements a small, honest, incremental parser for the Hermes-style format
(``<think>...</think>`` reasoning, then ``<tool_call>{"name":..., "arguments":...}</tool_call>``
blocks), with a toy tokenizer in which the four markers are atomic special
tokens and every other character is one token. Because it cannot decode the
reference model's real token ids, ``units`` always encodes ``raw_output`` with
the toy tokenizer (a real adapter must prefer ``output_token_ids``).

Faults can be injected with ``$REFERENCE_ADAPTER_BUG`` so tests can prove the
checks and the runner catch them. The value is ``kind`` or ``kind@substring``
(only fixtures whose id contains ``substring`` are affected); several faults
are separated by commas:

``leak``        ``</think>`` is kept at the end of reasoning_content (leakage)
``drop-one``    the stream drops every tool call when the whole output arrives
                in one delta (the vLLM/SGLang bug class seen in the spike)
``raise``       the "engine" parser raises ValueError (a parse outcome)
``harness``     the adapter itself fails (worker answers ok: false -> error)
``crash``       the worker process dies mid-replay (exit code 3)
``hang``        the worker never answers (runner timeout)
``no-special``  the adapter does not expose special token ids
``noisy``       the "engine" prints to stdout and writes to file descriptor 1
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from canitoolcall.adapters.base import Adapter, ReplayInput, Support, ToolSpec
from canitoolcall.chunking import TokensPerStep
from canitoolcall.results import ParsedToolCall, ParseResult

THINK_OPEN, THINK_CLOSE, CALL_OPEN, CALL_CLOSE = "<think>", "</think>", "<tool_call>", "</tool_call>"
MARKERS: tuple[str, ...] = (THINK_OPEN, THINK_CLOSE, CALL_OPEN, CALL_CLOSE)
_MARKER_ID = {m: i + 1 for i, m in enumerate(MARKERS)}
_ID_MARKER = {i: m for m, i in _MARKER_ID.items()}
_CHAR_BASE = 1000

BUG_ENV = "REFERENCE_ADAPTER_BUG"


# --------------------------------------------------------------------------- toy tokenizer


def encode(text: str) -> list[int]:
    """Markers become one special id each; every other character is one id."""
    ids: list[int] = []
    i = 0
    while i < len(text):
        for m in MARKERS:
            if text.startswith(m, i):
                ids.append(_MARKER_ID[m])
                i += len(m)
                break
        else:
            ids.append(_CHAR_BASE + ord(text[i]))
            i += 1
    return ids


def decode(ids: Sequence[int]) -> str:
    return "".join(_ID_MARKER[i] if i in _ID_MARKER else chr(i - _CHAR_BASE) for i in ids)


# --------------------------------------------------------------------------- parser


class _Trimmed:
    """Emits text with leading and trailing whitespace removed, chunk-invariantly.

    Leading whitespace is dropped; trailing whitespace is held back until more
    non-whitespace text arrives, and dropped at the end.
    """

    def __init__(self) -> None:
        self.started = False
        self.held = ""

    def push(self, text: str) -> str:
        if not self.started:
            text = text.lstrip()
            if not text:
                return ""
            self.started = True
        body = text.rstrip()
        tail = text[len(body) :]
        if not body:
            self.held += tail
            return ""
        out = self.held + body
        self.held = tail
        return out


@dataclass
class _Delta:
    content: str = ""
    reasoning: str = ""
    calls: list[tuple[str, str]] = field(default_factory=list)


class HermesStreamParser:
    """Incremental state machine: start -> reasoning? -> content <-> call."""

    def __init__(self, in_reasoning: bool = False, leak_think_close: bool = False) -> None:
        self.state = "reasoning" if in_reasoning else "start"
        self.buf = ""
        self.content = _Trimmed()
        self.reasoning = _Trimmed()
        self.leak = leak_think_close

    @staticmethod
    def _safe_len(buf: str, marker: str) -> int:
        """Length of ``buf`` that cannot be the start of ``marker``."""
        for k in range(min(len(marker) - 1, len(buf)), 0, -1):
            if marker.startswith(buf[-k:]):
                return len(buf) - k
        return len(buf)

    def feed(self, text: str, final: bool) -> _Delta:
        self.buf += text
        out = _Delta()
        while True:
            if self.state == "start":
                if self.buf.startswith(THINK_OPEN):
                    self.buf = self.buf[len(THINK_OPEN) :]
                    self.state = "reasoning"
                elif THINK_OPEN.startswith(self.buf) and not final:
                    break
                else:
                    self.state = "content"
            elif self.state == "reasoning":
                i = self.buf.find(THINK_CLOSE)
                if i >= 0:
                    out.reasoning += self.reasoning.push(self.buf[:i] + (THINK_CLOSE if self.leak else ""))
                    self.buf = self.buf[i + len(THINK_CLOSE) :]
                    self.state = "content"
                    continue
                n = len(self.buf) if final else self._safe_len(self.buf, THINK_CLOSE)
                out.reasoning += self.reasoning.push(self.buf[:n])
                self.buf = self.buf[n:]
                break
            elif self.state == "content":
                i = self.buf.find(CALL_OPEN)
                if i >= 0:
                    out.content += self.content.push(self.buf[:i])
                    self.buf = self.buf[i + len(CALL_OPEN) :]
                    self.state = "call"
                    continue
                n = len(self.buf) if final else self._safe_len(self.buf, CALL_OPEN)
                out.content += self.content.push(self.buf[:n])
                self.buf = self.buf[n:]
                break
            else:  # call
                i = self.buf.find(CALL_CLOSE)
                if i < 0:
                    if final:  # truncated call: pass the raw text through as content
                        out.content += self.content.push(CALL_OPEN + self.buf)
                        self.buf = ""
                    break
                body, self.buf = self.buf[:i], self.buf[i + len(CALL_CLOSE) :]
                self.state = "content"
                try:
                    obj = json.loads(body)
                    name, args = obj["name"], obj["arguments"]
                    if not isinstance(name, str) or not isinstance(args, dict):
                        raise TypeError("bad call shape")
                except (ValueError, KeyError, TypeError):
                    out.content += self.content.push(CALL_OPEN + body + CALL_CLOSE)
                    continue
                out.calls.append((name, json.dumps(args, ensure_ascii=False)))
        return out


class _Accumulator:
    """OpenAI-client style accumulation of stream deltas."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[ParsedToolCall] = []

    def add(self, d: _Delta) -> None:
        self.content.append(d.content)
        self.reasoning.append(d.reasoning)
        self.calls.extend(ParsedToolCall(n, a) for n, a in d.calls)

    def result(self) -> ParseResult:
        return ParseResult(
            content="".join(self.content) or None,
            reasoning_content="".join(self.reasoning) or None,
            tool_calls=tuple(self.calls),
        )


# --------------------------------------------------------------------------- adapter


def _bugs(fixture_id: str) -> set[str]:
    out: set[str] = set()
    for item in os.environ.get(BUG_ENV, "").split(","):
        kind, _, where = item.strip().partition("@")
        if kind and (not where or where in fixture_id):
            out.add(kind)
    return out


class ReferenceAdapter(Adapter):
    """Pure-Python Hermes-style parser used only by the core tests."""

    name: ClassVar[str] = "reference"
    pinned_version: ClassVar[str] = "1.0"
    supports_text_deltas: ClassVar[bool] = True
    FAMILIES: ClassVar[tuple[str, ...]] = ("qwen3-hermes",)

    def version(self) -> str:
        return "1.0"

    def engine_details(self) -> dict[str, Any]:
        return {"parser": "reference-hermes"}

    def supports(self, family: str, model: str) -> Support:
        if family in self.FAMILIES:
            return Support(True)
        return Support(False, f"reference adapter implements only {', '.join(self.FAMILIES)}")

    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        return {"parser": "reference-hermes", "starts_in_reasoning": self._in_reasoning(raw)}

    def units(self, raw: ReplayInput) -> list[int]:
        return encode(raw.text)

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        if "no-special" in _bugs(raw.fixture_id):
            return None
        return frozenset(_ID_MARKER)

    @staticmethod
    def _in_reasoning(raw: ReplayInput) -> bool:
        gp = (raw.generation_prompt or "").rstrip()
        return gp.endswith(THINK_OPEN)

    def _faults(self, raw: ReplayInput) -> set[str]:
        bugs = _bugs(raw.fixture_id)
        if "noisy" in bugs:
            print("engine chatter on sys.stdout")
            os.write(1, b"engine chatter on fd 1\n")
        if "harness" in bugs:
            raise RuntimeError("reference adapter: injected harness failure")
        if "crash" in bugs:
            os._exit(3)
        if "hang" in bugs:
            time.sleep(3600)
        return bugs

    def _run(self, raw: ReplayInput, deltas: Iterator[str] | Sequence[str], bugs: set[str]) -> ParseResult:
        if "raise" in bugs:
            try:
                raise ValueError("reference parser: injected failure")
            except ValueError as e:  # engine exceptions are outcomes (contract rule 6)
                return ParseResult(exception=f"{type(e).__name__}: {e}")
        parser = HermesStreamParser(self._in_reasoning(raw), leak_think_close="leak" in bugs)
        acc = _Accumulator()
        items = list(deltas)
        for i, text in enumerate(items):
            acc.add(parser.feed(text, final=i == len(items) - 1))
        if not items:
            acc.add(parser.feed("", final=True))
        return acc.result()

    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        return self._run(raw, [raw.text], self._faults(raw))

    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        bugs = self._faults(raw)
        result = self._run(raw, [decode(c) for c in chunks], bugs)
        if "drop-one" in bugs and len(chunks) == 1:
            return ParseResult(result.content, result.reasoning_content, (), result.exception)
        return result

    def parse_stream_text(self, raw: ReplayInput, deltas: Sequence[str], tools: Sequence[ToolSpec]) -> ParseResult:
        return self._run(raw, deltas, self._faults(raw))


class OneTokenReferenceAdapter(ReferenceAdapter):
    """The reference adapter as an engine whose server streams one token per event (like Ollama)."""

    tokens_per_step: ClassVar[TokensPerStep] = "one"
