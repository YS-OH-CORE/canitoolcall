"""The engine adapter contract.

An adapter replays a fixture's raw output through ONE engine's real parser
code, pinned to a version. It runs inside that engine's isolated venv (see
``canitoolcall.adapters.worker``), so this module is standard-library only.

Contract (every adapter MUST honour these; each was found by the engine spike
breaking without it — see docs/DESIGN.md "Faithfulness requirements"):

1. **Token ids first.** :meth:`Adapter.units` returns the output token ids:
   ``ReplayInput.token_ids`` when the fixture has them, else the engine's
   tokenizer encoding of ``ReplayInput.text``. Chunks are groups of these ids.
2. **Engine detokenization.** Text deltas are produced from the id groups with
   the engine's own (incremental) detokenizer and its effective request
   settings (e.g. vLLM ``adjust_request`` + v1 ``IncrementalDetokenizer``;
   SGLang ``skip_special_tokens=False``; llama.cpp/Ollama render only
   ``preserved_tokens`` from a vocab-only GGUF).
3. **Serving-layer entry points.** Call the same functions the engine's
   OpenAI-compatible server calls, with the same arguments (e.g. vLLM
   ``parse_delta(..., prompt_token_ids=...)``, SGLang reasoning parser before
   the function-call parser, ``parse_stream_end`` at finish).
4. **Engine exceptions are outcomes.** If the engine's parser raises, catch it
   and return ``ParseResult(exception="Type: message")``. Only raise
   (anything) for *harness* problems; the runner records those as ``error``.
5. **Pin everything.** :meth:`Adapter.parser_config` returns every input to the
   parser configuration (parser names, template source + sha256,
   enable_thinking / reasoning_format, tokenizer mode) so results reproduce.

Stop tokens (DESIGN.md adapter rule 5): fixtures end *before* the stop token.
An adapter may re-append the stop token that ended generation, but never when
:attr:`ReplayInput.truncated` is set: that output was cut by ``max_tokens``, so
the engine must see ``finish_reason: "length"`` and no stop token.
"""

from __future__ import annotations

import abc
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from canitoolcall.chunking import TokensPerStep
from canitoolcall.fixtures import Family, Fixture, TokenizerPin
from canitoolcall.results import ParseResult

ToolSpec = Mapping[str, Any]
"""An OpenAI-format tool: ``{"type": "function", "function": {...}}``."""


REMOTE_CODE_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # Kimi tokenizers are tiktoken-based and only load through the repo's own
        # tokenization_kimi.py. Pinned (repo, revision) pairs; nothing else runs repo code.
        ("moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50"),  # adapter tests
        ("moonshotai/Kimi-K2-Instruct-0905", "ac6c49f04883bd0a0598b790693a72061c676629"),
        ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"),
        ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"),
    }
)
"""Hub ``(repo, revision)`` pairs whose tokenizer needs ``trust_remote_code``.

Repo and revision come from fixture data (a ``tokenizer`` pin or family.json),
which may be a third-party corpus. Loading with ``trust_remote_code=True``
would run Python from that repo, so adapters and ``scripts/engines/gguf_vocab.sh``
enable it only for these reviewed pins (SECURITY.md: code execution from a
fixture is in scope).
"""


def trusts_remote_code(repo: str, revision: str | None) -> bool:
    """True only for the reviewed pins in :data:`REMOTE_CODE_ALLOWLIST`."""
    return revision is not None and (repo, revision) in REMOTE_CODE_ALLOWLIST


@dataclass(frozen=True)
class ReplayInput:
    """Everything an adapter needs to know about the raw output being replayed."""

    fixture_id: str
    family: str
    model: str
    """Reference model (``fixture.models[0]``): selects tokenizer, template, parser."""
    text: str
    """``fixture.raw_output``: exact completion text, special tokens literal."""
    token_ids: tuple[int, ...] | None = None
    tokenizer: TokenizerPin | None = None
    generation_prompt: str | None = None
    """Fixture override, else the family reference model's default, else None."""
    thinking: bool | None = None
    stop_tokens: tuple[str, ...] = ()
    tokenizer_mode: str = "hf"
    truncated: bool = False
    """Generation hit ``max_tokens`` (fixture tag ``truncated``): the finish
    reason is ``length`` and no stop token may be appended (DESIGN.md adapter rule 5)."""
    revision: str | None = None
    """Pinned revision of the reference model (``family.reference_models[].revision``).

    Adapters use it to load the tokenizer/template when the fixture has no
    ``tokenizer`` pin. ``None`` when the family is unknown.
    """

    @classmethod
    def from_fixture(cls, fixture: Fixture, family: Family | None = None) -> ReplayInput:
        ref = family.reference_for(fixture.reference_model) if family else None
        return cls(
            fixture_id=fixture.id,
            family=fixture.family,
            model=fixture.reference_model,
            text=fixture.raw_output,
            token_ids=fixture.output_token_ids,
            tokenizer=fixture.tokenizer,
            generation_prompt=(
                fixture.generation_prompt
                if fixture.generation_prompt is not None
                else (ref.default_generation_prompt if ref else None)
            ),
            thinking=fixture.thinking,
            stop_tokens=ref.stop_tokens if ref else (),
            tokenizer_mode=(fixture.tokenizer.mode if fixture.tokenizer else (ref.tokenizer_mode if ref else "hf")),
            revision=ref.revision if ref else None,
            truncated=fixture.truncated,
        )


@dataclass(frozen=True)
class Support:
    """Whether an adapter can replay a (family, model) pair, and why not."""

    supported: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.supported


class AdapterUnavailable(RuntimeError):
    """The engine (or its compiled harness) is not installed in this environment."""


class Adapter(abc.ABC):
    """Base class for engine adapters. One subclass per engine module."""

    name: ClassVar[str]
    """Engine name: ``vllm``, ``sglang``, ``llamacpp``, ``ollama``, ``transformers``."""

    supports_text_deltas: ClassVar[bool] = False
    """True if :meth:`parse_stream_text` is implemented (enables ``char:<seed>``)."""

    pinned_version: ClassVar[str]
    """The engine version (or commit) this adapter is written and tested against."""

    tokens_per_step: ClassVar[TokensPerStep] = "many"
    """How many tokens the engine's server can put in one streamed delta.

    ``"one"`` for servers that emit one event per generated token (llama-server,
    Ollama, transformers ``serve``): multi-token chunking strategies are then
    still run but reported as synthetic and never counted toward the case
    status. ``"many"`` for servers that coalesce tokens (vLLM, SGLang).
    """

    @abc.abstractmethod
    def version(self) -> str:
        """Engine version actually loaded (``vllm.__version__``, git sha, ...).

        Raises :class:`AdapterUnavailable` if the engine cannot be imported/run.
        """

    def engine_details(self) -> dict[str, Any]:
        """Extra pins recorded in results (key deps, build flags, harness sha)."""
        return {}

    def commit(self) -> str | None:
        """Engine source commit, when known."""
        return None

    @abc.abstractmethod
    def supports(self, family: str, model: str) -> Support:
        """Whether this engine has a parser configuration for ``(family, model)``.

        Must not import the engine; must be cheap. Unsupported pairs are
        reported as ``unsupported`` in the matrix, never as failures.
        """

    @abc.abstractmethod
    def parser_config(self, raw: ReplayInput) -> dict[str, Any]:
        """The exact parser configuration used for ``raw`` (recorded in results)."""

    @abc.abstractmethod
    def units(self, raw: ReplayInput) -> list[int]:
        """Output token ids that chunking strategies group (contract rule 1)."""

    @abc.abstractmethod
    def parse(self, raw: ReplayInput, tools: Sequence[ToolSpec]) -> ParseResult:
        """Non-streaming parse of the whole output, as the engine's server does it."""

    @abc.abstractmethod
    def parse_stream(self, raw: ReplayInput, chunks: Sequence[Sequence[int]], tools: Sequence[ToolSpec]) -> ParseResult:
        """Streaming parse: feed each group of token ids as one engine step.

        ``chunks`` partition :meth:`units` in order. The final chunk is the
        finishing step. Deltas are accumulated OpenAI-client style (see
        :class:`~canitoolcall.results.ParseResult`).
        """

    def special_token_ids(self, raw: ReplayInput) -> Collection[int] | None:
        """Ids the engine's tokenizer treats as special/control tokens.

        Optional. Enables the ``special`` chunking strategy (split at
        special-token boundaries). For HF tokenizers this is typically
        ``tokenizer.all_special_ids`` plus added tokens with ``special=True``.
        Return ``None`` (the default) when unknown; the worker then reports
        the strategy as skipped instead of guessing.
        """
        return None

    def parse_stream_text(self, raw: ReplayInput, deltas: Sequence[str], tools: Sequence[ToolSpec]) -> ParseResult:
        """Optional synthetic stress path: stream raw TEXT deltas (``char:<seed>``).

        Only called when :attr:`supports_text_deltas` is True. Results from this
        path never count toward a case's status (it can split special tokens,
        which no real server does).
        """
        raise NotImplementedError(f"{type(self).__name__} has no text-delta streaming path")

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release subprocesses/resources. Called once by the worker at shutdown."""
