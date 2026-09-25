"""Seeded, reproducible chunking strategies for stream replay.

Streams are built from groups of *units*. For every realistic strategy the
units are the **token ids** of the raw output (engines never split a token, and
special tokens are atomic); each adapter then turns the id groups into text
deltas with its engine's own incremental detokenizer. The synthetic ``char``
strategy works on characters and exists only as an opt-in stress test.

Strategy ids (stable, recorded in results):

=====================  ======================================================
``one``                whole output as a single delta
``special``            split at special-token boundaries: every special unit
                       is its own delta, and each run of ordinary units
                       between them is one delta (needs the engine's set of
                       special token ids; see ``Adapter.special_token_ids``)
``token``              one unit per delta
``rand:<seed>:<max>``  groups of 1..max units drawn from ``random.Random(seed)``
``char:<seed>``        synthetic character groups (seed 0 = per character,
                       else 1..8 chars from ``random.Random(seed)``)
=====================  ======================================================

Standard library only (imported inside engine venvs).
"""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar("T")

StrategyKind = Literal["one", "special", "token", "rand", "char"]

TokensPerStep = Literal["one", "many"]
"""How many output tokens an engine's server can put in one streamed delta.

``many``: deltas may carry several tokens (vLLM and SGLang output coalescing,
``stream_interval``, speculative decoding). ``one``: the server emits one
event per token (llama-server, Ollama on top of it, transformers ``serve``),
so a multi-token delta never reaches the parser.
"""


@dataclass(frozen=True)
class ChunkStrategy:
    """A deterministic way to split a sequence of units into stream deltas."""

    kind: StrategyKind
    seed: int = 0
    max_group: int = 8

    @property
    def id(self) -> str:
        """Stable string id, e.g. ``"rand:3:8"``; inverse of :meth:`parse`."""
        if self.kind == "rand":
            return f"rand:{self.seed}:{self.max_group}"
        if self.kind == "char":
            return f"char:{self.seed}"
        return self.kind

    @property
    def realistic(self) -> bool:
        """False for ``char``: it can split special tokens, which engines never do."""
        return self.kind != "char"

    @property
    def multi_unit(self) -> bool:
        """True if a delta can hold more than one unit (``one``, ``special``, ``rand`` with max > 1)."""
        return self.kind in ("one", "special") or (self.kind == "rand" and self.max_group > 1)

    def realistic_for(self, tokens_per_step: TokensPerStep) -> bool:
        """Whether an engine that streams ``tokens_per_step`` can produce these deltas.

        ``char`` never can. Multi-unit strategies only count for engines whose
        deltas can carry several tokens; for one-token-per-step engines they
        are still run and reported, but as synthetic (like ``char``).
        """
        if not self.realistic:
            return False
        return tokens_per_step == "many" or not self.multi_unit

    @classmethod
    def parse(cls, spec: str) -> ChunkStrategy:
        """Parse a strategy id (``one``, ``special``, ``token``, ``rand:S[:M]``, ``char:S``)."""
        parts = spec.strip().split(":")
        kind = parts[0]
        if kind in ("one", "special", "token") and len(parts) == 1:
            return cls(kind)  # type: ignore[arg-type]
        if kind == "rand" and len(parts) in (2, 3):
            max_group = int(parts[2]) if len(parts) == 3 else 8
            if max_group < 1:
                raise ValueError(f"max group must be >= 1 in {spec!r}")
            return cls("rand", seed=int(parts[1]), max_group=max_group)
        if kind == "char" and len(parts) == 2:
            return cls("char", seed=int(parts[1]))
        raise ValueError(f"unknown chunking strategy {spec!r}")


DEFAULT_STRATEGIES: tuple[ChunkStrategy, ...] = (
    ChunkStrategy("one"),
    ChunkStrategy("special"),
    ChunkStrategy("token"),
    *(ChunkStrategy("rand", seed=s, max_group=8) for s in range(1, 6)),
)
"""The default replay set: one chunk, special-token boundaries, per token, and
5 seeded random groupings."""


def special_group_sizes(special_mask: Sequence[bool]) -> list[int]:
    """Group sizes for the ``special`` strategy.

    Each unit flagged special forms its own group; each maximal run of
    ordinary units between them forms one group. With no special units the
    result equals the ``one`` strategy.
    """
    sizes: list[int] = []
    run = 0
    for is_special in special_mask:
        if is_special:
            if run:
                sizes.append(run)
                run = 0
            sizes.append(1)
        else:
            run += 1
    if run:
        sizes.append(run)
    return sizes


def group_sizes(n_units: int, strategy: ChunkStrategy, special_mask: Sequence[bool] | None = None) -> list[int]:
    """Sizes of consecutive groups covering ``n_units`` units (sum == n_units).

    An empty input yields ``[]`` for every strategy. The result depends only on
    ``(n_units, strategy)`` (and ``special_mask`` for ``special``), so it is
    identical on every machine.

    Raises ``ValueError`` for ``special`` without a mask of length ``n_units``.
    """
    if strategy.kind == "special":
        if special_mask is None or len(special_mask) != max(n_units, 0):
            raise ValueError("the 'special' strategy needs a special-unit mask with one flag per unit")
        return special_group_sizes(special_mask)
    if n_units <= 0:
        return []
    if strategy.kind == "one":
        return [n_units]
    if strategy.kind == "token" or (strategy.kind == "char" and strategy.seed == 0):
        return [1] * n_units
    rng = random.Random(strategy.seed)
    hi = strategy.max_group
    sizes: list[int] = []
    left = n_units
    while left > 0:
        k = min(rng.randint(1, hi), left)
        sizes.append(k)
        left -= k
    return sizes


def split(units: Sequence[T], strategy: ChunkStrategy, special: Collection[T] | None = None) -> list[list[T]]:
    """Split ``units`` into consecutive groups according to ``strategy``.

    ``special`` is the set of special units (token ids) and is required by the
    ``special`` strategy only; other strategies ignore it.
    """
    mask = [u in special for u in units] if special is not None and strategy.kind == "special" else None
    out: list[list[T]] = []
    i = 0
    for k in group_sizes(len(units), strategy, mask):
        out.append(list(units[i : i + k]))
        i += k
    return out


def split_text(text: str, strategy: ChunkStrategy) -> list[str]:
    """Character-level split (for the synthetic ``char`` strategy)."""
    return ["".join(g) for g in split(list(text), strategy)]


def synthetic_strategies(strategies: Sequence[str], tokens_per_step: TokensPerStep) -> tuple[str, ...]:
    """The strategy ids (in order) that do not count toward an engine's case status.

    Unknown ids are treated as counting, so a result is never silently hidden.
    """
    out: list[str] = []
    for sid in strategies:
        try:
            strat = ChunkStrategy.parse(sid)
        except ValueError:
            continue
        if not strat.realistic_for(tokens_per_step):
            out.append(sid)
    return tuple(dict.fromkeys(out))


def parse_strategies(specs: Sequence[str] | None) -> tuple[ChunkStrategy, ...]:
    """Parse CLI strategy ids; ``None`` or empty means :data:`DEFAULT_STRATEGIES`."""
    if not specs:
        return DEFAULT_STRATEGIES
    return tuple(ChunkStrategy.parse(s) for s in specs)
