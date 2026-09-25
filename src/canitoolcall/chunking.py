"""Seeded, reproducible chunking strategies for stream replay.

Streams are built from groups of *units*. For every realistic strategy the
units are the **token ids** of the raw output (engines never split a token, and
special tokens are atomic); each adapter then turns the id groups into text
deltas with its engine's own incremental detokenizer. The synthetic ``char``
strategy works on characters and exists only as an opt-in stress test.

Strategy ids (stable, recorded in results):

=====================  ======================================================
``one``                whole output as a single delta
``token``              one unit per delta
``rand:<seed>:<max>``  groups of 1..max units drawn from ``random.Random(seed)``
``char:<seed>``        synthetic character groups (seed 0 = per character,
                       else 1..8 chars from ``random.Random(seed)``)
=====================  ======================================================

Standard library only (imported inside engine venvs).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar("T")

StrategyKind = Literal["one", "token", "rand", "char"]


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

    @classmethod
    def parse(cls, spec: str) -> ChunkStrategy:
        """Parse a strategy id (``one``, ``token``, ``rand:S[:M]``, ``char:S``)."""
        parts = spec.strip().split(":")
        kind = parts[0]
        if kind in ("one", "token") and len(parts) == 1:
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
    ChunkStrategy("token"),
    *(ChunkStrategy("rand", seed=s, max_group=8) for s in range(1, 6)),
)
"""The default replay set: one chunk, per token, and 5 seeded random groupings."""


def group_sizes(n_units: int, strategy: ChunkStrategy) -> list[int]:
    """Sizes of consecutive groups covering ``n_units`` units (sum == n_units).

    An empty input yields ``[]`` for every strategy. The result depends only on
    ``(n_units, strategy)``, so it is identical on every machine.
    """
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


def split(units: Sequence[T], strategy: ChunkStrategy) -> list[list[T]]:
    """Split ``units`` into consecutive groups according to ``strategy``."""
    out: list[list[T]] = []
    i = 0
    for k in group_sizes(len(units), strategy):
        out.append(list(units[i : i + k]))
        i += k
    return out


def split_text(text: str, strategy: ChunkStrategy) -> list[str]:
    """Character-level split (for the synthetic ``char`` strategy)."""
    return ["".join(g) for g in split(list(text), strategy)]


def parse_strategies(specs: Sequence[str] | None) -> tuple[ChunkStrategy, ...]:
    """Parse CLI strategy ids; ``None`` or empty means :data:`DEFAULT_STRATEGIES`."""
    if not specs:
        return DEFAULT_STRATEGIES
    return tuple(ChunkStrategy.parse(s) for s in specs)
