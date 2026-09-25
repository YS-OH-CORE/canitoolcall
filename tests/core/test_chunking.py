from __future__ import annotations

import pytest

from canitoolcall.chunking import (
    DEFAULT_STRATEGIES,
    ChunkStrategy,
    group_sizes,
    parse_strategies,
    split,
    split_text,
)


@pytest.mark.parametrize("spec", ["one", "token", "rand:3:8", "rand:0:1", "char:0", "char:7"])
def test_id_roundtrip(spec: str) -> None:
    assert ChunkStrategy.parse(spec).id == spec


def test_rand_default_max() -> None:
    assert ChunkStrategy.parse("rand:4") == ChunkStrategy("rand", seed=4, max_group=8)


@pytest.mark.parametrize("bad", ["", "rand", "rand:x", "rand:1:0", "char", "token:1", "bytes:1"])
def test_parse_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        ChunkStrategy.parse(bad)


@pytest.mark.parametrize("strategy", [*DEFAULT_STRATEGIES, ChunkStrategy("char", 0), ChunkStrategy("char", 9)])
@pytest.mark.parametrize("n", [0, 1, 2, 17, 100])
def test_split_partitions_in_order(strategy: ChunkStrategy, n: int) -> None:
    units = list(range(n))
    groups = split(units, strategy)
    assert [u for g in groups for u in g] == units
    assert all(1 <= len(g) <= max(strategy.max_group, n) for g in groups)
    assert sum(group_sizes(n, strategy)) == n


def test_one_and_token_shapes() -> None:
    assert split([1, 2, 3], ChunkStrategy("one")) == [[1, 2, 3]]
    assert split([1, 2, 3], ChunkStrategy("token")) == [[1], [2], [3]]


def test_rand_is_seeded_and_bounded() -> None:
    s = ChunkStrategy("rand", seed=1, max_group=4)
    a, b = group_sizes(200, s), group_sizes(200, s)
    assert a == b
    assert max(a) <= 4
    assert group_sizes(200, ChunkStrategy("rand", seed=2, max_group=4)) != a


def test_rand_is_stable_across_releases() -> None:
    # Pinned: results record strategy ids, so the grouping for an id must never change.
    assert group_sizes(20, ChunkStrategy("rand", seed=1, max_group=8)) == group_sizes(
        20, ChunkStrategy.parse("rand:1:8")
    )
    assert group_sizes(20, ChunkStrategy.parse("rand:1:8")) == [3, 2, 5, 2, 8]


def test_char_is_marked_unrealistic() -> None:
    assert not ChunkStrategy.parse("char:0").realistic
    assert all(s.realistic for s in DEFAULT_STRATEGIES)
    assert split_text("abc", ChunkStrategy("char", 0)) == ["a", "b", "c"]


def test_parse_strategies_default() -> None:
    assert parse_strategies(None) == DEFAULT_STRATEGIES
    assert [s.id for s in parse_strategies(["one", "rand:9:2"])] == ["one", "rand:9:2"]
