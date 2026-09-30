"""Speculative reuse cache (step 3.6)."""

import pytest

from src.retrieval.cache import SpeculativeCache, query_similarity


def test_query_similarity():
    assert query_similarity("telemetry trace coverage", "telemetry trace coverage detail") == pytest.approx(0.75)
    assert query_similarity("alpha", "beta") == 0.0


def test_lookup_hit_and_miss_counts():
    c = SpeculativeCache(similarity=0.75)
    c.put("telemetry trace coverage gate", "run-1")
    hit = c.lookup("telemetry trace coverage gate detail")
    assert hit is not None and hit.entry == "run-1" and hit.similarity == pytest.approx(0.8)
    assert c.lookup("reranker cache keys") is None
    assert (c.hits, c.misses) == (1, 1)


def test_best_match_wins():
    c = SpeculativeCache(similarity=0.5)
    c.put("a b c d", "weak")
    c.put("a b c e", "strong")
    assert c.lookup("a b c e f").entry == "strong"


def test_invalid_threshold():
    with pytest.raises(ValueError):
        SpeculativeCache(similarity=0)
