"""Anti-fragmentation guard (step 3.2)."""

import numpy as np
import pytest

from src.decompose.dedupe import AntiFragmentationGuard, GuardConfig, is_simple_utterance
from src.retrieval.dense import HashingEmbedder
from src.schemas import SubQuery


def sq(*texts):
    return [SubQuery(id=f"q{i + 1}", text=t) for i, t in enumerate(texts)]


@pytest.fixture
def guard():
    return AntiFragmentationGuard(HashingEmbedder(), GuardConfig(max_subqueries=3, merge_similarity=0.85))


def test_simple_utterance_detection():
    assert is_simple_utterance("describe the reranker scores")
    assert not is_simple_utterance("describe the reranker and the cache")
    assert not is_simple_utterance("What is it? How is it used?")


def test_short_circuit_overrides_over_splitting(guard):
    r = guard.apply(sq("describe the reranker", "reranker scores"), "describe the reranker scores")
    assert [q.text for q in r.subqueries] == ["describe the reranker scores"] and "short_circuit" in r.notes[0]


def test_thin_fragment_folds_into_previous(guard):
    t = "What should the telemetry schema record, and how long can it be?"
    r = guard.apply(sq("What should the telemetry schema record", "how long can it be?"), t)
    assert len(r.subqueries) == 1 and r.subqueries[0].id == "q1"
    assert "how long can it be" in r.subqueries[0].text


def test_near_duplicates_merge(guard):
    t = "what is the cache hit rate, and what is the cache hit rate today"
    r = guard.apply(sq("what is the cache hit rate", "what is the cache hit rate today"), t)
    assert len(r.subqueries) == 1 and any("near_duplicate" in n for n in r.notes)


def test_distinct_intents_survive(guard):
    t = "How is the reranker trained, and which gate measures telemetry coverage?"
    r = guard.apply(sq("How is the reranker trained", "which gate measures telemetry coverage?"), t)
    assert len(r.subqueries) == 2 and r.notes == []


def test_cap_folds_overflow(guard):
    t = "alpha beta, gamma delta, epsilon zeta, eta theta, iota kappa"
    r = guard.apply(sq("alpha beta", "gamma delta", "epsilon zeta", "eta theta", "iota kappa"), t)
    assert len(r.subqueries) == 3 and r.subqueries[2].text.count(" and ") == 2
    assert r.notes.count("capped") == 2


def test_constraints_are_kept_on_merge(guard):
    subs = [SubQuery(id="q1", text="the gate for Delhi", constraints=["Delhi"]),
            SubQuery(id="q2", text="it", constraints=["12 people"])]
    r = guard.apply(subs, "the gate for Delhi, and it")
    assert r.subqueries[0].constraints == ["Delhi", "12 people"]


def test_invalid_cap_rejected():
    with pytest.raises(ValueError):
        AntiFragmentationGuard(HashingEmbedder(), GuardConfig(max_subqueries=0))


def test_embedder_similarity_is_cosine():
    v = HashingEmbedder().embed(["same text", "same text"])
    assert float(v[0] @ v[1]) == pytest.approx(1.0) and np.all(np.isfinite(v))
