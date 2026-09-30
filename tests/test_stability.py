"""Stability probe (step 2.3)."""

import pytest

from src.corpus.chunker import load_chunks
from src.retrieval.bm25 import BM25Retriever
from src.stream.stability import StabilityProbe, entities, is_incomplete, jaccard


@pytest.fixture(scope="module")
def bm25():
    return BM25Retriever(load_chunks())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I want to ask about the", True),
        ("tell me about retrieval and", True),
        ("the gates for…", True),
        ("the gates for...", True),
        ("which gate measures latency,", True),
        ("", True),
        ("which gate measures latency?", False),
        ("and what the threshold is.", False),
        ("do you know what it is!", False),
        ("describe the telemetry schema", False),
    ],
)
def test_is_incomplete(text, expected):
    assert is_incomplete(text) is expected


def test_entities_are_numbers_and_inner_capitals():
    assert entities("Plan a meeting in Delhi for 12 people") == ["Delhi", "12"]
    assert entities("Hello there") == []


def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0


def test_first_probe_has_zero_score(bm25):
    p = StabilityProbe(bm25).probe("describe the reranker component responsibilities", 0.0)
    assert p.first_probe and p.stability_score == 0.0 and p.sufficient


def test_score_rises_when_topk_is_stable(bm25):
    probe = StabilityProbe(bm25, k=3)
    probe.probe("citation hallucination fabricated", 0.0)
    same = probe.probe("citation hallucination fabricated um", 0.4)  # filler: top-k unchanged
    assert same.stability_score == 1.0
    moved = probe.probe("citation hallucination fabricated ids", 0.8)
    assert 0.0 < moved.stability_score < 1.0 and moved.stability_score == moved.jaccard


def test_insufficient_content_scores_zero(bm25):
    probe = StabilityProbe(bm25, min_content_terms=2, min_tokens=3)
    probe.probe("um so the", 0.0)
    p = probe.probe("um so the uh", 0.4)
    assert not p.sufficient and p.stability_score == 0.0 and p.top_ids == ()


def test_fillers_do_not_count_as_tokens(bm25):
    assert StabilityProbe(bm25).content("um uh er gates")[2] == 1


def test_reset_clears_history(bm25):
    probe = StabilityProbe(bm25)
    probe.probe("evaluation gates", 0.0)
    probe.reset()
    assert probe.probe("evaluation gates", 0.1).first_probe
