"""Dev scenario files are well-formed and their gold labels resolve to real corpus sections."""

from collections import Counter

from eval.scenarios import load_scenarios
from src.corpus.chunker import load_chunks
from src.synthesis.citations import CitationIndex


def test_scenarios_load_and_cover_all_categories():
    scenarios = load_scenarios()
    assert len(scenarios) == 30
    counts = Counter(s.category for s in scenarios)
    assert set(counts) == {
        "single_intent", "multi_intent", "late_constraint", "presentation_only", "noisy_fragment", "no_evidence"
    }


def test_gold_labels_resolve_to_corpus_sections():
    index = CitationIndex(load_chunks())
    for s in load_scenarios():
        for t in s.turns:
            for sub in t.gold_sub_intents:
                for c in sub.supporting + sub.also_relevant:
                    assert index.is_valid(c), f"{s.id} turn {t.turn}: {c}"


def test_category_invariants():
    for s in load_scenarios():
        if s.category == "multi_intent":
            assert len(s.turns[0].gold_sub_intents) >= 2, s.id
        if s.category == "single_intent":
            assert len(s.turns[0].gold_sub_intents) == 1, s.id
        if s.category == "presentation_only":
            assert not s.turns[-1].retrieval_required and s.turns[-1].reuses_turn, s.id
        if s.category == "late_constraint":
            assert len(s.turns) >= 2, s.id
        if s.category == "no_evidence":
            assert all(not sub.supporting for t in s.turns for sub in t.gold_sub_intents), s.id
            assert all(t.expect_uncertainty for t in s.turns), s.id
