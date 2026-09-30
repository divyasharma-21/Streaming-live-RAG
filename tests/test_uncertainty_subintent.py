"""Per-sub-intent uncertainty (step 4.5)."""

from src.schemas import CorpusChunk, ScoredChunk
from src.synthesis.uncertainty import compose_uncertainty, subintent_uncertainty, turn_clarification


def hit(body):
    return ScoredChunk(chunk=CorpusChunk(chunk_id="Doc_01 §3#1", doc_id="Doc_01", section="3", text=body),
                       score=1.0, rank=1, source="fusion")


EV = [hit("session memory is ephemeral and scoped to one conversation")]


def test_answered_and_verified_has_no_note():
    assert subintent_uncertainty("session memory rule", "session memory", EV, 0.34, 2, 0) is None


def test_no_evidence():
    assert subintent_uncertainty("hotel refund", "hotel refund", [], 0.34, 0, 0).startswith("No evidence")


def test_weak_evidence_names_missing_terms_and_asks_to_clarify():
    note = subintent_uncertainty("hotel refund deadline", "hotel refund deadline", EV, 0.34, 0, 0)
    assert "deadline" in note and "hotel" in note and "clarify" in note


def test_verification_failures():
    assert "withheld" in subintent_uncertainty("session memory", "session memory", EV, 0.34, 0, 2)
    assert subintent_uncertainty("session memory", "session memory", EV, 0.34, 1, 1).startswith("1 statement")


def test_generator_note_passes_through():
    assert subintent_uncertainty("session memory", "session memory", EV, 0.34, 1, 0, "scope unclear") == "scope unclear"


def test_turn_clarification_and_compose():
    assert "incomplete" in turn_clarification("insufficient_content", "so um can you")
    assert "prize" in turn_clarification("no_corpus_terms", "x", ["prize", "deadline"])
    assert compose_uncertainty([None, "a", "a", "b"]) == "a b" and compose_uncertainty([None]) is None
