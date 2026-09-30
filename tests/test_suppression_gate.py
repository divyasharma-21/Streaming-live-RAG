"""Suppression gate (step 2.2). Test phrases are generic and deliberately differ from the
dev scenarios so the gate is not validated on its own tuning data."""

from pathlib import Path

import pytest

from src.corpus.chunker import load_chunks
from src.retrieval.bm25 import BM25Retriever
from src.stream.suppression import PresentationGate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "The controller decides when to retrieve and when to wait. [Doc_01 §2.1]"


@pytest.fixture(scope="module")
def gate():
    return PresentationGate(BM25Retriever(load_chunks()).vocabulary)


@pytest.mark.parametrize(
    "utterance",
    [
        "Make it shorter.",
        "Could you rewrite your last answer as bullet points?",
        "Turn that into a table please",
        "Repeat what you just said.",
        "Translate it into French.",
        "Rephrase the previous response more formally.",
    ],
)
def test_presentation_requests_are_suppressed(gate, utterance):
    r = gate.evaluate(utterance, PRIOR)
    assert r.presentation_only and not r.retrieval_required
    assert r.reason == "presentation_restructure"


@pytest.mark.parametrize(
    "utterance",
    [
        "What does the reranker do?",
        "Summarize the evaluation gates.",  # transformation verb but a new topic
        "Shorten the list of pitfalls about citation hallucination",
        "How is reproducibility validated?",
    ],
)
def test_information_requests_are_not_suppressed(gate, utterance):
    r = gate.evaluate(utterance, PRIOR)
    assert not r.presentation_only and r.retrieval_required


def test_no_prior_output_never_suppresses(gate):
    r = gate.evaluate("Make it shorter.", None)
    assert r.retrieval_required and r.reason == "no_prior_output"


def test_ambiguous_case_uses_classifier(gate):
    r = gate.evaluate("In bullets.", PRIOR)  # cue, no explicit reference, no new content
    assert r.method == "classifier" and r.presentation_only


def test_terms_already_in_prior_output_are_not_new(gate):
    r = gate.evaluate("Rewrite that part about the controller as a list.", PRIOR)
    assert r.presentation_only and r.new_terms == ()


def test_gate_module_contains_no_dev_scenario_text():
    import json

    src = (ROOT / "src" / "stream" / "suppression.py").read_text(encoding="utf-8").lower()
    for path in (ROOT / "eval" / "dev_scenarios").glob("*.json"):
        for turn in json.loads(path.read_text(encoding="utf-8"))["turns"]:
            for chunk in turn["chunks"]:
                text = chunk["text"].strip().lower()
                if len(text.split()) >= 3:
                    assert text not in src, f"scenario text from {path.name} found in gate code"
