"""Presentation-only path (step 4.7)."""

import json

import pytest

from src.config import Settings
from src.llm.client import MockLLMClient
from src.retrieval.factory import build_retrieval
from src.schemas import Claim
from src.session.pipeline import SessionPipeline
from src.stream.simulator import chunks_from_text
from src.synthesis.citations import extract_citations
from src.synthesis.presentation import detect_transform, present
from src.telemetry.logger import TelemetryLogger

CLAIMS = [
    Claim(id="a.c1", text="Rule one holds.", subintent_id="a", chunk_ids=["Doc_01 §3#1"]),
    Claim(id="a.c2", text="Rule two holds.", subintent_id="a", chunk_ids=["Doc_01 §3#1"]),
    Claim(id="b.c1", text="The gate is G4.", subintent_id="b", chunk_ids=["Doc_01 §5#1"]),
]


@pytest.mark.parametrize(
    "instruction, expected",
    [("Give me that in two bullets", ("bullets", 2)), ("as a numbered list", ("numbered", None)),
     ("Make it a table", ("table", None)), ("a bullet list", ("bullets", None)), ("Shorter please", ("shorter", None)),
     ("Say it again", ("repeat", None)), ("Translate it into French", ("translate", None)),
     ("Can you jazz it up", ("unknown", None))],
)
def test_detect_transform(instruction, expected):
    assert detect_transform(instruction) == expected


async def test_bullets_respect_count_and_cover_intents_first():
    text, note = await present("in two bullets", CLAIMS)
    assert text == "- Rule one holds. [Doc_01 §3]\n- The gate is G4. [Doc_01 §5]" and note is None


@pytest.mark.parametrize("instruction", ["as a table", "numbered list", "shorter", "repeat that", "bullets"])
async def test_transforms_never_add_citations(instruction):
    text, _ = await present(instruction, CLAIMS)
    assert set(extract_citations(text)) <= {"Doc_01 §3", "Doc_01 §5"}


async def test_translation_without_llm_is_not_faked():
    text, note = await present("translate it into Hindi", CLAIMS)
    assert text.startswith("Rule one holds.") and "needs a configured LLM" in note


async def test_llm_transform_that_adds_a_citation_is_rejected():
    bad = MockLLMClient([json.dumps({"text": "Regel eins. [Doc_01 §3] Extra. [Doc_01 §8]"})])
    text, note = await present("translate into German", CLAIMS, client=bad)
    assert "Doc_01 §8" not in text and "changed the citations" in note
    good = MockLLMClient([json.dumps({"text": "Regel eins gilt. [Doc_01 §3]"})])
    text, note = await present("translate into German", CLAIMS, client=good)
    assert text == "Regel eins gilt. [Doc_01 §3]" and note is None


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


async def test_pipeline_presentation_turn_makes_zero_retrieval_calls(stack, tmp_path):
    p = SessionPipeline(Settings(log_dir=tmp_path, llm_provider="none"), stack=stack, telemetry=TelemetryLogger(None))
    calls = []
    original = p.engine._search
    p.engine._search = lambda q: calls.append(q) or original(q)
    first = await p.handle_turn("s", chunks_from_text(["What are the hard engineering rules?"]))
    n_first = len(calls)
    out = await p.handle_turn("s", chunks_from_text(["Could you rewrite that", "as bullet points?"]))
    assert out.kind == "presentation_only" and len(calls) == n_first and out.retrieval_calls == 0
    assert out.answer_version == first.answer_version  # content unchanged, version unchanged
    assert set(out.record.citations) <= set(first.record.citations)
    assert out.record.answer.startswith("- ") and out.ledger_diff["added"] == [] and out.ledger_diff["removed"] == []


async def test_presentation_request_after_an_empty_answer_does_not_search(stack, tmp_path):
    """The user saw only an uncertainty note; 'shorten that' must not trigger a corpus search."""
    p = SessionPipeline(Settings(log_dir=tmp_path, llm_provider="none"), stack=stack, telemetry=TelemetryLogger(None))
    first = await p.handle_turn("s", chunks_from_text(["What is the refund deadline for hotel bookings in Mumbai?"]))
    assert first.record.answer == "" and first.record.uncertainty
    out = await p.handle_turn("s", chunks_from_text(["Can you shorten that", "to one sentence?"]))
    assert out.kind == "presentation_only" and out.retrieval_calls == 0
    assert "no earlier answer" in out.record.uncertainty.lower()
