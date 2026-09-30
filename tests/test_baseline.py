"""Baseline pipeline, reranking, citations and uncertainty (step 1.8)."""

import json
from pathlib import Path

import jsonschema
import pytest

from src.baseline import BaselinePipeline
from src.config import Settings
from src.llm.client import MockLLMClient
from src.retrieval.factory import build_retrieval
from src.retrieval.rerank import IdentityReranker, LexicalReranker, make_reranker
from src.schemas import Claim, CorpusChunk, ScoredChunk
from src.synthesis.citations import CitationIndex, extract_citations
from src.synthesis.generator import ExtractiveGenerator, LLMGenerator, render_answer
from src.synthesis.uncertainty import evidence_score, uncertainty_note
from src.telemetry.logger import TelemetryLogger

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "output_record.schema.json").read_text(encoding="utf-8"))

# Test inputs only. They are written against the corpus text and are not benchmark prompts.
IN_CORPUS = "What are the technical evaluation gates and their target thresholds?"
NOT_IN_CORPUS = "What is the refund deadline for hotel bookings in Mumbai?"


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    return Settings(index_dir=d / "idx", log_dir=d / "logs", llm_provider="none")


@pytest.fixture(scope="module")
def stack(settings):
    return build_retrieval(settings)


@pytest.fixture(scope="module")
def pipeline(settings, stack):
    return BaselinePipeline(settings, stack=stack)


async def test_answer_has_valid_citations_and_matches_schema(pipeline, stack):
    result = await pipeline.answer(IN_CORPUS, utterance_end_s=2.1)
    record = result.record.model_dump()
    jsonschema.validate(record, SCHEMA)
    assert record["retrieval_events"] == [{"timestamp_s": 2.1, "query": IN_CORPUS, "trigger": "final"}]
    assert record["sub_queries"] == [IN_CORPUS]
    assert record["answer"] and record["citations"] and record["uncertainty"] is None
    assert "Doc_01 §5" in record["citations"]
    index = CitationIndex(stack.chunks)
    assert result.invalid_citations == []
    assert all(index.is_valid(c) for c in extract_citations(record["answer"]))
    assert extract_citations(record["answer"]) == record["citations"]


async def test_telemetry_trace_is_complete_and_written(pipeline, settings):
    result = await pipeline.answer(IN_CORPUS, request_id="req-telemetry")
    names = [e.event for e in result.telemetry]
    assert names == [
        "request_started", "retrieval_started", "retrieval_completed", "rerank_completed",
        "citation_check", "answer_emitted", "request_completed",
    ]
    assert result.telemetry[1].trigger == "final"
    lines = (settings.log_dir / "telemetry.jsonl").read_text().splitlines()
    assert sum(json.loads(line)["request_id"] == "req-telemetry" for line in lines) == len(names)


async def test_no_evidence_request_yields_uncertainty_not_a_guess(pipeline):
    result = await pipeline.answer(NOT_IN_CORPUS)
    assert result.record.answer == ""
    assert result.record.citations == []
    assert "does not contain sufficient evidence" in result.record.uncertainty


async def test_empty_content_request(pipeline):
    result = await pipeline.answer("what is the")
    assert result.record.citations == [] and result.record.uncertainty


async def test_llm_path_drops_fabricated_ids(settings, stack):
    def reply(prompt: str) -> str:
        real = prompt.split('<chunk id="', 1)[1].split('"', 1)[0]
        return json.dumps({
            "claims": [
                {"text": "Supported statement.", "chunk_ids": [real]},
                {"text": "Fabricated statement.", "chunk_ids": ["Doc_999 §1#1"]},
            ],
            "uncertainty": None,
        })

    telemetry = TelemetryLogger(None)
    pipe = BaselinePipeline(settings, stack=stack, generator=LLMGenerator(MockLLMClient(reply)), telemetry=telemetry)
    result = await pipe.answer(IN_CORPUS)
    assert [c.text for c in result.claims] == ["Supported statement."]
    assert "Doc_999" not in result.record.answer
    assert result.invalid_citations == []
    llm_events = [e for e in result.telemetry if e.event == "llm_call"]
    assert len(llm_events) == 1 and llm_events[0].tokens_in > 0


async def test_llm_path_with_no_valid_claims_reports_uncertainty(settings, stack):
    reply = json.dumps({"claims": [{"text": "x", "chunk_ids": ["nope"]}], "uncertainty": None})
    pipe = BaselinePipeline(settings, stack=stack, generator=LLMGenerator(MockLLMClient([reply])),
                            telemetry=TelemetryLogger(None))
    result = await pipe.answer(IN_CORPUS)
    assert result.record.answer == "" and result.record.uncertainty


async def test_extractive_claims_are_verbatim_corpus_spans(stack):
    hits = stack.hybrid.search(IN_CORPUS, k=3)
    synth = await ExtractiveGenerator().synthesize(IN_CORPUS, hits)
    assert synth.claims
    for claim in synth.claims:
        words = claim.text.split()[:4]
        chunk = next(c for c in stack.chunks if c.chunk_id == claim.chunk_ids[0])
        assert all(w.strip(";:") in chunk.text for w in words)


def test_render_answer_neutralises_citation_markers_in_claim_text():
    claim = Claim(id="c1", text="Cite as [Doc_ID §Section] always.", subintent_id="q1", chunk_ids=["Doc_01 §3#1"])
    answer = render_answer([claim])
    assert extract_citations(answer) == ["Doc_01 §3"]


def _sc(cid, text, rank):
    return ScoredChunk(chunk=CorpusChunk(chunk_id=cid, doc_id="D", section="1", text=text), score=1.0 / rank,
                       rank=rank, source="rrf")


def test_lexical_reranker_orders_by_query_coverage():
    hits = [_sc("a#1", "alpha", 1), _sc("b#1", "alpha beta gamma", 2)]
    out = LexicalReranker().rerank("alpha beta gamma", hits, top_k=2)
    assert [h.chunk.chunk_id for h in out] == ["b#1", "a#1"] and out[0].score == 1.0
    assert [h.chunk.chunk_id for h in IdentityReranker().rerank("x", hits, 2)] == ["a#1", "b#1"]
    with pytest.raises(ValueError):
        make_reranker("unknown")


def test_uncertainty_threshold():
    hits = [_sc("a#1", "alpha beta", 1)]
    assert evidence_score("alpha beta", "alpha beta") == 1.0
    assert uncertainty_note("alpha beta", hits, 0.5) is None
    note = uncertainty_note("alpha zeta omega", hits, 0.5)
    assert note and "omega" in note and "zeta" in note
    assert uncertainty_note("alpha", [], 0.5).startswith("No evidence")
