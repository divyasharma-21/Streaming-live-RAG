"""Per-sub-intent synthesis (step 4.3)."""

import json

from src.llm.client import MockLLMClient
from src.schemas import Claim, CorpusChunk, ScoredChunk, SubQuery
from src.synthesis.citations import extract_citations
from src.synthesis.generator import (
    SUBINTENT_SYSTEM_PROMPT,
    ExtractiveGenerator,
    LLMGenerator,
    render_answer,
    synthesize_subintents,
)


def hit(cid, body, heading="H"):
    sec = cid.split("§")[1].split("#")[0]
    return ScoredChunk(chunk=CorpusChunk(chunk_id=cid, doc_id="Doc_01", section=sec, heading=heading,
                                         text=f"T — §{sec} {heading}\n\n{body}"), score=1.0, rank=1, source="fusion")


ITEMS = [
    (SubQuery(id="s1", text="what is the reranker"), [hit("Doc_01 §2.1#1", "The reranker orders chunks.")]),
    (SubQuery(id="s2", text="what is the cache"), [hit("Doc_01 §7#1", "The cache reuses searches.")]),
]


async def test_llm_claims_are_tagged_and_restricted_to_own_evidence():
    reply = json.dumps({
        "claims": [
            {"subintent_id": "s1", "text": "The reranker orders chunks.", "chunk_ids": ["Doc_01 §2.1#1"]},
            {"subintent_id": "s2", "text": "Cites the other intent's chunk.", "chunk_ids": ["Doc_01 §2.1#1"]},
            {"subintent_id": "s2", "text": "The cache reuses searches.", "chunk_ids": ["Doc_01 §7#1"]},
            {"subintent_id": "s9", "text": "Unknown intent.", "chunk_ids": ["Doc_01 §7#1"]},
            {"subintent_id": "s1", "text": "Fabricated.", "chunk_ids": ["Doc_999 §1#1"]},
        ],
        "uncertain": [{"subintent_id": "s2", "note": "pricing not covered"}],
    })
    client = MockLLMClient([reply])
    out = await synthesize_subintents(LLMGenerator(client), ITEMS)
    assert [c.text for c in out.claims["s1"]] == ["The reranker orders chunks."]
    assert [c.text for c in out.claims["s2"]] == ["The cache reuses searches."]
    assert out.dropped_claims == 3 and out.uncertainty == {"s2": "pricing not covered"}
    prompt = client.prompts[0]
    assert '<subquestion id="s1">' in prompt and '<chunk id="Doc_01 §7#1">' in prompt
    assert "Use only that evidence" in SUBINTENT_SYSTEM_PROMPT


async def test_extractive_fallback_per_subintent():
    out = await synthesize_subintents(ExtractiveGenerator(), ITEMS)
    assert [c.subintent_id for c in out.claims["s1"]] == ["s1"] and out.claims["s2"][0].subintent_id == "s2"
    assert out.claims["s2"][0].chunk_ids == ["Doc_01 §7#1"]


async def test_subintents_without_evidence_are_skipped():
    out = await synthesize_subintents(ExtractiveGenerator(), [(SubQuery(id="s1", text="x"), [])])
    assert out.claims == {}


def test_rendering_uses_doc_section_citations():
    claims = [Claim(id="a", text="The cache reuses searches.", subintent_id="s2", chunk_ids=["Doc_01 §7#1"])]
    answer = render_answer(claims)
    assert answer == "The cache reuses searches. [Doc_01 §7]" and extract_citations(answer) == ["Doc_01 §7"]
