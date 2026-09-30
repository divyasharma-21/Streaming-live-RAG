"""Grounding verifier (step 4.4)."""

import json

import pytest

from src.corpus.chunker import load_chunks
from src.llm.client import LLMError, MockLLMClient
from src.schemas import Claim
from src.synthesis.citations import CitationIndex
from src.synthesis.grounding import (
    GroundingVerifier,
    LexicalSupportJudge,
    LLMSupportJudge,
    make_judge,
)


@pytest.fixture(scope="module")
def index():
    return CitationIndex(load_chunks())


def c(text, *ids, sid="s1"):
    return Claim(id="x", text=text, subintent_id=sid, chunk_ids=list(ids))


async def test_supported_claim_is_kept(index):
    v = GroundingVerifier(index, LexicalSupportJudge(0.8))
    claim = c("Memory is strictly ephemeral and scoped to the active conversation session.", "Doc_01 §3#1")
    report = await v.verify([claim])
    assert report.kept == [claim] and report.failed == [] and report.checks[0].verdict.score == 1.0


async def test_unsupported_claim_is_dropped(index):
    report = await GroundingVerifier(index, LexicalSupportJudge(0.8)).verify(
        [c("Venue B offers vegan catering for 200 guests.", "Doc_01 §3#1")])
    assert report.kept == [] and report.failed[0].failure == "unsupported"
    assert "vegan" in report.failed[0].verdict.reason


async def test_fabricated_ids_are_removed_or_fail(index):
    v = GroundingVerifier(index, LexicalSupportJudge(0.8))
    only_fake = await v.verify([c("Anything.", "Doc_999 §1#1")])
    assert only_fake.failed[0].failure == "fabricated_id" and only_fake.fabricated_ids == ["Doc_999 §1#1"]
    mixed = await v.verify([c("Memory is strictly ephemeral.", "Doc_01 §3#1", "Doc_12 §2#1")])
    assert mixed.kept[0].chunk_ids == ["Doc_01 §3#1"] and mixed.fabricated_ids == ["Doc_12 §2#1"]


async def test_flag_mode_keeps_but_records(index):
    report = await GroundingVerifier(index, LexicalSupportJudge(0.8), mode="flag").verify(
        [c("Venue B offers vegan catering.", "Doc_01 §3#1")])
    assert len(report.kept) == 1 and report.failed_subintents() == {"s1"}


async def test_wrong_citation_fails_lexical_support(index):
    """The same true sentence cited to an unrelated section is not supported."""
    report = await GroundingVerifier(index, LexicalSupportJudge(0.8)).verify(
        [c("Memory is strictly ephemeral and scoped to the active conversation session.", "Doc_01 §7#1")])
    assert report.kept == []


async def test_llm_judge_with_mock_and_error_handling():
    ok = LLMSupportJudge(MockLLMClient([json.dumps({"supported": True, "reason": "stated"})]))
    assert (await ok.judge("claim", "evidence")).supported

    def boom(prompt):
        raise LLMError("down")

    bad = LLMSupportJudge(MockLLMClient(boom))
    v = await bad.judge("claim", "evidence")
    assert not v.supported and "judge error" in v.reason


def test_make_judge_and_mode_validation(index):
    assert make_judge("lexical").name == "lexical"
    with pytest.raises(ValueError):
        make_judge("llm", None)
    with pytest.raises(ValueError):
        make_judge("other")
    with pytest.raises(ValueError):
        GroundingVerifier(index, LexicalSupportJudge(), mode="ignore")
