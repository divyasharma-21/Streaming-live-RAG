"""Claim ledger (step 4.2)."""

import pytest

from src.schemas import Claim, CorpusChunk, ScoredChunk, SubQuery
from src.session.ledger import ClaimLedger


def hit(cid):
    return ScoredChunk(chunk=CorpusChunk(chunk_id=cid, doc_id="Doc_01", section=cid.split("§")[1].split("#")[0],
                                         text="t"), score=1.0, rank=1, source="fusion")


def claim(ledger, sid, text, chunk, constraints=()):
    return Claim(id=ledger.next_claim_id(sid), text=text, subintent_id=sid, chunk_ids=[chunk],
                 version=ledger.version + 1, constraints=list(constraints))


@pytest.fixture
def ledger():
    lg = ClaimLedger()
    lg.upsert_subintent(SubQuery(id="t1.q1", text="rule A"))
    lg.upsert_subintent(SubQuery(id="t1.q2", text="rule B", constraints=["Pune"]))
    lg.add_evidence("t1.q1", [hit("Doc_01 §3#1")])
    lg.add_evidence("t1.q2", [hit("Doc_01 §5#1")])
    lg.add_claims([claim(lg, "t1.q1", "A holds", "Doc_01 §3#1"),
                   claim(lg, "t1.q2", "B holds", "Doc_01 §5#1", ["Pune"])])
    lg.commit_version(None)
    return lg


def test_claim_fields_and_version(ledger):
    c = ledger.claims["t1.q2.c1"]
    assert (c.text, c.subintent_id, c.chunk_ids, c.version, c.constraints) == (
        "B holds", "t1.q2", ["Doc_01 §5#1"], 1, ["Pune"])
    assert ledger.version == 1 and ledger.current.version == 1 and len(ledger.current.claims) == 2


def test_ledger_stores_subqueries_and_evidence(ledger):
    assert [q.id for q in ledger.subqueries()] == ["t1.q1", "t1.q2"]
    assert ledger.add_evidence("t1.q1", [hit("Doc_01 §3#1"), hit("Doc_01 §6#1")]) == ["Doc_01 §6#1"]
    assert [h.chunk.chunk_id for h in ledger.subintents["t1.q1"].evidence] == ["Doc_01 §3#1", "Doc_01 §6#1"]


def test_invalidate_and_diff_touch_only_named_claims(ledger):
    before = ledger.snapshot()
    ledger.invalidate(["t1.q2.c1"])
    ledger.add_claims([claim(ledger, "t1.q2", "B holds in Mumbai", "Doc_01 §5#1", ["Mumbai"])])
    ledger.commit_version("note")
    d = ClaimLedger.diff(before, ledger.snapshot())
    assert d == {"added": ["t1.q2.c2"], "removed": ["t1.q2.c1"], "changed": [], "unchanged": ["t1.q1.c1"]}
    assert ledger.claims["t1.q2.c2"].version == 2 and ledger.claims["t1.q1.c1"].version == 1
    assert [c.id for c in ledger.invalidated] == ["t1.q2.c1"]
    assert [v.version for v in ledger.history] == [1, 2] and ledger.current.uncertainty == "note"


def test_citations_follow_subintent_order(ledger):
    assert ledger.citations() == ["Doc_01 §3", "Doc_01 §5"]


def test_integrity_checks(ledger):
    with pytest.raises(KeyError):
        ledger.add_claims([Claim(id="x", text="t", subintent_id="missing", chunk_ids=["Doc_01 §3#1"])])
    with pytest.raises(ValueError):
        ledger.add_claims([ledger.claims["t1.q1.c1"]])


def test_clear_resets_everything(ledger):
    ledger.clear()
    assert ledger.version == 0 and not ledger.claims and not ledger.subintents and not ledger.history
