"""Evidence fusion across sub-queries (step 3.5)."""

from src.retrieval.fusion import fuse_evidence
from src.retrieval.rerank import IdentityReranker, LexicalReranker
from src.schemas import CorpusChunk, ScoredChunk


def hit(cid, body, rank=1, score=1.0):
    chunk = CorpusChunk(chunk_id=cid, doc_id="D", section=cid.split("§")[1].split("#")[0], text=f"T — H\n\n{body}")
    return ScoredChunk(chunk=chunk, score=score, rank=rank, source="rrf")


def test_dedupe_by_chunk_id_records_all_sources():
    r = fuse_evidence({"q1": [hit("D §1#1", "alpha beta")], "q2": [hit("D §1#1", "alpha beta")]},
                      {"q1": "alpha", "q2": "beta"}, LexicalReranker())
    assert [h.chunk.chunk_id for h in r.evidence] == ["D §1#1"]
    assert r.sources["D §1#1"] == ["q1", "q2"] and r.dropped_duplicates == 1


def test_near_duplicates_are_dropped():
    r = fuse_evidence({"q1": [hit("D §1#1", "one two three four five"), hit("D §2#1", "one two three four five")]},
                      {"q1": "one two"}, LexicalReranker(), near_duplicate=0.9)
    assert len(r.evidence) == 1 and r.dropped_near_duplicates == 1


def test_quota_prevents_one_intent_crowding_out_others():
    strong = [hit(f"D §{i}#1", f"alpha beta gamma item{i}", rank=i) for i in range(1, 7)]
    weak = [hit("D §9#1", "delta only", rank=1)]
    subs = {"q1": "alpha beta gamma", "q2": "delta epsilon zeta"}
    with_quota = fuse_evidence({"q1": strong, "q2": weak}, subs, LexicalReranker(), top_k=4, quota=2)
    assert "D §9#1" in [h.chunk.chunk_id for h in with_quota.evidence]
    assert len(with_quota.evidence) == 4
    no_quota = fuse_evidence({"q1": strong, "q2": weak}, subs, LexicalReranker(), top_k=4, quota=0)
    assert "D §9#1" not in [h.chunk.chunk_id for h in no_quota.evidence]


def test_evidence_is_ranked_and_capped():
    r = fuse_evidence({"q1": [hit(f"D §{i}#1", f"word{i} alpha") for i in range(1, 9)]}, {"q1": "alpha"},
                      IdentityReranker(), top_k=5)
    assert [h.rank for h in r.evidence] == [1, 2, 3, 4, 5] and all(h.source == "fusion" for h in r.evidence)


def test_empty_input():
    assert fuse_evidence({}, {}, LexicalReranker()).evidence == []
