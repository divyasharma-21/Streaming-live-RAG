"""BM25, dense (hashing backend), RRF and hybrid retrieval (step 1.6)."""

import numpy as np
import pytest

from src.config import Settings
from src.corpus.chunker import load_chunks
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, HashingEmbedder, make_embedder
from src.retrieval.factory import build_retrieval
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rrf import rrf_fuse
from src.retrieval.text import tokenize
from src.schemas import CorpusChunk, ScoredChunk


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def stack(chunks, tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")), chunks)


def _ids(hits):
    return [h.chunk.chunk_id for h in hits]


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("What are the Gates and queries?") == ["gate", "query"]


@pytest.mark.parametrize(
    "query, expected_section",
    [
        ("Reciprocal Rank Fusion deduplication reranker", "7"),
        ("citation support fabricated hallucinated document IDs gate", "5"),
        ("presentation_restructure two bullets", "4.3"),
    ],
)
def test_bm25_finds_the_section_that_contains_the_terms(stack, query, expected_section):
    hits = stack.bm25.search(query, k=3)
    assert hits and hits[0].chunk.section == expected_section
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_returns_nothing_for_stopword_only_query(stack):
    assert stack.bm25.search("what is the", k=5) == []


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=256)
    a, b = e.embed(["session state"]), e.embed(["session state"])
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_dense_retrieves_a_chunk_from_its_own_text(stack, chunks):
    target = chunks[5]
    assert stack.dense.search(target.text, k=1)[0].chunk.chunk_id == target.chunk_id


def test_dense_embedding_cache_round_trip(chunks, tmp_path):
    first = DenseRetriever(chunks, HashingEmbedder(), cache_dir=tmp_path)
    second = DenseRetriever(chunks, HashingEmbedder(), cache_dir=tmp_path)
    assert not first.cache_hit and second.cache_hit
    assert np.array_equal(first.matrix, second.matrix)


def test_unknown_dense_backend_is_rejected():
    with pytest.raises(ValueError):
        make_embedder("nope")


def _sc(cid, rank):
    chunk = CorpusChunk(chunk_id=cid, doc_id="D", section=cid, text=cid)
    return ScoredChunk(chunk=chunk, score=0.0, rank=rank, source="x")


def test_rrf_math_and_tie_break():
    fused = rrf_fuse([[_sc("a", 1), _sc("b", 2)], [_sc("b", 1), _sc("c", 2)]], k=60)
    assert _ids(fused) == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert all(h.source == "rrf" for h in fused)


def test_hybrid_fuses_both_retrievers(stack):
    hybrid = HybridRetriever([stack.bm25, stack.dense], rrf_k=60, candidates=10)
    hits = hybrid.search("uncertainty indicator when evidence is insufficient", k=5)
    assert len(hits) == 5
    assert len(set(_ids(hits))) == 5
    bm25_top = stack.bm25.search("uncertainty indicator when evidence is insufficient", k=10)
    assert _ids(hits)[0] in _ids(bm25_top)


def test_build_index_writes_snapshot(tmp_path, monkeypatch):
    import json

    from src.corpus import build_index

    monkeypatch.setenv("PRISM_INDEX_DIR", str(tmp_path))
    assert build_index.main() == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    lines = (tmp_path / "chunks.jsonl").read_text().splitlines()
    assert manifest["chunks"] == len(lines) >= 16
