"""Sparse retrieval: BM25 (Okapi) over corpus chunks (step 1.6)."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.retrieval.text import tokenize
from src.schemas import CorpusChunk, ScoredChunk


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[CorpusChunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    @property
    def vocabulary(self) -> frozenset[str]:
        """Normalised terms that occur anywhere in the indexed corpus."""
        return frozenset(self._bm25.idf)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        hits = [i for i in order if scores[i] > 0][:k]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]), rank=r, source=self.name)
            for r, i in enumerate(hits, start=1)
        ]
