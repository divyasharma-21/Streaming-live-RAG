"""Hybrid retrieval (step 1.6): BM25 + dense candidate lists fused with RRF."""

from __future__ import annotations

from typing import Protocol

from src.retrieval.rrf import rrf_fuse
from src.schemas import ScoredChunk


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]: ...


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, candidates: int = 20):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        n = max(k, self.candidates)
        rankings = [r.search(query, n) for r in self.retrievers]
        return rrf_fuse(rankings, k=self.rrf_k, top_n=k)
