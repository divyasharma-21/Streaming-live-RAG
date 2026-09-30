"""Reranking interface (step 1.8).

* `lexical` (default): orders candidates by the fraction of the query's content terms that
  appear in the chunk, ties broken by the fused rank. No model, microseconds per call.
* `cross_encoder`: sentence-transformers CrossEncoder (default
  `cross-encoder/ms-marco-MiniLM-L-6-v2`). Optional dependency; not tested in Phase 1.
* `none`: keep the fused order.
"""

from __future__ import annotations

from typing import Protocol

from src.retrieval.text import content_terms
from src.schemas import ScoredChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


def _rerank_by(hits: list[ScoredChunk], scores: list[float], top_k: int, source: str) -> list[ScoredChunk]:
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].rank))[:top_k]
    return [
        ScoredChunk(chunk=hits[i].chunk, score=float(scores[i]), rank=r, source=source)
        for r, i in enumerate(order, start=1)
    ]


class IdentityReranker:
    name = "none"

    def rerank(self, query: str, hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return _rerank_by(hits, [h.score for h in hits], top_k, self.name)


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q = content_terms(query)
        if not q:
            return IdentityReranker().rerank(query, hits, top_k)
        scores = [len(q & content_terms(h.chunk.text)) / len(q) for h in hits]
        return _rerank_by(hits, scores, top_k, self.name)


class CrossEncoderReranker:  # pragma: no cover - optional dependency, not installed in Phase 1
    name = "cross_encoder"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "PRISM_RERANK_BACKEND=cross_encoder needs `pip install -r requirements-optional.txt`"
            ) from exc
        self._model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.text) for h in hits]).tolist()
        return _rerank_by(hits, scores, top_k, self.name)


def make_reranker(backend: str, model_name: str = "") -> Reranker:
    if backend == "lexical":
        return LexicalReranker()
    if backend == "none":
        return IdentityReranker()
    if backend == "cross_encoder":
        return CrossEncoderReranker(model_name)
    raise ValueError(f"unknown rerank backend {backend!r} (expected lexical, cross_encoder or none)")
