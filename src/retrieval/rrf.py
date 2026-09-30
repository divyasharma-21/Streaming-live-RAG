"""Reciprocal Rank Fusion (step 1.6): score(d) = sum over rankings of 1 / (k + rank(d))."""

from __future__ import annotations

from src.schemas import CorpusChunk, ScoredChunk


def rrf_fuse(rankings: list[list[ScoredChunk]], k: int = 60, top_n: int | None = None) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, CorpusChunk] = {}
    for ranking in rankings:
        for hit in ranking:
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + hit.rank)
            chunks[cid] = hit.chunk
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    if top_n is not None:
        order = order[:top_n]
    return [ScoredChunk(chunk=chunks[cid], score=scores[cid], rank=r, source="rrf") for r, cid in enumerate(order, 1)]
