"""Speculative reuse cache (step 3.6).

Remembers the searches issued during one utterance. If a later sub-query (typically the
final version of a sub-query first searched provisionally) is close to an earlier query
(content-term Jaccard >= `similarity`), the earlier search is reused instead of searching
again, and the hit is logged. Entries hold the retrieval run itself, so a hit on a search
that is still in flight simply waits for its results.

The cache is created per utterance by the engine and discarded afterwards: no state
survives the turn, let alone the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.text import content_terms


def query_similarity(a: str, b: str) -> float:
    ta, tb = content_terms(a), content_terms(b)
    if not ta and not tb:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class CacheHit:
    query: str
    matched_query: str
    similarity: float
    entry: Any


@dataclass
class SpeculativeCache:
    similarity: float = 0.75
    entries: list[tuple[str, Any]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity <= 1.0:
            raise ValueError("cache similarity must be in (0, 1]")

    def put(self, query: str, entry: Any) -> None:
        self.entries.append((query, entry))

    def lookup(self, query: str) -> CacheHit | None:
        best: CacheHit | None = None
        for cached_query, entry in self.entries:
            sim = query_similarity(query, cached_query)
            if sim >= self.similarity and (best is None or sim > best.similarity):
                best = CacheHit(query, cached_query, sim, entry)
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
        return best
