"""Anti-fragmentation guard (step 3.2), applied to the output of any decomposer.

1. Short-circuit: an utterance with no intent boundary and no coordination ("and", "or",
   comma, semicolon) is a simple single-intent request -> exactly one sub-query (the whole
   utterance), whatever the decomposer proposed.
2. Thin fragments: a sub-query with fewer than `min_content_terms` content words (for
   example "how long can it be") is folded into its neighbour.
3. Near-duplicates: sub-queries whose embedding cosine similarity is >= `merge_similarity`
   are merged.
4. Cap: at most `max_subqueries`; the overflow is folded into the last kept sub-query.

Merged sub-queries keep the id of the first one, so caching and telemetry stay stable.
Cost: one embedding per sub-query (hashing embedder: microseconds; no model call).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.decompose.decomposer import split_clauses
from src.retrieval.dense import Embedder
from src.retrieval.text import content_terms
from src.schemas import SubQuery

_COORDINATION_RE = re.compile(r"\b(and|or)\b|[,;]", re.I)


@dataclass(frozen=True)
class GuardConfig:
    max_subqueries: int = 4
    merge_similarity: float = 0.85
    min_content_terms: int = 2


@dataclass
class GuardResult:
    subqueries: list[SubQuery]
    notes: list[str] = field(default_factory=list)


def is_simple_utterance(transcript: str) -> bool:
    return len(split_clauses(transcript)) <= 1 and not _COORDINATION_RE.search(transcript)


def _merge(a: SubQuery, b: SubQuery) -> SubQuery:
    constraints = list(dict.fromkeys(a.constraints + b.constraints))
    return SubQuery(id=a.id, text=f"{a.text.rstrip(' ?.')} and {b.text}", constraints=constraints)


class AntiFragmentationGuard:
    def __init__(self, embedder: Embedder, config: GuardConfig | None = None):
        self.embedder = embedder
        self.config = config or GuardConfig()
        if self.config.max_subqueries < 1:
            raise ValueError("max_subqueries must be >= 1")

    def apply(self, subqueries: list[SubQuery], transcript: str) -> GuardResult:
        notes: list[str] = []
        subs = list(subqueries)
        if len(subs) <= 1:
            return GuardResult(subs, notes)

        if is_simple_utterance(transcript):
            notes.append(f"short_circuit: {len(subs)} -> 1")
            return GuardResult([SubQuery(id=subs[0].id, text=transcript.strip())], notes)

        # thin fragments fold into the previous sub-query (or the next one if first)
        i = 0
        while len(subs) > 1 and i < len(subs):
            if len(content_terms(subs[i].text)) < self.config.min_content_terms:
                if i == 0:
                    subs[0:2] = [_merge(subs[0], subs[1])]
                else:
                    subs[i - 1:i + 1] = [_merge(subs[i - 1], subs[i])]
                    i -= 1
                notes.append("thin_fragment_merged")
            else:
                i += 1

        # near-duplicate merge by embedding similarity
        merged = True
        while merged and len(subs) > 1:
            merged = False
            vecs = self.embedder.embed([q.text for q in subs])
            sims = vecs @ vecs.T
            np.fill_diagonal(sims, -1.0)
            a, b = np.unravel_index(int(np.argmax(sims)), sims.shape)
            sim = float(sims[a, b])
            if sim >= self.config.merge_similarity:
                a, b = sorted((int(a), int(b)))
                subs[a] = _merge(subs[a], subs[b])
                del subs[b]
                notes.append(f"near_duplicate_merged: sim={sim:.2f}")
                merged = True

        # cap
        cap = self.config.max_subqueries
        while len(subs) > cap:
            subs[cap - 1] = _merge(subs[cap - 1], subs[cap])
            del subs[cap]
            notes.append("capped")
        return GuardResult(subs, notes)
