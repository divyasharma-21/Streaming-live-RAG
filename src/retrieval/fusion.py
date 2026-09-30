"""Evidence fusion across sub-queries (step 3.5).

Each sub-query's results are already RRF-fused (BM25 + dense) by the hybrid retriever.
Across sub-queries:

1. dedupe by chunk id (a chunk hit by several sub-queries is kept once, with all of them
   recorded as its sources);
2. dedupe near-duplicates: chunks whose body token sets have Jaccard >= `near_duplicate`
   keep only the better-scored one;
3. rerank every candidate against each sub-query that retrieved it (configured reranker:
   lexical by default, cross-encoder when enabled) and keep the best score;
4. per-sub-intent quota: each sub-query first gets up to `quota` of its own best
   candidates, then the remaining slots up to `top_k` go to the best remaining scores.
   One intent therefore cannot crowd the others out of the evidence set.

Cost: set operations plus one rerank call per sub-query over at most `candidates` chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.rerank import Reranker
from src.retrieval.text import content_terms
from src.schemas import CorpusChunk, ScoredChunk


@dataclass
class FusionResult:
    evidence: list[ScoredChunk]
    sources: dict[str, list[str]] = field(default_factory=dict)  # chunk id -> sub-query ids
    dropped_duplicates: int = 0
    dropped_near_duplicates: int = 0


def _body_terms(chunk: CorpusChunk) -> set[str]:
    body = chunk.text.split("\n\n", 1)[1] if "\n\n" in chunk.text else chunk.text
    return content_terms(body)


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def fuse_evidence(
    sub_results: dict[str, list[ScoredChunk]],
    sub_queries: dict[str, str],
    reranker: Reranker,
    top_k: int = 6,
    quota: int = 2,
    near_duplicate: float = 0.9,
) -> FusionResult:
    order = [sid for sid in sub_queries if sid in sub_results]
    # 1. dedupe by chunk id, remembering every sub-query that retrieved the chunk
    chunks: dict[str, CorpusChunk] = {}
    sources: dict[str, list[str]] = {}
    total = 0
    for sid in order:
        for hit in sub_results[sid]:
            total += 1
            cid = hit.chunk.chunk_id
            chunks.setdefault(cid, hit.chunk)
            if sid not in sources.setdefault(cid, []):
                sources[cid].append(sid)
    dropped_dupes = total - len(chunks)

    # 3. rerank each candidate against its own sub-queries (best score wins)
    per_sub: dict[str, dict[str, float]] = {}
    for sid in order:
        cands = [ScoredChunk(chunk=chunks[cid], score=0.0, rank=i + 1, source="fusion")
                 for i, cid in enumerate(c for c in chunks if sid in sources[c])]
        ranked = reranker.rerank(sub_queries[sid], cands, len(cands))
        per_sub[sid] = {h.chunk.chunk_id: h.score for h in ranked}
    best = {cid: max(per_sub[s].get(cid, float("-inf")) for s in sources[cid]) for cid in chunks}

    # 2. near-duplicate removal (keep the better-scored chunk)
    kept: list[str] = []
    terms: dict[str, set[str]] = {}
    dropped_near = 0
    for cid in sorted(chunks, key=lambda c: (-best[c], c)):
        terms[cid] = _body_terms(chunks[cid])
        dup_of = next((k for k in kept if _jaccard(terms[cid], terms[k]) >= near_duplicate), None)
        if dup_of is None:
            kept.append(cid)
        else:
            dropped_near += 1
            for s in sources[cid]:
                if s not in sources[dup_of]:
                    sources[dup_of].append(s)

    # 4. per-sub-intent quota, then fill by global score
    selected: list[str] = []
    for sid in order:
        mine = sorted((c for c in kept if sid in sources[c]), key=lambda c: (-per_sub[sid].get(c, float("-inf")), c))
        for cid in [c for c in mine if c not in selected][:quota]:
            if len(selected) < top_k:
                selected.append(cid)
    for cid in kept:
        if len(selected) >= top_k:
            break
        if cid not in selected:
            selected.append(cid)

    final = sorted(selected, key=lambda c: (-best[c], c))
    evidence = [ScoredChunk(chunk=chunks[c], score=float(best[c]), rank=i, source="fusion")
                for i, c in enumerate(final, start=1)]
    return FusionResult(evidence, {c: sources[c] for c in final}, dropped_dupes, dropped_near)
