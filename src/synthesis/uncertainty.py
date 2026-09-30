"""Uncertainty handling (baseline level).

The evidence score of a chunk is the fraction of the request's content terms that occur in
it (lexical coverage, independent of which reranker is configured). If no retrieved chunk
reaches `min_score`, the system states that the corpus lacks evidence and asks for
clarification instead of answering. Phase 4 adds per-sub-intent notes that also cover
claim-level verification failures (`subintent_uncertainty`) and clarification requests for
turns that could not be searched (`turn_clarification`).
"""

from __future__ import annotations

from src.retrieval.text import content_terms
from src.schemas import ScoredChunk


def evidence_score(query: str, text: str) -> float:
    q = content_terms(query)
    if not q:
        return 0.0
    return len(q & content_terms(text)) / len(q)


def supported_hits(query: str, hits: list[ScoredChunk], min_score: float) -> list[ScoredChunk]:
    return [h for h in hits if evidence_score(query, h.chunk.text) >= min_score]


def uncertainty_note(query: str, hits: list[ScoredChunk], min_score: float) -> str | None:
    """Return an explicit uncertainty / clarification note, or None when evidence suffices."""
    q = content_terms(query)
    if not q:
        return "The request contains no searchable content. Please restate what you would like to know."
    if not hits:
        return "No evidence for this request was found in the corpus. Please clarify or rephrase the request."
    best = max(evidence_score(query, h.chunk.text) for h in hits)
    if best >= min_score:
        return None
    covered = set().union(*(content_terms(h.chunk.text) for h in hits)) & q
    missing = sorted(q - covered)
    detail = f" Terms not found in the retrieved evidence: {', '.join(missing)}." if missing else ""
    return (
        "The corpus does not contain sufficient evidence to answer this request reliably"
        f" (best evidence coverage {best:.2f} < {min_score:.2f}).{detail}"
        " Please clarify or narrow the request."
    )


# ---------------------------------------------------------------------- per sub-intent (Phase 4, step 4.5)


def subintent_uncertainty(
    subquery_text: str,
    query_text: str,
    evidence: list[ScoredChunk],
    min_score: float,
    claims_kept: int,
    claims_failed: int,
    generator_note: str | None = None,
) -> str | None:
    """Explicit note for one sub-intent, or None when it is answered and verified.

    Emitted when the sub-intent has no evidence, its best evidence is below `min_score`,
    some or all of its claims failed grounding verification, or the generator reported a gap."""
    label = f'"{subquery_text.strip().rstrip("?.")}"'
    if not evidence:
        return f"No evidence in the corpus for {label}."
    q = content_terms(query_text)
    best = max((evidence_score(query_text, h.chunk.text) for h in evidence), default=0.0)
    if q and best < min_score:
        covered = set().union(*(content_terms(h.chunk.text) for h in evidence)) & q
        missing = ", ".join(sorted(q - covered))
        return (f"The corpus does not contain sufficient evidence for {label}"
                + (f" (not found: {missing})" if missing else "") + ". Could you clarify or narrow this part?")
    if claims_failed and not claims_kept:
        return f"Statements for {label} could not be verified against the cited corpus sections and were withheld."
    if claims_failed:
        return f"{claims_failed} statement(s) for {label} could not be verified and were withheld."
    if claims_kept == 0:
        return f"The retrieved evidence did not yield a supported answer for {label}."
    return generator_note


def turn_clarification(reason: str, utterance: str, unmatched_terms: list[str] | None = None) -> str:
    """Note for a turn that could not be searched at all (controller suppression)."""
    if reason == "no_corpus_terms":
        terms = ", ".join(unmatched_terms or [])
        return ("The corpus contains no information about this request"
                + (f" ({terms})" if terms else "") + ", so it cannot be answered from the provided documents.")
    return f'The request seems incomplete ("{utterance.strip()}"). What would you like to know?'


def compose_uncertainty(notes: list[str | None]) -> str | None:
    kept = [n for n in dict.fromkeys(n for n in notes if n)]
    return " ".join(kept) if kept else None
