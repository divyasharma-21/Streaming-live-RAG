"""Delta engine (step 4.6): classify a follow-up input against the session's ledger.

The follow-up utterance is decomposed into clauses (the Phase 3 planner) and every clause is
classified against the current sub-intents:

* `contradiction`   - a negation cue ("not", "instead of", "rather than", ...) directly
                      precedes a term that an existing sub-intent contains. Target: the most
                      recent sub-intent whose text or constraints contain it (what the user
                      said), else whose claims contain it (what was answered).
* `new_subintent`   - the clause is a question or request (wh-word, request verb, "?") that
                      brings content words no existing sub-intent contains.
* `parameter_update`- otherwise: a statement or constraint about existing material ("the
                      trip was international", "only phase two"). Target: the sub-intent with
                      the largest content overlap (sub-query text, constraints and claims); if
                      there is no overlap, the most recent sub-intent when the clause refers back
                      ("it", "that", "also", ...) or when only one sub-intent exists; else the
                      clause becomes a new sub-intent.
* `presentation_only` is decided for the whole turn by the Phase 2 suppression gate.

The turn's overall kind is the highest-priority clause kind
(contradiction > parameter_update > new_subintent). All cue lists are generic English.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.retrieval.text import content_terms
from src.session.ledger import ClaimLedger

DeltaKind = Literal["new_subintent", "parameter_update", "contradiction", "presentation_only"]
PRIORITY = {"contradiction": 3, "parameter_update": 2, "new_subintent": 1, "presentation_only": 0}

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|isn't|wasn't|aren't|weren't|don't|doesn't|didn't|instead\s+of|rather\s+than|except|excluding)"
    r"\s+(?:the\s+|a\s+|an\s+|for\s+|in\s+|about\s+)?([\w-]+(?:\s+[\w-]+)?)",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(?:and\s+|also\s+|so\s+|but\s+)?(?:specifically\s+)?"
    r"(?:what|which|who|whom|whose|when|where|why|how|is|are|does|do|can|could|should|would|will|"
    r"tell|explain|describe|list|give|show|compare|define|include|summari[sz]e|outline|clarify)\b",
    re.I,
)
_REFERS_BACK_RE = re.compile(r"\b(it|its|that|this|those|these|they|them|also|too|as well|same|actually)\b", re.I)


@dataclass
class ClauseDelta:
    clause: str
    subquery_id: str
    kind: DeltaKind
    target: str | None = None  # affected existing sub-intent id
    negated_terms: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class DeltaDecision:
    kind: DeltaKind
    clauses: list[ClauseDelta] = field(default_factory=list)
    reason: str = ""

    @property
    def affected(self) -> list[str]:
        return list(dict.fromkeys(c.target for c in self.clauses if c.target))


def _subintent_terms(ledger: ClaimLedger, sid: str) -> set[str]:
    state = ledger.subintents[sid]
    text = " ".join([state.subquery.text, *state.subquery.constraints, *(c.text for c in ledger.claims_for(sid))])
    return content_terms(text)


def negated_terms(clause: str) -> list[str]:
    """Content terms right after a negation cue, minus terms the rest of the clause also uses
    ("not the X rule, I meant the Y rule" negates X, not "rule")."""
    out: list[str] = []
    rest = clause
    for m in _NEGATION_RE.finditer(clause):
        out += [t for t in content_terms(m.group(1)) if t not in out]
        rest = rest.replace(m.group(0), " ")
    kept = content_terms(rest)
    return [t for t in out if t not in kept]


class DeltaClassifier:
    def __init__(self, min_overlap: float = 0.2):
        self.min_overlap = min_overlap

    def classify_clause(self, clause: str, subquery_id: str, ledger: ClaimLedger) -> list[ClauseDelta]:
        """One entry per affected sub-intent (a contradiction can affect several)."""
        sids = list(ledger.subintents)
        if not sids:
            return [ClauseDelta(clause, subquery_id, "new_subintent", reason="no prior sub-intents")]
        terms = content_terms(clause)
        per_sub = {sid: _subintent_terms(ledger, sid) for sid in sids}

        negated = negated_terms(clause)
        if negated:
            # every sub-intent whose stated text/constraints hold a negated term; if none, whose claims do
            stated = {sid: content_terms(" ".join([ledger.subintents[sid].subquery.text,
                                                   *ledger.subintents[sid].subquery.constraints])) for sid in sids}
            for pool in (stated, per_sub):
                hits = {sid: [t for t in negated if t in pool[sid]] for sid in sids}
                affected = [sid for sid in sids if hits[sid]]
                if affected:
                    return [ClauseDelta(clause, subquery_id, "contradiction", sid, hits[sid],
                                        f"negates {', '.join(hits[sid])}") for sid in affected]

        known = set().union(*per_sub.values())
        new_terms = terms - known
        if _QUESTION_RE.search(clause) and new_terms:
            return [ClauseDelta(clause, subquery_id, "new_subintent",
                                reason=f"question with new terms: {', '.join(sorted(new_terms)[:5])}")]

        overlap = {sid: len(terms & per_sub[sid]) / len(terms) for sid in sids} if terms else {}
        best = max(overlap, key=lambda s: (overlap[s], sids.index(s)), default=None)
        if best is not None and overlap[best] >= self.min_overlap:
            return [ClauseDelta(clause, subquery_id, "parameter_update", best, reason=f"overlap {overlap[best]:.2f}")]
        if _REFERS_BACK_RE.search(clause) or len(sids) == 1:
            return [ClauseDelta(clause, subquery_id, "parameter_update", sids[-1], reason="refers back to prior answer")]
        return [ClauseDelta(clause, subquery_id, "new_subintent", reason="no overlap with prior sub-intents")]

    def classify(self, clauses: list[tuple[str, str]], ledger: ClaimLedger, presentation_only: bool = False) -> DeltaDecision:
        """`clauses`: (clause text, sub-query id) pairs from the planner, in order."""
        if presentation_only:
            return DeltaDecision("presentation_only", [], "suppression gate: presentation_restructure")
        decided = [d for text, qid in clauses for d in self.classify_clause(text, qid, ledger)]
        if not decided:
            return DeltaDecision("new_subintent", [], "no clauses")
        top = max(decided, key=lambda c: PRIORITY[c.kind])
        return DeltaDecision(top.kind, decided, "; ".join(f"{c.kind}: {c.reason}" for c in decided))
