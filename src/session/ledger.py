"""Claim ledger (step 4.2): the per-session record of what has been answered and why.

* Claims: id, text, sub-intent id, supporting chunk ids, dependent constraints, version.
* Sub-intents: the current sub-queries (with their carried constraints) and the evidence set
  retrieved for each.
* Versions: every committed answer is an `AnswerVersion` snapshot. Claims keep the version
  in which they were created, so a refinement that keeps a claim keeps its id, version and
  citations unchanged.
* `snapshot()` / `diff()` expose exactly which claims a turn added, removed or kept (used by
  the delta tests to prove an update touched only the affected claims).

The ledger lives inside a `Session` (src/session/store.py) and dies with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import AnswerVersion, Claim, ScoredChunk, SubQuery
from src.synthesis.citations import chunk_to_citation


@dataclass
class SubIntentState:
    subquery: SubQuery
    evidence: list[ScoredChunk] = field(default_factory=list)
    uncertainty: str | None = None
    created_version: int = 1


class ClaimLedger:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.version = 0  # answer version; 0 = nothing answered yet
        self.subintents: dict[str, SubIntentState] = {}
        self.claims: dict[str, Claim] = {}
        self.invalidated: list[Claim] = []
        self.history: list[AnswerVersion] = []
        self._claim_counters: dict[str, int] = {}

    # ------------------------------------------------------------------ sub-intents

    def upsert_subintent(self, subquery: SubQuery) -> SubIntentState:
        state = self.subintents.get(subquery.id)
        if state is None:
            state = self.subintents[subquery.id] = SubIntentState(subquery, created_version=self.version + 1)
        else:
            state.subquery = subquery
        return state

    def add_evidence(self, subintent_id: str, hits: list[ScoredChunk]) -> list[str]:
        """Add hits to a sub-intent's evidence set; returns the chunk ids that were new."""
        state = self.subintents[subintent_id]
        have = {h.chunk.chunk_id for h in state.evidence}
        new = [h for h in hits if h.chunk.chunk_id not in have]
        state.evidence.extend(new)
        return [h.chunk.chunk_id for h in new]

    def subqueries(self) -> list[SubQuery]:
        return [s.subquery for s in self.subintents.values()]

    # ------------------------------------------------------------------ claims

    def next_claim_id(self, subintent_id: str) -> str:
        n = self._claim_counters.get(subintent_id, 0) + 1
        self._claim_counters[subintent_id] = n
        return f"{subintent_id}.c{n}"

    def add_claims(self, claims: list[Claim]) -> None:
        for c in claims:
            if c.subintent_id not in self.subintents:
                raise KeyError(f"claim {c.id} refers to unknown sub-intent {c.subintent_id}")
            if c.id in self.claims:
                raise ValueError(f"duplicate claim id {c.id}")
            self.claims[c.id] = c

    def invalidate(self, claim_ids: list[str]) -> list[Claim]:
        removed = [self.claims.pop(cid) for cid in claim_ids if cid in self.claims]
        self.invalidated.extend(removed)
        return removed

    def claims_for(self, subintent_id: str) -> list[Claim]:
        return [c for c in self.claims.values() if c.subintent_id == subintent_id]

    def active_claims(self) -> list[Claim]:
        """Active claims in sub-intent order, then creation order."""
        order = {sid: i for i, sid in enumerate(self.subintents)}
        added = {cid: i for i, cid in enumerate(self.claims)}  # dicts keep insertion order
        return sorted(self.claims.values(), key=lambda c: (order.get(c.subintent_id, len(order)), added[c.id]))

    def citations(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.active_claims():
            for cid in c.chunk_ids:
                seen.setdefault(chunk_to_citation(cid), None)
        return list(seen)

    # ------------------------------------------------------------------ versions

    def commit_version(self, uncertainty: str | None) -> AnswerVersion:
        self.version += 1
        answer = AnswerVersion(version=self.version, claims=self.active_claims(), uncertainty=uncertainty)
        self.history.append(answer)
        return answer

    @property
    def current(self) -> AnswerVersion | None:
        return self.history[-1] if self.history else None

    def snapshot(self) -> dict[str, tuple]:
        """Immutable view of the active claims: id -> (text, chunk ids, version, constraints, sub-intent)."""
        return {c.id: (c.text, tuple(c.chunk_ids), c.version, tuple(c.constraints), c.subintent_id)
                for c in self.claims.values()}

    @staticmethod
    def diff(before: dict[str, tuple], after: dict[str, tuple]) -> dict[str, list[str]]:
        return {
            "added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
            "unchanged": sorted(k for k in set(before) & set(after) if before[k] == after[k]),
        }
