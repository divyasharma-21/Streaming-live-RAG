"""Multi-intent decomposer (step 3.1).

Input: the transcript so far plus the current live sub-query set. Output: diff operations

    {"op": "add",   "text": "..."}                       new sub-query
    {"op": "keep",  "id": "q1", "text": "..."}           keep q1 (text may be refined)
    {"op": "merge", "ids": ["q1", "q2"], "text": "..."}  fold several into one (keeps the first id)

Live sub-queries that no operation references are dropped. `apply_ops` turns the diff
into the new live set; invalid operations (unknown ids, empty text) are ignored.

Two implementations share this interface:
* `LLMDecomposer`: the playbook design. One LLM call with a JSON schema via
  src/llm/client.py; the prompt is generic and contains no scenario text.
* `RuleDecomposer`: the CPU-only default used when no LLM is configured. It splits at
  generic English intent boundaries (question marks, semicolons, "..., and what/how/...",
  enumerated wh-clauses), treats a leading non-question phrase as context for the clause
  that follows, and diffs the clauses against the live set by term overlap.

Anti-fragmentation (step 3.2) and context carry-over (step 3.3) run after either
decomposer, in src/decompose/dedupe.py.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from src.llm.client import LLMClient, LLMError
from src.retrieval.text import content_terms
from src.schemas import SubQuery
from src.telemetry.logger import RequestTrace

WH = r"(?:what|which|who|whom|whose|when|where|why|how|whether|is|are|does|do|can|could|should)"
REQUEST_VERBS = r"(?:tell|explain|describe|list|summari[sz]e|give|show|compare|define|include|outline|clarify)"
LEAD = rf"(?:{WH}|{REQUEST_VERBS})"
# hard boundaries: a coordinated new question / request, a question mark, a semicolon
_HARD_RE = re.compile(rf"(?:,\s*|\s+)and\s+(?:also\s+)?(?={LEAD}\b)|(?<=\?)\s+|;\s*|,\s*(?:and\s+)?also\s+(?={LEAD}\b)", re.I)
# soft boundary: ", what ..." without a conjunction (enumeration, or a topic phrase before the question)
_SOFT_RE = re.compile(rf",\s*(?={WH}\b)", re.I)
_HAS_LEAD_RE = re.compile(rf"\b{LEAD}\b", re.I)
_LEADING_CONJ_RE = re.compile(r"^(?:(?:and|also|so|then|plus)\b[\s,]*)+", re.I)


class DiffOp(BaseModel):
    op: Literal["add", "keep", "merge"]
    id: str | None = None
    ids: list[str] = Field(default_factory=list)
    text: str = ""


class DecompositionDiff(BaseModel):
    ops: list[DiffOp]


class Decomposer(Protocol):
    name: str

    async def decompose(self, transcript: str, live: list[SubQuery], trace: RequestTrace | None = None) -> list[DiffOp]: ...


@dataclass
class IdGenerator:
    prefix: str = "q"
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    def __call__(self) -> str:
        return f"{self.prefix}{next(self._counter)}"


def apply_ops(live: list[SubQuery], ops: list[DiffOp], new_id: IdGenerator) -> list[SubQuery]:
    """Apply a diff to the live set. Returns the new live set (in operation order)."""
    by_id = {q.id: q for q in live}
    used: set[str] = set()
    out: list[SubQuery] = []
    for op in ops:
        text = op.text.strip()
        if op.op == "add":
            if text:
                out.append(SubQuery(id=new_id(), text=text))
        elif op.op == "keep":
            if op.id in by_id and op.id not in used:
                used.add(op.id)
                old = by_id[op.id]
                changed = bool(text) and text != old.text
                out.append(old.model_copy(update={"text": text, "status": "pending"}) if changed else old)
        elif op.op == "merge":
            ids = [i for i in dict.fromkeys(op.ids) if i in by_id and i not in used]
            if ids and text:
                used.update(ids)
                out.append(SubQuery(id=ids[0], text=text))
    return out


def split_clauses(transcript: str) -> list[str]:
    """Split an utterance into candidate intent clauses (generic English cues only)."""
    text = re.sub(r"\s+", " ", transcript).strip()
    if not text:
        return []
    clauses: list[str] = []
    for hard in _HARD_RE.split(text):
        hard = _LEADING_CONJ_RE.sub("", hard.strip(" ,")).strip(" ,")
        if not hard:
            continue
        pieces = [_LEADING_CONJ_RE.sub("", p.strip(" ,")) for p in _SOFT_RE.split(hard)]
        pieces = [p for p in pieces if p]
        merged: list[str] = []
        for piece in pieces:
            if merged and not _HAS_LEAD_RE.search(merged[-1]):
                merged[-1] = f"{merged[-1]}, {piece}"  # leading topic phrase -> context of this clause
            else:
                merged.append(piece)
        clauses.extend(merged)
    return clauses


def _coverage(of: str, by: str) -> float:
    a = content_terms(of)
    return len(a & content_terms(by)) / len(a) if a else 0.0


def diff_against_live(clauses: list[str], live: list[SubQuery], match_threshold: float = 0.5) -> list[DiffOp]:
    """Keep live sub-queries that a clause extends, merge several into one clause, add the rest."""
    ops: list[DiffOp] = []
    claimed: set[str] = set()
    for clause in clauses:
        matches = [q.id for q in live if q.id not in claimed and _coverage(q.text, clause) >= match_threshold]
        if len(matches) == 1:
            ops.append(DiffOp(op="keep", id=matches[0], text=clause))
        elif len(matches) > 1:
            ops.append(DiffOp(op="merge", ids=matches, text=clause))
        else:
            ops.append(DiffOp(op="add", text=clause))
        claimed.update(matches)
    return ops


class RuleDecomposer:
    name = "rules"

    async def decompose(self, transcript: str, live: list[SubQuery], trace: RequestTrace | None = None) -> list[DiffOp]:
        return diff_against_live(split_clauses(transcript), live)


DECOMPOSER_SYSTEM = (
    "You split a user's spoken request into the distinct questions it contains, so each can be "
    "searched separately. Do not split one question into near-identical parts. Keep each "
    "sub-query self-contained. Reply with JSON only."
)


def decomposer_prompt(transcript: str, live: list[SubQuery]) -> str:
    current = json.dumps([{"id": q.id, "text": q.text} for q in live], ensure_ascii=False)
    return (
        f"Transcript so far: {transcript!r}\n"
        f"Current sub-queries: {current}\n"
        "Return the operations that turn the current sub-queries into the right set for the transcript:\n"
        '{"ops": [{"op": "keep", "id": "<existing id>", "text": "<possibly refined text>"}, '
        '{"op": "add", "text": "<new sub-query>"}, '
        '{"op": "merge", "ids": ["<id>", "<id>"], "text": "<combined sub-query>"}]}\n'
        "Every current sub-query that is still needed must appear in a keep or merge operation."
    )


class LLMDecomposer:
    """One JSON-constrained LLM call per decomposition. Falls back to the rule decomposer on
    an LLM error, so a flaky endpoint never blocks retrieval."""

    name = "llm"

    def __init__(self, client: LLMClient, fallback: RuleDecomposer | None = None):
        self.client = client
        self.fallback = fallback or RuleDecomposer()
        self.calls = 0
        self.fallbacks = 0

    async def decompose(self, transcript: str, live: list[SubQuery], trace: RequestTrace | None = None) -> list[DiffOp]:
        self.calls += 1
        try:
            resp = await self.client.generate(
                decomposer_prompt(transcript, live), schema=DecompositionDiff, system=DECOMPOSER_SYSTEM, trace=trace
            )
        except LLMError:
            self.fallbacks += 1
            return await self.fallback.decompose(transcript, live, trace)
        return resp.parsed.ops


def make_decomposer(kind: str, client: LLMClient | None = None) -> Decomposer:
    if kind == "rules":
        return RuleDecomposer()
    if kind == "llm":
        if client is None:
            raise ValueError("decomposer 'llm' needs an LLM (set PRISM_LLM_PROVIDER / PRISM_LLM_MODEL)")
        return LLMDecomposer(client)
    raise ValueError(f"unknown decomposer {kind!r} (expected rules or llm)")
