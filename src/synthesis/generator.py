"""Answer synthesis for the baseline.

* `ExtractiveGenerator` (used when PRISM_LLM_PROVIDER=none): selects the evidence
  sentences that overlap most with the request and cites each one. It never writes new
  text, so every claim is a verbatim corpus span. It does not reason or summarise.
* `LLMGenerator`: asks the configured LLM (via src/llm/client.py) for JSON claims, each with
  the ids of the evidence chunks supporting it. Claims citing ids outside the supplied
  evidence are dropped. The prompt is generic and contains no scenario-specific text.
* `synthesize_subintents` (Phase 4, step 4.3): claims per sub-intent for the session
  pipeline. With an LLM: one JSON call over all sub-intents, the model is told to use only
  the evidence given for each sub-question, and a claim citing a chunk outside its own
  sub-intent's evidence is dropped. Without an LLM: the extractive generator per sub-intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.retrieval.text import content_terms
from src.schemas import Claim, ScoredChunk, SubQuery
from src.synthesis.citations import CITE_RE, chunk_to_citation, format_citation
from src.telemetry.logger import RequestTrace

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(*])")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}")
_LIST_MARK_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(?:\[[ xX]\]\s+)?")


@dataclass
class Synthesis:
    claims: list[Claim]
    uncertainty: str | None = None
    dropped_claims: int = 0
    notes: list[str] = field(default_factory=list)


def neutralise_markers(text: str) -> str:
    """Unwrap citation-like markers inside claim text ("[Doc_ID §Section]" -> "Doc_ID §Section")
    so the only bracketed citations in an answer are the ones the pipeline attached."""
    return CITE_RE.sub(r"\1", text)


def render_answer(claims: list[Claim]) -> str:
    parts = []
    for claim in claims:
        cites = []
        for cid in claim.chunk_ids:
            c = format_citation(chunk_to_citation(cid))
            if c not in cites:
                cites.append(c)
        parts.append(f"{neutralise_markers(claim.text).rstrip()} {' '.join(cites)}".strip())
    return "\n".join(parts)


def _clean(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _units(chunk_text: str) -> list[str]:
    """Candidate evidence units: sentences of paragraphs, list items, and table rows
    (each row labelled with its column headers; header rows are not units)."""
    body = chunk_text.split("\n\n", 1)[1] if "\n\n" in chunk_text else chunk_text
    units: list[str] = []
    in_code = False
    header: list[str] | None = None
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        is_row = not in_code and line.lstrip().startswith("|")
        if not is_row:
            header = None
        if not line.strip() or _TABLE_SEP_RE.match(line):
            continue
        if in_code:
            units.append(line.strip())
        elif is_row:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
                continue
            pairs = zip(header, cells) if len(header) == len(cells) else (("", c) for c in cells)
            units.append("; ".join(f"{h}: {c}" if h else c for h, c in pairs if c))
        else:
            units.extend(s for s in _SENT_RE.split(_LIST_MARK_RE.sub("", line).strip()) if s)
    return [_clean(u) for u in units if _clean(u)]


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_claims: int = 4, per_chunk: int = 2):
        self.max_claims = max_claims
        self.per_chunk = per_chunk

    async def synthesize(self, query: str, evidence: list[ScoredChunk], trace: RequestTrace | None = None) -> Synthesis:
        q = content_terms(query)
        claims: list[Claim] = []
        for hit in evidence:
            units = _units(hit.chunk.text)
            scored = []
            for pos, unit in enumerate(units):
                overlap = len(q & content_terms(unit))
                if overlap:
                    scored.append((-overlap, pos, unit))
            if not scored and q & content_terms(hit.chunk.heading):
                # the request matches the section heading, not its sentences: use the leading units
                scored = [(0, pos, unit) for pos, unit in enumerate(units)]
            for _, _, unit in sorted(scored)[: self.per_chunk]:
                claims.append(
                    Claim(id=f"c{len(claims) + 1}", text=unit, subintent_id="q1", chunk_ids=[hit.chunk.chunk_id])
                )
                if len(claims) >= self.max_claims:
                    return Synthesis(claims=claims)
        return Synthesis(claims=claims)


class LLMClaim(BaseModel):
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    claims: list[LLMClaim]
    uncertainty: str | None = None


SYSTEM_PROMPT = (
    "You answer questions strictly from the evidence chunks provided. Do not use any other "
    "knowledge. Every claim must cite the ids of the evidence chunks that support it. If the "
    "evidence does not answer the request or part of it, say what is missing in 'uncertainty' "
    "instead of guessing. Reply with JSON only."
)


def build_prompt(query: str, evidence: list[ScoredChunk]) -> str:
    blocks = "\n\n".join(f"<chunk id=\"{h.chunk.chunk_id}\">\n{h.chunk.text}\n</chunk>" for h in evidence)
    return (
        f"Evidence:\n{blocks}\n\n"
        f"Request: {query}\n\n"
        'Return JSON: {"claims": [{"text": "<one factual statement>", "chunk_ids": ["<id from evidence>"]}], '
        '"uncertainty": "<what the evidence does not cover, or null>"}'
    )


class LLMGenerator:
    name = "llm"

    def __init__(self, client: LLMClient):
        self.client = client

    async def synthesize(self, query: str, evidence: list[ScoredChunk], trace: RequestTrace | None = None) -> Synthesis:
        allowed = {h.chunk.chunk_id for h in evidence}
        resp = await self.client.generate(build_prompt(query, evidence), schema=LLMAnswer, system=SYSTEM_PROMPT, trace=trace)
        out: LLMAnswer = resp.parsed
        claims, dropped = [], 0
        for c in out.claims:
            ids = [cid for cid in c.chunk_ids if cid in allowed]
            if not ids or not c.text.strip():
                dropped += 1
                continue
            claims.append(Claim(id=f"c{len(claims) + 1}", text=c.text.strip(), subintent_id="q1", chunk_ids=ids))
        notes = [f"dropped {dropped} claim(s) without valid evidence ids"] if dropped else []
        uncertainty = out.uncertainty.strip() if out.uncertainty and out.uncertainty.strip() else None
        return Synthesis(claims=claims, uncertainty=uncertainty, dropped_claims=dropped, notes=notes)


# ---------------------------------------------------------------------- per-sub-intent synthesis (4.3)

SubIntentInput = tuple[SubQuery, list[ScoredChunk]]


@dataclass
class SubIntentSynthesis:
    claims: dict[str, list[Claim]]  # sub-intent id -> claims (temporary ids; the ledger assigns final ids)
    uncertainty: dict[str, str] = field(default_factory=dict)  # sub-intent id -> note from the generator
    dropped_claims: int = 0


class LLMSubClaim(BaseModel):
    subintent_id: str
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class LLMSubUncertainty(BaseModel):
    subintent_id: str
    note: str


class LLMSubAnswer(BaseModel):
    claims: list[LLMSubClaim]
    uncertain: list[LLMSubUncertainty] = Field(default_factory=list)


SUBINTENT_SYSTEM_PROMPT = (
    "You answer a user's request, split into sub-questions, strictly from the evidence chunks "
    "given for each sub-question. Use only that evidence; do not use any other knowledge. Every "
    "claim is one factual statement, belongs to one sub-question, and cites the ids of the "
    "evidence chunks (from that sub-question) that support it. If the evidence for a "
    "sub-question is missing or insufficient, add it to 'uncertain' with a short note instead "
    "of guessing. Reply with JSON only."
)


def build_subintent_prompt(items: list[SubIntentInput]) -> str:
    parts = []
    for sub, evidence in items:
        blocks = "\n".join(f'<chunk id="{h.chunk.chunk_id}">\n{h.chunk.text}\n</chunk>' for h in evidence)
        constraints = f" (context: {', '.join(sub.constraints)})" if sub.constraints else ""
        parts.append(f'<subquestion id="{sub.id}">{sub.text}{constraints}\n{blocks or "(no evidence)"}\n</subquestion>')
    return (
        "\n\n".join(parts)
        + '\n\nReturn JSON: {"claims": [{"subintent_id": "<sub-question id>", "text": "<one factual statement>", '
        '"chunk_ids": ["<id from that sub-question\'s evidence>"]}], '
        '"uncertain": [{"subintent_id": "<id>", "note": "<what the evidence does not cover>"}]}'
    )


def _search_text(sub: SubQuery) -> str:
    return " ".join([sub.text, *sub.constraints]).strip()


async def synthesize_subintents(generator, items: list[SubIntentInput], trace: RequestTrace | None = None,
                                query_override: dict[str, str] | None = None) -> SubIntentSynthesis:
    """Claims per sub-intent. LLMGenerator: one JSON call covering all sub-intents; any claim
    citing a chunk outside its own sub-intent's evidence is dropped. ExtractiveGenerator:
    verbatim evidence units per sub-intent. `query_override` lets a delta turn rank evidence
    by the new constraint instead of the original sub-query."""
    items = [(s, ev) for s, ev in items if ev]
    out = SubIntentSynthesis(claims={s.id: [] for s, _ in items})
    if not items:
        return out
    if isinstance(generator, LLMGenerator):
        resp = await generator.client.generate(build_subintent_prompt(items), schema=LLMSubAnswer,
                                               system=SUBINTENT_SYSTEM_PROMPT, trace=trace)
        parsed: LLMSubAnswer = resp.parsed
        allowed = {s.id: {h.chunk.chunk_id for h in ev} for s, ev in items}
        for c in parsed.claims:
            ids = [cid for cid in c.chunk_ids if cid in allowed.get(c.subintent_id, set())]
            if c.subintent_id not in allowed or not ids or not c.text.strip():
                out.dropped_claims += 1
                continue
            bucket = out.claims[c.subintent_id]
            bucket.append(Claim(id=f"tmp{len(bucket) + 1}", text=c.text.strip(), subintent_id=c.subintent_id,
                                chunk_ids=ids))
        for u in parsed.uncertain:
            if u.subintent_id in allowed and u.note.strip():
                out.uncertainty[u.subintent_id] = u.note.strip()
        return out
    for sub, evidence in items:
        query = (query_override or {}).get(sub.id) or _search_text(sub)
        synth = await generator.synthesize(query, evidence, trace=trace)
        out.claims[sub.id] = [c.model_copy(update={"subintent_id": sub.id}) for c in synth.claims]
        if synth.uncertainty:
            out.uncertainty[sub.id] = synth.uncertainty
    return out
