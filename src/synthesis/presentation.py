"""Presentation-only path (step 4.7): transform the existing answer without any retrieval.

Transforms work on the ledger's active claims, and every output line carries the citations
of the claim it came from, so a transform can drop citations but never add one.
* bullets / numbered list / table / shorter (one statement per sub-intent) / repeat:
  deterministic, no model. A count in the request ("two bullets") limits the output;
  statements are picked round-robin across sub-intents so no intent disappears first.
* translate (and any request the rules do not recognise): needs an LLM. With an LLM the
  rewritten text is accepted only if it introduces no citation absent from the answer;
  otherwise, or without an LLM, the answer is shown unchanged with an explicit note.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from src.llm.client import LLMClient, LLMError
from src.schemas import Claim
from src.synthesis.citations import chunk_to_citation, extract_citations, format_citation
from src.synthesis.generator import neutralise_markers, render_answer

_NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
_COUNT_RE = re.compile(r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\b", re.I)
_KINDS = [
    ("translate", re.compile(r"\btranslat\w*|\bin (?:hindi|french|german|spanish|tamil|telugu|bengali|marathi|"
                             r"kannada|japanese|chinese|arabic|portuguese|italian)\b", re.I)),
    ("table", re.compile(r"\btable\b|\btabular\b|\btabulate\b", re.I)),
    ("bullets", re.compile(r"\bbullet\w*|\bpoints?\b", re.I)),
    ("numbered", re.compile(r"\bnumbered\b|\bnumber them\b|\blist\b", re.I)),
    ("shorter", re.compile(r"\bshort\w*|\bbrief\w*|\bconcise\w*|\bsimpl\w*|\bsummar\w*|\bcondense\w*|"
                           r"\bone sentence\b|\btl;?dr\b|\bin a nutshell\b", re.I)),
    ("repeat", re.compile(r"\brepeat\b|\bagain\b|\brestate\b|\bsay (?:it|that)\b", re.I)),
]


def detect_transform(instruction: str) -> tuple[str, int | None]:
    kind = next((k for k, rx in _KINDS if rx.search(instruction)), "unknown")
    m = _COUNT_RE.search(instruction)
    count = None
    if m:
        tok = m.group(1).lower()
        count = int(tok) if tok.isdigit() else _NUMBER_WORDS[tok]
        count = count if count > 0 else None
    return kind, count


def _cites(claim: Claim) -> str:
    return " ".join(dict.fromkeys(format_citation(chunk_to_citation(c)) for c in claim.chunk_ids))


def _line(claim: Claim) -> str:
    return f"{neutralise_markers(claim.text).rstrip()} {_cites(claim)}".strip()


def _round_robin(claims: list[Claim], limit: int | None) -> list[Claim]:
    groups: dict[str, list[Claim]] = {}
    for c in claims:
        groups.setdefault(c.subintent_id, []).append(c)
    out: list[Claim] = []
    depth = 0
    while any(len(g) > depth for g in groups.values()):
        out += [g[depth] for g in groups.values() if len(g) > depth]
        depth += 1
    return out[:limit] if limit else out


class TransformedText(BaseModel):
    text: str


def transform(kind: str, count: int | None, claims: list[Claim]) -> str:
    if kind == "shorter":
        firsts = {}
        for c in claims:
            firsts.setdefault(c.subintent_id, c)
        chosen = _round_robin(list(firsts.values()), count)
        return " ".join(_line(c) for c in chosen)
    chosen = _round_robin(claims, count)
    if kind == "bullets":
        return "\n".join(f"- {_line(c)}" for c in chosen)
    if kind == "numbered":
        return "\n".join(f"{i}. {_line(c)}" for i, c in enumerate(chosen, 1))
    if kind == "table":
        rows = [f"| {i} | {neutralise_markers(c.text).replace('|', '/')} | {_cites(c)} |" for i, c in enumerate(chosen, 1)]
        return "\n".join(["| # | Statement | Source |", "|---|---|---|", *rows])
    return render_answer(claims)  # repeat


async def present(instruction: str, claims: list[Claim], client: LLMClient | None = None,
                  trace=None) -> tuple[str, str | None]:
    """Return (text, note). Never retrieves; never adds a citation."""
    original = render_answer(claims)
    if not claims:
        return original, "There is no earlier answer to reformat."
    kind, count = detect_transform(instruction)
    if kind not in ("translate", "unknown"):
        return transform(kind, count, claims), None
    if client is None:
        what = "Translation" if kind == "translate" else "This formatting request"
        return original, f"{what} needs a configured LLM; the answer is shown unchanged."
    allowed = set(extract_citations(original))
    prompt = (f"Rewrite the answer below as requested. Keep every [Doc_ID §Section] marker attached to the "
              f"statement it supports; do not add, change or invent any marker or fact.\n\n"
              f"Request: {instruction}\n\nAnswer:\n{original}\n\nReturn JSON: {{\"text\": \"<rewritten answer>\"}}")
    try:
        resp = await client.generate(prompt, schema=TransformedText, trace=trace)
    except LLMError:
        return original, "The reformatting model is unavailable; the answer is shown unchanged."
    text = resp.parsed.text.strip()
    if not text or set(extract_citations(text)) - allowed:
        return original, "The reformatted text changed the citations, so the original answer is shown."
    return text, None
