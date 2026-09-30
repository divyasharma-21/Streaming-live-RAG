"""Stability probe (step 2.3).

On each partial transcript: run a cheap BM25 probe (top-k chunk ids), compute the Jaccard
overlap with the previous probe's top-k, and check that the fragment has enough content.

    stability_score = Jaccard(top_k(prev), top_k(now))   if the fragment is sufficient
                    = 0                                  otherwise (or on the first probe)

Content sufficiency: at least `min_tokens` non-filler tokens and at least
`min_content_terms` slots/entities, where a slot is a content term that exists in the corpus
vocabulary and an entity is a number or a capitalised word inside the sentence.
`incomplete` flags fragments that visibly continue (trailing ellipsis/comma, or ending on a
function word such as "in", "and", "the", unless the fragment ends with . ? or !). All word lists are generic English.

Cost: one BM25 scoring pass over the corpus per chunk (well under 1 ms on this corpus).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.text import tokenize

FILLERS = frozenset("um umm uh uhh er erm hmm mm ah oh okay ok yeah well basically actually wait".split())
DANGLING = frozenset(
    """a an the to of in on at for with from by and or but nor about into onto than as so if
    that which what who how where when why i we you my our your their is are was were be am
    need want like can could would should will shall do does did has have had its his her
    this these those some any please also then because while""".split()
)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def surface_tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text) if w.lower() not in FILLERS]


def is_incomplete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(("…", "...", ",", ";", ":", "-", "—")):
        return True
    if stripped.endswith((".", "?", "!")):
        return False  # sentence-final punctuation closes the fragment
    words = _WORD_RE.findall(stripped)
    return not words or words[-1].lower() in DANGLING


def entities(text: str) -> list[str]:
    """Numbers and capitalised words that are not sentence-initial."""
    out = []
    for sentence in re.split(r"[.!?]\s+", text):
        words = _WORD_RE.findall(sentence)
        for i, w in enumerate(words):
            if any(ch.isdigit() for ch in w) or (i > 0 and w[0].isupper() and w.lower() not in FILLERS and w != "I"):
                out.append(w)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ProbeResult:
    ts: float
    top_ids: tuple[str, ...]
    jaccard: float
    slots: tuple[str, ...]
    entities: tuple[str, ...]
    n_tokens: int
    incomplete: bool
    sufficient: bool
    stability_score: float
    first_probe: bool = False


@dataclass
class StabilityProbe:
    bm25: BM25Retriever
    k: int = 3
    min_content_terms: int = 2
    min_tokens: int = 3
    history: list[ProbeResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._vocab = self.bm25.vocabulary

    def reset(self) -> None:
        self.history = []

    def content(self, transcript: str) -> tuple[list[str], list[str], int]:
        slots = [t for t in dict.fromkeys(tokenize(transcript)) if t in self._vocab]
        ents = list(dict.fromkeys(entities(transcript)))
        return slots, ents, len(surface_tokens(transcript))

    def probe(self, transcript: str, ts: float) -> ProbeResult:
        slots, ents, n_tokens = self.content(transcript)
        sufficient = n_tokens >= self.min_tokens and (len(slots) + len(ents)) >= self.min_content_terms
        top = tuple(h.chunk.chunk_id for h in self.bm25.search(transcript, k=self.k)) if slots else ()
        prev = self.history[-1] if self.history else None
        j = jaccard(set(prev.top_ids), set(top)) if prev is not None else 0.0
        result = ProbeResult(
            ts=ts,
            top_ids=top,
            jaccard=round(j, 4),
            slots=tuple(slots),
            entities=tuple(ents),
            n_tokens=n_tokens,
            incomplete=is_incomplete(transcript),
            sufficient=sufficient,
            stability_score=round(j, 4) if sufficient else 0.0,
            first_probe=prev is None,
        )
        self.history.append(result)
        return result
