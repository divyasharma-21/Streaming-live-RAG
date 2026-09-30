"""Context carry-over (step 3.3).

Extract shared entities and constraints from each sub-query (quantities with units, dates
and relative time words, locations after a preposition, other multi-word proper names) and
inject them into the sub-queries that follow, so a later clause such as "and the refund
terms" is still searched with the location and head-count stated earlier.

Propagation is forward only: a constraint stated in clause i is carried into clauses
i+1..n, never back into earlier clauses. A later sub-query that refers back with a pronoun
("... and how is it handled") also receives the previous sub-query's topic terms.
All patterns are generic English; nothing is corpus- or scenario-specific.
"""

from __future__ import annotations

import re

from src.retrieval.text import tokenize
from src.schemas import SubQuery

UNITS = (
    "people|persons|person|attendees|guests|participants|users|employees|staff|seats|rooms|days|nights|"
    "hours|hour|minutes|minute|seconds|second|ms|weeks|week|months|month|years|year|items|pages|page|words|"
    "chunks|tokens|percent|%|km|miles|kg|usd|dollars|euros|rupees|inr"
)
_QUANTITY_RE = re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{UNITS})\b|\b\d+(?:[.,]\d+)?%", re.I)
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_DATE_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})(?:,?\s+\d{{4}})?|(?:{_MONTHS})(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?(?:,?\s+\d{{4}})?|"
    rf"(?:next|last|this)\s+(?:week|month|year|{_DAYS})|{_DAYS}|today|tomorrow|yesterday|tonight)\b",
    re.I,
)
_LOCATION_RE = re.compile(r"\b(?:in|at|from|to|near|around)\s+((?:[A-Z][\w-]+)(?:\s+[A-Z][\w-]+)*)")
_PROPER_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-Z][\w-]*(?:\s+[A-Z0-9][\w-]*)+)\b")
_PRONOUN_RE = re.compile(r"\b(it|its|this|that|they|them|their|those|these)\b", re.I)
_SENTENCE_START_WORDS = {"I", "What", "Which", "How", "Why", "When", "Where", "Who", "Tell", "Explain", "And", "Also"}


def extract_constraints(text: str) -> list[str]:
    found: list[str] = []
    for rx in (_QUANTITY_RE, _DATE_RE):
        found += [m.group(0).strip() for m in rx.finditer(text)]
    found += [m.group(1).strip() for m in _LOCATION_RE.finditer(text)]
    for m in _PROPER_RE.finditer(text):
        phrase = m.group(1).strip()
        if phrase.split()[0] not in _SENTENCE_START_WORDS:
            found.append(phrase)
    out: list[str] = []
    for c in found:
        low = c.lower()
        if not any(low in o.lower() or o.lower() in low for o in out):
            out.append(c)
    return out


def _mentions(text: str, constraint: str) -> bool:
    return constraint.lower() in text.lower()


def carry_context(subqueries: list[SubQuery], max_topic_terms: int = 4) -> list[SubQuery]:
    """Return copies of the sub-queries with forward-propagated constraints filled in."""
    carried: list[str] = []
    out: list[SubQuery] = []
    previous: SubQuery | None = None
    for q in subqueries:
        own = extract_constraints(q.text)
        inject = [c for c in carried if not _mentions(q.text, c)]
        if previous is not None and _PRONOUN_RE.search(q.text):
            own_terms = set(tokenize(q.text))
            topic = [t for t in dict.fromkeys(tokenize(previous.text)) if t not in own_terms][:max_topic_terms]
            if topic:
                inject.append(" ".join(topic))
        constraints = list(dict.fromkeys(q.constraints + inject))
        new = q.model_copy(update={"constraints": constraints})
        out.append(new)
        for c in own:
            if c not in carried:
                carried.append(c)
        previous = q
    return out


def search_text(q: SubQuery) -> str:
    """Text used for retrieval: the sub-query plus its carried constraints."""
    return " ".join([q.text, *q.constraints]).strip()
