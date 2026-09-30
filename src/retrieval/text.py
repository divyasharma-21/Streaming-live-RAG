"""Shared, dependency-free text normalisation for sparse retrieval and lexical scoring."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Generic English function words. Contains no corpus- or benchmark-specific terms.
STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from into onto over under
    about as is are was were be been being am do does did done have has had having it its this
    that these those there here what which who whom whose when where why how i me my we our you
    your he she they them their his her not no nor so than too very can could should would may
    might must will shall just also any all each some such only own same other more most few
    both up down out off again further once s t please tell give show explain describe list
    know want need like""".split()
)


def _stem(token: str) -> str:
    """Very light plural folding ('gates' -> 'gate', 'queries' -> 'query')."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, stopwords removed, plurals folded."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    return set(tokenize(text))
