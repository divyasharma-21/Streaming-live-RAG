"""Citation handling: `[Doc_ID §Section]` markers in answer text.

A citation is valid only if its (doc id, section) exists in the loaded corpus. Chunk ids
(`Doc_01 §3#1`) map to section-level citations (`Doc_01 §3`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.schemas import CorpusChunk

CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+ §[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)\]")


def chunk_to_citation(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def format_citation(citation: str) -> str:
    return f"[{citation}]"


def extract_citations(text: str) -> list[str]:
    """Citation keys in order of first appearance, without duplicates."""
    seen: dict[str, None] = {}
    for m in CITE_RE.finditer(text):
        seen.setdefault(m.group(1), None)
    return list(seen)


class CitationIndex:
    def __init__(self, chunks: Iterable[CorpusChunk]):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.citations = {c.citation for c in self.chunks.values()}

    def is_valid(self, citation: str) -> bool:
        return citation in self.citations

    def is_valid_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self.chunks

    def check(self, citations: Iterable[str]) -> tuple[list[str], list[str]]:
        valid, invalid = [], []
        for c in citations:
            (valid if self.is_valid(c) else invalid).append(c)
        return valid, invalid
