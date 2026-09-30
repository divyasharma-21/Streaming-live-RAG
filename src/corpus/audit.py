"""Corpus audit (step 1.5): document/section counts, chunk-length histogram,
duplicate chunks, empty chunks, and chunks outside the target size.

    python -m src.corpus.audit [--json]

Exit code 1 if any empty or duplicate chunk is found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field

from src.config import get_settings
from src.corpus.chunker import chunk_documents, count_tokens
from src.corpus.loader import load_corpus
from src.schemas import CorpusChunk

HIST_BINS = [(0, 49), (50, 99), (100, 199), (200, 299), (300, 399), (400, 10**9)]
SHORT_CHUNK_TOKENS = 30


@dataclass
class AuditReport:
    documents: int
    sections: int
    chunks: int
    tokens_total: int
    tokens_min: int
    tokens_max: int
    tokens_mean: float
    histogram: dict[str, int]
    duplicate_chunks: list[list[str]] = field(default_factory=list)
    empty_chunks: list[str] = field(default_factory=list)
    short_chunks: list[str] = field(default_factory=list)
    oversized_chunks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.duplicate_chunks and not self.empty_chunks


def _body(chunk: CorpusChunk) -> str:
    """Chunk text without the title/heading prefix, whitespace-normalised."""
    body = chunk.text.split("\n\n", 1)[1] if "\n\n" in chunk.text else ""
    return re.sub(r"\s+", " ", body).strip().lower()


def audit_chunks(n_docs: int, n_sections: int, chunks: list[CorpusChunk], max_tokens: int) -> AuditReport:
    sizes = [c.n_tokens for c in chunks]
    hist = {}
    for lo, hi in HIST_BINS:
        label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"
        hist[label] = sum(lo <= s <= hi for s in sizes)

    by_body: dict[str, list[str]] = {}
    for c in chunks:
        by_body.setdefault(_body(c), []).append(c.chunk_id)
    dupes = [ids for body, ids in by_body.items() if body and len(ids) > 1]

    return AuditReport(
        documents=n_docs,
        sections=n_sections,
        chunks=len(chunks),
        tokens_total=sum(sizes),
        tokens_min=min(sizes, default=0),
        tokens_max=max(sizes, default=0),
        tokens_mean=round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        histogram=hist,
        duplicate_chunks=dupes,
        empty_chunks=[c.chunk_id for c in chunks if not _body(c)],
        short_chunks=[c.chunk_id for c in chunks if count_tokens(_body(c)) < SHORT_CHUNK_TOKENS],
        oversized_chunks=[c.chunk_id for c in chunks if count_tokens(_body(c)) > max_tokens],
    )


def run_audit(settings=None) -> AuditReport:
    s = settings or get_settings()
    docs = load_corpus(s.corpus_dir)
    chunks = chunk_documents(docs, s.chunk_max_tokens, s.chunk_overlap_tokens)
    return audit_chunks(len(docs), sum(len(d.sections) for d in docs), chunks, s.chunk_max_tokens)


def format_report(r: AuditReport) -> str:
    width = max(r.histogram.values(), default=1) or 1
    lines = [
        "Corpus audit",
        f"  documents        {r.documents}",
        f"  sections         {r.sections}",
        f"  chunks           {r.chunks}",
        f"  tokens (words)   total={r.tokens_total} min={r.tokens_min} max={r.tokens_max} mean={r.tokens_mean}",
        "  chunk-length histogram (words, incl. title/heading prefix):",
    ]
    for label, n in r.histogram.items():
        lines.append(f"    {label:>9} | {'#' * round(20 * n / width):<20} {n}")
    lines += [
        f"  duplicate chunks {len(r.duplicate_chunks)} {r.duplicate_chunks or ''}",
        f"  empty chunks     {len(r.empty_chunks)} {r.empty_chunks or ''}",
        f"  oversized chunks {len(r.oversized_chunks)} {r.oversized_chunks or ''}",
        f"  short chunks (<{SHORT_CHUNK_TOKENS} body words, informational) {len(r.short_chunks)} {r.short_chunks or ''}",
        f"  status           {'OK' if r.ok else 'PROBLEMS FOUND'}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_audit()
    print(json.dumps(asdict(report), indent=2) if args.json else format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
