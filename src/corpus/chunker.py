"""Section-aware chunker (step 1.4).

* Chunks never cross a section boundary, so every chunk maps to one `Doc_ID §Section`.
* A section body is split into atomic blocks: fenced code blocks, tables, list items and
  paragraphs. Blocks are packed greedily up to `max_tokens`; a block is only split when it
  alone exceeds the budget (tables by whole rows with the header repeated, code by lines,
  paragraphs by sentences). Tables and lists are never split mid-row / mid-item.
* Consecutive chunks of one section overlap by the trailing block(s) of the previous chunk,
  up to `overlap_tokens`.
* Each chunk is prefixed with the document title and section heading.

Token counts are whitespace-delimited words: an approximation that needs no tokenizer
download. Model tokenizers typically produce somewhat more tokens than words.
"""

from __future__ import annotations

import re

from src.corpus.loader import Document, Section
from src.schemas import CorpusChunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(*])")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


def count_tokens(text: str) -> int:
    return len(text.split())


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split a section body into (kind, text) blocks: code, table, item, para."""
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                j += 1
            blocks.append(("code", "\n".join(lines[i : j + 1])))
            i = j + 1
        elif _is_table_line(line):
            j = i
            while j < len(lines) and _is_table_line(lines[j]):
                j += 1
            blocks.append(("table", "\n".join(lines[i:j])))
            i = j
        elif _LIST_ITEM_RE.match(line):
            j = i + 1
            # continuation lines of the same item are indented and not a new item
            while j < len(lines) and lines[j].strip() and not _LIST_ITEM_RE.match(lines[j]) and lines[j].startswith(" "):
                j += 1
            blocks.append(("item", "\n".join(lines[i:j])))
            i = j
        else:
            j = i + 1
            while (
                j < len(lines)
                and lines[j].strip()
                and not _is_table_line(lines[j])
                and not _LIST_ITEM_RE.match(lines[j])
                and not lines[j].lstrip().startswith("```")
            ):
                j += 1
            blocks.append(("para", "\n".join(lines[i:j])))
            i = j
    return blocks


def _split_oversized(kind: str, text: str, max_tokens: int) -> list[str]:
    """Split one block that alone exceeds max_tokens, at safe boundaries only."""
    if kind == "table":
        rows = text.splitlines()
        header, body = rows[:2], rows[2:]
        units = []
        cur: list[str] = []
        for row in body:
            if cur and count_tokens("\n".join(header + cur + [row])) > max_tokens:
                units.append("\n".join(header + cur))
                cur = []
            cur.append(row)
        if cur:
            units.append("\n".join(header + cur))
        return units
    if kind == "code":
        lines = text.splitlines()
        fence = lines[0].strip()
        inner = lines[1:-1] if lines[-1].lstrip().startswith("```") else lines[1:]
        # each piece is re-wrapped in its own fence so no chunk has an unbalanced fence
        return [f"{fence}\n{piece}\n```" for piece in _pack(inner, max_tokens - 2, "\n")]
    return _pack(_SENTENCE_RE.split(text), max_tokens, " ")


def _pack(pieces: list[str], max_tokens: int, sep: str) -> list[str]:
    units, cur = [], []
    for piece in pieces:
        if cur and count_tokens(sep.join(cur + [piece])) > max_tokens:
            units.append(sep.join(cur))
            cur = []
        cur.append(piece)
    if cur:
        units.append(sep.join(cur))
    return units


def chunk_section(
    doc: Document, section: Section, max_tokens: int = 400, overlap_tokens: int = 40
) -> list[CorpusChunk]:
    prefix = f"{doc.title} — §{section.section_id} {section.heading}"
    units: list[str] = []
    for kind, text in split_blocks(section.text):
        if count_tokens(text) > max_tokens:
            units.extend(_split_oversized(kind, text, max_tokens))
        else:
            units.append(text)

    groups: list[list[str]] = []
    cur: list[str] = []
    for unit in units:
        if cur and count_tokens("\n\n".join(cur + [unit])) > max_tokens:
            groups.append(cur)
            # overlap: carry trailing whole units from the previous chunk
            carry: list[str] = []
            for prev in reversed(cur):
                if count_tokens("\n\n".join([prev] + carry)) > overlap_tokens:
                    break
                carry.insert(0, prev)
            cur = carry if count_tokens("\n\n".join(carry + [unit])) <= max_tokens else []
        cur.append(unit)
    if cur:
        groups.append(cur)
    if not groups:  # heading-only section: keep it retrievable by its heading
        groups = [[]]

    chunks = []
    for part, group in enumerate(groups, start=1):
        body = "\n\n".join(group)
        text = f"{prefix}\n\n{body}" if body else prefix
        chunks.append(
            CorpusChunk(
                chunk_id=f"{doc.doc_id} §{section.section_id}#{part}",
                doc_id=doc.doc_id,
                section=section.section_id,
                text=text,
                title=doc.title,
                heading=section.heading,
                part=part,
                n_tokens=count_tokens(text),
            )
        )
    return chunks


def chunk_documents(docs: list[Document], max_tokens: int = 400, overlap_tokens: int = 40) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for doc in docs:
        for section in doc.sections:
            chunks.extend(chunk_section(doc, section, max_tokens, overlap_tokens))
    return chunks


def load_chunks(settings=None) -> list[CorpusChunk]:
    """Load and chunk the configured corpus."""
    from src.config import get_settings
    from src.corpus.loader import load_corpus

    s = settings or get_settings()
    return chunk_documents(load_corpus(s.corpus_dir), s.chunk_max_tokens, s.chunk_overlap_tokens)
