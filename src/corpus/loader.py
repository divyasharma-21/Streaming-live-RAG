"""Corpus loader: parse data/corpus/*.md into documents and sections (step 1.4).

Corpus file format (see data/PROVENANCE.md):

    ---
    doc_id: Doc_01
    title: ...
    ---
    ## [§1] Heading
    body...
    ### [§1.1] Sub-heading
    body...

Section ids are written explicitly in the `[§...]` marker, so they are stable and never
inferred. A section's text is only the lines up to the next section marker (sub-sections
are separate sections).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SECTION_RE = re.compile(r"^(#{1,6})\s+\[§([0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)\]\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s*```")


class CorpusFormatError(ValueError):
    pass


@dataclass
class Section:
    section_id: str
    heading: str
    level: int
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    path: Path
    meta: dict[str, str] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    def section_ids(self) -> list[str]:
        return [s.section_id for s in self.sections]


def _split_front_matter(text: str, path: Path) -> tuple[dict[str, str], list[str]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise CorpusFormatError(f"{path.name}: missing front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise CorpusFormatError(f"{path.name}: unterminated front matter") from exc
    meta = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, lines[end + 1 :]


def parse_document(path: Path) -> Document:
    meta, body = _split_front_matter(path.read_text(encoding="utf-8"), path)
    doc_id = meta.get("doc_id")
    if not doc_id:
        raise CorpusFormatError(f"{path.name}: front matter has no doc_id")
    if doc_id != path.stem:
        raise CorpusFormatError(f"{path.name}: doc_id {doc_id!r} does not match file name")

    doc = Document(doc_id=doc_id, title=meta.get("title", doc_id), path=path, meta=meta)
    current: Section | None = None
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        if current is not None:
            current.text = "\n".join(buf).strip("\n")
            doc.sections.append(current)

    for line in body:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else SECTION_RE.match(line)
        if match:
            flush()
            current = Section(section_id=match.group(2), heading=match.group(3), level=len(match.group(1)), text="")
            buf = []
        elif current is not None:
            buf.append(line)
        elif line.strip():
            raise CorpusFormatError(f"{path.name}: text before the first section marker: {line[:60]!r}")
    flush()

    if in_fence:
        raise CorpusFormatError(f"{path.name}: unterminated code fence")
    ids = doc.section_ids()
    dupes = sorted({s for s in ids if ids.count(s) > 1})
    if dupes:
        raise CorpusFormatError(f"{path.name}: duplicate section ids {dupes}")
    if not ids:
        raise CorpusFormatError(f"{path.name}: no sections")
    return doc


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(Path(corpus_dir).glob("*.md"))
    if not paths:
        raise CorpusFormatError(f"no .md documents in {corpus_dir}")
    docs = [parse_document(p) for p in paths]
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise CorpusFormatError(f"duplicate doc ids in {corpus_dir}")
    return docs
