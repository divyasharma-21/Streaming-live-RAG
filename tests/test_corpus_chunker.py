"""Loader and section-aware chunker (step 1.4)."""

from pathlib import Path

import pytest

from src.corpus.chunker import chunk_documents, chunk_section, count_tokens, split_blocks
from src.corpus.loader import CorpusFormatError, Document, Section, load_corpus, parse_document

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"

EXPECTED_SECTIONS = [
    "0", "1", "1.1", "2", "2.1", "3", "4", "4.1", "4.1.1", "4.1.2", "4.2", "4.3", "5", "6", "7", "8",
]


@pytest.fixture(scope="module")
def docs():
    return load_corpus(CORPUS)


def test_loader_reads_all_sections(docs):
    assert [d.doc_id for d in docs] == ["Doc_01"]
    assert docs[0].section_ids() == EXPECTED_SECTIONS
    assert docs[0].title == "Streaming Live RAG"
    assert all(s.text.strip() for s in docs[0].sections)


def test_section_markers_inside_code_are_ignored(tmp_path):
    p = tmp_path / "Doc_X.md"
    p.write_text("---\ndoc_id: Doc_X\ntitle: T\n---\n## [§1] A\n```\n## [§9] not a heading\n```\n", encoding="utf-8")
    doc = parse_document(p)
    assert doc.section_ids() == ["1"]


@pytest.mark.parametrize(
    "content",
    [
        "no front matter\n## [§1] A\n",
        "---\ntitle: T\n---\n## [§1] A\n",  # no doc_id
        "---\ndoc_id: Doc_X\n---\n## [§1] A\n## [§1] B\n",  # duplicate section
        "---\ndoc_id: Doc_X\n---\nstray text\n## [§1] A\n",
    ],
)
def test_loader_rejects_malformed_documents(tmp_path, content):
    p = tmp_path / "Doc_X.md"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(CorpusFormatError):
        parse_document(p)


def test_chunk_ids_are_stable_and_unique(docs):
    a = [c.chunk_id for c in chunk_documents(docs)]
    b = [c.chunk_id for c in chunk_documents(load_corpus(CORPUS))]
    assert a == b
    assert len(a) == len(set(a))
    assert all(cid.startswith("Doc_01 §") and "#" in cid for cid in a)


def test_every_section_has_a_chunk_with_prefix(docs):
    chunks = chunk_documents(docs)
    assert {c.section for c in chunks} == set(EXPECTED_SECTIONS)
    for c in chunks:
        assert c.text.startswith(f"Streaming Live RAG — §{c.section} {c.heading}")


def test_chunks_respect_budget(docs):
    for c in chunk_documents(docs, max_tokens=400):
        prefix = count_tokens(f"{c.title} — §{c.section} {c.heading}")
        assert c.n_tokens - prefix <= 400


def test_tables_never_split_mid_row(docs):
    for c in chunk_documents(docs, max_tokens=60, overlap_tokens=10):
        in_code = False
        for line in c.text.splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if not in_code and line.lstrip().startswith("|"):
                assert line.rstrip().endswith("|"), f"broken row in {c.chunk_id}: {line!r}"


def test_oversized_table_repeats_header(docs):
    gates = next(s for s in docs[0].sections if s.section_id == "5")
    chunks = chunk_section(docs[0], gates, max_tokens=60, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert "| Gate | Criterion | Target Threshold | Validation Method |" in c.text


def test_oversized_code_block_keeps_fences_balanced(docs):
    diagram = next(s for s in docs[0].sections if s.section_id == "2")
    chunks = chunk_section(docs[0], diagram, max_tokens=40, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.count("```") % 2 == 0, c.chunk_id


def test_list_items_are_atomic():
    blocks = split_blocks("- first item\n  continued\n- second item\n\nA paragraph.")
    assert blocks == [("item", "- first item\n  continued"), ("item", "- second item"), ("para", "A paragraph.")]


def test_small_budget_creates_overlap():
    doc = Document(doc_id="Doc_T", title="T", path=Path("Doc_T.md"))
    body = "\n\n".join(f"Paragraph {i} " + "word " * 8 for i in range(6))
    sec = Section(section_id="1", heading="H", level=2, text=body)
    chunks = chunk_section(doc, sec, max_tokens=25, overlap_tokens=12)
    assert len(chunks) >= 3
    first_tail = chunks[0].text.split("\n\n")[-1]
    assert first_tail in chunks[1].text


def test_no_body_text_lost(docs):
    chunks = chunk_documents(docs)
    joined = "\n".join(c.text for c in chunks)
    for section in docs[0].sections:
        for line in section.text.splitlines():
            if line.strip():
                assert line in joined, line
