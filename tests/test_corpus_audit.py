"""Corpus audit (step 1.5)."""

from src.corpus.audit import audit_chunks, format_report, run_audit
from src.schemas import CorpusChunk


def _chunk(cid, section, body):
    return CorpusChunk(chunk_id=cid, doc_id="Doc_T", section=section, text=f"T — §{section} H\n\n{body}".rstrip(),
                       n_tokens=len(body.split()) + 4)


def test_real_corpus_audit_is_clean():
    report = run_audit()
    assert report.documents == 1
    assert report.sections == 16
    assert report.chunks >= report.sections
    assert report.ok, format_report(report)
    assert not report.oversized_chunks


def test_audit_detects_duplicates_and_empties():
    chunks = [
        _chunk("Doc_T §1#1", "1", "same words here"),
        _chunk("Doc_T §2#1", "2", "Same   words here"),
        _chunk("Doc_T §3#1", "3", ""),
    ]
    report = audit_chunks(1, 3, chunks, max_tokens=400)
    assert report.duplicate_chunks == [["Doc_T §1#1", "Doc_T §2#1"]]
    assert report.empty_chunks == ["Doc_T §3#1"]
    assert not report.ok
    assert "PROBLEMS FOUND" in format_report(report)
