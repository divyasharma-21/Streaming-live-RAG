# Corpus provenance

## Source

| Field | Value |
|---|---|
| Source file | `Theme_4_Guide_RAG.pdf` (the Theme 4 problem statement supplied by the project owner) |
| Source SHA-256 | `319adef6b15661e62b133b4805b4399cee58ef3e11012c054051b6974fd829db` |
| Source size | 1,996,121 bytes, 6 pages (US Letter) |
| Text layer | **None.** The PDF was produced by a print-to-PDF driver. Every page is a mosaic of image tiles with no extractable text (`pypdf` and `PyMuPDF` both return empty text). |
| Corpus output | `data/corpus/Doc_01.md` (single document, `Doc_ID = Doc_01`) |
| Corpus file SHA-256 | recorded in `data/corpus/MANIFEST.json` and checked by `tests/test_corpus_integrity.py` |

The source PDF itself is **not** committed to the repository, because every page carries
personal information in its footer and the PDF metadata carries a personal author field.

## Extraction method

1. Each page was rendered to PNG with PyMuPDF (130 dpi full page, 220 dpi crops of the
   dense text regions). PyMuPDF was used only as a one-off reading tool and is not a project
   dependency.
2. The rendered pages were **transcribed manually by reading the images** (no OCR engine was
   available in the build environment). Wording, punctuation, list numbering, table rows and
   code blocks were copied as they appear in the PDF.
3. No content was added, paraphrased, summarised, or supplemented from any external source.

## Structure and stable identifiers

The PDF's own numbered headings (1-8) are used as section numbers. Sub-headings that the PDF
shows without a number were given a dotted number beneath their parent. The title block is `§0`.

| Section id | Heading (as in PDF) | PDF page |
|---|---|---|
| §0 | Streaming Live RAG (title + subtitle) | 1 |
| §1 | 1. Executive Summary & Problem Overview | 1 |
| §1.1 | The Target Challenge | 1 |
| §2 | 2. System Architecture & Pipeline (diagram) | 2 |
| §2.1 | Core Pipeline Components (table) | 2 |
| §3 | 3. Hard Engineering Rules & Constraints | 3 |
| §4 | 4. Input & Output Event Specifications (intro) | 3 |
| §4.1 | Example 1: Incremental Multi-Intent Utterance | 3 |
| §4.1.1 | Stream Processing Timeline (table) | 3 |
| §4.1.2 | Structured Output Event Record (JSON) | 4 |
| §4.2 | Example 2: Late-Arriving Detail (Refine, Do Not Restart) | 4 |
| §4.3 | Example 3: Query Suppression (No Retrieval Required) | 4 |
| §5 | 5. Technical Evaluation Gates (table, spans pages 4-5) | 4-5 |
| §6 | 6. Common Technical Pitfalls | 5 |
| §7 | 7. Implementation & Architecture Roadmap | 5 |
| §8 | 8. Engineering Deliverables Checklist | 6 |

Citations take the form `[Doc_01 §<section id>]`, e.g. `[Doc_01 §3]`. Section ids are
written explicitly in the corpus file (`## [§3] ...`) so they never depend on parser heuristics.
Changing a section id is a breaking change to every gold label and must be recorded here.

## Content deliberately excluded

| Excluded item | Where it appears | Reason |
|---|---|---|
| Diagonal "Samsung Electronics" watermark | every page | watermark, not content |
| Footer line with a person's name, e-mail handle, job titles and organisation | every page | personal information |
| Barcode and the numeric identifier printed above it | every page | tracking mark, not content |
| PDF metadata (author, title field, producer) | file metadata | personal information / not content |
| Empty checkbox glyphs in section 8 | page 6 | rendered as Markdown `- [ ]` instead |

## Known transcription limitations

- **Truncated lines in the source.** In the JSON example (§4.1.2) the `"answer"` line ends at
  `...Venue A provide` and the `"uncertainty"` line ends at `...retrieved corpus.` without a
  closing quote. The text runs off the right edge of the page in the PDF itself; nothing more
  is recoverable. The corpus reproduces the truncation exactly, so that block is *not* valid JSON.
- **Diagram glyphs.** The architecture diagram in §2 uses box-drawing characters and arrow
  glyphs. It is reproduced with ASCII `+ - |`, `-->` and `v`; all words are unchanged.
- **Typography.** Bold/italic are kept as Markdown `**`/`*`. Curly vs. straight quotes follow
  the PDF (the Example 1 timeline mixes both).
- **Example data is not a knowledge base.** Items such as "Venue A", "Doc_12 §2" or the travel
  reimbursement rule are *illustrations inside the problem statement*. They are indexed as
  text of this document, but the corpus contains no real venue, catering or reimbursement
  policy documents. Questions about those topics have no real evidence in this corpus.

## Scope note

This corpus is the only corpus used in Phase 1, per the project owner's instruction. If the
organisers later supply a separate evaluation corpus, add each document as its own
`data/corpus/Doc_NN.md` with an explicit section header per section, update `MANIFEST.json`
(`python scripts/corpus_manifest.py --write`) and extend this file.
