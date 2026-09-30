"""Citation traceability: resolve a `[Doc_ID §Section]` citation to its corpus text.

    python -m src.corpus.show "Doc_01 §5"          # the section's text and its chunk ids
    python -m src.corpus.show "[Doc_01 §4.1.2]"

Exit code 1 if the citation does not exist in the corpus (a fabricated id).
"""

from __future__ import annotations

import argparse
import sys

from src.config import get_settings
from src.corpus.chunker import load_chunks
from src.corpus.loader import load_corpus
from src.synthesis.citations import CITE_RE


def resolve(citation: str) -> tuple[str, str, list[str]] | None:
    """(heading, section text, chunk ids) for a citation, or None if it does not exist."""
    m = CITE_RE.search(citation) or CITE_RE.search(f"[{citation.strip()}]")
    if not m:
        return None
    doc_id, section = m.group(1).split(" §", 1)
    s = get_settings()
    for doc in load_corpus(s.corpus_dir):
        if doc.doc_id != doc_id:
            continue
        for sec in doc.sections:
            if sec.section_id == section:
                ids = [c.chunk_id for c in load_chunks(s) if c.doc_id == doc_id and c.section == section]
                return sec.heading, sec.text, ids
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("citation")
    args = parser.parse_args()
    found = resolve(args.citation)
    if found is None:
        print(f"{args.citation}: not found in the corpus (fabricated or malformed citation)")
        return 1
    heading, text, ids = found
    print(f"{args.citation.strip('[]')}  —  {heading}\nchunks: {', '.join(ids)}\n\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
