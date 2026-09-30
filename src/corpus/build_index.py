"""Build BM25 + dense indexes from data/corpus/ and write an inspectable snapshot.

    python -m src.corpus.build_index

Writes PRISM_INDEX_DIR/chunks.jsonl (every chunk) and manifest.json (counts, backends,
build time). Exit code 1 if the corpus audit finds empty or duplicate chunks.
"""

from __future__ import annotations

import json
import sys
import time

from src.config import get_settings
from src.corpus.audit import format_report, run_audit
from src.retrieval.factory import build_retrieval


def main() -> int:
    s = get_settings()
    report = run_audit(s)
    print(format_report(report))
    if not report.ok:
        return 1

    t0 = time.perf_counter()
    stack = build_retrieval(s)
    build_ms = (time.perf_counter() - t0) * 1000

    s.index_dir.mkdir(parents=True, exist_ok=True)
    with (s.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in stack.chunks:
            f.write(c.model_dump_json() + "\n")
    manifest = {
        "chunks": len(stack.chunks),
        "sparse": "bm25okapi",
        "dense": stack.dense.embedder.name,
        "dense_dim": int(stack.dense.matrix.shape[1]),
        "dense_cache_hit": stack.dense.cache_hit,
        "rrf_k": s.rrf_k,
        "build_ms": round(build_ms, 1),
    }
    (s.index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexes built: {json.dumps(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
