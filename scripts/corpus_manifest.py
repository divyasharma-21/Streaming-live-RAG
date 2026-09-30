"""Check or rewrite the SHA-256 hashes in data/corpus/MANIFEST.json.

    python scripts/corpus_manifest.py          # check, exit 1 on mismatch
    python scripts/corpus_manifest.py --write  # update hashes after a deliberate corpus edit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
MANIFEST = CORPUS_DIR / "MANIFEST.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for doc in manifest["documents"]:
        digest = hashlib.sha256((CORPUS_DIR / doc["file"]).read_bytes()).hexdigest()
        if digest != doc["sha256"]:
            ok = False
            print(f"{doc['file']}: manifest {doc['sha256'][:12]}... != actual {digest[:12]}...")
            doc["sha256"] = digest
    listed = {d["file"] for d in manifest["documents"]}
    unlisted = sorted(p.name for p in CORPUS_DIR.glob("*.md") if p.name not in listed)
    if unlisted:
        ok = False
        print(f"not in manifest (add them by hand with doc_id/title/source): {unlisted}")

    if args.write:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("manifest written")
        return 0
    print("manifest OK" if ok else "manifest MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
