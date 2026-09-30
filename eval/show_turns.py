"""Print one line per turn from a replay's results.json (demo helper).

    python eval/show_turns.py results/demo/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/demo/results.json")
    res = json.loads(path.read_text(encoding="utf-8"))
    for t in res["turns"]:
        diff = {k: v for k, v in t["ledger_diff"].items() if v}
        first = "none" if t["first_retrieval_s"] is None else f"{t['first_retrieval_s']}s"
        print(f"{t['scenario']:<11} t{t['turn']}  v{t['version']}  {t['kind']:<17} "
              f"retrievals={t['retrieval_calls']}  first_retrieval={first} / utterance_end={t['utterance_end_s']}s")
        print(f"             claims {diff}")
        if t["uncertainty"]:
            print(f"             uncertainty: {t['uncertainty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
