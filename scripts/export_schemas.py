"""Write JSON Schema files for the pydantic models in src/schemas.py to schemas/.

    python scripts/export_schemas.py          # write
    python scripts/export_schemas.py --check  # exit 1 if a committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas import EXPORTED_SCHEMAS  # noqa: E402

OUT_DIR = ROOT / "schemas"


def render(name: str) -> str:
    return json.dumps(EXPORTED_SCHEMAS[name].model_json_schema(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    stale = []
    for name in EXPORTED_SCHEMAS:
        path = OUT_DIR / f"{name}.schema.json"
        text = render(name)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print(f"stale schema files (run `make schema`): {stale}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
