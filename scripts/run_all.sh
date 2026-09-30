#!/usr/bin/env bash
# One-command clean-machine runner (the "clean CLI runner" alternative to `docker compose up`, [Doc_01 §8]).
#   bash scripts/run_all.sh
# Creates .venv, installs the pinned dependencies, runs the test suite, builds the indexes and runs
# the full replay (gates G1-G6 -> results/replay/). The LLM profile comes from PRISM_* environment
# variables: unset = offline CPU profile; the intended profile needs a running Ollama with the model
# (checked before the replay starts; see `python -m src.llm.client --check`).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else "Python 3.11 is required (found %d.%d)" % sys.version_info[:2])'
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.corpus.build_index > /dev/null
.venv/bin/python eval/replay.py --out results/replay
