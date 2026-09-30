# Phase 5 report: Telemetry, Benchmarking & Packaging

Phase 4 was merged to `main` (`a82a0ea`); this work starts there. Scope agreed with the owner:
complete everything that does not need Docker, including the intended LLM integration, so that
Docker/clean-machine runtime verification is the only environment-dependent task left.

## Done

| Step | What | Commit |
|---|---|---|
| 5.1 | Per-turn telemetry completed: one request id per turn across the engine and pipeline; `answer_emitted` records the version transition, sub-queries, claim -> source mapping, per-stage latencies, token counts and cost; baseline likewise. `src/telemetry/coverage.py` (G6 checker, CLI exits 1 below 100%); `docs/TELEMETRY.md`; `make coverage`. | `cd9b79e` |
| 5.2 (LLM) | Intended 7-8B Ollama runtime in `src/llm/client.py`: transient-error retries with backoff, `num_ctx` / `keep_alive` / retries / intended-size settings, `health()` (reachable, model pulled, parameter size + 7-8B class), `check_llm` / `require_llm` (fail fast, never a silent fallback), `python -m src.llm.client --check`. Test-only fake Ollama server (`tests/fake_ollama.py`); end-to-end HTTP tests of every LLM path. `.env.example` shows the intended profile. | `92cbe39` |
| 5.2 | `eval/replay.py`: full streaming system + baseline over all scenarios, fresh telemetry per run, gates G1-G6, `results.json` + `summary.md`; `run_eval.py --system full --metrics all`. Docker: `docker compose up` = Ollama -> one-shot model pull -> replay (intended profile); `docker-compose.offline.yml`; `PRISM_IN_CONTAINER` for G1. `make replay / replay-llm / replay-offline / llm-check`. Fixed: the async runner called the synchronous health check (would have broken every real LLM run); baseline and streaming request ids no longer collide. | `dd48853` |
| 5.3 | `reports/BENCHMARK.md`: streaming vs baseline on latency, accuracy (new citation recall/precision metric), grounding and cost; gates; 3 ablation families; 4 analysed edge-case failures; a verification-by-category table. | `d94d26f` |
| 5.4 | `reports/ARCHITECTURE_BRIEF.md` (~1,800 words): design, trigger logic, decomposition, fusion and grounding, refinement, provenance, per-component cost, failure-mode mitigations, deployment profiles. A test enforces the 6-page budget. | `bb19461` |
| 5.5 | `reports/DEMO_SCRIPT.md` (5-minute storyboard with exact commands and scenario files, every command run while writing it), `python -m src.corpus.show` (citation -> corpus text; exit 1 for fabricated ids), `eval/show_turns.py`, `make demo`. Fixed: identical claim text from two sub-intents is rendered once. | `6cebb9e` |
| 5.6 | `scripts/run_all.sh` / `make all`: one command from a fresh clone (Python check, venv, pinned deps, tests, indexes, replay). Verified on a fresh clone in 26 s, exit 0; also run with the intended profile against the fake server. `docker compose config` validates both compose files. Fixed: after an empty answer, the uncertainty note is the session's last output, so "shorten that" is suppressed instead of searching; a Python 3.11 f-string error. | `e469585` |
| 5.7 | `reports/FINAL_CHECKLIST.md`; this report; README; repo-wide test that no scenario, gold or PDF-example text is hard-coded in `src/`; `make ablate-llm`; replay snapshots (offline, and a clearly labelled fake-LLM wiring check). | this commit |

## Verification

All commands were actually run in the build container (Python 3.11.15, CPU only).

**`make test`: 278 passed.**

**`python eval/run_eval.py --system full --metrics all`** (offline profile, exit 0):

| Gate | Result | Status |
|---|---|---|
| G1 | CLI one-command replay completed; container not run | NOT VERIFIED (requires Docker) |
| G2 | 33/34 = 0.971, false triggers 0/6 | PASS |
| G3 | 9/12 = 0.75 | PASS |
| G4 | 131/131 = 1.0, 0 fabricated, shuffled control 0.0 | PASS |
| G5 | 5/5 | PASS |
| G6 | 80/80 = 1.0 | PASS |

**Other runs:**
- **Clean machine:** fresh `git clone` + `bash scripts/run_all.sh` completed in 26 s, exit 0, with a clean tree.
- **Intended profile on the clean clone:** `--check` reports `intended_7_8b_class=True` (fake 8.0B
  model); the full replay completes over HTTP with ~29k input tokens counted and G6 = 80/80.
  Its G3 = 0.5 comes from the crude fake decomposer and is not a result.
- **Unreachable model:** the replay stops with an actionable error (exit 2).
- **LLM ablation rows** produced against the fake server by all three ablation scripts (wiring only).
- **Compose files:** `docker compose -f docker-compose.yml config` and `… -f docker-compose.offline.yml config` both exit 0.
- **Demo:** `make demo` ran at speaking speed.

**Not run, and why**

| Check | Reason |
|---|---|
| `docker compose up` / `docker build` (G1) | no Docker daemon in this environment; the owner asked not to install infrastructure |
| a real 7-8B model | no Ollama server or suitable hardware here (the owner's laptop has ~4 GB RAM); everything around it is implemented and tested against a fake server |
| demo video | a human recording task |

## Decisions

- **The intended LLM architecture is kept, not downgraded.**
  - Ollama + a 7-8B instruct model is the default of `docker compose up` and of `.env.example`.
  - The code default stays offline (`PRISM_LLM_PROVIDER=none`), so tests and CPU-only machines
    never need a model.
  - The model tag is configuration (`llama3.1:8b` default, to be confirmed by the owner).
- **No silent fallback.** A configured LLM that is unreachable, or whose model is not pulled,
  stops the run. `PRISM_DECOMPOSER=llm` / `PRISM_GROUNDING_JUDGE=llm` without an LLM is a
  configuration error.
- **Model size is reported, never assumed.** The Ollama health check reads the model's reported
  parameter size, and every replay records `intended_7_8b_class`, so a small model cannot be
  mistaken for the intended one.
- **The fake Ollama server lives in `tests/` only.** A test proves no `src/` module imports it,
  and its outputs are labelled wiring checks everywhere they appear.
- **G1 is only marked PASS inside the container.** The CLI runner is verified and recorded
  separately; the playbook allows it as the repository deliverable ("docker compose up or clean
  CLI runner", [Doc_01 §8]), but the G1 gate names the container.
- **Accuracy vs baseline uses turn-level citation recall/precision against gold sections**,
  counting only the claims a turn added, so refinement turns are not inflated by earlier claims.

## Known issues

1. **No real 7-8B measurements.** G4 on LLM-written claims, LLM-decomposer G3, the LLM judge's
   discrimination, generation latency and memory are unmeasured until the Docker run.
2. **G1 is unverified** until `docker compose up` runs on a clean machine.
3. **The dev scenarios are still drafts** (not human-reviewed), and small (40 turns). All numbers
   are dev-set numbers.
4. **Analysed failures that remain open** (BENCHMARK F1-F4): cross-turn back-references, lexical
   answerability, hybrid retrieval dominated by BM25, and the lexical G3 metric.
5. **Post-utterance CPU latency is higher for streaming than baseline** (5-6 vs 2.5-2.7 ms), because
   fusion, synthesis and verification run after speech ends. It is negligible in absolute terms.

## Questions for the human

1. Is `llama3.1:8b` the model to standardise on, or does the teammate prefer another 7-8B instruct model?
2. Should the dev scenarios be reviewed and extended (contradictions, noun-phrase multi-intents)
   before the final Docker benchmark run?

## Ready?

**Yes.** Phase 5 is complete except for what needs Docker and a machine able to run a 7-8B model.
A single `docker compose up` on such a machine verifies both (see `FINAL_CHECKLIST.md`,
"Remaining work").

PHASE 5 COMPLETE, awaiting "final docker verification"
