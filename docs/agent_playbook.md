# PRISM: Streaming Live RAG, Agent Build Playbook

Give this file to your coding agent (Claude Code, Gemini CLI, Aider, OpenCode, etc.) together with the problem statement PDF.
The agent works **one phase at a time** and **stops at every gate** for human review.

---

## 0. Standing rules (the agent re-reads these at the start of every phase)

1. **Phase discipline.** Do only the current phase. When its steps are done, run the verify commands, write `reports/PHASE_N_REPORT.md`, then **STOP** and print: `PHASE N COMPLETE, awaiting "proceed to phase N+1"`. Never start the next phase on your own.
2. **Hard constraints from the problem statement (never violate):**
   - Corpus isolation: every fact comes from the provided corpus only. No web, no parametric knowledge in answers.
   - No hardcoding or precomputation of prompts, queries or canned answers from the benchmark. The held-out replay set is private.
   - Every factual claim carries a `[Doc_ID §Section]` citation that exists in the corpus. If evidence is missing, emit an uncertainty note or a clarification request.
   - Session state is ephemeral and per-session. No cross-session memory or user tracking.
   - Architectural parsimony: no multi-agent frameworks. Every component must justify its latency and compute cost in the report.
3. **Free / local tools only** (see the stack below). Keep the LLM behind one interface (`llm/client.py`) so the model can be swapped.
4. **Commit after every numbered step** with message `phaseN.stepM: <what>`.
5. **Test as you go.** Each step adds or updates a test. A phase is not done until `make test` passes.
6. **Ask, don't guess.** If the corpus location or format, the allowed models, or the hardware is unknown, ask the human before building on an assumption.
7. **Dev data is for evaluation only.** Files under `eval/` must never be imported by anything under `src/`.

## Suggested stack (all free)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11, `pydantic`, `asyncio` | |
| Sparse retrieval | `bm25s` or `rank_bm25` | |
| Dense retrieval | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) + `faiss-cpu` | verify the model is allowed and downloadable |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| Grounding check | `cross-encoder/nli-deberta-v3-small`, or the LLM as judge | compare both in the ablation |
| LLM | Ollama with a local instruct model (for example a 7-8B class) | free, offline, reproducible |
| Packaging | Docker + docker compose, `uv` or `pip-tools` lockfile | single-command run (gate G1) |
| Tests | `pytest` | |

## Target repo layout

```
prism-live-rag/
├── docker-compose.yml  Dockerfile  Makefile  README.md  pyproject.toml  uv.lock
├── data/corpus/                 # provided corpus (read-only)
├── eval/
│   ├── dev_scenarios/*.json     # our own streaming test cases
│   └── run_eval.py
├── reports/                     # PHASE_N_REPORT.md, benchmark, architecture brief
├── logs/                        # telemetry JSONL output
├── src/
│   ├── schemas.py
│   ├── llm/client.py
│   ├── corpus/{loader.py,chunker.py,audit.py}
│   ├── retrieval/{bm25.py,dense.py,hybrid.py,rrf.py,rerank.py,cache.py}
│   ├── stream/{simulator.py,controller.py,stability.py}
│   ├── decompose/{decomposer.py,dedupe.py}
│   ├── session/{store.py,ledger.py,delta.py}
│   ├── synthesis/{generator.py,grounding.py,uncertainty.py}
│   ├── telemetry/{events.py,logger.py}
│   ├── baseline.py              # non-streaming baseline pipeline
│   └── engine.py                # end-to-end streaming engine
└── tests/
```

## Report template (`reports/PHASE_N_REPORT.md`)

```
# Phase N report
## Done            (steps completed, with commit hashes)
## Verification    (commands run + actual output/metrics)
## Decisions       (choices made, alternatives rejected, why)
## Known issues    (what is weak or unfinished)
## Questions for the human
## Ready for the next phase?  (yes / no + reason)
```

---

# PHASE 1: Framing & Foundation

**Goal:** a reproducible repo, a clean indexed corpus, a working non-streaming baseline, structured schemas, and our own dev scenarios.

### Steps

**1.1 Intake check.** Locate the corpus and read its format. Confirm with the human: allowed models, hardware (CPU or GPU, RAM), and any rules text beyond the PDF. Write the answers to `reports/PHASE_1_REPORT.md`.

**1.2 Scaffold.** Create the repo layout above, `pyproject.toml` with pinned dependencies, a `Dockerfile`, `docker-compose.yml`, and a `Makefile` with targets `test`, `index`, `baseline`, `replay`. Start Ollama as a compose service or document its requirement.

**1.3 Schemas (`src/schemas.py`).** Define pydantic models: `TranscriptChunk(ts, text, is_final)`, `ControllerDecision(action: wait|retrieve|suppress, reason, trigger)`, `SubQuery(id, text, status)`, `CorpusChunk(doc_id, section, text, chunk_id)`, `Claim(id, text, subintent_id, chunk_ids, version)`, `AnswerVersion(version, claims, uncertainty)`, and `TelemetryEvent`. The final output record must match the PDF example: `retrieval_events`, `sub_queries`, `answer`, `citations`, `uncertainty`. Add a JSON-schema validation test.

**1.4 Corpus loader and chunker.** Parse every document into sections, then chunk section-aware (roughly 200-400 tokens, with a small overlap). Prefix each chunk with its doc title and section heading. Give every chunk a stable id in the `Doc_ID §Section` form. Do not split tables or lists mid-row.

**1.5 Corpus audit (`corpus/audit.py`).** Print document count, section count, chunk-length histogram, duplicate chunks, and empty chunks. Fix any chunking problem it exposes.

**1.6 Indexes.** Build BM25 and dense (FAISS) indexes at container start from `data/corpus/`. Expose `search(query, k) -> ranked chunks` for each and for `hybrid.py` (RRF fusion, k=60 as a starting point).

**1.7 LLM client.** One async interface with `generate(prompt, schema=None)` that returns text plus token counts, and logs both. Support JSON-constrained output.

**1.8 Baseline pipeline (`baseline.py`).** Full utterance in, hybrid retrieval, rerank, LLM answer with citations, output record. This is the reference system for all later benchmarks. Keep it simple.

**1.9 Dev scenarios (`eval/dev_scenarios/`).** Create about 30 scenario JSON files derived from the corpus: multi-intent utterances, late-constraint follow-ups, presentation-only turns, noisy or incomplete fragments, and no-evidence questions. Each has timestamped chunks, gold sub-intents, gold supporting chunk ids, and `retrieval_required`. **List the scenarios and gold labels for human review before finalizing.**

**1.10 Baseline metrics.** `eval/run_eval.py` runs the baseline over the dev scenarios and reports recall@5, citation validity (does the cited id exist), and latency.

### Verify
```
make test
docker compose up            # builds indexes, answers a sample query
python eval/run_eval.py --system baseline
```
**Pass when:** the compose command works on a clean checkout, all citations in baseline output resolve to real chunk ids, the schema tests pass, and baseline metrics print.

**STOP.** Write the report and wait for approval.

---

# PHASE 2: Controller & Live Stream Simulation

**Goal:** decide Wait / Retrieve / Suppress incrementally, and start retrieval early without thrashing.

### Steps

**2.1 Stream simulator (`stream/simulator.py`).** Replays a scenario's timestamped chunks through an async generator. Support simulated time (fast tests) and wall-clock time (demo). Emit an utterance-end event.

**2.2 Suppression gate.** Detect presentation-only turns (reformat, shorten, repeat, translate) that operate on prior output. Use generic patterns plus a small classifier fallback. Never put benchmark phrases in code. Output `retrieval_required=false, reason=presentation_restructure`.

**2.3 Stability probe (`stream/stability.py`).** On each partial transcript, run a cheap BM25 probe. Compute top-k overlap (Jaccard) with the previous probe, and check that the query has enough content (entities or slots present, minimum token count). Expose a `stability_score`.

**2.4 Controller (`stream/controller.py`).** Policy: `wait` while the score is low or the fragment is incomplete; `retrieve` (provisional) once stability holds across N consecutive chunks; `retrieve` (final) at utterance end if new content arrived since the last retrieval; `suppress` per 2.2. Make thresholds config values.

**2.5 Provisional retrieval wiring.** When the controller says retrieve, dispatch the search, log `retrieval_started` with timestamp, query, and trigger (`provisional`, `multi_intent`, `final`).

**2.6 Controller metrics.** Early retrieval rate (G2: retrieval starts before utterance end, target 80% or more of eligible queries), false-trigger rate on no-retrieval cases, and seconds gained versus baseline.

**2.7 Ablation.** Compare rule-only, rule+stability, and a small LLM-based controller. Tune thresholds on dev scenarios. Record the trade-off table for the benchmark report.

### Verify
```
make test
python eval/run_eval.py --system controller_only --metrics g2
```
**Pass when:** early retrieval is at or above 80% on eligible dev queries, presentation-only turns never trigger retrieval, and the ablation table exists.

**STOP.** Report and wait.

---

# PHASE 3: Multi-Intent Parsing & Evidence Fusion

**Goal:** split compound utterances into non-redundant sub-queries, retrieve in parallel, and fuse evidence cleanly.

### Steps

**3.1 Decomposer (`decompose/decomposer.py`).** LLM call with a JSON schema. Input: transcript so far plus the current live sub-query set. Output: diff operations (`add`, `merge`, `keep`). The prompt is generic and contains no scenario-specific text.

**3.2 Anti-fragmentation (`decompose/dedupe.py`).** Short-circuit to a single query for simple single-intent utterances. Merge sub-queries whose embedding similarity exceeds a threshold. Cap the sub-query count.

**3.3 Context carry-over.** Extract shared entities and constraints from the utterance (location, quantity, dates) and inject them into each sub-query, so nothing loses conversational context.

**3.4 Parallel retrieval.** `asyncio.gather` over sub-queries, each running hybrid retrieval. Log one `retrieval_events` entry per sub-query with trigger `multi_intent`.

**3.5 Fusion and rerank.** RRF within each sub-query, then merge across sub-queries: dedupe by chunk id, then by near-duplicate similarity, then cross-encoder rerank. Enforce a per-sub-intent quota so one intent cannot crowd out the others.

**3.6 Speculative reuse cache (`retrieval/cache.py`).** If the final sub-query is close to a provisional one, reuse its results rather than searching again. Log cache hits.

**3.7 Metrics and ablation.** G3: at least 70% of compound queries yield 2 or more correct sub-intents. Ablations: hybrid versus dense-only, and with versus without the anti-fragmentation guard. Analyze at least one decomposition failure in detail.

### Verify
```
make test
python eval/run_eval.py --system decompose_fusion --metrics g3
```
**Pass when:** G3 is at least 70% on compound dev cases, simple queries are not over-split, and both ablations are recorded.

**STOP.** Report and wait.

---

# PHASE 4: Session Refinement & Grounding

**Goal:** update answers in place when late constraints arrive, and guarantee that every claim is grounded.

### Steps

**4.1 Session store (`session/store.py`).** In-memory, keyed by session id, destroyed when the session ends. No disk persistence, no cross-session lookup. Add a test proving isolation between two sessions.

**4.2 Claim ledger (`session/ledger.py`).** Each claim holds its id, text, sub-intent id, supporting chunk ids, dependent constraints, and version. The ledger also stores the current sub-queries and evidence set.

**4.3 Synthesis (`synthesis/generator.py`).** The LLM returns structured claims, each with chunk ids. Render them to text with `[Doc_ID §Section]` citations. Instruct the model to use only the supplied evidence.

**4.4 Grounding verifier (`synthesis/grounding.py`).** For each claim: (a) cited ids exist in the corpus, (b) the NLI model or LLM judge confirms the cited chunk supports the claim. Drop or flag failures. Compare NLI against LLM-judge in an ablation.

**4.5 Uncertainty (`synthesis/uncertainty.py`).** If a sub-intent's best evidence falls below a threshold, or verification fails, emit an explicit `uncertainty` note or a targeted clarification request instead of guessing.

**4.6 Delta engine (`session/delta.py`).** Classify each new user input as one of: `new_subintent`, `parameter_update`, `contradiction`, `presentation_only`. For the first three, re-query only the affected sub-queries, invalidate only the dependent claims, keep the untouched claims and their citations, add delta citations, and bump the answer version.

**4.7 Presentation-only path.** Transform the existing answer (bullets, shorter, translated) with no retrieval. Reuse existing citations and assert that no new ids appear.

**4.8 Metrics.** G4: at least 85% citation support, with zero fabricated ids. G5: late constraints update the answer without clearing session state or re-running full-corpus search. Analyze at least one refinement failure.

### Verify
```
make test
python eval/run_eval.py --system full --metrics g4,g5
```
**Pass when:** G4 and G5 thresholds are met, a delta update touches only affected claims (checked by a test that compares ledger diffs), and suppression turns run with zero retrieval calls.

**STOP.** Report and wait.

---

# PHASE 5: Telemetry, Benchmarking & Packaging

**Goal:** full observability, a benchmark against the baseline, documents, and one-command reproducibility.

### Steps

**5.1 Telemetry completion (`telemetry/`).** Structured JSONL for every request: timestamps, retrieval decisions and triggers, sub-queries, source mappings, answer version transitions, latencies per stage, token counts, and an estimated cost. Add a coverage checker that fails when any request lacks a complete trace (G6: 100%).

**5.2 Replay runner.** `docker compose up` runs the full replay suite non-interactively and writes `results.json` plus a summary table covering G1-G6.

**5.3 Benchmark report (`reports/BENCHMARK.md`).** Streaming system versus the baseline on latency, accuracy, grounding, and cost. Include at least three analyzed edge-case failures and two architectural ablations (already collected in phases 2-4). Use tables, not prose, for the numbers.

**5.4 Architecture brief (6 pages max).** System design rationale, retrieval trigger logic, decomposition strategy, data provenance, trade-offs, and failure-mode mitigations. Justify each component's latency and compute cost.

**5.5 Demo script for the 5-minute video.** Storyboard covering: early retrieval, multi-intent decomposition, late-detail refinement, presentation-query suppression, citation traceability, and live telemetry. Provide the exact commands and scenario files to show.

**5.6 Clean-machine test.** Fresh clone, a single command, no manual steps. Fix anything that breaks.

**5.7 Final gate checklist.** Write `reports/FINAL_CHECKLIST.md` marking G1-G6 pass or fail with evidence, plus the deliverables list: repository, architecture brief, benchmark report, demo video, telemetry schema.

### Verify
```
docker compose up            # on a clean checkout
python eval/run_eval.py --system full --metrics all
```
**Pass when:** all six gates meet their thresholds and every deliverable exists.

**STOP.** Final report to the human.

---
