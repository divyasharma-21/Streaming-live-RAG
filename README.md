# PRISM Streaming Live RAG

An event-driven retrieval-augmented generation engine that listens to a live transcript,
retrieves early, decomposes multi-intent requests, refines answers when late constraints
arrive, and grounds every claim in a `[Doc_ID §Section]` citation.
The problem statement is the Theme 4 guide (transcribed into `data/corpus/Doc_01.md`).
The build plan is `docs/agent_playbook.md`.

> **Status: all five phases complete (Phase 5: telemetry, benchmarking & packaging).**
> The only remaining environment-dependent task is the final Docker verification: `docker compose up` on a
> machine with Docker and enough RAM for a 7-8B model. That run verifies gate G1 and the intended 7-8B
> runtime together. See `reports/FINAL_CHECKLIST.md`.
>
> | Gate | Target | Dev-set result (offline profile) |
> |---|---|---|
> | G1 reproducibility | one command, container, clean machine | one-command CLI runner verified on a fresh clone; **container not run (requires Docker)** |
> | G2 early retrieval | >= 80% | 0.971 |
> | G3 multi-intent | >= 70% | 0.75 |
> | G4 grounding | >= 85% support, 0 fabricated | 1.0, 0 fabricated (extractive claims) |
> | G5 session refinement | state continuity | 5/5 |
> | G6 telemetry | 100% trace coverage | 80/80 |
>
> Deliverables:
> * `reports/ARCHITECTURE_BRIEF.md`, `reports/BENCHMARK.md` and `reports/DEMO_SCRIPT.md` (the video still has to be recorded);
> * `docs/TELEMETRY.md` + `schemas/`;
> * per-phase reports and ablations in `reports/`.

## Quick start

Requires Python 3.11. **One command** from a fresh clone (creates `.venv`, installs pinned
dependencies, runs the tests, builds the indexes and runs the full replay with gates G1-G6):

```bash
bash scripts/run_all.sh          # or: make all
cat results/replay/summary.md    # G1-G6 + streaming-vs-baseline table
```

### Runtime profiles

| profile | how | needs |
|---|---|---|
| **intended** (agent playbook) | `docker compose up`: starts Ollama, pulls `PRISM_LLM_MODEL` (default `llama3.1:8b`, a 7-8B instruct model), runs the replay with LLM synthesis, LLM decomposer and LLM judge. Without Docker: run Ollama yourself, then `make llm-check` and `make replay-llm` | Docker (or Ollama) and enough RAM for a 7-8B model |
| **offline** | code default (`PRISM_LLM_PROVIDER=none`): extractive answers, rule decomposer, lexical judge. `make replay-offline` or `docker compose -f docker-compose.offline.yml up` | nothing beyond Python |

`python -m src.llm.client --check` reports whether the configured Ollama server is reachable, whether
the model is pulled, and its parameter size (flagged if outside the intended 7-8B class). A configured
but unavailable model stops a run; there is no silent fallback.

### Individual commands

```bash
make install          # pinned runtime + dev dependencies (pydantic, rank-bm25, numpy, pytest, jsonschema)
make test             # full test suite
make replay           # full replay suite, gates G1-G6 -> results/replay/ (= python eval/run_eval.py --system full --metrics all)
make demo             # wall-clock demo of the storyboard scenarios (reports/DEMO_SCRIPT.md)
make coverage         # G6 trace coverage of logs/telemetry.jsonl
make audit            # corpus audit
make index            # build BM25 + dense indexes, write indexes/chunks.jsonl
make baseline Q="What are the technical evaluation gates and their target thresholds?"
make eval             # baseline over eval/dev_scenarios: recall@5, citation validity, latency
make stream U="What does the stability probe compute? | And which gate measures | early retrieval?"   # streaming demo
make controller / decompose / full    # per-phase metrics (G2 / G3 / G4+G5)
make ablate / ablate3 / ablate4       # per-phase ablations; make ablate-llm adds the LLM rows (needs the model)
python -m src.corpus.show "Doc_01 §5" # resolve a citation to its corpus text
```

Add `--clock wall` to `python -m src.engine ...` to replay fragments at real speed,
`--prior "..."` to simulate an earlier answer in the session (needed for suppression), and
`--no-decompose` for the Phase 2 single-query behaviour.

Without `make`: `python -m pytest -q`, `python -m src.corpus.build_index`,
`python -m src.baseline --query "..."`, `python eval/run_eval.py --system baseline`.

Output of the baseline is the record defined in the problem statement:

```json
{"retrieval_events": [{"timestamp_s": 0.0, "query": "...", "trigger": "final"}],
 "sub_queries": ["..."], "answer": "... [Doc_01 §5]", "citations": ["Doc_01 §5"], "uncertainty": null}
```

Telemetry for every request is appended to `logs/telemetry.jsonl`
(schema: `schemas/telemetry_event.schema.json`).

## Architecture

Streaming path (Phases 2-3):

```
transcript chunks ─► simulator ─► suppression gate ─► stability probe ─► controller (wait / retrieve / suppress)
                                                                                │ retrieve, or stable new content
                                                                                ▼
          planner: decomposer (add/keep/merge) ─► anti-fragmentation guard ─► context carry-over
                                                                                │ new / changed sub-queries
                                                                                ▼
      speculative cache ─(miss)─► parallel hybrid search per sub-query (asyncio.gather, trigger multi_intent)
                                                                                │
                                                                                ▼
            fusion: dedupe by id ─► near-duplicates ─► rerank per sub-query ─► per-intent quota ─► evidence
```

Session path (Phase 4, `src/session/pipeline.py`, one turn at a time):

```
session store (per session id, in memory) ─► streaming engine (above), given the session's last output
   ├─ presentation-only turn ─► transform existing answer (bullets / table / shorter / …), no retrieval, no new ids
   ├─ nothing searchable     ─► clarification request, or per-intent "no evidence" notes
   └─ otherwise ─► delta engine per clause vs. the claim ledger:
                     new_subintent     ─► new sub-intent with its own evidence and claims
                     parameter_update  ─► + constraint, + delta evidence, + delta claims (old claims kept)
                     contradiction     ─► invalidate only claims depending on the negated term, add replacements
                 ─► synthesis per sub-intent ─► grounding verifier (ids exist + support judge) ─► uncertainty per intent
                 ─► ledger commits answer version N+1 ─► output record
```

Baseline path (Phase 1):

```
utterance ─► hybrid retrieval ─► rerank ─► evidence threshold ─► synthesis ─► citation check ─► OutputRecord
             BM25 + dense, RRF    lexical     (uncertainty)       extractive      ids must exist      + telemetry
             (k=60)               (or cross-                      or LLM (JSON)   in the corpus
                                   encoder)
```

| Path | Role |
|---|---|
| `data/corpus/` | read-only corpus (`Doc_01.md`) + `MANIFEST.json` (SHA-256) |
| `data/PROVENANCE.md` | how the corpus was produced, what was excluded, stable id table |
| `src/schemas.py` | pydantic models: `TranscriptChunk`, `ControllerDecision`, `SubQuery`, `CorpusChunk`, `Claim`, `AnswerVersion`, `OutputRecord`, `TelemetryEvent` |
| `schemas/*.schema.json` | JSON Schemas exported from `src/schemas.py` (`make schema`) |
| `src/config.py` | all settings from environment variables (`.env.example`) |
| `src/corpus/` | `loader.py` (sections), `chunker.py` (section-aware chunks), `audit.py`, `build_index.py` |
| `src/retrieval/` | `bm25.py`, `dense.py` (embedder interface), `rrf.py`, `hybrid.py`, `rerank.py`, `factory.py` |
| `src/llm/client.py` | **the only place that talks to an LLM** (ollama / openai-compatible / mock) |
| `src/synthesis/` | `generator.py` (extractive or LLM), `citations.py`, `uncertainty.py` |
| `src/telemetry/` | JSONL event logger and event types |
| `src/baseline.py` | non-streaming reference pipeline |
| `src/stream/` | `simulator.py` (replay), `suppression.py` (presentation-only gate), `stability.py` (probe), `controller.py` (policy + LLM controller option) |
| `src/decompose/` | `decomposer.py` (rule / LLM, diff ops), `dedupe.py` (anti-fragmentation guard), `context.py` (carry-over), `planner.py` |
| `src/retrieval/fusion.py`, `src/retrieval/cache.py` | cross-sub-query evidence fusion; per-utterance speculative reuse cache |
| `src/session/` | `store.py` (ephemeral sessions), `ledger.py` (claims, sub-intents, evidence, versions, diffs), `delta.py` (delta engine), `pipeline.py` (full system) |
| `src/synthesis/` (Phase 4 parts) | `generator.py` `synthesize_subintents`, `grounding.py` (verifier + judges), `uncertainty.py` per-intent notes, `presentation.py` (transforms) |
| `src/engine.py` | streaming engine: controller decisions, planning, parallel retrieval, fusion, telemetry |
| `eval/` | dev scenarios, `run_eval.py`, per-phase evaluators (`controller_eval.py`, `decompose_eval.py`, `full_eval.py`) and ablations (`ablate_controller.py`, `ablate_decompose.py`, `ablate_grounding.py`). **Never imported by `src/`** (enforced by a test) |

### Citations and ids

* Chunk id: `Doc_01 §4.1.2#1` (document, section, part). Stable across runs.
* Citation: `[Doc_01 §4.1.2]` in answer text, `"Doc_01 §4.1.2"` in `citations`.
* A citation is emitted only for evidence chunks the pipeline actually used. The citation
  check fails the request if any cited id does not exist in the corpus.
* If no retrieved chunk covers enough of the request (`PRISM_MIN_EVIDENCE_SCORE`), the answer
  is empty and `uncertainty` explains what is missing and asks for clarification.

## Configuration

Copy `.env.example` to `.env` (or export the variables). The defaults are CPU-light and need
no downloads:

| Setting | Default | Alternatives |
|---|---|---|
| `PRISM_DENSE_BACKEND` | `hashing` (numpy feature hashing, **not semantic**) | `sentence_transformers` (+ `PRISM_DENSE_MODEL`, default `BAAI/bge-small-en-v1.5`) |
| `PRISM_RERANK_BACKEND` | `lexical` (query-term coverage) | `cross_encoder` (+ `PRISM_RERANK_MODEL`), `none` |
| `PRISM_LLM_PROVIDER` | `none` (extractive answers, no LLM) | `ollama`, `openai_compatible` (+ `PRISM_LLM_MODEL`, `PRISM_LLM_BASE_URL`, `PRISM_LLM_API_KEY`) |
| `PRISM_RRF_K` / `PRISM_CANDIDATES` / `PRISM_TOP_K` | 60 / 20 / 5 | |
| `PRISM_MIN_EVIDENCE_SCORE` | 0.34 | not tuned; see report |
| `PRISM_CHUNK_MAX_TOKENS` / `PRISM_CHUNK_OVERLAP_TOKENS` | 400 / 40 | |
| `PRISM_CONTROLLER_MODE` | `rule_only` (tuned) | `rule_stability`, `llm` (needs an LLM) |
| `PRISM_STABILITY_THRESHOLD` / `PRISM_STABLE_CHUNKS` / `PRISM_MAX_PROVISIONAL` | 0.2 / 1 / 1 | tuned in step 2.7 |
| `PRISM_PROBE_K` / `PRISM_PROBE_MIN_CONTENT_TERMS` / `PRISM_PROBE_MIN_TOKENS` | 3 / 2 / 4 | tuned in step 2.7 |
| `PRISM_GATE_THRESHOLD` | 0.5 | suppression-gate classifier fallback |
| `PRISM_DECOMPOSER` | `rules` | `llm` (needs an LLM) |
| `PRISM_ANTI_FRAGMENTATION` / `PRISM_MAX_SUBQUERIES` / `PRISM_SUBQUERY_MERGE_SIMILARITY` / `PRISM_SUBQUERY_MIN_CONTENT_TERMS` | 1 / 4 / 0.85 / 2 | |
| `PRISM_FUSION_TOP_K` / `PRISM_FUSION_QUOTA` / `PRISM_FUSION_NEAR_DUPLICATE` | 6 / 2 / 0.9 | |
| `PRISM_CACHE_SIMILARITY` | 0.75 | 0 disables the speculative cache |
| `PRISM_GROUNDING_JUDGE` / `PRISM_GROUNDING_MODE` / `PRISM_LEXICAL_SUPPORT_THRESHOLD` | `lexical` / `drop` / 0.8 | `nli` (optional model), `llm` (needs an LLM); `flag` keeps failures but reports them |

## Final verification on suitable hardware (teammate)

Everything that does not need Docker or a 7-8B model is implemented and verified: see
`reports/FINAL_CHECKLIST.md` for what was verified locally, what was verified with the test-only fake
Ollama server (`tests/fake_ollama.py`, wiring only), and what still needs the real runtime.

1. **Docker + 7-8B runtime (one step).** On a machine with Docker and enough free RAM for a
   quantised 7-8B model, run `docker compose up`.
   - Expect: the `ollama-pull` job pulls `PRISM_LLM_MODEL`, and the `prism` container exits 0.
   - Expect: `results/replay/summary.md` shows G1 PASS, and the run reports `intended_7_8b_class=True`.
   - Pin the `ollama/ollama` image tag once verified, and confirm or replace the default model tag
     (`PRISM_LLM_MODEL`).
2. **LLM ablation rows.** `docker compose run --rm prism make ablate-llm` (or `make ablate-llm`
   with a local Ollama) fills the LLM controller, LLM decomposer and LLM judge rows of the Phase 2-4
   ablation reports.
3. **Optional models.** `make install-optional` installs sentence-transformers + faiss-cpu
   (PyTorch) for `PRISM_DENSE_BACKEND=sentence_transformers`, `PRISM_RERANK_BACKEND=cross_encoder`
   and the NLI judge (`PRISM_GROUNDING_JUDGE=nli`). Pin the versions in
   `requirements-optional.txt` once tested.
4. **Hosted endpoint instead of Ollama** (optional): `PRISM_LLM_PROVIDER=openai_compatible`,
   `PRISM_LLM_BASE_URL=https://.../v1`, `PRISM_LLM_API_KEY=...` (never commit `.env`), and
   `PRISM_COST_PER_1K_INPUT/OUTPUT` if billed.
5. **Lockfile.** `uv.lock` covers the optional extras; regenerate it with `make lock` after changing
   `pyproject.toml`. `requirements*.txt` are kept in sync by hand.

## Rules this repo enforces (from the problem statement, [Doc_01 §3])

* Corpus isolation: answers come only from `data/corpus/`; the extractive path quotes it verbatim.
* No hardcoded prompts, queries or answers in `src/`; dev scenarios live in `eval/` only.
* Every claim carries a citation that exists in the corpus, or the system emits uncertainty.
* No cross-session state: sessions live only in memory, are looked up by exact id, and are destroyed on `end_session`.
* Presentation-only turns are suppressed and never reach the corpus.
* No multi-agent frameworks.
