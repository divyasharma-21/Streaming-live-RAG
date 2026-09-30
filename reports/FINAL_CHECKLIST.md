# Final gate checklist

Evidence comes from `python eval/run_eval.py --system full --metrics all` (the replay runner,
`eval/replay.py`) on the 30 dev scenarios, offline CPU profile. Snapshot:
`reports/REPLAY_SUMMARY_OFFLINE.md`; details: `reports/BENCHMARK.md`. The held-out benchmark set
is private and was not available.

## Gates G1-G6

| Gate | Target | Result | Status | Evidence |
|---|---|---|---|---|
| G1 Reproducibility | container launches via one command on a clean machine; the replay completes unattended | one-command CLI runner (`bash scripts/run_all.sh`) verified on a fresh clone (venv + pinned deps + 278 tests + indexes + replay, exit 0, 26 s); `docker compose config` validates both compose files | **NOT VERIFIED — requires Docker** | the container was never built or run (no Docker daemon). `eval/replay.py` marks G1 PASS automatically when it completes inside the container (`PRISM_IN_CONTAINER=1`) |
| G2 Early retrieval | >= 80% of eligible queries | 33/34 = 0.971; false triggers 0/6 | **PASS** | results.json `gates.G2`; ablation `reports/PHASE_2_CONTROLLER_ABLATION.md` |
| G3 Multi-intent identification | >= 70% of compound queries | 9/12 = 0.75; 0/22 simple queries over-split | **PASS** | `gates.G3`; `reports/PHASE_3_ABLATION.md` |
| G4 Factual grounding | >= 85% citation support, 0 fabricated ids | 131/131 = 1.0; 0 fabricated; shuffled control 0.0 | **PASS (offline profile)** | `gates.G4`; the claims are verbatim extractive spans, so G4 on LLM-written claims requires the 7-8B runtime |
| G5 Session refinement | verified state continuity | 5/5 refinement turns; presentation turns 5/5 with zero retrieval | **PASS** | `gates.G5`; ledger-diff tests in `tests/test_delta.py` |
| G6 Telemetry & observability | 100% trace coverage | 80/80 requests | **PASS** | `gates.G6`; `python -m src.telemetry.coverage results/replay/telemetry.jsonl` |

## Deliverables ([Doc_01 §8])

| Deliverable | Status | Where |
|---|---|---|
| Reproducible repository: source, pinned lockfiles, environment templates, one-command run | **done** (container run pending Docker) | `src/`, `requirements*.txt` + `uv.lock`, `.env.example`, `scripts/run_all.sh` / `make all`, `docker-compose.yml`, `docker-compose.offline.yml`, `Dockerfile` |
| System architecture brief (<= 6 pages) | **done** (~1,800 words) | `reports/ARCHITECTURE_BRIEF.md` |
| Benchmarking & evaluation report (vs baseline, >= 3 edge-case failures, 2 ablations) | **done** (4 failures analysed, 3 ablation families) | `reports/BENCHMARK.md` + phase ablation reports |
| System demonstration video (<= 5 min) | **script done; video not recorded** (a human recording task) | `reports/DEMO_SCRIPT.md`, `make demo` |
| Telemetry & observability schema | **done** | `docs/TELEMETRY.md`, `schemas/telemetry_event.schema.json`, `src/telemetry/coverage.py` |

## Standing rules (agent playbook §0 / problem statement [Doc_01 §3])

| Rule | How it is enforced | Test |
|---|---|---|
| Corpus isolation | corpus is `data/corpus/` only (SHA-256 manifest); synthesis sees only retrieved evidence; the LLM prompt restricts it to supplied evidence, and claims citing other chunks are dropped | `test_corpus_integrity.py`, `test_synthesis_subintents.py` |
| No hard-coded benchmark prompts or answers | no scenario / gold / PDF-example text in `src/`; prompts are generic | `test_scaffold.py::test_no_scenario_or_benchmark_text_is_hardcoded_in_src`, `test_suppression_gate.py` |
| Every claim cited with an existing `[Doc_ID §Section]`, or uncertainty | grounding verifier (id existence + support judge), rendering assertion, per-intent uncertainty | `test_grounding.py`, `test_delta.py`, `test_uncertainty_subintent.py` |
| Session state ephemeral, per session | in-memory store, exact-id lookup, no disk or listing API, `end()` destroys state; replay ends every session | `test_session_store.py`, `test_full_eval.py` |
| Architectural parsimony | single asyncio process, no multi-agent framework; per-component cost table | `reports/ARCHITECTURE_BRIEF.md` §8 |
| LLM behind one interface | only `src/llm/client.py` does HTTP or imports an LLM SDK | `test_scaffold.py::test_all_llm_calls_live_in_llm_client`, `test_llm_runtime.py::test_fake_server_is_test_only` |
| `eval/` never imported by `src/` | AST check | `test_scaffold.py::test_src_never_imports_eval` |
| Commit per step, `make test` passes | `phaseN.stepM:` commits; 278 tests pass | `git log` |

## Verification status by category

| Item | 1. verified locally (offline profile) | 2. verified with the test/fake LLM | 3. requires the 7-8B Ollama runtime | 4. requires Docker |
|---|---|---|---|---|
| G2-G6 on the dev set | yes | wiring (`reports/REPLAY_SUMMARY_FAKE_LLM_WIRING.md`) | numbers with LLM decomposition / synthesis / judge | — |
| G1 | CLI one-command run on a fresh clone | — | — | **yes** |
| LLM client (Ollama and OpenAI-compatible): JSON mode, retries, health check, 7-8B size classification, token/cost telemetry | stub tests | yes (`tests/test_llm_runtime.py`, `tests/test_replay.py`) | real latency, memory and quality | — |
| LLM synthesis, LLM decomposer, LLM judge, LLM controller, LLM translation | — | yes (every path end to end over HTTP) | yes | — |
| LLM rows of the ablations | — | wiring (`make ablate-llm` ran against the fake) | yes (`make ablate-llm`) | — |
| Optional semantic embedder / cross-encoder / NLI judge | — | — | needs the optional models (`make install-optional`) | — |
| `docker compose up` (Ollama + model pull + replay) | compose files validated with `docker compose config` | — | yes | **yes** |

## Remaining work

1. **Final Docker verification (environment-dependent).** On a machine with Docker and enough
   free RAM for a 7-8B model:
   1. run `docker compose up`;
   2. confirm the `prism` container exits 0 and `results/replay/summary.md` shows G1 PASS;
   3. confirm the run reports `intended_7_8b_class=True`.

   This one step verifies G1 and the intended 7-8B runtime together. Then run
   `docker compose run --rm prism make ablate-llm` to fill the LLM ablation rows.
2. **Human tasks (not environment-dependent):**
   - record the demo video from `reports/DEMO_SCRIPT.md`;
   - review and approve the dev scenarios (`draft_pending_human_review`);
   - confirm the default model tag (`llama3.1:8b`) and pin the `ollama/ollama` image tag.
