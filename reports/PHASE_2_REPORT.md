# Phase 2 report: Controller & Live Stream Simulation

Goal (playbook): decide wait / retrieve / suppress incrementally, and start retrieval early
without thrashing. Phase 1 was merged to `main` (`f525a8f`); this work starts from there.

## Done

| Step | What | Commit |
|---|---|---|
| 2.1 | `src/stream/simulator.py`: async `replay()` over `TranscriptChunk`s with `simulated` (instant) and `wall` (real offsets, `speed`) clocks. Emits `chunk` and `utterance_end` events, plus an implicit end if no chunk is final. Validates timestamp order. | `7650ad5` |
| 2.2 | `src/stream/suppression.py`: presentation-only gate. Decisive generic rules (transformation cue + reference to prior output + no new corpus terms), then a hand-weighted logistic fallback for ambiguous cases. Outputs `retrieval_required=false, reason=presentation_restructure`. Never fires without prior output. Adds `BM25Retriever.vocabulary`. | `1ad39b2` |
| 2.3 | `src/stream/stability.py`: per-chunk BM25 top-k probe, Jaccard vs the previous probe, content sufficiency (corpus-vocabulary slots + numeric/capitalised entities, minimum tokens), completeness check. Exposes `stability_score`. | `618487a` |
| 2.4 | `src/stream/controller.py`: `wait` / `retrieve(provisional)` after N stable chunks (at most `max_provisional`) / `retrieve(final)` at utterance end only if new content arrived / `suppress` (presentation, insufficient content). Modes `rule_only` and `rule_stability`. All thresholds in `src/config.py` / `.env.example`. `ControllerDecision` gains optional `ts` and `stability_score`. | `e10f102` |
| 2.5 | `src/engine.py`: `StreamingEngine.run_turn()` replays the stream and asks the controller on each event. On `retrieve` it dispatches hybrid search + rerank as a background task and logs `retrieval_started` (stream ts, keyword query, trigger), then `retrieval_completed`. Also logs `controller_decision`, `utterance_end` and `request_completed`. CLI: `python -m src.engine --utterance "a \| b"`. | `635af53` |
| 2.6 | `eval/controller_eval.py` + `run_eval.py --system controller_only --metrics g2`: G2 early-retrieval rate, false-trigger rate, presentation retrievals, seconds gained vs baseline, retrievals per turn, first/final recall@5, engine time. | `45a5731` |
| 2.7 | `eval/ablate_controller.py`: rule_only (18-config grid) vs rule_stability (60-config grid) vs LLM controller. Pre-stated selection rule, tuned defaults, `reports/PHASE_2_CONTROLLER_ABLATION.md`. `LLMController` (one LLM call per chunk via `src/llm/client.py`). `make controller / ablate / stream`. | `219ce2a` |
| 2.7 (fix) | Failure analysis found a completeness bug (sentence-final punctuation was ignored) and a mislabelled reason (`no_corpus_terms` added). Fixed with tests; ablation regenerated. | `b579c3d` |
| — | This report and the README update | this commit |

## Verification

All commands were actually run in the build container (Python 3.11.15, CPU only).

**`make test`**: `134 passed` (72 from Phase 1 + 62 new).

**Clean checkout.** A fresh `git clone` into a venv with the pinned `requirements-dev.txt`:
`make test` passed 134; G2 and the baseline eval printed the numbers below; `git status` stayed clean.

**`python eval/run_eval.py --system controller_only --metrics g2`** (defaults, simulated clock):

| metric | value |
|---|---|
| G2 early retrieval rate | 33/34 = **0.9706** (target >= 0.8: **PASS**) |
| eligible turns with no retrieval at all | 1 (`noevid_02`, reason `no_corpus_terms`) |
| false-trigger rate on no-retrieval turns | 0/6 = **0.0** |
| presentation-only turns that retrieved | **0/5** |
| seconds gained vs baseline (mean over eligible / over early) | 0.86 / 0.89 |
| retrievals per eligible turn | 1.29 |
| recall@5 of first / final retrieval | 0.95 / 0.95 |
| engine ms per turn (mean / max) | about 4.6 / 10 |

**Ablation** (`reports/PHASE_2_CONTROLLER_ABLATION.md`, 78 configurations, about 12 s):

| controller | G2 | false triggers | presentation retrievals | s gained | retrievals / turn |
|---|---|---|---|---|---|
| rule_only, start values | 0.9706 | 0 | 0 | 0.885 | 1.32 |
| **rule_only, tuned (default)** | **0.9706** | **0** | **0** | 0.862 | 1.29 |
| rule_stability, start values | 0.7059 | 0 | 0 | 0.377 | 1.00 |
| rule_stability, tuned | 0.8824 | 0 | 0 | 0.465 | 1.00 |
| llm (one call per chunk) | not run: no LLM configured | | | | |

The thresholds are reproduced by the ablation script: `rule_only` N=1, min content terms 2,
min tokens 4; `rule_stability` threshold 0.2, N=1, k=3.

**Phase 1 regression check.** `python eval/run_eval.py --system baseline` is unchanged:
recall@5 0.9516, citations 64/64 valid, uncertainty 1/5, false uncertainty 1/30.

**Pass criteria (playbook).**
- Early retrieval >= 80%: **yes**, 0.97.
- Presentation-only turns never trigger retrieval: **yes**, 0/5. A unit test also asserts zero search calls on a presentation turn.
- The ablation table exists: **yes**, though its LLM row is not measured.

**Not run, and why**

| Check | Reason |
|---|---|
| LLM-based controller on the dev set | No LLM available (infrastructure is the teammate's job). `LLMController` is implemented and unit-tested with the mock client only. The ablation row says "not run". |
| Wall-clock demo on real speech timing | Covered only by a short unit test (0.3 s of stream time at 2x). No live audio or ASR input exists. |
| `docker compose up` | Still out of scope; no Docker daemon. |

## Decisions

- **Default mode `rule_only`, not `rule_stability`.** Both pass G2 after tuning. `rule_only`
  starts retrieval earlier (0.97 vs 0.88, +0.4 s gained) at the cost of about 0.3 extra
  retrievals per turn. The selection rule (fixed in advance: zero false triggers, then max G2)
  picks `rule_only`. `rule_stability` stays available via `PRISM_CONTROLLER_MODE` and is the
  better choice if retrieval cost matters more than earliness.
- **Why `rule_stability` is structurally later.** Jaccard needs two probes, so the first chunk
  can never be "stable". The dev utterances are only 2-3 chunks long, so this costs one chunk
  (0.8 s) on most turns. Longer real utterances would hurt it less.
- **The stability metric was not bent to pass the gate.** The first-probe score stays 0 by definition.
- **The suppression gate needs prior output.** A transformation request with nothing to
  transform is handled like any other request.
- **"New content" for the gate means corpus-searchable terms** absent from the prior output.
  Words outside the corpus vocabulary (for example a target language) cannot trigger retrieval.
- **The classifier fallback is hand-weighted, not trained.** Training it on the dev
  scenarios would leak evaluation data into `src/`. A test also asserts that no dev-scenario
  utterance appears in the gate's code.
- **At most one provisional retrieval per utterance** (`PRISM_MAX_PROVISIONAL=1`), plus a
  final one only when new corpus terms arrived. This is the anti-thrashing guard: at most 2
  retrievals per turn, 1.29 on average.
- **At utterance end the content bar is lower than mid-stream.** A complete one-concept
  question still retrieves. An unfinished one below the minimums is suppressed with
  `insufficient_content` (clarification). A complete question with zero corpus terms is
  suppressed with `no_corpus_terms` (no evidence).
- **Prior output in eval** is the previous turn's utterance plus its retrieved evidence text,
  because answer synthesis per turn arrives in Phase 4. Nothing crosses scenarios.
- **Search queries are keyword strings** (content words of the transcript so far), matching the
  style of the problem statement's `retrieval_events`.
- **Component cost (architectural parsimony).** The gate is regex and set operations (microseconds).
  The probe is one BM25 pass over 16 chunks per chunk event (sub-millisecond). The whole
  engine takes about 4-5 ms per turn on CPU in simulated time. The LLM controller would add one
  LLM round-trip per chunk and was not adopted as the default.

## Known issues

1. **Tuned on draft data.** The 30 dev scenarios are still `draft_pending_human_review`, and
   thresholds were tuned on them, so G2 = 0.97 is optimistic for the held-out set.
2. **Premature provisional retrieval with `rule_only`.** It retrieves on the first complete,
   content-sufficient fragment. Example: `late_01` turn 1 retrieved on "What does the
   retrieval controller" before "decide?" arrived. Recall was unaffected here because the
   final retrieval re-queries when new terms arrive, but this is the "eager retrieval"
   pitfall ([Doc_01 §6]) in mild form.
3. **Scenario timing is uniform** (0.8 s steps, end +0.5 s). "Seconds gained" measures
   simulated stream time, not real ASR latency.
4. **The gate is lexical.** Paraphrased formatting requests without any cue word ("make
   it less wordy") go to the fallback, and phrasings without a cue at all are not caught. The
   dev set has only 5 presentation turns.
5. **`no_corpus_terms` suppression depends on vocabulary overlap.** A no-evidence question
   that shares common words with the corpus still retrieves (e.g. `noevid_01`). Phase 4
   grounding has to catch those.
6. **The LLM controller is unmeasured**, so the ablation's "model-based" arm is empty.
7. **`multi_intent` triggers** are not produced yet (Phase 3).

## Questions for the human

1. Are the dev scenarios approved? Tuning should be repeated on approved or extended scenarios, ideally with longer utterances.
2. Which matters more for the demo and benchmark: earliest retrieval (`rule_only`) or fewer
   retrievals (`rule_stability`)? Both pass G2 on the dev set.
3. Once the teammate's LLM is available, should the LLM-controller ablation be run before Phase 3?

## Ready for the next phase?

**Yes.** All Phase 2 steps are implemented, committed and tested (134 tests). G2 = 0.97 >= 0.8,
presentation turns never retrieve, and the ablation table exists. The only unmeasured arm is
the LLM controller, which needs the teammate's LLM.

PHASE 2 COMPLETE, awaiting "proceed to phase 3"
