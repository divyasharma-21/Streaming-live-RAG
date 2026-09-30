# PRISM Streaming Live RAG: benchmark and evaluation report

## 1. Setup

| item | value |
|---|---|
| Corpus | `Doc_01` (the Theme 4 guide, 16 sections / 16 chunks; `data/PROVENANCE.md`) |
| Test set | 30 dev scenarios, 40 turns (`eval/dev_scenarios/`, still `draft_pending_human_review`); the held-out benchmark set is private and was not available |
| Systems | **streaming**: `SessionPipeline` (controller -> decomposition -> parallel retrieval -> fusion -> delta refinement -> grounded synthesis); **baseline**: `BaselinePipeline` (full utterance -> hybrid retrieval -> rerank -> synthesis) |
| Profile measured | **offline CPU profile**: `PRISM_LLM_PROVIDER=none`, extractive synthesis, rule decomposer, lexical grounding judge, hashing dense embedder, lexical reranker |
| Profile not measured | **intended profile**: local Ollama 7-8B instruct model (LLM synthesis, LLM decomposer, LLM judge). It needs hardware and a pulled model that were not available (see section 7) |
| Clock | simulated stream time (chunks every 0.8 s, utterance end +0.5 s); compute latencies are real CPU wall time |
| Hardware | CPU-only build container, Python 3.11.15 |
| Command | `python eval/run_eval.py --system full --metrics all` (= `eval/replay.py`); raw output in `results/replay/`, snapshot in `reports/REPLAY_SUMMARY_OFFLINE.md` |

## 2. Gates

| Gate | Target | Result | Status |
|---|---|---|---|
| G1 reproducibility | container launches via one command on a clean machine; replay completes unattended | one-command CLI runner verified on a clean clone (section 6 of `FINAL_CHECKLIST.md`); container not run | **NOT VERIFIED** (requires Docker) |
| G2 early retrieval | >= 80% of eligible queries | 33/34 = **0.971**; false triggers 0/6 | PASS |
| G3 multi-intent identification | >= 70% of compound queries | 9/12 = **0.75** | PASS |
| G4 factual grounding | >= 85% citation support, 0 fabricated ids | 131/131 = **1.0**, 0 fabricated; shuffled-citation control 0.0 | PASS (see the caveat in section 3) |
| G5 session refinement | verified state continuity | **5/5** refinement turns; presentation turns 5/5 with zero retrieval | PASS |
| G6 telemetry | 100% trace coverage | **80/80** requests (40 streaming + 40 baseline) | PASS |

## 3. Streaming system vs baseline

| dimension | metric | streaming | baseline |
|---|---|---|---|
| latency | retrieval starts before the user finishes (eligible turns) | **0.971** | 0.0 (by design: starts at utterance end) |
| latency | mean retrieval head start vs utterance end (stream s) | **0.86 s** | 0 s |
| latency | post-utterance compute latency p50 / p95 (ms) | 5.9 / 9.1 | **2.7 / 3.1** |
| accuracy | answer cites gold sections: recall (answerable turns) | **0.887** | 0.844 |
| accuracy | answer cites gold sections: precision | 0.700 | **0.753** |
| accuracy | uncertainty emitted on no-evidence turns | **3/5** | 1/5 |
| accuracy | false uncertainty on answerable turns | **0/30** | 1/30 |
| accuracy | late constraints refined in place (version bumped, state kept) | **5/5** | not supported (stateless) |
| grounding | fabricated citation ids | 0 | 0 |
| grounding | claims literally contained in their cited chunk | 1.0 | not computed |
| cost | no-retrieval turns that searched the corpus | **0/6** | 6/6 |
| cost | LLM tokens in / out, estimated cost (offline profile) | 0 / 0, $0 | 0 / 0, $0 |
| cost | peak RSS of the whole replay (both systems) | 52.5 MB | (same process) |

Mean per-stage latency of the streaming system (ms per turn, CPU):

| controller | planning | retrieval (all sub-queries) | fusion | synthesis | verification | presentation |
|---|---|---|---|---|---|---|
| 0.51 | 0.20 | 3.45 | 0.88 | 0.75 | 0.35 | 0.05 |

**Reading the numbers**
- **Latency.** Streaming starts retrieval on average 0.86 s of stream time before the user stops
  talking, which is the point of the design. On this CPU path its *post-utterance* compute is
  higher than the baseline's (5.9 vs 2.7 ms). Fusion, per-intent synthesis and verification
  run after the end of speech, and both figures are far below conversational thresholds.
  With an LLM, post-utterance time is dominated by generation, which the streaming system
  overlaps with speech only for retrieval, not for synthesis.
- **Accuracy.** Streaming cites more of the gold sections (recall +0.04), because each
  sub-intent gets its own evidence quota. It also cites more non-gold sections (precision
  -0.05), from the same extra per-intent evidence. The citation metrics are lexical, and the
  dev set is small (31 answerable turns).
- **G4 caveat.** The offline generator emits verbatim corpus spans, so support near 1.0 is
  expected by construction. The meaningful checks here are 0 fabricated ids and the 0.0
  shuffled control, which shows the support test would catch a misattributed claim. G4 on
  LLM-written claims requires the 7-8B runtime.
- **Cost.** Streaming never searched on presentation-only or incomplete turns (0/6); the baseline
  searched on all 6.

## 4. Architectural ablations

**A1. Retrieval controller: rule-only vs rule+stability vs LLM** (Phase 2,
`reports/PHASE_2_CONTROLLER_ABLATION.md`; 78 configurations, selection rule fixed in advance)

| controller | G2 | false triggers | presentation retrievals | head start (s) | retrievals / turn |
|---|---|---|---|---|---|
| rule-only, tuned (**default**) | **0.971** | 0 | 0 | **0.86** | 1.29 |
| rule + stability, tuned | 0.882 | 0 | 0 | 0.46 | **1.00** |
| rule + stability, untuned | 0.706 | 0 | 0 | 0.38 | 1.00 |
| LLM controller (one call per chunk) | requires 7-8B runtime | | | | |

Rule-only retrieves earlier; the stability probe saves about 0.3 retrievals per turn but loses
about 0.4 s of head start, because Jaccard needs two probes, so the first chunk can never be stable.

**A2. Hybrid vs dense-only retrieval, and the anti-fragmentation guard** (Phase 3, `reports/PHASE_3_ABLATION.md`)

| configuration | G3 | single-intent over-split | intent evidence recall | retrievals / turn |
|---|---|---|---|---|
| default: rules + guard, hybrid | 0.75 | **0/22** | 0.976 | 1.56 |
| dense-only (hashing embedder) | 0.75 | 0/22 | **1.000** | 1.56 |
| BM25-only | 0.75 | 0/22 | 0.976 | 1.56 |
| guard OFF | 0.75 | 2/22 | 0.976 | 1.62 |
| fusion quota OFF | 0.75 | 0/22 | 0.952 | 1.56 |
| LLM decomposer | requires 7-8B runtime | | | |

- Hybrid equals BM25-only here, because RRF over 16 chunks is dominated by the BM25 ranking.
- Dense-only finds the one intent hybrid misses (`present_02`). The "dense" retriever is a lexical
  hashing embedder, not a semantic model.
- The guard is what prevents over-splitting (2/22 -> 0/22).
- The quota is what keeps a weak intent's evidence in the fused set.

**A3. Grounding judge** (Phase 4, `reports/PHASE_4_GROUNDING_ABLATION.md`; 131 emitted claims plus corruptions)

| judge | accepts original claims | rejects shuffled citations | rejects swapped text |
|---|---|---|---|
| lexical (CPU default) | 1.0 | 0.992 | 0.985 |
| NLI (`cross-encoder/nli-deberta-v3-small`) | optional model not installed | | |
| LLM judge | requires 7-8B runtime (wiring verified with a fake server) | | |

## 5. Edge-case failures (analysed)

**F1. Cross-turn back-reference is not resolved** (`late_01` turn 2; G5 passes, but the answer is empty).
- **What happened:** turn 2 "…how does it avoid thrashing on noisy partial input?" was correctly
  classified `new_subintent`, and §6 ranked first. No claim was produced; the uncertainty note
  lists "avoid, partial, specifically".
- **Cause:** evidence coverage was 2/6 = 0.33, just under the 0.34 threshold. "specifically" counts as a
  content word, and "it" (= the retrieval controller from turn 1) is not carried across turns,
  because carry-over works within one utterance.
- **Mitigation proposed (not applied, to avoid tuning on seen data):** carry the referenced
  sub-intent's topic into back-referring clauses; add discourse adverbs to the stopword list.

**F2. Topical but unanswerable questions get answered** (`noevid_01` GPU requirement, `noevid_03` Venue A cancellation terms).
- **What happened:** both expect an uncertainty note. Instead each was answered with verbatim
  corpus sentences that share words with the question: "required / run / system", and the
  Venue A example text in §4.1.
- **Cause:** the evidence threshold and the lexical judge check that a claim is *in* its chunk, not
  that it *answers* the question. `noevid_03` is the corpus's own illustration being treated as policy.
- **Mitigation:** an answerability-aware judge (the LLM judge in the intended profile, or NLI).
  Measurement requires the 7-8B runtime.

**F3. Hybrid retrieval misses a component row** (`present_02` turn 1 "What does the multi-intent decomposer do?").
- **What happened:** gold §2.1 (the component table) is not in the top-5 evidence. The diagram (§2)
  and Example 1 (§4.1) outrank it.
- **Cause:** BM25 dominates RRF on this 16-chunk corpus; "multi-intent" and "decomposer" occur in
  several sections. A dense-only run finds it (ablation A2).
- **Mitigation:** a real semantic embedder (`sentence_transformers`, optional dependency) or
  cross-encoder reranking. Not installed in this environment.

**F4. The G3 metric under-credits correct splits** (`late_04` t2, `late_05` t2).
- The splits are structurally right, but lexical matching misses "two" vs "2", "it" vs "phase
  two", and "handle" vs "handling". G3 = 0.75 is therefore a conservative figure.

## 6. Cost and resource summary

| item | offline profile (measured) | intended profile |
|---|---|---|
| model downloads | none | one 7-8B Ollama model (pulled by `docker compose up`) |
| RAM | ~53 MB peak for the whole replay | Ollama model memory + ~0.1 GB application; needs suitable hardware |
| LLM calls per answered turn | 0 | 1 synthesis + 1 decomposition per planning step + 1 judge call per claim |
| wall time, 80-request replay | 0.42 s | dominated by generation; not measured |
| est. cost | $0 | $0 for a local model (token counts are recorded per request) |

## 7. What is verified where

| claim | verified locally (offline profile) | verified with the fake/test LLM | requires the 7-8B Ollama runtime | requires Docker |
|---|---|---|---|---|
| G2, G3, G5, G6 numbers above | yes | pipeline wiring only | quality with LLM decomposer/synthesis | — |
| G4 numbers above | yes (extractive claims) | wiring + 0 fabricated ids | G4 on LLM-written claims | — |
| LLM client: health check, retries, JSON mode, token accounting | stub tests | yes (`tests/test_llm_runtime.py`, `tests/test_replay.py`) | real latency/quality | — |
| LLM controller / decomposer / judge ablation rows | — | wiring | yes | — |
| G1 | CLI single command on a clean clone | — | — | yes |
