# Phase 3 report: Multi-Intent Parsing & Evidence Fusion

Goal (playbook): split compound utterances into non-redundant sub-queries, retrieve in
parallel, and fuse evidence cleanly. Phase 2 was merged to `main` (`ea9ab6e`); this work starts there.

## Done

| Step | What | Commit |
|---|---|---|
| 3.1 | `src/decompose/decomposer.py`: diff operations `add` / `keep` / `merge` over the live sub-query set, plus `apply_ops`. **`LLMDecomposer`**: generic prompt + JSON schema via `src/llm/client.py`, falls back to rules on an LLM error. **`RuleDecomposer`**: CPU default; splits at generic English intent boundaries and diffs clauses against the live set. `PRISM_DECOMPOSER=rules\|llm`. | `7584d62` |
| 3.2 | `src/decompose/dedupe.py`: anti-fragmentation guard. Short-circuits simple utterances to one query, folds thin fragments (< 2 content words), merges sub-queries with embedding cosine >= 0.85, caps at 4. | `5029d3b` |
| 3.3 | `src/decompose/context.py`: extracts quantities with units, dates / relative time, locations and proper names; carries them forward into later sub-queries; pronoun references get the previous sub-query's topic terms. `SubQuery.constraints` added to the schema. | `b024c46` |
| 3.4 | `src/decompose/planner.py` (decomposer -> guard -> carry-over) and `src/engine.py`: the planner runs on retrieve decisions and on stable new content after the provisional search. New or changed sub-queries are searched with `asyncio.gather`, one `retrieval_started` each, trigger `multi_intent`. `decompose=False` keeps the Phase 2 behaviour. | `4ee618a` |
| 3.5 | `src/retrieval/fusion.py`: RRF within each sub-query (hybrid retriever), then across sub-queries: dedupe by chunk id, near-duplicate removal (Jaccard >= 0.9), rerank against each originating sub-query, per-sub-intent quota (2), cap (6). Records which sub-queries each evidence chunk serves. | `7818348` |
| 3.6 | `src/retrieval/cache.py`: per-utterance speculative reuse cache (content-term Jaccard >= 0.75). A final sub-query close to a provisional one reuses that search (even one still in flight). Logged as `cache_hit`; discarded after the utterance. | `9db531a` |
| 3.7 (fix) | A race found while checking ablation determinism: a stale provisional search finishing last overwrote a sub-query's newer results. Fixed. The regression test fails without the fix and passes with it. | `5cad870` |
| 3.7 | `eval/decompose_eval.py` (G3, over-split, per-intent evidence recall), `run_eval.py --system decompose_fusion --metrics g3`, `eval/ablate_decompose.py` -> `reports/PHASE_3_ABLATION.md`, `make decompose / ablate3`. | `679552e` |
| — | This report and the README update | this commit |

## Verification

All commands were actually run in the build container (Python 3.11.15, CPU only, no LLM).

**`make test`**: `190 passed` (134 before Phase 3, 56 new). The same result holds in a fresh
clone with the pinned venv, and `git status` stayed clean.

**`python eval/run_eval.py --system decompose_fusion --metrics g3`** (defaults: rule decomposer + guard, hybrid):

| metric | value |
|---|---|
| G3: compound turns with >= 2 correct sub-intents | 9/12 = **0.75** (target >= 0.70: **PASS**) |
| G3 failures | `late_04` t2, `late_05` t2, `noevid_02` t1 (analysed below) |
| single-intent turns over-split | **0/22** |
| sub-queries per compound turn | 1.92 |
| gold sub-intents whose evidence is in the fused set | 0.976 |
| matched sub-query recall@5 | 0.974 |
| retrievals per eligible turn / cache hits | 1.56 / 2 |
| G2 early retrieval (full pipeline) | 0.97 |
| no-retrieval turns that retrieved | 0 |

**Matching rule** (fixed before any results were inspected):
- Each gold sub-intent is matched one-to-one to a predicted sub-query.
- The match score is the share of the gold description's content words present in the sub-query's own text; carried constraints are excluded.
- A match counts as correct when its score is at least 1/3.

**Ablations** (`reports/PHASE_3_ABLATION.md`; deterministic, 3 identical runs):

| config | G3 | over-split | intent evidence recall | retrievals / turn | cache hits |
|---|---|---|---|---|---|
| full: rules + guard, hybrid (default) | 0.75 | 0/22 | 0.976 | 1.56 | 2 |
| **retrieval: dense-only** (hashing embedder) | 0.75 | 0/22 | **1.000** | 1.56 | 2 |
| retrieval: bm25-only (reference) | 0.75 | 0/22 | 0.976 | 1.56 | 2 |
| **anti-fragmentation guard OFF** | 0.75 | **2/22** | 0.976 | 1.62 | 4 |
| speculative cache OFF | 0.75 | 0/22 | 0.976 | 1.62 | 0 |
| fusion quota OFF | 0.75 | 0/22 | 0.952 | 1.56 | 2 |
| decomposer: llm | not run (no LLM) | | | | |

- **Hybrid vs dense-only:** decomposition is unaffected, since G3 depends only on the planner.
  Hybrid misses one intent's evidence: `present_02` ("What does the multi-intent decomposer
  do?"; gold §2.1). This is the same BM25-dominated miss seen in Phase 1. Dense-only finds it.
  Hybrid equals BM25-only here because RRF over this 16-chunk corpus is dominated by the BM25
  ranking. Caveat: the "dense" retriever is the lexical hashing embedder, not a semantic model.
- **Guard on vs off:** without the guard, `single_02` ("...and how is it validated?") and
  `single_05` ("...and how long can it be?") are over-split into a second, pronoun-only
  fragment. With the guard, over-split is 0/22. G3 is unchanged, because the guard never merged
  a gold-distinct intent in the dev set.
- **Quota off** drops `multi_04`'s second intent (presentation-only handling, §4.3) out of the
  evidence set: 0.976 -> 0.952.
- **Cache off** costs 2 extra searches (1.56 -> 1.62 per turn).

**Regression checks.**
- `--system controller_only --metrics g2`: still 33/34 = 0.97, 0/5 presentation retrievals.
- `--system baseline`: recall@5 0.9516, citations 64/64 valid (unchanged).

**Pass criteria (playbook).**
- G3 >= 70%: **yes**, 0.75.
- Simple queries not over-split: **yes**, 0/22.
- Both required ablations recorded: **yes**. The LLM-decomposer row is not measured.

**Not run, and why**

| Check | Reason |
|---|---|
| LLM decomposer (the playbook's primary design for 3.1) on the dev set | No LLM available. Implemented and unit-tested with the mock client only; the ablation row says "not run". |
| Cross-encoder rerank in fusion | Optional dependency not installed. Fusion uses the configured reranker (lexical by default). |
| `docker compose up` | Out of scope; no Docker daemon. |

## Decomposition failure analysis

**1. `noevid_02` t1: a real pipeline gap.** "What is the submission deadline and how much
prize money is awarded?" None of its words occur in the corpus. The Phase 2 controller
therefore suppresses the turn with `no_corpus_terms`, and the planner never runs, so neither
intent is identified (0/2). The decomposition itself would have worked:
`split_clauses` gives "What is the submission deadline" / "how much prize money is awarded?".
Retrieval is correctly skipped. What is lost is the per-intent structure that Phase 4 needs to
say "no evidence for *each* of these two questions". Proposed fix (not applied, to avoid
tuning on seen results): run the planner, without retrieval, on `no_corpus_terms` turns.

**2. `late_04` t2: correct split, missed by the metric.** "Only phase two, and which gate
measures it?" is split into "Only phase two" / "which gate measures it?", which is the right
structure. Both matches then fall below 1/3:
- gold "phase 2 work items" vs "Only phase two" scores 1/4, because "two" is not "2";
- gold "G2 early retrieval gate" vs "which gate measures it" scores 1/4, because "it" means phase two.

The pronoun carry-over does add "phase two" to the second sub-query's constraints, but
constraints are excluded from matching by design. This shows the metric is strict and
lexical, so G3 = 0.75 may understate the decomposer.

**3. `late_05` t2: correct split, stemming miss.** "…how the system should handle it"
vs gold "expected handling of a late detail": "handle" vs "handling" is not folded by the
light stemmer.

**4. Under-split outside the dev set.** The guard's thin-fragment rule (< 2 content words)
merged "Tell me what the probe does" back into the next clause during a demo run, because
"probe" is its only content word. The dev set contains no such case, so neither G3 nor the
over-split rate shows it. The threshold was not tuned after seeing this.

## Decisions

- **Rule decomposer as the CPU default, LLM decomposer behind the same interface.** The playbook
  specifies an LLM call. Without an LLM, a generic rule splitter keeps the pipeline runnable,
  and it produces the same diff operations, so switching is one setting.
- **The guard runs after either decomposer.** It short-circuits simple utterances even if an LLM
  over-splits them. Its similarity uses the configured embedder, with no extra model.
- **Noun-phrase coordination is not split by the rules** ("the capacity and the catering
  options"). "retrieval and fusion" is usually one intent, and a rule cannot tell the two
  apart. The LLM decomposer is meant to handle this.
- **Constraints propagate forward only**, so a later clause's entity cannot leak into an
  earlier, unrelated clause.
- **A single live sub-query is never re-searched mid-stream.** The Phase 2 one-provisional-
  search limit still holds; only a multi-intent split may search mid-stream.
- **Fusion reranks against each chunk's own sub-queries** (best score wins), then applies the
  quota, so a strong intent cannot crowd out a weak one (see the quota ablation).
- **The cache is per utterance**, so no state crosses turns or sessions.
- **The G3 matching rule was fixed in advance** and excludes carried constraints, so context
  injection cannot inflate G3.
- **Component cost (architectural parsimony).**
  - Rule decomposition, guard and carry-over are regex and set operations plus one hashing
    embedding per sub-query (well under 1 ms).
  - Each sub-query adds one hybrid search (about 1-2 ms on this corpus).
  - Fusion is one rerank call per sub-query.
  - The full ablation over 6 configurations x 40 turns runs in about 2 s on CPU.
  - The LLM decomposer would add one LLM round-trip per planning call.

## Known issues

1. **The LLM decomposer is unmeasured.** The rule fallback handles grammatical boundaries only (no
   noun-phrase coordination, no semantic splits).
2. **Tuned and evaluated on draft data**: 12 compound turns is a small sample, and the dev
   scenarios are still `draft_pending_human_review`. No Phase 3 threshold was tuned, but the
   defaults were chosen with this data in view.
3. **The G3 metric is lexical** (number words, pronouns, "-ing" forms). See failures 2 and 3.
4. **Mid-stream re-search of a growing sub-query.** In the demo, "which gate measures"
   (0.8 s) and "which gate measures early retrieval" (1.6 s) were both searched: their
   similarity is 0.5 < 0.75, so the cache did not apply. That adds retrievals (1.56 per turn).
5. **`no_corpus_terms` turns skip planning** (failure 1).
6. **The guard can under-split a clause whose only content word is its topic** (failure 4).
7. **Hybrid misses `present_02`'s evidence** (known since Phase 1). This is where a real dense
   embedding model (teammate setup) would be expected to help.

## Questions for the human

1. Should the planner also run, without retrieval, on `no_corpus_terms` turns, so Phase 4 can
   report uncertainty per sub-intent (failure 1)? This would be applied in Phase 4.
2. Are the dev scenarios approved? More compound scenarios (especially noun-phrase coordination
   and 3+ intents) would make G3 more meaningful.
3. When the teammate's LLM is available, should the LLM-decomposer ablation row be filled in
   before Phase 4?

## Ready for the next phase?

**Yes.**
- All Phase 3 steps are implemented, committed and tested (190 tests).
- G3 = 0.75 >= 0.70, 0/22 simple queries over-split, and both required ablations are recorded.
- Phase 2 and Phase 1 metrics are unchanged.
- Open items: the LLM-decomposer measurement (needs the teammate's LLM) and the failure 1 design question.

PHASE 3 COMPLETE, awaiting "proceed to phase 4"
