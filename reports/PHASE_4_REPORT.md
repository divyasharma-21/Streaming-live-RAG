# Phase 4 report: Session Refinement & Grounding

Goal (playbook): update answers in place when late constraints arrive, and guarantee that
every claim is grounded. Phase 3 was merged to `main` (`a9507a2`); this work starts there.

## Done

| Step | What | Commit |
|---|---|---|
| 4.1 | `src/session/store.py`: in-memory `SessionStore`. Exact-id lookup only (no listing/search API), an unknown id is an error, `end()` destroys the session, no disk I/O. Isolation tests, including an AST check that the module imports no persistence library. | `f7876b2` |
| 4.2 | `src/session/ledger.py`: `ClaimLedger` with claims (id, text, sub-intent id, chunk ids, **dependent constraints** (new `Claim.constraints` field), version), current sub-queries and their evidence sets, invalidated claims, answer-version history, `snapshot()` / `diff()`. | `09c37f7` |
| 4.3 | `src/synthesis/generator.py`: `synthesize_subintents`. With an LLM: one JSON call over all sub-intents, the model is told to use only each sub-question's evidence, and a claim citing a chunk outside its own sub-intent's evidence is dropped. Without an LLM: the extractive generator per sub-intent. Rendered with `[Doc_ID §Section]`. | `da195e0` |
| 4.4 | `src/synthesis/grounding.py`: `GroundingVerifier`. (a) cited ids must exist (fabricated ids removed; a claim with none left fails); (b) a support judge must accept the claim. Judges: `lexical` (CPU default), `nli` (optional model), `llm` (LLM as judge). Failures are dropped or flagged. | `0fa7eef` |
| 4.5 | `src/synthesis/uncertainty.py`: `subintent_uncertainty` (no evidence / best evidence below threshold, naming the missing terms and asking to clarify / claims failing verification / generator gap); `turn_clarification` for unsearchable turns. | `6a82912` |
| 4.6 | `src/session/delta.py` (clause-level classifier: `new_subintent` / `parameter_update` / `contradiction` / `presentation_only`) and `src/session/pipeline.py` (`SessionPipeline`, the full system): targeted search of the new utterance only, parameter updates add constraint + delta evidence + delta claims, contradictions invalidate only the dependent claims (in every affected sub-intent), untouched claims keep id/text/citations/version, answer version bumps. Ledger-diff tests. | `abe0dbd` |
| 4.7 | `src/synthesis/presentation.py`: bullets / numbered / table / shorter / repeat, with no retrieval. Each line keeps its claim's citations; the pipeline asserts no new id appears. Translation requires an LLM (output accepted only if the citation set is unchanged); without one the answer is shown unchanged with a note. | `7ac3b58` |
| 4.8 | `eval/full_eval.py` + `run_eval.py --system full --metrics g4,g5`; `eval/ablate_grounding.py` -> `reports/PHASE_4_GROUNDING_ABLATION.md`; `make full / ablate4`. | `c3d3551` |
| — | This report and the README update | this commit |

## Verification

All commands were actually run in the build container (Python 3.11.15, CPU only, no LLM).

**`make test`**: `252 passed` (190 before Phase 4, 62 new). The same result holds in a fresh
clone with the pinned venv, and `git status` stayed clean.

**`python eval/run_eval.py --system full --metrics g4,g5`** (defaults: extractive generator,
lexical grounding judge, rule decomposer). Deterministic: two runs gave identical output.

| metric | value |
|---|---|
| G4 citation support (literal check, independent of the pipeline's judge) | 131/131 = **1.0** (target >= 0.85) |
| fabricated ids | **0** |
| pre-verification support (pipeline judge) / claims dropped | 0.985 / 2 |
| shuffled-citation control (same check, claims re-cited to another section) | **0.0** |
| G5 refinement turns with state continuity | **5/5** (3 parameter_update, 2 new_subintent, 0 contradiction) |
| presentation-only turns: zero retrieval calls and no new citation | **5/5** (0 retrieval calls) |
| uncertainty on turns that expect it / false uncertainty on answerable turns | 3/5 / 0/30 |

**Reading G4 honestly.**
- The generator is extractive, so emitted claims are verbatim corpus spans, and support near 1.0 is expected by construction.
- What G4 = 1.0 does show: every claim is attached to a chunk that literally contains it, and no id is fabricated. The shuffled control falling to 0.0 shows the check is not vacuous.
- A G4 that tests generated (paraphrased) text needs the LLM generator plus an NLI or LLM judge. Neither is available here.

**G5 check per refinement turn:**
- every earlier sub-intent is still in the session, and the answer version is the previous version + 1;
- no claim outside the affected sub-intents was removed or changed (from the ledger diff);
- no earlier turn's retrieval query was issued again (only the new utterance's clauses are searched).

**Delta decisions on the dev follow-ups** (for review; the scenarios carry no gold delta labels):

| turn | utterance (abridged) | clause decisions |
|---|---|---|
| late_01 t2 | "…how does it avoid thrashing on noisy partial input?" | new_subintent |
| late_02 t2 | "Actually, focus on grounding, and include the threshold…" | parameter_update(t1.q1) + new_subintent |
| late_03 t2 | "…what does the final output record look like for it?" | new_subintent |
| late_04 t2 | "Only phase two, and which gate measures it?" | parameter_update(t1.q1) + new_subintent |
| late_05 t2 | "Actually, I mean the late-arriving constraints problem…, and how…" | parameter_update(t1.q1) + new_subintent |

`contradiction` does not occur in the dev scenarios. It is covered by unit and pipeline tests
(`tests/test_delta.py`). One of them proves with a ledger diff that exactly the claims
depending on the negated term are removed and the original answer's claims are untouched.

**Grounding-judge ablation** (`reports/PHASE_4_GROUNDING_ABLATION.md`, 131 claims):

| judge | accepts original claims | rejects shuffled citations | rejects swapped text |
|---|---|---|---|
| lexical (0.80) | 1.0 | 0.992 | 0.985 |
| NLI | not run: optional model not installed | | |
| LLM judge | not run: no LLM configured | | |

**Regression checks** (all unchanged):
- G3 = 0.75 (0/22 over-split);
- G2 = 0.97 (0/5 presentation retrievals);
- baseline recall@5 = 0.9516, citations 64/64 valid.

**Pass criteria (playbook).**
- G4 (>= 85% support, zero fabricated ids): **yes**, with the caveat above.
- G5: **yes**, 5/5.
- A delta update touches only the affected claims, checked by ledger-diff tests: **yes**.
- Suppression turns run with zero retrieval calls: **yes**, 5/5 on the dev set plus a unit test that counts search calls.

**Not run, and why**

| Check | Reason |
|---|---|
| NLI vs LLM-judge comparison (the playbook's 4.4 ablation) | Neither judge can run here (optional model not installed, no LLM). Both are implemented; the LLM judge is tested with a mock client, and the NLI judge is untested. Only the lexical CPU judge is measured. |
| LLM synthesis (the playbook's 4.3 design) on the dev set | No LLM. Implemented and tested with a mock client (own-evidence id restriction, dropped fabrications). |
| Translation in the presentation path | Needs an LLM. Without one the answer is shown unchanged with an explicit note. |
| `docker compose up` | Out of scope; no Docker daemon. |

## Refinement failure analysis

**`late_01` turn 2: a follow-up question that gets no answer.**
- **Turn 1:** "What does the retrieval controller decide?" was answered from §2.1.
- **Turn 2:** "And specifically how does it avoid thrashing on noisy partial input?" was
  correctly classified `new_subintent` and searched on its own clause.
- **Outcome:** §6 ("Eager / Premature Retrieval on Noise… The controller must wait for semantic
  intent boundaries") ranked first in the evidence, yet no claim was produced. The note reads:
  *"The corpus does not contain sufficient evidence for … (not found: avoid, partial, specifically)"*.

Causes:
1. **Evidence-threshold arithmetic.** The query has 6 content terms (specifically, avoid,
   thrashing, noisy, partial, input); §6 contains 2 (thrashing, noisy, since "Noise" ≠ "noisy"):
   2/6 = 0.33, just under `PRISM_MIN_EVIDENCE_SCORE` = 0.34.
2. **Filler words count.** "specifically" is a discourse word, not a content word, but the
   generic stopword list does not contain it.
3. **No cross-turn carry-over.** "it" means *the retrieval controller* from turn 1, but Phase 3
   carry-over works within one utterance only. The clause is therefore searched without its topic.

The system behaved safely (an explicit uncertainty note, not a guess), but the answer was in the
corpus. Proposed fixes, not applied to avoid tuning on seen results:
- carry the referenced sub-intent's topic terms into a `new_subintent` clause that refers back;
- treat discourse adverbs ("specifically", "exactly", "basically") as stopwords.

## Decisions

- **Clause-level delta classification.** A follow-up often mixes a constraint and a new
  question ("Only phase two, and which gate measures it?"). Classifying each planner clause and
  deriving the turn's kind by priority (contradiction > parameter_update > new_subintent) handles
  both parts.
- **"Actually" is not a contradiction.** Contradiction needs an explicit negation of a term the
  session holds. "Actually, the trip was international" is a parameter update, as in the PDF
  example ([Doc_01 §4.2]).
- **Negated terms exclude words the correction re-uses.** "Not the X rule, I meant the Y rule"
  negates X, not "rule". A contradiction affects every sub-intent whose stated text or constraints
  hold the negated term, and falls back to claims only if none do.
- **Parameter updates keep existing claims and add delta claims.** This mirrors the PDF's refined
  response ("The standard reimbursement rule still applies. However…"): prior citations are
  preserved and delta citations added.
- **Targeted search.** Only the follow-up's clauses are searched (the PDF: "dispatches targeted
  queries"). Earlier sub-intents are never re-searched; their evidence stays in the ledger. This is
  how "without re-running full-corpus search" is implemented. Each targeted search still ranks the
  whole 16-chunk corpus, as any search does.
- **A presentation turn keeps the answer version** (the content is unchanged); `last_output`
  becomes the transformed text so a later transform works on what the user saw.
- **The presentation assertion is a hard error.** A transform that introduced a citation would
  raise, not silently pass.
- **G4 uses a check independent of the pipeline judge**, plus a shuffled control. Re-using the
  judge that already filtered the claims would report 100% by construction.
- **Suppressed `no_corpus_terms` turns now run the planner without retrieval**, so each intent
  gets its own "no evidence" note. This was the open Phase 3 question (`noevid_02`): it now
  yields two per-intent notes.
- **Component cost (architectural parsimony).**
  - The store, ledger, delta classifier and presentation transforms are dictionaries, regex and set operations.
  - The lexical judge is one set intersection per claim.
  - A full 30-scenario, 40-turn evaluation runs in about 1 s on CPU.
  - With an LLM configured, each answered turn adds one synthesis call, plus one judge call per claim if `PRISM_GROUNDING_JUDGE=llm`.

## Known issues

1. **Unmeasured LLM/NLI paths.** LLM synthesis, the LLM judge, the NLI judge and translation are
   implemented but untested against real models. G4 is therefore measured on verbatim extractive
   claims only.
2. **No-evidence detection 3/5.** `noevid_01` ("What GPU is required…") and `noevid_03` (Venue A
   cancellation terms) are still answered with verbatim but irrelevant corpus sentences. The
   lexical judge checks that a claim is *in* its chunk, not that it *answers* the question.
   An answerability judge (NLI or LLM) is needed.
3. **Extractive answers are verbose and sometimes off-topic**, e.g. a §8 deliverables row
   answering "hard engineering rules". Relevance is lexical.
4. **No cross-turn carry-over for back-references** (the failure above).
5. **Contradiction handling is exercised only by unit tests**; the dev set has no contradiction case.
6. **Delta classification is lexical.** Paraphrased constraints with no overlapping word attach
   to the latest sub-intent only if they contain a back-reference word.
7. **The dev scenarios are still drafts, with no gold delta labels.** The table above is for review.

## Questions for the human

1. Should gold delta labels (`new_subintent` / `parameter_update` / `contradiction`) be added to
   the late-constraint scenarios, together with 1-2 contradiction scenarios, so delta
   classification can be measured rather than listed?
2. Should the two proposed fixes from the failure analysis (cross-turn topic carry, discourse
   stopwords) be applied, and if so, in Phase 5 or as a Phase 4 follow-up?
3. Once the teammate's LLM is available, should the unmeasured paths be run and reported before
   the Phase 5 benchmark: LLM synthesis, LLM judge, and the NLI judge if the model is allowed?

## Ready for the next phase?

**Yes.**
- All Phase 4 steps are implemented, committed and tested (252 tests).
- G4 and G5 thresholds are met, ledger-diff tests prove delta updates touch only affected
  claims, and presentation turns make zero retrieval calls.
- Main caveat: the grounding numbers come from the extractive CPU path, and the LLM/NLI comparison
  still needs the teammate's infrastructure.

PHASE 4 COMPLETE, awaiting "proceed to phase 5"
