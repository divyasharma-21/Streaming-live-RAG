# Phase 1 report

## Intake (step 1.1)

| Question | Answer (source) |
|---|---|
| Corpus location and format | No separate corpus was supplied. The owner instructed that the initial corpus is built **only** from `Theme_4_Guide_RAG.pdf`. It is transcribed to `data/corpus/Doc_01.md` (Markdown with explicit `[§N]` section headers). See `data/PROVENANCE.md`. |
| Allowed models | Not fixed yet. Owner: "do not require a large local model"; the final LLM infrastructure is decided by a teammate. The playbook suggests Ollama with a local instruct model, `BAAI/bge-small-en-v1.5`, and `cross-encoder/ms-marco-MiniLM-L-6-v2`. None of them were downloaded or verified in Phase 1. |
| Hardware | CPU-only, about 4 GB RAM (owner). |
| Rules beyond the PDF | None supplied. The hard constraints are PDF §3 and the playbook's standing rules. |
| Infrastructure | Docker, WSL, Ollama and cloud are **out of scope** for this phase (owner). Dockerfile and compose are configuration only. |

## Done

| Step | What | Commit |
|---|---|---|
| 1.1 | Intake answers; corpus `Doc_01.md` (16 sections) transcribed from the image-only PDF; `MANIFEST.json` hashes; `data/PROVENANCE.md`; integrity test | `1c51cc3` |
| 1.2 | Repo layout; `pyproject.toml` + pinned `requirements*.txt` + `uv.lock`; `Dockerfile`, `docker-compose.yml` (config only); `Makefile` (`test index baseline replay eval audit schema manifest lock`); `.env.example`; `src/config.py`; placeholders for Phase 2-4 modules; layout + eval-isolation + LLM-isolation tests | `d7cd449` |
| 1.3 | `src/schemas.py` (all playbook models + `OutputRecord` matching the PDF record); JSON Schemas in `schemas/`; validation tests including the PDF example | `76c0679` |
| 1.4 | Loader (explicit section markers, fence-aware, strict validation) and section-aware chunker (atomic tables/lists/code, header-repeating table splits, overlap, title+heading prefix, ids `Doc_01 §S#n`) | `3e33d99` |
| 1.5 | `corpus/audit.py`: counts, length histogram, duplicate/empty/short/oversized chunks | `5c17be8` |
| 1.6 | BM25 (`rank_bm25`), dense retriever with pluggable embedder (hashing default, sentence-transformers optional, FAISS optional, on-disk embedding cache), RRF (k=60), hybrid; `build_index.py` | `0578df8` |
| 1.7 | `src/llm/client.py`: async `generate(prompt, schema=None)` returning text + token counts; JSON mode with parse/validate and one retry; Ollama, OpenAI-compatible and mock providers; `llm_call` telemetry with tokens and cost estimate; JSONL telemetry logger | `c9f76dc` |
| 1.8 | `baseline.py`: hybrid retrieval -> rerank -> evidence threshold -> extractive or LLM synthesis -> citation check -> uncertainty -> `OutputRecord`, with a full per-request telemetry trace | `f364c02` |
| 1.9 | 30 dev scenarios in `eval/dev_scenarios/` (status `draft_pending_human_review`) + `eval/scenarios.py` schema + validation tests | `085d68f` |
| 1.10 | `eval/run_eval.py --system baseline`: recall@5, citation validity, latency, plus uncertainty, suppression and per-retriever recall | `a6afed6` |
| — | README and this report | this commit |

## Verification

All commands below were actually run in the build container (Linux, Python 3.11.15, CPU).

**`make test`**: `72 passed in 2.90s`.

**Clean checkout.** `git clone` of the repo into a fresh directory, then a fresh venv with
`pip install -r requirements-dev.txt`. Installed pins: pydantic 2.13.5, rank-bm25 0.2.2,
numpy 2.4.6, pytest 9.1.1, pytest-asyncio 1.4.0, jsonschema 4.26.0. Results:
`make test` 72 passed; `make index`, `make baseline` and `make eval` succeeded; `git status` clean afterwards.

**`make audit`**

```
  documents        1
  sections         16
  chunks           16
  tokens (words)   total=1971 min=17 max=197 mean=123.2
         0-49 | #####                3
        50-99 | ####                 2
      100-199 | #################### 11
      200-299 |                      0
  duplicate chunks 0 / empty chunks 0 / oversized chunks 0
  short chunks (<30 body words, informational) 3 ['Doc_01 §0#1', 'Doc_01 §4#1', 'Doc_01 §4.1#1']
  status           OK
```

**`make index`**: `{"chunks": 16, "sparse": "bm25okapi", "dense": "hashing-1024", "dense_dim": 1024, "build_ms": 20.9}`

**`make baseline`** (sample query "What are the technical evaluation gates and their target thresholds?").
It returned a schema-valid record with citations `["Doc_01 §5", "Doc_01 §6"]`,
`invalid_citations=[]`, `uncertainty: null`, latency 4.9 ms.

**`python eval/run_eval.py --system baseline`** (30 scenarios, 40 turns; defaults: hashing
dense, lexical rerank, no LLM, extractive synthesis):

| metric | value |
|---|---|
| recall@5 (mean over 31 turns with gold evidence) | 0.9516 (29 perfect, 1 zero) |
| citation validity | 64/64 = 1.0 (**0 fabricated**) |
| latency ms p50 / p95 / mean | 4.03 / 5.06 / 4.23 |
| uncertainty emitted on the 5 turns that expect it | 1/5 |
| false uncertainty on 30 answerable turns | 1/30 |
| no-retrieval turns that retrieved anyway | 6/6 (expected: the baseline has no controller) |
| retrieval-only recall@5 bm25 / dense(hashing) / hybrid | 0.9677 / 0.8280 / 0.9677 |

Peak memory of the full eval process: about 50 MB RSS. Wall time 0.84 s.

**Other checks run.** `scripts/export_schemas.py --check` passed (schemas current).
`scripts/corpus_manifest.py` passed (hashes match). `docker-compose.yml` parses as YAML
(PyYAML) with services `prism` and `ollama`.

**Not run, and why**

| Check | Reason |
|---|---|
| `docker compose up` (playbook verify, gate G1) | Docker/infra is out of scope for this phase (owner). The Docker CLI is present in the container, but there is no daemon (`/var/run/docker.sock` missing). The image has never been built. |
| Any real LLM call (Ollama or hosted) | No LLM server available; out of scope. The HTTP clients were tested only against a local stub server imitating the documented response shapes. |
| `sentence-transformers` dense backend, cross-encoder reranker, FAISS | Optional heavy dependencies (PyTorch) not installed. Code paths are marked `pragma: no cover` and untested. |
| NLI / LLM-judge grounding | Phase 4. |

The playbook's pass criteria for Phase 1:
- **Schema tests pass:** yes.
- **All baseline citations resolve to real chunk ids:** yes, 64/64.
- **Baseline metrics print:** yes.
- **The compose command works on a clean checkout:** not verified (see above).

## Decisions

- **One document, sections from the PDF's own numbering.** The source is one guide.
  Splitting it into invented "documents" would fabricate structure. Unnumbered sub-headings
  get dotted ids (`§1.1`, `§4.1.2`), and the ids are written into the file, not inferred.
- **The PDF is not committed.** Every page carries personal footer information. Provenance is
  recorded as a SHA-256 hash instead.
- **One chunk per section, even when shorter than 200 tokens.** The whole corpus is about
  2,000 words. Merging sections to reach the 200-400 token target would make one chunk span
  several `§Section`s and weaken citation precision. The chunker supports splitting (tested
  with small budgets) for a future larger corpus.
- **Token counts are whitespace words.** This avoids downloading a tokenizer. They
  under-count model tokens.
- **Hashing embedder as the default dense backend.** It keeps the pipeline runnable on
  4 GB / CPU with no downloads, behind the same interface as sentence-transformers. It is
  lexical, not semantic, and scores lower than BM25 on the dev set (0.83 vs 0.97 recall@5),
  so the "dense" ablation numbers are **not** representative of a real embedding model.
- **The extractive generator is the default instead of an LLM.** With no LLM configured, the
  baseline quotes corpus sentences verbatim and cites each one. That guarantees grounding but
  does not summarise or reason. The LLM path is implemented and tested with a mock client.
- **Lexical coverage is used as the evidence score for uncertainty**, independent of the
  reranker, so the threshold means the same thing whichever reranker is configured.
  `PRISM_MIN_EVIDENCE_SCORE=0.34` was chosen a priori and has not been tuned.
- **One logic fix made after the first eval run.** When a request matched a section
  heading but no body sentence, the extractive generator produced nothing and reported false
  uncertainty. It now falls back to the section's leading sentences. False uncertainty went
  from 3/30 to 1/30; recall and citation validity were unchanged. No thresholds were tuned
  on the dev set.
- **Citation-like text inside the corpus is neutralised.** Corpus text such as
  `[Doc_ID §Section]` is unwrapped in claim text, so the only bracketed citations in an answer
  are ones the pipeline attached.
- **The LLM client uses stdlib `urllib`**, not an SDK, to keep dependencies minimal. It is the
  only module allowed to do HTTP, which a test enforces.
- **Placeholders for Phase 2-4 modules contain docstrings only** (layout step 1.2). No Phase 2 logic was written.
- **Component cost (for architectural parsimony).** Retrieval + rerank + extractive synthesis
  take about 4 ms per request and about 50 MB RSS. BM25 and hashing index build takes about 21 ms.

## Known issues

1. **recall@5 is close to its ceiling and says little.** There are only 16 chunks, so top-5
   covers 31% of the corpus. The metric becomes informative only with a larger corpus.
2. **Weak no-evidence detection (1/5).**
   - `noevid_01` ("What GPU is required to run the system?") passes the threshold on generic
     words (*required*, *run*, *system*).
   - `noevid_03` (Venue A cancellation terms) matches the example text in §4.1.1/§4.1.2.
     The baseline cites the illustration as if it were policy, which is exactly the
     "illustration vs. knowledge" risk noted in the provenance file.
   - `noisy_03` ("so for the corpus can you") is answered instead of prompting a clarification.
   - Lexical coverage cannot tell topical match from answerability. This needs Phase 4
     grounding (NLI / LLM judge) and the Phase 2 controller.
3. **Recall failure `present_02` turn 1** ("What does the multi-intent decomposer do?"): gold
   §2.1 is not in the top 5. The diagram in §2 and the example in §4.1 outrank the component
   table. Candidate for the Phase 3 hybrid/dense ablation.
4. **Extractive answers are fragmentary.** At most 2 units per chunk and 4 per answer, so the
   sample query lists only G1-G2 of six gates. Diagram lines can surface as claims (e.g.
   `| [2] Multi-Intent Decomposer |`). This is inherent to extractive synthesis without an LLM.
5. **Partial-evidence multi-intent (`multi_07`)** answers the grounded part and emits no
   uncertainty for the missing part. Per-sub-intent uncertainty needs Phase 3 decomposition.
6. **Presentation-only turns retrieve (6/6)**, as expected for the baseline. Suppression is Phase 2.
7. **G1 is unverified**: Docker was never built (see Verification).
8. The `requirements*.txt` files and `pyproject.toml` are synced by hand. The optional extras
   are unpinned (`>=`) until someone tests them.

## Dev scenarios for human review (step 1.9)

All 30 files are marked `review_status: draft_pending_human_review`. They were written by
reading the corpus. Gold labels are section-level citations, and a test checks that each
one exists in the corpus. Every utterance is our own phrasing; none is copied from the
PDF's examples. Please check especially:
- `late_04` turn 2: the phase 2 -> G2 mapping is inferred from matching wording.
- `noisy_03`: incomplete utterance, labelled no-retrieval + clarification.
- `noevid_03`: an illustration entity labelled as no evidence.
- `single_05`: two facets of one deliverable, labelled single intent.

| Scenario | Category | Turn | Utterance | Retrieval required | Gold sub-intents -> supporting sections | Expect uncertainty |
|---|---|---|---|---|---|---|
| late_01 | late_constraint | 1 | What does the retrieval controller decide? | yes | s1: retrieval controller decisions -> Doc_01 §2.1 (also: Doc_01 §2) | no |
| late_01 | late_constraint | 2 | And specifically how does it avoid thrashing on noisy partial input? | yes | s2: avoiding premature retrieval on noise -> Doc_01 §6 (also: Doc_01 §2.1) | no |
| late_02 | late_constraint | 1 | Summarize the hard engineering rules. | yes | s1: hard engineering rules -> Doc_01 §3 | no |
| late_02 | late_constraint | 2 | Actually, focus on grounding, and include the threshold it is evaluated against. | yes | s2: factual grounding rule -> Doc_01 §3; s3: G4 grounding threshold -> Doc_01 §5 | no |
| late_03 | late_constraint | 1 | What does Example 1 show? | yes | s1: Example 1 incremental multi-intent utterance -> Doc_01 §4.1 (also: Doc_01 §4.1.1) | no |
| late_03 | late_constraint | 2 | And what does the final output record look like for it? | yes | s2: Example 1 structured output event record -> Doc_01 §4.1.2 | no |
| late_04 | late_constraint | 1 | What are the phases of the implementation roadmap? | yes | s1: roadmap phases -> Doc_01 §7 | no |
| late_04 | late_constraint | 2 | Only phase two, and which gate measures it? | yes | s2: phase 2 work items -> Doc_01 §7; s3: G2 early retrieval gate -> Doc_01 §5 | no |
| late_05 | late_constraint | 1 | What problem does streaming live RAG address? | yes | s1: problem overview -> Doc_01 §1 (also: Doc_01 §0, Doc_01 §1.1) | no |
| late_05 | late_constraint | 2 | Actually, I mean the late-arriving constraints problem specifically, and how the system should handle it. | yes | s2: late-arriving constraints problem -> Doc_01 §1; s3: expected handling of a late detail -> Doc_01 §4.2 (also: Doc_01 §1.1, Doc_01 §2.1) | no |
| multi_01 | multi_intent | 1 | I'd like to know the target for early retrieval, and also how multi-intent identification is measured. | yes | s1: G2 early retrieval target threshold -> Doc_01 §5; s2: G3 multi-intent identification validation method -> Doc_01 §5 | no |
| multi_02 | multi_intent | 1 | What is the session-bound state rule, and what should the demo video show? | yes | s1: session-bound state rule -> Doc_01 §3; s2: demonstration video contents -> Doc_01 §8 | no |
| multi_03 | multi_intent | 1 | Tell me what the evidence fusion component does, what the citation hallucination pitfall is, and what the grounding gate threshold is. | yes | s1: evidence fusion and reranking responsibilities -> Doc_01 §2.1 (also: Doc_01 §2); s2: citation hallucination pitfall -> Doc_01 §6; s3: G4 factual grounding threshold -> Doc_01 §5 | no |
| multi_04 | multi_intent | 1 | How should the system handle late-arriving constraints and what does it do for presentation-only requests? | yes | s1: handling of late-arriving constraints -> Doc_01 §4.2 (also: Doc_01 §1, Doc_01 §1.1, Doc_01 §2.1); s2: handling of presentation-only requests -> Doc_01 §4.3 (also: Doc_01 §6) | no |
| multi_05 | multi_intent | 1 | What does the telemetry schema deliverable need to capture, and which gate is about trace coverage? | yes | s1: telemetry and observability schema deliverable -> Doc_01 §8; s2: G6 telemetry trace coverage gate -> Doc_01 §5 | no |
| multi_06 | multi_intent | 1 | Which phase covers the answer delta engine, and what does architectural parsimony mean? | yes | s1: roadmap phase containing the answer delta engine -> Doc_01 §7; s2: architectural parsimony rule -> Doc_01 §3 | no |
| multi_07 | multi_intent | 1 | What is the target threshold for factual grounding, and what is the budget for cloud hosting? | yes | s1: G4 factual grounding threshold -> Doc_01 §5; s2: cloud hosting budget -> (none) | yes |
| multi_08 | multi_intent | 1 | In the example timeline, what happens at 0.8 seconds, and what fields does the output record contain? | yes | s1: controller action at 0.8 s in the example timeline -> Doc_01 §4.1.1; s2: fields of the structured output event record -> Doc_01 §4.1.2 | no |
| noevid_01 | no_evidence | 1 | What GPU is required to run the system? | yes | s1: required GPU -> (none) | yes |
| noevid_02 | no_evidence | 1 | What is the submission deadline and how much prize money is awarded? | yes | s1: submission deadline -> (none); s2: prize money -> (none) | yes |
| noevid_03 | no_evidence | 1 | What are the cancellation terms for Venue A in Pune? | yes | s1: Venue A cancellation terms -> (none) (also: Doc_01 §4.1.1, Doc_01 §4.1.2) | yes |
| noisy_01 | noisy_fragment | 1 | um so the the what's the uh threshold for early retrieval again? | yes | s1: G2 early retrieval threshold -> Doc_01 §5 | no |
| noisy_02 | noisy_fragment | 1 | I need, wait, no, what does the delta engine... the answer delta thing mutate when constraints arrive? | yes | s1: answer delta engine behaviour -> Doc_01 §7 (also: Doc_01 §2.1, Doc_01 §4.2) | no |
| noisy_03 | noisy_fragment | 1 | so for the corpus can you | no | (none) | yes |
| present_01 | presentation_only | 1 | What are the common technical pitfalls? | yes | s1: common technical pitfalls -> Doc_01 §6 | no |
| present_01 | presentation_only | 2 | Can you shorten that to one sentence? | no | reuse turn 1, no retrieval | no |
| present_02 | presentation_only | 1 | What does the multi-intent decomposer do? | yes | s1: multi-intent decomposer role -> Doc_01 §2.1 (also: Doc_01 §2) | no |
| present_02 | presentation_only | 2 | Put that as a numbered list please. | no | reuse turn 1, no retrieval | no |
| present_03 | presentation_only | 1 | What are the engineering deliverables? | yes | s1: engineering deliverables -> Doc_01 §8 | no |
| present_03 | presentation_only | 2 | Translate your previous answer into Hindi. | no | reuse turn 1, no retrieval | no |
| present_04 | presentation_only | 1 | Which evaluation gates exist? | yes | s1: evaluation gates -> Doc_01 §5 | no |
| present_04 | presentation_only | 2 | Say that again but more simply. | no | reuse turn 1, no retrieval | no |
| present_05 | presentation_only | 1 | What is the target challenge? | yes | s1: target challenge -> Doc_01 §1.1 | no |
| present_05 | presentation_only | 2 | Reformat that as a table. | no | reuse turn 1, no retrieval | no |
| single_01 | single_intent | 1 | What is the core engineering challenge of the retrieval controller? | yes | s1: retrieval controller core engineering challenge -> Doc_01 §2.1 | no |
| single_02 | single_intent | 1 | Which gate covers reproducibility and how is it validated? | yes | s1: G1 reproducibility gate and its validation method -> Doc_01 §5 (also: Doc_01 §8) | no |
| single_03 | single_intent | 1 | What does the corpus isolation rule require? | yes | s1: corpus isolation rule -> Doc_01 §3 | no |
| single_04 | single_intent | 1 | Why is over-fragmenting sub-queries a problem? | yes | s1: over-fragmenting sub-queries pitfall -> Doc_01 §6 (also: Doc_01 §2.1) | no |
| single_05 | single_intent | 1 | What should the system architecture brief cover and how long can it be? | yes | s1: architecture brief contents and page limit -> Doc_01 §8 | no |
| single_06 | single_intent | 1 | What happens in phase three of the implementation roadmap? | yes | s1: roadmap phase 3 work items -> Doc_01 §7 | no |

## Questions for the human

1. **Corpus:** is the Theme 4 guide the only corpus, or will the organisers supply an
   evaluation corpus? The whole pipeline assumes `data/corpus/*.md` with explicit section
   markers, so a new corpus needs a conversion step.
2. **Dev scenarios:** please approve, edit or reject the 30 scenarios above (flip
   `review_status` to `approved`). Phase 2 thresholds should be tuned only on approved scenarios.
3. **Models:** which LLM / endpoint will the teammate provide, and are
   `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` allowed and within the RAM budget?
4. Should the source PDF be stored anywhere (e.g. privately) for provenance, given it cannot be committed?

## Ready for the next phase?

**Yes, with one open item.**
- Every Phase 1 step is implemented, committed and tested (72 tests).
- Baseline citations are 100% valid, and the metrics print.
- **Open:** the `docker compose up` part of the verification (gate G1) has not been run. By
  the owner's instruction it belongs to the infrastructure teammate.
- The dev scenarios should be reviewed before Phase 2 uses them for tuning.

PHASE 1 COMPLETE, awaiting "proceed to phase 2"
