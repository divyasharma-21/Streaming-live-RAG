# PRISM Streaming Live RAG: architecture brief

## 1. Problem and constraints

A conversational assistant must answer from a supplied corpus while the user is still speaking.
It must handle several questions in one utterance, refine answers when late details arrive, and
ground every claim ([Doc_01 §1], [Doc_01 §1.1]).

The hard rules ([Doc_01 §3]) shape every decision:
- corpus isolation;
- no hard-coded benchmark prompts or answers;
- every claim cited as `[Doc_ID §Section]`, or an explicit uncertainty note;
- ephemeral per-session state;
- architectural parsimony: every component must justify its latency and compute.

The target deployment is a single CPU machine running a local 7-8B instruct model through Ollama.
The same code also runs fully offline on 4 GB machines, with extractive answers.

## 2. System design

```
transcript chunks ─► controller (wait / retrieve / suppress) ──► planner: decomposer ─► anti-fragmentation guard ─► context carry-over
      │                    │ presentation-only? ─► suppress                         │ new / changed sub-queries
      │                    ▼                                                        ▼
      │           stability probe (BM25 top-k)          speculative cache ─► parallel hybrid search (BM25 + dense, RRF) per sub-query
      ▼                                                                             ▼
session store ─► claim ledger ◄── delta engine ◄── grounding verifier ◄── synthesis per sub-intent ◄── fusion (dedupe, rerank, quota)
                     │
                     ▼
         output record {retrieval_events, sub_queries, answer, citations, uncertainty} + JSONL telemetry per request
```

| layer | module | role |
|---|---|---|
| stream | `src/stream/simulator.py` | replays timestamped chunks (simulated or wall clock) and emits the utterance end |
| control | `src/stream/{suppression,stability,controller}.py` | decides when to search, and when not to |
| planning | `src/decompose/{decomposer,dedupe,context,planner}.py` | turns an utterance into non-redundant, self-contained sub-queries |
| retrieval | `src/retrieval/{bm25,dense,rrf,hybrid,rerank,fusion,cache}.py` | hybrid search per sub-query, fused evidence per intent |
| session | `src/session/{store,ledger,delta,pipeline}.py` | ephemeral state, claim versions, refinement in place |
| synthesis | `src/synthesis/{generator,grounding,uncertainty,presentation,citations}.py` | cited claims, verification, uncertainty, format transforms |
| LLM | `src/llm/client.py` | the only module that talks to a model (Ollama / OpenAI-compatible) |
| telemetry | `src/telemetry/{events,logger,coverage}.py` | per-request JSONL traces and the G6 coverage check |

A single `asyncio` process runs everything. There is no multi-agent framework: every "agent-like"
decision is a function call with a typed input and output.

## 3. Retrieval trigger logic

On every chunk the controller runs three cheap checks, in this order:

1. **Suppression gate.** A turn is presentation-only when the utterance contains a
   transformation cue (shorten, bullets, table, repeat, translate, …), refers back to earlier
   output, and adds no new corpus-searchable term. It is suppressed with
   `presentation_restructure` ([Doc_01 §4.3]). Ambiguous cases go to a hand-weighted logistic
   scorer; it is not trained, so no evaluation data leaks into `src/`.
2. **Completeness and content.**
   - A fragment ending in a dangling function word, comma or ellipsis waits.
   - So does a fragment below the content minimum (2 corpus terms or entities, 4 tokens).
3. **Stability.** In `rule_stability` mode, the Jaccard overlap between consecutive BM25 top-3
   probes must reach a threshold.

The default is `rule_only`, which skips check 3. It was chosen by an ablation with a selection
rule fixed in advance (BENCHMARK A1). It reaches G2 = 0.97 with zero false triggers, while
`rule_stability` reaches 0.88 but searches 0.3 fewer times per turn.

**Thrashing guards:**
- at most one provisional search per utterance;
- a final search only if new corpus terms arrived since the last one;
- a mid-stream search only when a new intent appears.

**Unsearchable turns:**
- An unfinished utterance becomes a clarification request.
- A complete one with no corpus word becomes a per-intent "no evidence" note.

## 4. Decomposition strategy

The decomposer returns **diff operations** (`add` / `keep` / `merge`) against the live sub-query
set, so sub-query ids stay stable as the utterance grows and only new or changed sub-queries are
searched ([Doc_01 §4.1.1]).

- **Intended profile:** one JSON-schema LLM call per planning step, with a generic prompt.
- **Offline profile:** a rule splitter at grammatical intent boundaries ("…, and what…",
  "?", enumerated wh-clauses). A leading topic phrase becomes context of the clause that follows.

Every proposal then passes the **anti-fragmentation guard**:
- a simple single-intent utterance is short-circuited to one query;
- a fragment with fewer than 2 content words ("how long can it be") is folded into its neighbour;
- near-duplicates (embedding cosine >= 0.85) are merged;
- at most 4 sub-queries.

**Context carry-over** copies quantities, dates, locations and names forward into later
sub-queries (never backwards), and gives pronoun-only clauses the previous clause's topic.

Results: G3 = 0.75 with 0/22 simple queries over-split. Without the guard, 2/22 are over-split.

## 5. Evidence fusion and grounding

- **Search.** Each sub-query runs BM25 and dense retrieval fused by RRF (k = 60), then a reranker.
- **Fusion across sub-queries:**
  1. dedupe by chunk id, recording which sub-queries each chunk serves;
  2. drop near-duplicates (Jaccard >= 0.9);
  3. rerank against each chunk's own sub-queries;
  4. apply a **per-intent quota** (2) before filling the remaining slots, so a strong intent
     cannot crowd out a weak one. Without the quota, intent evidence recall falls 0.976 -> 0.952.

**Synthesis:**
- **Intended profile:** one JSON call covering all sub-intents. The model sees each sub-question
  with only its own evidence, and any claim citing a chunk outside that evidence is dropped.
- **Offline profile:** verbatim evidence units.

**Verification.** Every claim passes the grounding verifier:
- (a) its cited ids must exist; fabricated ids are removed, and a claim with no valid id fails;
- (b) a judge must accept the claim: `lexical` on CPU, `llm` in the intended profile, or `nli` if the optional model is installed.

**Uncertainty.** It is set per sub-intent (no evidence, weak evidence naming the missing terms,
failed verification) instead of guessing ([Doc_01 §3]). A pipeline-level assertion guarantees
that a rendered citation always resolves to a corpus section.

## 6. Session refinement

Sessions live only in memory, keyed by id: there is no listing or search API, no disk, and
`end()` destroys all state. Each session holds a **claim ledger**: claims with sub-intent, chunk
ids, dependent constraints and version; the sub-queries with their evidence; and the answer history.

The **delta engine** classifies each clause of a follow-up against the ledger ([Doc_01 §4.2]):

| kind | detection | action |
|---|---|---|
| `new_subintent` | a question or request with content words the session does not hold | new sub-intent, own evidence and claims |
| `parameter_update` | a statement or constraint overlapping an existing sub-intent (or referring back) | add the constraint and the delta evidence, add delta claims, keep existing claims |
| `contradiction` | a negation cue before a term an existing sub-intent holds | invalidate only the claims depending on that term; replace them |
| `presentation_only` | suppression gate | transform the existing answer, zero retrieval, no new citation ids |

Only the follow-up's own clauses are searched, never the earlier sub-intents. Untouched claims
keep id, text, citations and version, and the answer version is bumped. Ledger-diff tests prove
that an update touches only the affected claims (G5 = 5/5).

## 7. Data provenance

- **Source.** The corpus is the Theme 4 guide itself (`data/corpus/Doc_01.md`). It was transcribed
  by hand from the image-only PDF, because the PDF has no text layer.
- **Excluded.** Watermark, personal footer data, barcode and metadata. The PDF itself is not
  committed.
- **Identity.** A SHA-256 manifest is checked by tests. Section ids are written into the file,
  so chunk ids (`Doc_01 §4.1.2#1`) and citations (`[Doc_01 §4.1.2]`) are stable.
- **Isolation from evaluation.** Evaluation scenarios live in `eval/`, and a test enforces that no
  `src/` module imports them. The corpus's own illustrations (e.g. "Venue A") are indexed as text
  of the guide, not as facts about venues (BENCHMARK F2).

## 8. Trade-offs and component cost

| component | measured cost (CPU, mean per turn) | why it earns its place |
|---|---|---|
| suppression gate + stability/completeness checks + controller | 0.5 ms | early retrieval (0.86 s head start) with 0 false triggers; saves every search on presentation turns (0/6 vs 6/6) |
| decomposition + guard + carry-over (rules) | 0.2 ms | per-intent retrieval (G3 0.75); guard prevents over-splitting (2/22 -> 0/22) |
| hybrid retrieval (all sub-queries) | 3.5 ms | BM25 alone misses semantic matches once a real embedder is installed; RRF costs one sort |
| fusion (dedupe, rerank, quota) | 0.9 ms | keeps every intent in the evidence (quota ablation) |
| synthesis (extractive) / verification (lexical) | 0.75 / 0.35 ms | grounding by construction; 0 fabricated ids |
| speculative cache | saves ~0.06 searches per turn | avoids re-searching a provisional query that barely changed |
| LLM calls (intended profile) | one synthesis call per answered turn, one decomposition call per planning step, one judge call per claim | the dominant cost; used only where rules cannot do the job (paraphrased answers, semantic splits, entailment) |

The whole offline replay (80 requests, both systems) takes 0.42 s at a 52 MB peak.

Main trade-offs:
- `rule_only` vs `rule_stability`: earliness vs fewer searches;
- extractive vs LLM synthesis: guaranteed verbatim grounding vs fluent answers that need a judge;
- lexical vs LLM/NLI judge: cost vs answerability checking.

## 9. Failure modes and mitigations

| failure mode ([Doc_01 §6]) | mitigation in PRISM | residual risk |
|---|---|---|
| eager retrieval on noise | completeness + content minimums, provisional limit, final search only on new terms | `rule_only` can search a half-finished but complete-looking fragment |
| context loss on late constraints | claim ledger + delta engine; nothing is cleared, only dependent claims change | back-references across turns are not resolved (F1) |
| citation hallucination | own-evidence id restriction, id-existence check, support judge, rendering assertion | the lexical judge cannot detect wrong relations between present words |
| ignoring presentation-only turns | suppression gate before any search; transforms reuse citations only | cue-less paraphrases ("make it less wordy") rely on the fallback scorer |
| over-fragmenting sub-queries | short-circuit, thin-fragment fold, similarity merge, cap | a one-word topic clause can be merged back (under-split) |
| unanswerable but topical questions | per-intent evidence threshold and uncertainty | lexical overlap can pass (F2); needs the LLM or NLI judge |
| model unavailable | `require_llm` health check before a run; no silent fallback | — |

## 10. Deployment profiles

| profile | configuration | status |
|---|---|---|
| intended | `docker compose up`: Ollama + one-shot `ollama pull $PRISM_LLM_MODEL` (default `llama3.1:8b`, a 7-8B instruct model) + replay; `PRISM_DECOMPOSER=llm`, `PRISM_GROUNDING_JUDGE=llm` | code and configuration complete, and wiring verified against a fake Ollama server. A real 7-8B run requires suitable hardware; the container requires Docker |
| offline | `PRISM_LLM_PROVIDER=none` (code default), `make replay-offline` or `docker-compose.offline.yml` | verified locally; all gate numbers in `BENCHMARK.md` |

`python -m src.llm.client --check` reports the served model's parameter size. Every replay
records whether it ran inside the intended 7-8B class, so a smaller model can never be mistaken
for the intended one.
