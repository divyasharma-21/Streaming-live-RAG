> **WIRING CHECK ONLY.** Intended-profile replay (Ollama client, LLM decomposer, LLM synthesis, LLM judge) run against
> `tests/fake_ollama.py`, a deterministic TEST-ONLY fake server, NOT a language model. It proves every LLM code
> path runs end to end over HTTP with token accounting and full telemetry. Its quality numbers (e.g. G3) are
> meaningless and must not be reported as results; a real 7-8B run requires suitable hardware.

# Replay summary (llm profile)

Run 2026-09-30T03:16:46+0000 · commit e469585 · LLM ollama:llama3.1:8b size=8.0B intended_7_8b_class=True

| Gate | Criterion | Target | Result | Status | Evidence |
|---|---|---|---|---|---|
| G1 | Reproducibility | Pass/Fail | CLI runner completed | NOT VERIFIED | single-command CLI replay completed; the container itself was not exercised (requires Docker) |
| G2 | Early retrieval | >= 0.80 of eligible queries | 0.9706 | PASS | 33/34 eligible turns retrieved before utterance end; false triggers on no-retrieval turns 0/6 |
| G3 | Multi-intent identification | >= 0.70 of compound queries | 0.5 | FAIL | 6/12 compound turns with >= 2 correct sub-intents |
| G4 | Factual grounding | >= 0.85 citation support, 0 fabricated ids | 1.0 | PASS | 34/34 claims literally supported by their cited chunk; fabricated ids 0; shuffled-citation control 0.0 |
| G5 | Session refinement | verified state continuity | 5/5 | PASS | refinement kinds {'new_subintent': 2, 'parameter_update': 3, 'contradiction': 0}; presentation turns with zero retrieval 5/5 |
| G6 | Telemetry & observability | 100% trace coverage | 1.0 | PASS | 80/80 requests with a complete trace |

## Streaming system vs baseline

| metric | streaming (full) | baseline |
|---|---|---|
| retrieval starts before utterance end (eligible turns) | 0.9706 | 0.0 |
| mean seconds of retrieval head start vs utterance end | 0.862 | 0.0 |
| post-utterance latency p50 / p95 ms | 10.347 / 15.41 | 4.275 / 5.374 |
| no-retrieval turns (presentation-only / incomplete) that searched the corpus | 0/6 | 6/6 |
| answer cites the gold sections: recall / precision (answerable turns) | 0.704 / 0.982 | 0.785 / 1.0 |
| fabricated citation ids | 0 | 0 |
| claims literally supported by cited chunk | 1.0 | n/a (not computed for baseline) |
| uncertainty on no-evidence turns | 3/5 | 1/5 |
| false uncertainty on answerable turns | 1/30 | 1/30 |
| late-constraint turns refined in place (answer version bumped, state kept) | 5/5 | 0 (stateless) |
| LLM tokens in / out (total) | 29122 / 1963 | 15456 / 607 |
| estimated cost USD (total) | 0.0 | 0.0 |
