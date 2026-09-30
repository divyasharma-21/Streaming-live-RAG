# Demo script: 5-minute system video

Covers the six items the playbook requires: early retrieval, multi-intent decomposition,
late-detail refinement, presentation-query suppression, citation traceability and live
telemetry. Every command below was run while writing this script (offline profile). The video
itself still has to be recorded by the team.

## Before recording (once)

```bash
make install                                   # pinned dependencies (or: bash scripts/run_all.sh)
make index                                     # build indexes, print the corpus audit
# Intended runtime (optional, needs a machine that can run a 7-8B model):
make llm-check                                 # Ollama reachable, model pulled, parameter size in the 7-8B class
```

Two terminals: **A** runs the demo, **B** shows live telemetry.
Scenario files shown: `eval/dev_scenarios/single_02.json`, `multi_03.json`, `late_02.json`, `present_04.json`.
Offline profile commands are shown; for the intended profile prefix each `python eval/replay.py …` with
`PRISM_LLM_PROVIDER=ollama PRISM_LLM_MODEL=llama3.1:8b PRISM_DECOMPOSER=llm PRISM_GROUNDING_JUDGE=llm`.

## Storyboard

| time | scene | on screen (exact command) | what to point out |
|---|---|---|---|
| 0:00-0:25 | Problem + one command | `make replay` then `cat results/replay/summary.md` | batch RAG waits for the end of speech; the replay runs the whole suite in one command and prints G1-G6 plus streaming-vs-baseline |
| 0:25-1:10 | **Early retrieval** | A: `python eval/replay.py --scenario single_02 --clock wall --show --out results/demo` | chunks arrive at real speed; `controller_decision wait` on "Which gate covers reproducibility", then `retrieval_started provisional` at 0.0-0.8 s, **before** `utterance_end` |
| 1:10-1:55 | **Multi-intent decomposition** | A: `python eval/replay.py --scenario multi_03 --clock wall --show --out results/demo` | one utterance, three `retrieval_started multi_intent` events at the same timestamp (evidence fusion / citation hallucination / grounding threshold); one answer citing §2.1, §6 and §5 |
| 1:55-2:50 | **Late-detail refinement** | A: `python eval/replay.py --scenario late_02 --clock wall --show --out results/demo` then `python eval/show_turns.py results/demo/results.json` | turn 1 → v1; turn 2 ("Actually, focus on grounding, and include the threshold…") → **v2 parameter_update**: the 4 turn-1 claims are `unchanged`, only new claims are `added`, only the delta was searched; the unsupported part gets an explicit uncertainty note |
| 2:50-3:30 | **Presentation suppression** | A: `python eval/replay.py --scenario present_04 --clock wall --show --out results/demo` then `python eval/show_turns.py results/demo/results.json` | turn 2 "Say that again but more simply": the gate suppresses it, **no `retrieval_started`**, `retrievals=0`, same answer version, same citations |
| 3:30-4:15 | **Citation traceability** | A: `python -m src.corpus.show "Doc_01 §5"` then `python -m src.corpus.show "Doc_999 §1"` | every `[Doc_ID §Section]` in an answer resolves to the exact corpus text; a fabricated id is rejected (exit 1), which the grounding verifier does automatically (0 fabricated ids in the benchmark) |
| 4:15-4:50 | **Live telemetry** | B (start before a run): `tail -f results/demo/telemetry.jsonl`; after: `python -m src.telemetry.coverage results/demo/telemetry.jsonl` | one JSON line per event: timestamps, decisions, triggers, sub-queries, claim → source mapping, version transitions, per-stage latency, tokens and cost; coverage 100% |
| 4:50-5:00 | Wrap-up | `cat results/replay/summary.md` | gates G2-G6 pass on the dev set; G1 = container run (see `FINAL_CHECKLIST.md`) |

## Notes for the presenter

- `--clock wall` replays at speaking speed (0.8 s between chunks); without it the same run
  takes milliseconds. `make demo` runs the four scenarios above at speaking speed.
- The offline profile answers with verbatim corpus sentences. With the intended 7-8B model the
  same commands give fluent answers; say which profile the recording uses.
- Do not present dev-scenario numbers as held-out benchmark results; the held-out set is private.
