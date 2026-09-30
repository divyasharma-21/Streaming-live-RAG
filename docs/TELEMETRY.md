# Telemetry & observability schema

Every request writes structured JSON Lines to `PRISM_LOG_DIR/telemetry.jsonl` (the replay runner
writes its own file per run). The record schema is `TelemetryEvent`
(`src/schemas.py`, exported as `schemas/telemetry_event.schema.json`):

| field | meaning |
|---|---|
| `event` | event type (table below) |
| `request_id` | one id per user turn; the streaming engine and the session pipeline share it |
| `session_id` | the conversation session (ephemeral; never reused across sessions) |
| `wall_time` | Unix epoch seconds when the event was written |
| `elapsed_ms` | ms since the trace for this component started |
| `stage_latency_ms` | latency of the stage the event closes |
| `trigger` | `provisional` / `multi_intent` / `final` on retrieval events |
| `tokens_in`, `tokens_out`, `est_cost_usd` | on `llm_call` and `answer_emitted` (0 when no LLM ran) |
| `data` | event-specific fields (below) |

## Events

| event | emitted by | key `data` fields |
|---|---|---|
| `request_started` | engine, baseline | `system` (`streaming` / `baseline`), `controller`, `clock`, `decompose`, `has_prior_output`, `expects_answer` |
| `controller_decision` | engine, every chunk and at utterance end | `stream_ts`, `action` (wait/retrieve/suppress), `reason`, `trigger_decided`, `stability_score` |
| `utterance_end` | engine | `stream_ts`, `implicit` |
| `decomposition` | engine (planner) | `stream_ts`, `planner`, diff `ops`, `guard_notes`, `sub_queries` (id, text, constraints) |
| `cache_hit` | engine | `subquery_id`, `query`, `matched_query`, `similarity` |
| `retrieval_started` | engine, baseline | `stream_ts`, `query`, `subquery_id` (+ top-level `trigger`) |
| `retrieval_completed` | engine, baseline | `stream_ts`, `subquery_id`, `chunk_ids` |
| `rerank_completed` | baseline | `reranker`, `chunk_ids`, `scores` |
| `fusion_completed` | engine | `chunk_ids`, `sources` (chunk -> sub-queries), dropped duplicates |
| `llm_call` | `src/llm/client.py` | `provider`, `model`, `json_mode`, `attempts`, `tokens_estimated`, `ok` |
| `citation_check` | pipeline, baseline | `citations`, `invalid`, `fabricated_ids` |
| `answer_emitted` | pipeline, baseline | `answer_version_from` -> `answer_version`, `kind`, `sub_queries`, `sources` (claim id -> chunk ids), `citations`, `delta_citations`, `delta_clauses`, `ledger_diff`, `retrieval_calls`, `retrieval_triggers`, `first_retrieval_s`, `utterance_end_s`, `stage_ms` (controller / planning / retrieval / fusion / synthesis / verification / presentation), `uncertainty` |
| `request_completed` | engine, baseline | total latency, `retrievals`, `first_retrieval_s`, `retrieved_early`, `suppressed_reason`, `cache_hits`, `stage_ms` |

## Coverage (gate G6)

`python -m src.telemetry.coverage <file.jsonl>` groups events by `request_id` and fails (exit 1)
unless every request has a complete trace; the rules are in `src/telemetry/coverage.py`.
`make coverage` checks `logs/telemetry.jsonl`; the replay runner checks its own run file.

Cost: `est_cost_usd = tokens_in/1000 * PRISM_COST_PER_1K_INPUT + tokens_out/1000 * PRISM_COST_PER_1K_OUTPUT`
(both default 0 for local models). Without an LLM, token counts are recorded as 0, not omitted.
