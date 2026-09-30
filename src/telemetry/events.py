"""Telemetry event types. The record schema is `TelemetryEvent` in src/schemas.py
(exported to schemas/telemetry_event.schema.json)."""

from __future__ import annotations

from src.schemas import TelemetryEvent, TelemetryEventType

REQUEST_STARTED = "request_started"
CONTROLLER_DECISION = "controller_decision"
UTTERANCE_END = "utterance_end"
DECOMPOSITION = "decomposition"
CACHE_HIT = "cache_hit"
FUSION_COMPLETED = "fusion_completed"
RETRIEVAL_STARTED = "retrieval_started"
RETRIEVAL_COMPLETED = "retrieval_completed"
RERANK_COMPLETED = "rerank_completed"
LLM_CALL = "llm_call"
CITATION_CHECK = "citation_check"
ANSWER_EMITTED = "answer_emitted"
REQUEST_COMPLETED = "request_completed"
ERROR = "error"


def estimate_cost_usd(tokens_in: int, tokens_out: int, per_1k_in: float, per_1k_out: float) -> float:
    return round(tokens_in / 1000 * per_1k_in + tokens_out / 1000 * per_1k_out, 8)


__all__ = ["TelemetryEvent", "TelemetryEventType", "estimate_cost_usd"]
