"""Structured event schemas (playbook step 1.3).

The final output record mirrors the "Structured Output Event Record" in the problem
statement ([Doc_01 §4.1.2]): retrieval_events, sub_queries, answer, citations, uncertainty.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# "Doc_12 §2", "Doc_01 §4.1.2": a document id, one space, the section sign, a dotted section id.
CITATION_PATTERN = r"^[A-Za-z0-9_\-]+ §[0-9A-Za-z]+(\.[0-9A-Za-z]+)*$"

Trigger = Literal["provisional", "multi_intent", "final"]
Citation = Annotated[str, StringConstraints(pattern=CITATION_PATTERN)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- stream / controller


class TranscriptChunk(_Strict):
    ts: float = Field(ge=0, description="Seconds since the start of the utterance.")
    text: str
    is_final: bool = Field(default=False, description="True for the last chunk of an utterance.")


class ControllerDecision(_Strict):
    action: Literal["wait", "retrieve", "suppress"]
    reason: str
    trigger: Trigger | None = Field(default=None, description="Set when action == 'retrieve'.")
    ts: float | None = Field(default=None, ge=0, description="Stream time of the decision (s).")
    stability_score: float | None = Field(default=None, ge=0, le=1)


class SubQuery(_Strict):
    id: str
    text: str
    status: Literal["pending", "retrieved", "merged", "dropped"] = "pending"
    constraints: list[str] = Field(default_factory=list, description="Shared context carried into this sub-query.")


# ---------------------------------------------------------------- corpus


class CorpusChunk(_Strict):
    chunk_id: str = Field(description="Stable id: '<Doc_ID> §<Section>#<part>'.")
    doc_id: str
    section: str = Field(description="Section id without the § sign, e.g. '4.1.2'.")
    text: str = Field(description="Chunk text, prefixed with the document title and section heading.")
    title: str = ""
    heading: str = ""
    part: int = Field(default=1, ge=1)
    n_tokens: int = Field(default=0, ge=0)

    @property
    def citation(self) -> str:
        """Section-level citation key, e.g. 'Doc_01 §3'."""
        return f"{self.doc_id} §{self.section}"


class ScoredChunk(_Strict):
    chunk: CorpusChunk
    score: float
    rank: int = Field(ge=1)
    source: str = Field(description="Which retriever or stage produced this score.")


# ---------------------------------------------------------------- answer


class Claim(_Strict):
    id: str
    text: str
    subintent_id: str
    chunk_ids: list[str]
    version: int = Field(default=1, ge=1)
    constraints: list[str] = Field(default_factory=list, description="Constraints this claim depends on.")


class AnswerVersion(_Strict):
    version: int = Field(ge=1)
    claims: list[Claim]
    uncertainty: str | None = None


class RetrievalEvent(_Strict):
    timestamp_s: float = Field(ge=0)
    query: str
    trigger: Trigger


class OutputRecord(_Strict):
    """Final per-request record, matching the problem statement example."""

    retrieval_events: list[RetrievalEvent]
    sub_queries: list[str]
    answer: str
    citations: list[Citation]
    uncertainty: str | None = None


# ---------------------------------------------------------------- telemetry

TelemetryEventType = Literal[
    "request_started",
    "controller_decision",
    "utterance_end",
    "decomposition",
    "cache_hit",
    "retrieval_started",
    "retrieval_completed",
    "rerank_completed",
    "fusion_completed",
    "llm_call",
    "citation_check",
    "answer_emitted",
    "request_completed",
    "error",
]


class TelemetryEvent(_Strict):
    event: TelemetryEventType
    request_id: str
    session_id: str | None = None
    wall_time: float = Field(default_factory=time.time, description="Unix epoch seconds.")
    elapsed_ms: float = Field(default=0.0, ge=0, description="Milliseconds since request_started.")
    stage_latency_ms: float | None = Field(default=None, ge=0)
    trigger: Trigger | None = None
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    est_cost_usd: float | None = Field(default=None, ge=0)
    data: dict[str, Any] = Field(default_factory=dict)


EXPORTED_SCHEMAS: dict[str, type[BaseModel]] = {
    "output_record": OutputRecord,
    "telemetry_event": TelemetryEvent,
    "transcript_chunk": TranscriptChunk,
    "controller_decision": ControllerDecision,
    "sub_query": SubQuery,
    "corpus_chunk": CorpusChunk,
    "claim": Claim,
    "answer_version": AnswerVersion,
}
