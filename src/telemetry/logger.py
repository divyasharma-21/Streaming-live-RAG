"""Structured JSONL telemetry.

One `TelemetryLogger` per process writes to PRISM_LOG_DIR/telemetry.jsonl. A `RequestTrace`
stamps every event of one request with its request id and elapsed time, and keeps the
events in memory so callers and tests can inspect a complete trace.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.schemas import TelemetryEvent


class TelemetryLogger:
    """`echo`, if given, is called with every event as it is written (live display for demos)."""

    def __init__(self, path: Path | None = None, echo: Callable[[TelemetryEvent], None] | None = None):
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: TelemetryEvent) -> None:
        if self.echo is not None:
            self.echo(event)
        if self.path is None:
            return
        line = event.model_dump_json(exclude_none=True)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def trace(self, request_id: str | None = None, session_id: str | None = None) -> RequestTrace:
        return RequestTrace(self, request_id or uuid.uuid4().hex[:12], session_id)


class RequestTrace:
    def __init__(self, logger: TelemetryLogger, request_id: str, session_id: str | None = None):
        self.logger = logger
        self.request_id = request_id
        self.session_id = session_id
        self.t0 = time.perf_counter()
        self.events: list[TelemetryEvent] = []

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def emit(self, event: str, stage_latency_ms: float | None = None, **fields: Any) -> TelemetryEvent:
        known = {k: fields.pop(k) for k in ("trigger", "tokens_in", "tokens_out", "est_cost_usd") if k in fields}
        record = TelemetryEvent(
            event=event,
            request_id=self.request_id,
            session_id=self.session_id,
            elapsed_ms=round(self.elapsed_ms(), 3),
            stage_latency_ms=None if stage_latency_ms is None else round(stage_latency_ms, 3),
            data=fields,
            **known,
        )
        self.events.append(record)
        self.logger.write(record)
        return record


def null_trace() -> RequestTrace:
    """A trace that records in memory only (no file)."""
    return TelemetryLogger(None).trace()
