"""Stream simulator (step 2.1): replay timestamped transcript chunks as an async stream.

    async for event in replay(chunks, clock="simulated"):
        event.kind      # "chunk" or "utterance_end"
        event.ts        # stream time in seconds since the start of the utterance
        event.text      # chunk text ("" for utterance_end)
        event.transcript  # all chunk text received so far, joined

* `simulated` clock: events are yielded immediately (fast tests); `ts` carries stream time.
* `wall` clock: the generator sleeps so events arrive at their real offsets (demo).
  `speed` > 1 replays faster than real time.

The utterance-end event is emitted for the chunk with `is_final=True` (after the chunk
event if that chunk carries text). If no chunk is final, an implicit utterance end is
emitted at the last timestamp. The caller supplies the chunks; this module never reads
evaluation files.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Literal

from src.schemas import TranscriptChunk

Clock = Literal["simulated", "wall"]


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["chunk", "utterance_end"]
    ts: float
    text: str
    transcript: str
    index: int
    implicit_end: bool = False


def join_transcript(parts: Iterable[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def validate_chunks(chunks: list[TranscriptChunk]) -> None:
    if not chunks:
        raise ValueError("a stream needs at least one chunk")
    ts = [c.ts for c in chunks]
    if ts != sorted(ts):
        raise ValueError("chunk timestamps must be non-decreasing")
    finals = [i for i, c in enumerate(chunks) if c.is_final]
    if finals and finals != [len(chunks) - 1]:
        raise ValueError("only the last chunk may be final")


async def replay(chunks: list[TranscriptChunk], clock: Clock = "simulated", speed: float = 1.0) -> AsyncIterator[StreamEvent]:
    if clock not in ("simulated", "wall"):
        raise ValueError(f"unknown clock {clock!r}")
    if speed <= 0:
        raise ValueError("speed must be positive")
    validate_chunks(chunks)

    start = time.monotonic()
    received: list[str] = []

    async def wait_until(ts: float) -> None:
        if clock == "wall":
            delay = ts / speed - (time.monotonic() - start)
            if delay > 0:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)  # yield control so dispatched tasks can progress

    for i, chunk in enumerate(chunks):
        await wait_until(chunk.ts)
        if chunk.text.strip():
            received.append(chunk.text)
            yield StreamEvent("chunk", chunk.ts, chunk.text, join_transcript(received), i)
        if chunk.is_final:
            yield StreamEvent("utterance_end", chunk.ts, "", join_transcript(received), i)
            return
    last = chunks[-1]
    yield StreamEvent("utterance_end", last.ts, "", join_transcript(received), len(chunks) - 1, implicit_end=True)


def chunks_from_text(parts: list[str], step_s: float = 0.8, end_gap_s: float = 0.5) -> list[TranscriptChunk]:
    """Build a chunk stream from text fragments at a fixed cadence (demo / CLI helper)."""
    out = [TranscriptChunk(ts=round(i * step_s, 3), text=t, is_final=False) for i, t in enumerate(parts)]
    end = round(max(0, len(parts) - 1) * step_s + end_gap_s, 3)
    out.append(TranscriptChunk(ts=end, text="", is_final=True))
    return out
