"""Stream simulator (step 2.1)."""

import time

import pytest

from src.schemas import TranscriptChunk
from src.stream.simulator import chunks_from_text, replay


async def _collect(chunks, **kw):
    return [e async for e in replay(chunks, **kw)]


async def test_simulated_replay_emits_chunks_then_utterance_end():
    chunks = chunks_from_text(["first part", "second part"], step_s=0.8, end_gap_s=0.5)
    events = await _collect(chunks)
    assert [(e.kind, e.ts) for e in events] == [("chunk", 0.0), ("chunk", 0.8), ("utterance_end", 1.3)]
    assert events[1].transcript == "first part second part"
    assert events[-1].transcript == "first part second part" and not events[-1].implicit_end


async def test_final_chunk_with_text_yields_chunk_and_end():
    chunks = [TranscriptChunk(ts=0.0, text="a"), TranscriptChunk(ts=0.5, text="b", is_final=True)]
    events = await _collect(chunks)
    assert [e.kind for e in events] == ["chunk", "chunk", "utterance_end"]
    assert events[-1].ts == 0.5


async def test_implicit_end_when_no_final_chunk():
    events = await _collect([TranscriptChunk(ts=0.0, text="a"), TranscriptChunk(ts=0.4, text="b")])
    assert events[-1].kind == "utterance_end" and events[-1].implicit_end and events[-1].ts == 0.4


async def test_simulated_clock_is_fast():
    chunks = chunks_from_text(["x"] * 5, step_s=10.0)
    t = time.monotonic()
    await _collect(chunks)
    assert time.monotonic() - t < 0.5


async def test_wall_clock_respects_offsets():
    chunks = chunks_from_text(["a", "b"], step_s=0.2, end_gap_s=0.1)
    t = time.monotonic()
    await _collect(chunks, clock="wall", speed=2.0)  # 0.3 s of stream time at 2x
    assert 0.12 <= time.monotonic() - t < 1.0


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [TranscriptChunk(ts=1.0, text="a"), TranscriptChunk(ts=0.5, text="b")],
        [TranscriptChunk(ts=0.0, text="a", is_final=True), TranscriptChunk(ts=0.5, text="b")],
    ],
)
async def test_invalid_streams_are_rejected(chunks):
    with pytest.raises(ValueError):
        await _collect(chunks)
