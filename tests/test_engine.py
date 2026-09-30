"""Provisional retrieval wiring in the streaming engine (step 2.5)."""

import json

import pytest

from src.config import Settings
from src.engine import StreamingEngine, to_search_query
from src.retrieval.factory import build_retrieval
from src.stream.simulator import chunks_from_text
from src.telemetry.logger import TelemetryLogger


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


@pytest.fixture
def engine(stack, tmp_path):
    s = Settings(log_dir=tmp_path)
    eng = StreamingEngine(s, stack=stack, telemetry=TelemetryLogger(tmp_path / "t.jsonl"))
    eng.search_calls = 0
    original = eng._search

    def counting(query):
        eng.search_calls += 1
        return original(query)

    eng._search = counting
    return eng


def test_to_search_query_keeps_content_words():
    assert to_search_query("um so what is the uh Early retrieval target?") == "Early retrieval target"


async def test_provisional_retrieval_is_logged_before_utterance_end(engine, tmp_path):
    chunks = chunks_from_text(["Which gate covers", "early retrieval", "and how is it validated?"])
    result = await engine.run_turn(chunks, request_id="r1")
    assert result.retrieval_events[0].trigger == "provisional"
    assert result.retrieved_early and result.first_retrieval_s < result.utterance_end_s
    assert result.final_results and engine.search_calls == len(result.retrievals)

    names = [e.event for e in result.telemetry]
    assert names.index("retrieval_started") < names.index("utterance_end")
    started = next(e for e in result.telemetry if e.event == "retrieval_started")
    assert started.trigger == "provisional" and started.data["query"] and started.data["stream_ts"] == 0.8
    logged = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert {"retrieval_started", "retrieval_completed", "controller_decision", "request_completed"} <= {
        r["event"] for r in logged
    }


async def test_final_retrieval_at_end_when_nothing_stable_earlier(engine):
    result = await engine.run_turn(chunks_from_text(["what about grounding"]))
    assert [e.trigger for e in result.retrieval_events] == ["final"]
    assert not result.retrieved_early


async def test_presentation_turn_makes_zero_search_calls(engine):
    result = await engine.run_turn(chunks_from_text(["Could you rewrite that", "as a short list?"]),
                                   prior_output="Some earlier answer.")
    assert engine.search_calls == 0 and result.retrievals == []
    assert result.suppressed_reason == "presentation_restructure"


async def test_incomplete_utterance_is_suppressed_without_search(engine):
    result = await engine.run_turn(chunks_from_text(["so um", "can you"]))
    assert engine.search_calls == 0 and result.suppressed_reason == "insufficient_content"


# ------------------------------------------------------------------ Phase 3: parallel multi-intent retrieval (3.4)


async def test_multi_intent_subqueries_are_searched_in_parallel(engine):
    chunks = chunks_from_text(["How is the reranker scored, and which gate", "measures telemetry coverage?"])
    result = await engine.run_turn(chunks)
    multi = [e for e in result.retrieval_events if e.trigger == "multi_intent"]
    assert len(multi) == 2 and len({e.timestamp_s for e in multi}) == 1
    assert len(result.sub_queries) == 2 and all(result.sub_results[q.id] for q in result.sub_queries)
    names = [e.event for e in result.telemetry]
    starts = [i for i, n in enumerate(names) if n == "retrieval_started"]
    first_done = names.index("retrieval_completed")
    assert all(i < first_done for i in starts)  # both dispatched before either completed
    assert "decomposition" in names


async def test_new_intent_after_provisional_searches_only_the_new_subquery(engine):
    chunks = chunks_from_text(["How does the reranker deduplicate chunks?", "And how is the cache keyed?"])
    result = await engine.run_turn(chunks)
    triggers = [(e.trigger, e.timestamp_s) for e in result.retrieval_events]
    # q1 searched provisionally at 0.0; at 0.8 only the new q2 is searched (q1 unchanged, not re-searched)
    assert triggers == [("provisional", 0.0), ("multi_intent", 0.8)]
    assert [q.id for q in result.sub_queries] == ["q1", "q2"]


async def test_single_intent_is_not_split(engine):
    result = await engine.run_turn(chunks_from_text(["Which gate covers", "early retrieval", "and how is it validated?"]))
    assert len(result.sub_queries) == 1
    assert [e.trigger for e in result.retrieval_events] == ["provisional"]


async def test_decompose_off_keeps_phase2_behaviour(stack, tmp_path):
    eng = StreamingEngine(Settings(log_dir=tmp_path), stack=stack, telemetry=TelemetryLogger(None), decompose=False)
    result = await eng.run_turn(chunks_from_text(["How is the reranker scored, and which gate", "measures telemetry?"]))
    assert all(e.trigger != "multi_intent" for e in result.retrieval_events) and result.sub_queries == []


async def test_fused_evidence_covers_every_sub_intent(engine):
    chunks = chunks_from_text(["How is the reranker scored, and which gate", "measures telemetry coverage?"])
    result = await engine.run_turn(chunks)
    assert result.evidence and len(result.evidence) <= engine.settings.fusion_top_k
    covered = {s for sources in result.evidence_sources.values() for s in sources}
    assert covered == {q.id for q in result.sub_queries}
    assert "fusion_completed" in [e.event for e in result.telemetry]


async def test_no_decompose_evidence_is_last_retrieval(stack, tmp_path):
    eng = StreamingEngine(Settings(log_dir=tmp_path), stack=stack, telemetry=TelemetryLogger(None), decompose=False)
    result = await eng.run_turn(chunks_from_text(["describe the telemetry trace coverage gate"]))
    assert result.evidence == result.final_results and result.evidence


async def test_final_subquery_close_to_provisional_reuses_cached_search(engine):
    chunks = chunks_from_text(["Describe the telemetry trace coverage gate", "in detail"])
    result = await engine.run_turn(chunks)
    names = [e.event for e in result.telemetry]
    assert [e.trigger for e in result.retrieval_events] == ["provisional"]  # no second search
    assert "cache_hit" in names
    hit = next(e for e in result.telemetry if e.event == "cache_hit")
    assert hit.data["similarity"] >= engine.settings.cache_similarity
    assert result.sub_results[result.sub_queries[0].id] == result.retrievals[0].results


async def test_cache_disabled_searches_again(stack, tmp_path):
    eng = StreamingEngine(Settings(log_dir=tmp_path, cache_similarity=0.0), stack=stack, telemetry=TelemetryLogger(None))
    result = await eng.run_turn(chunks_from_text(["Describe the telemetry trace coverage gate", "in detail"]))
    assert len(result.retrieval_events) == 2


async def test_late_finishing_stale_search_does_not_overwrite_newer_results(stack, tmp_path):
    """A provisional search that completes after the sub-query's newer search must be ignored."""
    eng = StreamingEngine(Settings(log_dir=tmp_path, cache_similarity=0.0), stack=stack,
                          telemetry=TelemetryLogger(None))
    original = eng._search

    def slow_first(query, _calls=[0]):
        _calls[0] += 1
        if _calls[0] == 1:
            import time as _t
            _t.sleep(0.2)  # the first (provisional) search finishes last
        return original(query)

    eng._search = slow_first
    result = await eng.run_turn(chunks_from_text(["Describe the telemetry trace coverage gate", "and token cost"]))
    last_q1 = [r for r in result.retrievals if r.subquery_id == result.sub_queries[0].id][-1]
    assert result.sub_results[result.sub_queries[0].id] == last_q1.results
