"""Streaming engine: controller (Phase 2) + multi-intent planning and parallel retrieval (Phase 3).

For every transcript chunk and at the utterance end the controller decides wait / retrieve /
suppress. With decomposition enabled (default):

* on a `retrieve` decision, and on stable new content after the provisional retrieval
  (controller reason `provisional_limit`), the query planner (decomposer -> anti-
  fragmentation guard -> context carry-over) updates the live sub-query set;
* every new or changed sub-query is searched in parallel (`asyncio.gather`), one
  `retrieval_started` event each, trigger `multi_intent` when more than one sub-query is
  live (otherwise the controller's `provisional` / `final` trigger). A single live
  sub-query is never re-searched mid-stream, so the provisional limit still holds;
* searches run as background tasks so the stream keeps flowing;
* a sub-query whose query is close to one already searched in this utterance reuses that
  search (speculative reuse cache, logged as `cache_hit`);
* at the end, per-sub-query results are fused into one evidence set with a per-intent quota.

With `decompose=False` the Phase 2 behaviour is unchanged: one search of the whole
transcript per `retrieve` decision. Session refinement and answer synthesis are Phase 4.

    python -m src.engine --utterance "first fragment | second fragment" [--clock wall] [--prior "..."]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.decompose.context import search_text
from src.decompose.planner import QueryPlanner, build_planner
from src.retrieval.cache import SpeculativeCache
from src.retrieval.factory import RetrievalStack, build_retrieval
from src.retrieval.fusion import fuse_evidence
from src.retrieval.rerank import Reranker, make_reranker
from src.retrieval.text import STOPWORDS
from src.schemas import (
    ControllerDecision,
    RetrievalEvent,
    ScoredChunk,
    SubQuery,
    TelemetryEvent,
    TranscriptChunk,
    Trigger,
)
from src.stream.controller import RetrievalController, build_controller
from src.stream.simulator import Clock, chunks_from_text, replay
from src.stream.stability import FILLERS, surface_tokens
from src.telemetry import events as ev
from src.telemetry.logger import RequestTrace, TelemetryLogger


def to_search_query(transcript: str) -> str:
    """Keyword query from the transcript: content words in order, fillers/stopwords dropped."""
    seen: dict[str, str] = {}
    for w in surface_tokens(transcript):
        key = w.lower()
        if key not in STOPWORDS and key not in FILLERS and key not in seen:
            seen[key] = w
    return " ".join(seen.values())


@dataclass
class RetrievalRun:
    event: RetrievalEvent
    results: list[ScoredChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    subquery_id: str | None = None


@dataclass
class StreamTurnResult:
    utterance: str
    utterance_end_s: float
    decisions: list[ControllerDecision]
    retrievals: list[RetrievalRun]
    suppressed_reason: str | None
    telemetry: list[TelemetryEvent]
    sub_queries: list[SubQuery] = field(default_factory=list)
    sub_results: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    evidence: list[ScoredChunk] = field(default_factory=list)  # fused across sub-queries (3.5)
    evidence_sources: dict[str, list[str]] = field(default_factory=dict)  # chunk id -> sub-query ids
    stage_ms: dict[str, float] = field(default_factory=dict)  # controller / planning / retrieval / fusion
    request_id: str = ""

    @property
    def retrieval_events(self) -> list[RetrievalEvent]:
        return [r.event for r in self.retrievals]

    @property
    def first_retrieval_s(self) -> float | None:
        return min((r.event.timestamp_s for r in self.retrievals), default=None)

    @property
    def retrieved_early(self) -> bool:
        first = self.first_retrieval_s
        return first is not None and first < self.utterance_end_s

    @property
    def final_results(self) -> list[ScoredChunk]:
        return self.retrievals[-1].results if self.retrievals else []


@dataclass
class _TurnState:
    trace: RequestTrace
    runs: list[RetrievalRun] = field(default_factory=list)
    tasks: list[asyncio.Task] = field(default_factory=list)
    live: list[SubQuery] = field(default_factory=list)
    searched: dict[str, str] = field(default_factory=dict)  # sub-query id -> query last searched
    sub_results: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    planned_transcript: str | None = None
    cache: SpeculativeCache | None = None
    aliases: dict[str, RetrievalRun] = field(default_factory=dict)  # sub-query id -> reused run (3.6)
    stage_ms: dict[str, float] = field(default_factory=lambda: {"controller": 0.0, "planning": 0.0,
                                                                "retrieval": 0.0, "fusion": 0.0})


class StreamingEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        stack: RetrievalStack | None = None,
        controller: RetrievalController | None = None,
        reranker: Reranker | None = None,
        telemetry: TelemetryLogger | None = None,
        planner: QueryPlanner | None = None,
        decompose: bool = True,
    ):
        self.settings = s = settings or get_settings()
        self.stack = stack or build_retrieval(s)
        self.controller = controller or build_controller(self.stack, s)
        self.reranker = reranker or make_reranker(s.rerank_backend, s.rerank_model)
        self.telemetry = telemetry or TelemetryLogger(s.log_dir / "telemetry.jsonl")
        self.decompose = decompose
        self.planner = (planner or build_planner(self.stack.dense.embedder, s)) if decompose else None
        self.retriever = self.stack.hybrid

    # ------------------------------------------------------------------ search

    def _search(self, query: str) -> list[ScoredChunk]:
        hits = self.retriever.search(query, k=self.settings.candidates)
        return self.reranker.rerank(query, hits, self.settings.top_k)

    async def _run_one(self, run: RetrievalRun, st: _TurnState) -> None:
        t = time.perf_counter()
        run.results = await asyncio.to_thread(self._search, run.event.query)
        run.latency_ms = (time.perf_counter() - t) * 1000
        st.stage_ms["retrieval"] += run.latency_ms
        # only the search for the sub-query's latest query may set its results: an older
        # (e.g. provisional) search can finish later and must not overwrite newer results
        if run.subquery_id is not None and st.searched.get(run.subquery_id) == run.event.query:
            st.sub_results[run.subquery_id] = run.results
        st.trace.emit(
            ev.RETRIEVAL_COMPLETED,
            stage_latency_ms=run.latency_ms,
            trigger=run.event.trigger,
            stream_ts=run.event.timestamp_s,
            subquery_id=run.subquery_id,
            chunk_ids=[h.chunk.chunk_id for h in run.results],
        )

    def _dispatch(self, targets: list[tuple[str | None, str]], ts: float, trigger: Trigger, st: _TurnState) -> None:
        """Start one background task that searches all targets in parallel (asyncio.gather)."""
        runs = []
        for sub_id, query in targets:
            run = RetrievalRun(RetrievalEvent(timestamp_s=ts, query=query, trigger=trigger), subquery_id=sub_id)
            st.runs.append(run)
            runs.append(run)
            if st.cache is not None:
                st.cache.put(query, run)
            st.trace.emit(ev.RETRIEVAL_STARTED, trigger=trigger, stream_ts=ts, query=query, subquery_id=sub_id)
        st.tasks.append(asyncio.create_task(self._gather(runs, st)))

    async def _gather(self, runs: list[RetrievalRun], st: _TurnState) -> None:
        await asyncio.gather(*(self._run_one(r, st) for r in runs))

    # ------------------------------------------------------------------ planning

    async def _plan_and_search(self, transcript: str, ts: float, trigger: Trigger | None, st: _TurnState) -> None:
        """Update the live sub-query set and search the new or changed sub-queries.

        `trigger` is the controller trigger for a retrieve decision, or None for a stable
        chunk after the provisional retrieval (then only a multi-intent split may search)."""
        assert self.planner is not None
        t = time.perf_counter()
        plan = await self.planner.plan(transcript, st.live, st.trace)
        st.live, st.planned_transcript = plan.subqueries, transcript
        plan_ms = (time.perf_counter() - t) * 1000
        st.stage_ms["planning"] += plan_ms
        st.trace.emit(
            ev.DECOMPOSITION,
            stage_latency_ms=plan_ms,
            stream_ts=ts,
            planner=self.planner.name,
            ops=[o.model_dump(exclude_defaults=True) for o in plan.ops],
            guard_notes=plan.guard_notes,
            sub_queries=[{"id": q.id, "text": q.text, "constraints": q.constraints} for q in plan.subqueries],
        )
        multi = len(st.live) > 1
        if trigger is None and not multi:
            return
        targets = []
        for q in st.live:
            query = to_search_query(search_text(q))
            if query and st.searched.get(q.id) != query:
                st.searched[q.id] = query
                hit = st.cache.lookup(query) if st.cache is not None else None
                if hit is not None:
                    st.aliases[q.id] = hit.entry
                    st.trace.emit(ev.CACHE_HIT, stream_ts=ts, subquery_id=q.id, query=query,
                                  matched_query=hit.matched_query, similarity=round(hit.similarity, 4))
                    continue
                st.aliases.pop(q.id, None)
                targets.append((q.id, query))
        if targets:
            self._dispatch(targets, ts, "multi_intent" if multi else trigger or "provisional", st)

    # ------------------------------------------------------------------ turn

    async def run_turn(
        self,
        chunks: list[TranscriptChunk],
        prior_output: str | None = None,
        clock: Clock = "simulated",
        speed: float = 1.0,
        request_id: str | None = None,
        session_id: str | None = None,
        expects_answer: bool = False,
    ) -> StreamTurnResult:
        """`expects_answer`: the caller (the session pipeline) will emit answer events for this
        request id; recorded so the telemetry coverage checker knows what a complete trace is."""
        trace = self.telemetry.trace(request_id, session_id)
        trace.emit(ev.REQUEST_STARTED, system="streaming", controller=self.controller.name, clock=clock,
                   decompose=self.decompose, has_prior_output=bool(prior_output), expects_answer=expects_answer)
        self.controller.start_utterance(prior_output)
        if self.planner is not None:
            self.planner.reset()
        st = _TurnState(trace)
        if self.decompose and self.settings.cache_similarity > 0:
            st.cache = SpeculativeCache(self.settings.cache_similarity)  # per utterance, discarded after
        decisions: list[ControllerDecision] = []
        transcript, end_ts = "", 0.0

        async for event in replay(chunks, clock=clock, speed=speed):
            transcript = event.transcript
            is_end = event.kind == "utterance_end"
            t_ctrl = time.perf_counter()
            if is_end:
                end_ts = event.ts
                trace.emit(ev.UTTERANCE_END, stream_ts=event.ts, implicit=event.implicit_end)
                decision = await self.controller.on_utterance_end(transcript, event.ts)
            else:
                decision = await self.controller.on_chunk(transcript, event.ts)
            st.stage_ms["controller"] += (time.perf_counter() - t_ctrl) * 1000
            decisions.append(decision)
            trace.emit(ev.CONTROLLER_DECISION, stream_ts=event.ts, action=decision.action, reason=decision.reason,
                       trigger_decided=decision.trigger, stability_score=decision.stability_score)

            if not self.decompose:
                if decision.action == "retrieve" and decision.trigger is not None:
                    self._dispatch([(None, to_search_query(transcript))], event.ts, decision.trigger, st)
                continue
            if decision.action == "retrieve" and decision.trigger is not None:
                await self._plan_and_search(transcript, event.ts, decision.trigger, st)
            elif decision.action == "wait" and decision.reason == "provisional_limit":
                await self._plan_and_search(transcript, event.ts, None, st)
            elif is_end and decision.action == "wait" and st.searched and transcript != st.planned_transcript:
                await self._plan_and_search(transcript, event.ts, "final", st)

        if st.tasks:
            await asyncio.gather(*st.tasks)
        for sub_id, run in st.aliases.items():
            st.sub_results[sub_id] = run.results
        for q in st.live:
            if q.id in st.sub_results:
                q.status = "retrieved"
        last = decisions[-1] if decisions else None
        suppressed = last.reason if last is not None and last.action == "suppress" else None
        sub_results = {q.id: st.sub_results[q.id] for q in st.live if q.id in st.sub_results}
        result = StreamTurnResult(transcript, end_ts, decisions, st.runs, suppressed, [], list(st.live), sub_results)
        if sub_results:
            t = time.perf_counter()
            fused = fuse_evidence(sub_results, {q.id: search_text(q) for q in st.live}, self.reranker,
                                  top_k=self.settings.fusion_top_k, quota=self.settings.fusion_quota,
                                  near_duplicate=self.settings.fusion_near_duplicate)
            result.evidence, result.evidence_sources = fused.evidence, fused.sources
            st.stage_ms["fusion"] = (time.perf_counter() - t) * 1000
            trace.emit(ev.FUSION_COMPLETED, stage_latency_ms=st.stage_ms["fusion"],
                       chunk_ids=[h.chunk.chunk_id for h in fused.evidence], sources=fused.sources,
                       dropped_duplicates=fused.dropped_duplicates,
                       dropped_near_duplicates=fused.dropped_near_duplicates)
        elif st.runs:
            result.evidence = result.final_results
        trace.emit(ev.REQUEST_COMPLETED, stage_latency_ms=trace.elapsed_ms(), retrievals=len(st.runs),
                   sub_queries=len(st.live), first_retrieval_s=result.first_retrieval_s, utterance_end_s=end_ts,
                   retrieved_early=result.retrieved_early, suppressed_reason=suppressed,
                   cache_hits=st.cache.hits if st.cache is not None else 0,
                   stage_ms={k: round(v, 3) for k, v in st.stage_ms.items()})
        result.stage_ms = {k: round(v, 3) for k, v in st.stage_ms.items()}
        result.request_id = trace.request_id
        result.telemetry = list(trace.events)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream one utterance through the controller and retrieval.")
    parser.add_argument("--utterance", required=True, help="Fragments separated by '|'.")
    parser.add_argument("--step", type=float, default=0.8, help="Seconds between fragments.")
    parser.add_argument("--clock", choices=["simulated", "wall"], default="simulated")
    parser.add_argument("--prior", default=None, help="Previous output in this session, if any.")
    parser.add_argument("--no-decompose", action="store_true", help="Phase 2 behaviour: no multi-intent planning.")
    args = parser.parse_args()
    parts = [p.strip() for p in args.utterance.split("|") if p.strip()]
    if not parts:
        parser.error("--utterance is empty")

    engine = StreamingEngine(decompose=not args.no_decompose)
    result = asyncio.run(engine.run_turn(chunks_from_text(parts, step_s=args.step), args.prior, clock=args.clock))
    out = {
        "decisions": [d.model_dump(exclude_none=True) for d in result.decisions],
        "retrieval_events": [e.model_dump() for e in result.retrieval_events],
        "sub_queries": [q.model_dump() for q in result.sub_queries],
        "sub_results": {k: [h.chunk.chunk_id for h in v] for k, v in result.sub_results.items()},
        "evidence": [{"chunk_id": h.chunk.chunk_id, "score": round(h.score, 4),
                      "sub_queries": result.evidence_sources.get(h.chunk.chunk_id, [])} for h in result.evidence],
        "utterance_end_s": result.utterance_end_s,
        "retrieved_early": result.retrieved_early,
        "suppressed_reason": result.suppressed_reason,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
