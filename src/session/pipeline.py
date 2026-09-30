"""Full session pipeline (Phase 4): streaming engine + session ledger + delta refinement +
grounded synthesis + uncertainty, one turn at a time.

Per turn:
1. The streaming engine (Phase 2-3) runs the controller, decomposes the utterance into
   sub-queries, searches them (targeted: only this utterance's clauses are searched) and
   fuses evidence. The session's last output is passed so presentation-only turns are
   suppressed before any search.
2. Presentation-only turns -> the existing answer is transformed without retrieval (4.7).
   Turns with nothing searchable -> clarification request, or per-intent "no evidence" notes.
3. Otherwise the delta engine (4.6) classifies each clause against the ledger:
   new sub-intents get their own evidence and claims; a parameter update adds the clause as
   a constraint, adds its evidence (delta citations) and adds delta claims to the affected
   sub-intent while keeping its existing claims; a contradiction invalidates only the
   affected claims that depend on the negated term and adds replacements.
   Untouched sub-intents keep their claims, ids and citations.
4. New claims are synthesized (4.3), verified (4.4), and uncertainty is set per sub-intent (4.5).
5. The ledger commits a new answer version; the output record is built from all active claims.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.decompose.context import search_text
from src.engine import StreamingEngine, StreamTurnResult
from src.llm.client import make_llm_client
from src.retrieval.factory import RetrievalStack, build_retrieval
from src.retrieval.text import content_terms
from src.schemas import Claim, OutputRecord, ScoredChunk, SubQuery, TranscriptChunk
from src.session.delta import ClauseDelta, DeltaClassifier, DeltaDecision
from src.session.ledger import ClaimLedger
from src.session.store import Session, SessionStore
from src.synthesis.citations import CitationIndex, extract_citations
from src.synthesis.generator import ExtractiveGenerator, LLMGenerator, render_answer, synthesize_subintents
from src.synthesis.grounding import GroundingReport, GroundingVerifier, make_judge
from src.synthesis.presentation import present
from src.synthesis.uncertainty import compose_uncertainty, subintent_uncertainty, supported_hits, turn_clarification
from src.telemetry import events as ev
from src.telemetry.events import estimate_cost_usd
from src.telemetry.logger import RequestTrace, TelemetryLogger


@dataclass
class TurnOutcome:
    record: OutputRecord
    answer_version: int
    kind: str  # first_answer | new_subintent | parameter_update | contradiction | presentation_only | clarification | no_evidence
    stream: StreamTurnResult
    delta: DeltaDecision | None = None
    ledger_diff: dict[str, list[str]] = field(default_factory=dict)
    grounding: GroundingReport | None = None
    delta_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    request_id: str = ""
    version_from: int = 0
    stage_ms: dict[str, float] = field(default_factory=dict)  # engine stages + synthesis / verification / presentation
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float = 0.0

    @property
    def retrieval_calls(self) -> int:
        return len(self.stream.retrievals)


def unique_by_text(claims: list[Claim]) -> list[Claim]:
    """Claims to render: one per distinct text (two sub-intents can yield the same corpus span;
    the ledger keeps both claims, the answer shows the statement once)."""
    seen: set[str] = set()
    out = []
    for c in claims:
        if c.text not in seen:
            seen.add(c.text)
            out.append(c)
    return out


def _evidence_for(stream: StreamTurnResult, qid: str) -> list[ScoredChunk]:
    own = [h for h in stream.evidence if qid in stream.evidence_sources.get(h.chunk.chunk_id, [])]
    return own or stream.sub_results.get(qid, [])


class SessionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        stack: RetrievalStack | None = None,
        store: SessionStore | None = None,
        engine: StreamingEngine | None = None,
        generator=None,
        verifier: GroundingVerifier | None = None,
        delta: DeltaClassifier | None = None,
        telemetry: TelemetryLogger | None = None,
    ):
        self.settings = s = settings or get_settings()
        self.stack = stack or build_retrieval(s)
        self.telemetry = telemetry or TelemetryLogger(s.log_dir / "telemetry.jsonl")
        self.store = store or SessionStore()
        self.engine = engine or StreamingEngine(s, stack=self.stack, telemetry=self.telemetry, decompose=True)
        client = make_llm_client(s)
        self.generator = generator or (LLMGenerator(client) if client is not None else ExtractiveGenerator())
        self.index = CitationIndex(self.stack.chunks)
        self.verifier = verifier or GroundingVerifier(
            self.index, make_judge(s.grounding_judge, client, s.lexical_support_threshold), mode=s.grounding_mode)
        self.delta = delta or DeltaClassifier()

    def end_session(self, session_id: str) -> None:
        self.store.end(session_id)

    # ------------------------------------------------------------------ turn

    async def handle_turn(self, session_id: str, chunks: list[TranscriptChunk],
                          request_id: str | None = None, clock: str = "simulated", speed: float = 1.0) -> TurnOutcome:
        request_id = request_id or uuid.uuid4().hex[:12]  # one id for the engine and pipeline events
        session = self.store.get_or_create(session_id)
        session.turns += 1
        turn = session.turns
        ledger = session.ledger
        version_from = ledger.version
        stream = await self.engine.run_turn(chunks, prior_output=session.prior_output, clock=clock, speed=speed,
                                            request_id=request_id, session_id=session_id, expects_answer=True)
        trace = self.telemetry.trace(request_id, session_id)
        before = ledger.snapshot()
        stages: dict[str, float] = {}

        if stream.suppressed_reason == "presentation_restructure" and ledger.version > 0:
            outcome = await self._presentation(session, stream, trace, stages)
        elif stream.suppressed_reason is not None or not stream.sub_queries:
            outcome = await self._unsearchable(session, stream, turn)
        else:
            outcome = await self._answer(session, stream, turn, trace, stages)
        outcome.ledger_diff = ClaimLedger.diff(before, ledger.snapshot())
        outcome.request_id, outcome.version_from = request_id, version_from
        outcome.stage_ms = {**stream.stage_ms, **{k: round(v, 3) for k, v in stages.items()}}
        llm_events = [e for e in [*stream.telemetry, *trace.events] if e.event == ev.LLM_CALL]
        outcome.tokens_in = sum(e.tokens_in or 0 for e in llm_events)
        outcome.tokens_out = sum(e.tokens_out or 0 for e in llm_events)
        outcome.est_cost_usd = estimate_cost_usd(outcome.tokens_in, outcome.tokens_out,
                                                 self.settings.cost_per_1k_input, self.settings.cost_per_1k_output)
        self._emit(outcome, session.ledger, trace)
        return outcome

    # ------------------------------------------------------------------ paths

    async def _presentation(self, session: Session, stream: StreamTurnResult, trace: RequestTrace,
                            stages: dict[str, float]) -> TurnOutcome:
        """Step 4.7 path: the answer is re-rendered, never re-retrieved."""
        ledger = session.ledger
        current = ledger.current
        prior_citations = ledger.citations()
        t = time.perf_counter()
        text, note = await present(stream.utterance, unique_by_text(ledger.active_claims()), client=make_llm_client(self.settings),
                                   trace=trace)
        stages["presentation"] = (time.perf_counter() - t) * 1000
        new_ids = [c for c in extract_citations(text) if c not in prior_citations]
        if new_ids:  # a transform may never introduce citations
            raise AssertionError(f"presentation transform introduced new citations: {new_ids}")
        session.last_output = text
        record = OutputRecord(retrieval_events=stream.retrieval_events,
                              sub_queries=[q.text for q in ledger.subqueries()], answer=text,
                              citations=extract_citations(text),
                              uncertainty=compose_uncertainty([current.uncertainty if current else None, note]))
        return TurnOutcome(record, ledger.version, "presentation_only", stream,
                           DeltaDecision("presentation_only", [], "suppression gate"))

    async def _unsearchable(self, session: Session, stream: StreamTurnResult, turn: int) -> TurnOutcome:
        ledger = session.ledger
        reason = stream.suppressed_reason or "no_subqueries"
        if reason == "no_corpus_terms" and self.engine.planner is not None:
            # identify the intents anyway so each one gets its own "no evidence" note
            self.engine.planner.reset()
            plan = await self.engine.planner.plan(stream.utterance, [])
            for q in plan.subqueries:
                state = ledger.upsert_subintent(q.model_copy(update={"id": f"t{turn}.{q.id}"}))
                state.uncertainty = f'No evidence in the corpus for "{q.text.strip().rstrip("?.")}".'
            notes = [ledger.subintents[f"t{turn}.{q.id}"].uncertainty for q in plan.subqueries]
            answer = ledger.commit_version(compose_uncertainty(self._notes(ledger)))
            record = self._record(ledger, stream, answer.uncertainty)
            session.last_output = record.answer or record.uncertainty or None  # what the user saw
            return TurnOutcome(record, ledger.version, "no_evidence", stream,
                               DeltaDecision("new_subintent", [], "; ".join(n for n in notes if n)))
        vocab = self.stack.bm25.vocabulary
        unmatched = sorted(t for t in content_terms(stream.utterance) if t not in vocab)
        note = turn_clarification(reason, stream.utterance, unmatched)
        record = OutputRecord(retrieval_events=stream.retrieval_events, sub_queries=[], answer="",
                              citations=[], uncertainty=note)
        return TurnOutcome(record, ledger.version, "clarification", stream)

    async def _answer(self, session: Session, stream: StreamTurnResult, turn: int, trace: RequestTrace,
                      stages: dict[str, float]) -> TurnOutcome:
        ledger = session.ledger
        first = ledger.version == 0
        prior_citations = ledger.citations()
        clauses = [(q.text, q.id) for q in stream.sub_queries]
        decision = self.delta.classify(clauses, ledger)
        by_qid = {q.id: q for q in stream.sub_queries}

        to_synthesize: list[tuple[str, SubQuery, list[ScoredChunk], str, list[str]]] = []
        for cd in decision.clauses:
            q = by_qid[cd.subquery_id]
            evidence = _evidence_for(stream, q.id)
            if cd.kind == "new_subintent" or cd.target is None:
                sid = f"t{turn}.{q.id}"
                ledger.upsert_subintent(q.model_copy(update={"id": sid}))
                ledger.add_evidence(sid, evidence)
                to_synthesize.append((sid, ledger.subintents[sid].subquery, evidence, search_text(q), list(q.constraints)))
                continue
            state = ledger.subintents[cd.target]
            sub = state.subquery
            if cd.kind == "contradiction":
                self._invalidate_dependent(ledger, cd)
                kept_constraints = [c for c in sub.constraints if not (content_terms(c) & set(cd.negated_terms))]
            else:
                kept_constraints = list(sub.constraints)
            state.subquery = sub.model_copy(update={"constraints": list(dict.fromkeys(kept_constraints + [cd.clause]))})
            ledger.add_evidence(cd.target, evidence)
            to_synthesize.append((cd.target, state.subquery, evidence, cd.clause, [cd.clause]))

        grounding = await self._synthesize_and_verify(ledger, to_synthesize, trace, stages)
        answer = ledger.commit_version(compose_uncertainty(self._notes(ledger)))
        record = self._record(ledger, stream, answer.uncertainty)
        session.last_output = record.answer or record.uncertainty or None  # what the user saw
        kind = "first_answer" if first else decision.kind
        delta_citations = [c for c in ledger.citations() if c not in prior_citations]  # new in this version
        invalid = [c for c in extract_citations(record.answer) if not self.index.is_valid(c)]
        return TurnOutcome(record, ledger.version, kind, stream, decision, grounding=grounding,
                           delta_citations=delta_citations, invalid_citations=invalid)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _invalidate_dependent(ledger: ClaimLedger, cd: ClauseDelta) -> None:
        negated = set(cd.negated_terms)
        claims = ledger.claims_for(cd.target)
        state = ledger.subintents[cd.target]
        if content_terms(state.subquery.text) & negated:
            dependent = claims  # the sub-intent itself was about the negated term
        else:
            dependent = [c for c in claims if content_terms(" ".join([c.text, *c.constraints])) & negated]
        ledger.invalidate([c.id for c in dependent])

    async def _synthesize_and_verify(self, ledger: ClaimLedger, items, trace: RequestTrace,
                                     stages: dict[str, float]) -> GroundingReport:
        """items: (sub-intent id, sub-query, evidence, query, claim constraints). For a new
        sub-intent the query is its own search text; for a delta it is the delta clause."""
        s = self.settings
        inputs, queries, meta = [], {}, {}
        for sid, sub, evidence, query, constraints in items:
            inputs.append((sub, supported_hits(query, evidence, s.min_evidence_score)))
            queries[sid] = query
            label = sub.text if query == search_text(sub) else query
            meta[sid] = (label, evidence, query, constraints)
        t = time.perf_counter()
        synth = await synthesize_subintents(self.generator, inputs, trace=trace, query_override=queries)
        stages["synthesis"] = (time.perf_counter() - t) * 1000
        new_claims: list[Claim] = []
        duplicates: dict[str, int] = {}
        for sid, claims in synth.claims.items():
            existing = {c.text for c in ledger.claims_for(sid)}
            for c in claims:
                if c.text in existing:  # already part of the answer: nothing new to add
                    duplicates[sid] = duplicates.get(sid, 0) + 1
                    continue
                existing.add(c.text)
                new_claims.append(c.model_copy(update={
                    "id": ledger.next_claim_id(sid), "version": ledger.version + 1, "constraints": meta[sid][3]}))
        t = time.perf_counter()
        report = await self.verifier.verify(new_claims, trace)
        stages["verification"] = (time.perf_counter() - t) * 1000
        ledger.add_claims(report.kept)
        for sid, (label, evidence, query, _) in meta.items():
            kept = sum(c.subintent_id == sid for c in report.kept) + duplicates.get(sid, 0)
            failed = sum(chk.claim.subintent_id == sid for chk in report.failed)
            note = subintent_uncertainty(label, query, evidence, s.min_evidence_score, kept, failed,
                                         synth.uncertainty.get(sid))
            state = ledger.subintents[sid]
            state.uncertainty = compose_uncertainty([state.uncertainty, note]) if state.subquery.text != label else note
        return report

    @staticmethod
    def _notes(ledger: ClaimLedger) -> list[str | None]:
        return [s.uncertainty for s in ledger.subintents.values()]

    @staticmethod
    def _record(ledger: ClaimLedger, stream: StreamTurnResult, uncertainty: str | None) -> OutputRecord:
        answer = render_answer(unique_by_text(ledger.active_claims()))
        return OutputRecord(retrieval_events=stream.retrieval_events,
                            sub_queries=[q.text for q in ledger.subqueries()], answer=answer,
                            citations=extract_citations(answer), uncertainty=uncertainty)

    def _emit(self, o: TurnOutcome, ledger: ClaimLedger, trace: RequestTrace) -> None:
        """Per-turn trace: citation check, then the answer event carrying version transition,
        sub-queries, claim -> source mapping, per-stage latencies, token counts and cost."""
        trace.emit(ev.CITATION_CHECK, citations=o.record.citations, invalid=o.invalid_citations,
                   fabricated_ids=o.grounding.fabricated_ids if o.grounding else [])
        trace.emit(
            ev.ANSWER_EMITTED,
            stage_latency_ms=sum(v for k, v in o.stage_ms.items() if k in ("synthesis", "verification", "presentation")),
            tokens_in=o.tokens_in, tokens_out=o.tokens_out, est_cost_usd=o.est_cost_usd,
            answer_version_from=o.version_from, answer_version=o.answer_version, kind=o.kind,
            sub_queries=[{"id": q.id, "text": q.text, "constraints": q.constraints} for q in ledger.subqueries()],
            sources={c.id: c.chunk_ids for c in ledger.active_claims()},
            citations=o.record.citations, delta_citations=o.delta_citations,
            delta_clauses=[(c.kind, c.target) for c in o.delta.clauses] if o.delta else [],
            ledger_diff=o.ledger_diff, retrieval_calls=o.retrieval_calls,
            retrieval_triggers=[e.trigger for e in o.record.retrieval_events],
            first_retrieval_s=o.stream.first_retrieval_s, utterance_end_s=o.stream.utterance_end_s,
            stage_ms=o.stage_ms, uncertainty=o.record.uncertainty is not None,
        )
