"""Non-streaming baseline pipeline (step 1.8). The reference system for all later benchmarks.

    full utterance -> hybrid retrieval (BM25 + dense, RRF) -> rerank -> answer with citations
                   -> citation check -> uncertainty -> OutputRecord

Usage:
    python -m src.baseline --query "..." [--utterance-end-s 2.1] [--show-evidence]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.llm.client import make_llm_client
from src.retrieval.factory import RetrievalStack, build_retrieval
from src.retrieval.rerank import Reranker, make_reranker
from src.schemas import Claim, OutputRecord, RetrievalEvent, ScoredChunk, TelemetryEvent
from src.synthesis.citations import CitationIndex, chunk_to_citation, extract_citations
from src.synthesis.generator import ExtractiveGenerator, LLMGenerator, render_answer
from src.synthesis.uncertainty import supported_hits, uncertainty_note
from src.telemetry import events as ev
from src.telemetry.logger import TelemetryLogger


@dataclass
class BaselineResult:
    record: OutputRecord
    retrieved: list[ScoredChunk]
    reranked: list[ScoredChunk]
    evidence: list[ScoredChunk]
    claims: list[Claim]
    invalid_citations: list[str]
    latency_ms: float
    stage_ms: dict[str, float] = field(default_factory=dict)
    telemetry: list[TelemetryEvent] = field(default_factory=list)


class BaselinePipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        stack: RetrievalStack | None = None,
        reranker: Reranker | None = None,
        generator=None,
        telemetry: TelemetryLogger | None = None,
    ):
        self.settings = s = settings or get_settings()
        self.stack = stack or build_retrieval(s)
        self.reranker = reranker or make_reranker(s.rerank_backend, s.rerank_model)
        if generator is None:
            client = make_llm_client(s)
            generator = LLMGenerator(client) if client is not None else ExtractiveGenerator()
        self.generator = generator
        self.telemetry = telemetry or TelemetryLogger(s.log_dir / "telemetry.jsonl")
        self.citations = CitationIndex(self.stack.chunks)

    async def answer(
        self, utterance: str, utterance_end_s: float = 0.0, request_id: str | None = None, session_id: str | None = None
    ) -> BaselineResult:
        s = self.settings
        trace = self.telemetry.trace(request_id, session_id)
        stage: dict[str, float] = {}
        trace.emit(ev.REQUEST_STARTED, system="baseline", utterance=utterance, generator=self.generator.name)

        # retrieval: one final-trigger search over the full utterance
        trace.emit(ev.RETRIEVAL_STARTED, trigger="final", query=utterance, utterance_end_s=utterance_end_s)
        t = time.perf_counter()
        retrieved = self.stack.hybrid.search(utterance, k=s.candidates)
        stage["retrieval"] = (time.perf_counter() - t) * 1000
        trace.emit(
            ev.RETRIEVAL_COMPLETED,
            stage_latency_ms=stage["retrieval"],
            trigger="final",
            chunk_ids=[h.chunk.chunk_id for h in retrieved],
        )

        t = time.perf_counter()
        top = self.reranker.rerank(utterance, retrieved, s.top_k)
        stage["rerank"] = (time.perf_counter() - t) * 1000
        trace.emit(
            ev.RERANK_COMPLETED,
            stage_latency_ms=stage["rerank"],
            reranker=self.reranker.name,
            chunk_ids=[h.chunk.chunk_id for h in top],
            scores=[round(h.score, 4) for h in top],
        )

        # synthesis only over evidence that clears the threshold
        evidence = supported_hits(utterance, top, s.min_evidence_score)
        note = uncertainty_note(utterance, top, s.min_evidence_score)
        t = time.perf_counter()
        if evidence:
            synth = await self.generator.synthesize(utterance, evidence, trace=trace)
            claims = synth.claims
            note = synth.uncertainty or note
            if not claims and note is None:
                note = "The retrieved evidence did not support any answer to this request. Please clarify."
        else:
            claims = []
        stage["synthesis"] = (time.perf_counter() - t) * 1000

        answer = render_answer(claims)
        cited = extract_citations(answer)
        valid, invalid = self.citations.check(cited)
        claim_chunks = [cid for c in claims for cid in c.chunk_ids]
        trace.emit(
            ev.CITATION_CHECK,
            citations=cited,
            invalid=invalid,
            claim_chunk_ids=claim_chunks,
            all_claim_chunks_exist=all(self.citations.is_valid_chunk(c) for c in claim_chunks),
        )

        record = OutputRecord(
            retrieval_events=[RetrievalEvent(timestamp_s=utterance_end_s, query=utterance, trigger="final")],
            sub_queries=[utterance],
            answer=answer,
            citations=valid,
            uncertainty=note,
        )
        llm = [e for e in trace.events if e.event == ev.LLM_CALL]
        tokens_in, tokens_out = sum(e.tokens_in or 0 for e in llm), sum(e.tokens_out or 0 for e in llm)
        trace.emit(ev.ANSWER_EMITTED, tokens_in=tokens_in, tokens_out=tokens_out,
                   est_cost_usd=ev.estimate_cost_usd(tokens_in, tokens_out, s.cost_per_1k_input, s.cost_per_1k_output),
                   answer_version_from=0, answer_version=1, kind="baseline", citations=valid,
                   sub_queries=[{"id": "q1", "text": utterance, "constraints": []}],
                   sources={c.id: c.chunk_ids for c in claims}, stage_ms={k: round(v, 3) for k, v in stage.items()},
                   uncertainty=note is not None)
        latency = trace.elapsed_ms()
        trace.emit(ev.REQUEST_COMPLETED, stage_latency_ms=latency, stage_ms={k: round(v, 3) for k, v in stage.items()})
        return BaselineResult(
            record=record,
            retrieved=retrieved,
            reranked=top,
            evidence=evidence,
            claims=claims,
            invalid_citations=invalid,
            latency_ms=latency,
            stage_ms=stage,
            telemetry=list(trace.events),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer one utterance with the non-streaming baseline.")
    parser.add_argument("--query", required=True, help="The full user utterance.")
    parser.add_argument("--utterance-end-s", type=float, default=0.0)
    parser.add_argument("--show-evidence", action="store_true", help="Also print the reranked evidence ids.")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must not be empty")

    result = asyncio.run(BaselinePipeline().answer(args.query, args.utterance_end_s))
    print(json.dumps(result.record.model_dump(), indent=2, ensure_ascii=False))
    if args.show_evidence:
        for h in result.evidence:
            print(f"  evidence {h.rank}: {h.chunk.chunk_id} ({chunk_to_citation(h.chunk.chunk_id)}) score={h.score:.3f}")
    print(f"latency_ms={result.latency_ms:.1f} invalid_citations={result.invalid_citations}", file=sys.stderr)
    return 0 if not result.invalid_citations else 1


if __name__ == "__main__":
    sys.exit(main())
