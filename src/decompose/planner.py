"""Query planner: decomposer (3.1) -> anti-fragmentation guard (3.2) -> context carry-over (3.3).

Keeps sub-query ids stable across calls within one utterance, so the engine can tell which
sub-queries are new or changed and retrieve only those.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.decompose.context import carry_context
from src.decompose.decomposer import Decomposer, DiffOp, IdGenerator, apply_ops, make_decomposer
from src.decompose.dedupe import AntiFragmentationGuard, GuardConfig
from src.llm.client import make_llm_client
from src.retrieval.dense import Embedder
from src.schemas import SubQuery
from src.telemetry.logger import RequestTrace


@dataclass
class PlanResult:
    subqueries: list[SubQuery]
    ops: list[DiffOp]
    guard_notes: list[str] = field(default_factory=list)


class QueryPlanner:
    def __init__(self, decomposer: Decomposer, guard: AntiFragmentationGuard | None):
        self.decomposer = decomposer
        self.guard = guard
        self.new_id = IdGenerator()

    @property
    def name(self) -> str:
        return f"{self.decomposer.name}+{'guard' if self.guard else 'noguard'}"

    def reset(self) -> None:
        self.new_id = IdGenerator()

    async def plan(self, transcript: str, live: list[SubQuery], trace: RequestTrace | None = None) -> PlanResult:
        ops = await self.decomposer.decompose(transcript, live, trace)
        subs = apply_ops(live, ops, self.new_id)
        if not subs and transcript.strip():
            subs = [SubQuery(id=self.new_id(), text=transcript.strip())]
        notes: list[str] = []
        if self.guard is not None:
            guarded = self.guard.apply(subs, transcript)
            subs, notes = guarded.subqueries, guarded.notes
        subs = carry_context([q.model_copy(update={"constraints": []}) for q in subs])
        return PlanResult(subs, ops, notes)


def build_planner(embedder: Embedder, settings: Settings | None = None, decomposer: str | None = None,
                  anti_fragmentation: bool | None = None) -> QueryPlanner:
    s = settings or get_settings()
    kind = decomposer or s.decomposer
    client = make_llm_client(s) if kind == "llm" else None
    use_guard = s.anti_fragmentation if anti_fragmentation is None else anti_fragmentation
    guard = AntiFragmentationGuard(
        embedder,
        GuardConfig(s.max_subqueries, s.subquery_merge_similarity, s.subquery_min_content_terms),
    ) if use_guard else None
    return QueryPlanner(make_decomposer(kind, client), guard)
