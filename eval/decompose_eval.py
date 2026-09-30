"""Decomposition + fusion metrics (step 3.7) over the dev scenarios.

Matching (fixed before any results were inspected): each gold sub-intent is matched
one-to-one to a predicted sub-query by the share of the gold description's content terms
that appear in the sub-query's own text (carried constraints excluded, so injected context
cannot inflate the score). Greedy assignment, highest score first; a match counts as a
correctly identified sub-intent when its score >= MATCH_THRESHOLD.

* G3 (target >= 0.70): share of compound turns (>= 2 gold sub-intents, retrieval required)
  with >= 2 correctly identified sub-intents.
* over-split rate: share of single-intent retrieval turns that produced > 1 sub-query.
* intent evidence recall: share of gold sub-intents (with evidence) whose supporting
  section appears in the fused evidence set; `matched sub-query recall@5` checks the
  matched sub-query's own top-5 instead.
"""

from __future__ import annotations

import statistics
import time

from eval.scenarios import Scenario
from src.config import Settings, get_settings
from src.engine import StreamingEngine
from src.retrieval.text import content_terms
from src.schemas import SubQuery

MATCH_THRESHOLD = 1 / 3
G3_TARGET = 0.70


def match_subintents(gold: list, predicted: list[SubQuery]) -> dict[str, tuple[str, float]]:
    """gold sub-intent id -> (predicted sub-query id, score), one-to-one, score >= threshold."""
    pairs = []
    for g in gold:
        gt = content_terms(g.text)
        if not gt:
            continue
        for q in predicted:
            score = len(gt & content_terms(q.text)) / len(gt)
            pairs.append((score, g.id, q.id))
    matched: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for score, gid, qid in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if score >= MATCH_THRESHOLD and gid not in matched and qid not in used:
            matched[gid] = (qid, round(score, 4))
            used.add(qid)
    return matched


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


async def evaluate_decomposition(scenarios: list[Scenario], engine: StreamingEngine) -> dict:
    rows = []
    for scenario in scenarios:
        session_id = f"eval-{scenario.id}-{int(time.time() * 1000)}"
        prior: str | None = None
        for t in scenario.turns:
            result = await engine.run_turn(t.chunks, prior_output=prior,
                                           request_id=f"{scenario.id}-t{t.turn}", session_id=session_id)
            matched = match_subintents(t.gold_sub_intents, result.sub_queries)
            evidence_cites = {h.chunk.citation for h in result.evidence}
            intent_rows = []
            for g in t.gold_sub_intents:
                qid, score = matched.get(g.id, (None, 0.0))
                own = {h.chunk.citation for h in result.sub_results.get(qid, [])[:5]} if qid else set()
                intent_rows.append({
                    "gold": g.id, "text": g.text, "supporting": g.supporting, "matched": qid, "score": score,
                    "in_evidence": all(c in evidence_cites for c in g.supporting) if g.supporting else None,
                    "in_own_top5": all(c in own for c in g.supporting) if (g.supporting and qid) else None,
                })
            rows.append({
                "scenario": scenario.id, "turn": t.turn, "category": scenario.category,
                "utterance": result.utterance, "retrieval_required": t.retrieval_required,
                "gold_n": len(t.gold_sub_intents),
                "predicted": [{"id": q.id, "text": q.text, "constraints": q.constraints} for q in result.sub_queries],
                "correct": sum(1 for r in intent_rows if r["matched"]),
                "intents": intent_rows,
                "evidence": sorted(evidence_cites),
                "retrievals": len(result.retrievals),
                "cache_hits": sum(e.event == "cache_hit" for e in result.telemetry),
                "early": result.retrieved_early,
                "suppressed_reason": result.suppressed_reason,
            })
            ev_text = " ".join(h.chunk.text for h in result.evidence)
            prior = f"{result.utterance} {ev_text}".strip()
    return {"summary": summarize(rows, engine), "turns": rows}


def summarize(rows: list[dict], engine: StreamingEngine | None = None) -> dict:
    compound = [r for r in rows if r["retrieval_required"] and r["gold_n"] >= 2]
    single = [r for r in rows if r["retrieval_required"] and r["gold_n"] == 1]
    g3_pass = [r for r in compound if r["correct"] >= 2]
    intents = [i for r in rows if r["retrieval_required"] for i in r["intents"] if i["supporting"]]
    eligible = [r for r in rows if r["retrieval_required"]]
    rate = len(g3_pass) / len(compound) if compound else None
    return {
        "system": "decompose_fusion",
        "planner": engine.planner.name if engine is not None and engine.planner is not None else None,
        "retriever": getattr(engine.retriever, "name", None) if engine is not None else None,
        "turns": len(rows),
        "g3": {"compound_turns": len(compound), "passed": len(g3_pass),
               "rate": round(rate, 4) if rate is not None else None, "target": G3_TARGET,
               "pass": rate is not None and rate >= G3_TARGET,
               "failed": [f"{r['scenario']} t{r['turn']}" for r in compound if r["correct"] < 2]},
        "over_split": {"single_intent_turns": len(single), "over_split": sum(len(r["predicted"]) > 1 for r in single),
                       "rate": round(sum(len(r["predicted"]) > 1 for r in single) / len(single), 4) if single else None,
                       "cases": [f"{r['scenario']} t{r['turn']}" for r in single if len(r["predicted"]) > 1]},
        "subqueries_per_compound_turn": _mean([len(r["predicted"]) for r in compound]),
        "intent_evidence_recall": _mean([1.0 if i["in_evidence"] else 0.0 for i in intents]),
        "matched_subquery_recall_at_5": _mean([1.0 if i["in_own_top5"] else 0.0 for i in intents if i["matched"]]),
        "retrievals_per_eligible_turn": _mean([r["retrievals"] for r in eligible]),
        "cache_hits": sum(r["cache_hits"] for r in rows),
        "g2_early_rate": round(sum(r["early"] for r in eligible) / len(eligible), 4) if eligible else None,
        "no_retrieval_turns_retrieved": sum(r["retrievals"] > 0 for r in rows if not r["retrieval_required"]),
    }


def print_decompose_summary(s: dict) -> None:
    g3, os_ = s["g3"], s["over_split"]
    print(f"\nSystem: decompose_fusion (planner={s['planner']}, retriever={s['retriever']})   turns={s['turns']}")
    print("| metric | value |")
    print("|---|---|")
    print(f"| G3 compound turns with >= 2 correct sub-intents | {g3['passed']}/{g3['compound_turns']} = {g3['rate']} "
          f"(target >= {g3['target']}: {'PASS' if g3['pass'] else 'FAIL'}) |")
    print(f"| G3 failures | {', '.join(g3['failed']) or 'none'} |")
    print(f"| single-intent turns over-split | {os_['over_split']}/{os_['single_intent_turns']} = {os_['rate']} "
          f"{os_['cases'] or ''} |")
    print(f"| sub-queries per compound turn | {s['subqueries_per_compound_turn']} |")
    print(f"| gold sub-intent evidence in fused set | {s['intent_evidence_recall']} |")
    print(f"| matched sub-query recall@5 | {s['matched_subquery_recall_at_5']} |")
    print(f"| retrievals per eligible turn / cache hits | {s['retrievals_per_eligible_turn']} / {s['cache_hits']} |")
    print(f"| G2 early retrieval (full pipeline) | {s['g2_early_rate']} |")
    print(f"| no-retrieval turns that retrieved | {s['no_retrieval_turns_retrieved']} |")


def build_engine(settings: Settings | None = None, stack=None, telemetry=None, retriever: str = "hybrid",
                 anti_fragmentation: bool | None = None) -> StreamingEngine:
    from src.decompose.planner import build_planner
    from src.retrieval.factory import build_retrieval

    s = settings or get_settings()
    stack = stack or build_retrieval(s)
    planner = build_planner(stack.dense.embedder, s, anti_fragmentation=anti_fragmentation)
    engine = StreamingEngine(s, stack=stack, telemetry=telemetry, planner=planner, decompose=True)
    engine.retriever = stack.by_name(retriever)
    return engine
