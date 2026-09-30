"""Controller metrics (step 2.6) over the dev scenarios.

G2 early retrieval rate: share of eligible turns (`retrieval_required: true`) whose first
retrieval starts before the utterance-end timestamp. Target >= 0.80.
False-trigger rate: share of no-retrieval turns (`retrieval_required: false`) that retrieved.
Seconds gained vs baseline: the baseline starts retrieval at utterance end, so the gain on a
turn is `utterance_end_s - first_retrieval_s` (0 when the first retrieval is at the end).

Turns of one scenario run in order in one ephemeral session. A later turn's "prior output"
is the previous turn's utterance plus the text of its retrieved evidence (there is no answer
synthesis before Phase 4). Nothing is shared across scenarios.
"""

from __future__ import annotations

import statistics
import time

from eval.scenarios import Scenario
from src.config import Settings, get_settings
from src.engine import StreamingEngine
from src.retrieval.factory import RetrievalStack, build_retrieval
from src.stream.controller import RetrievalController, build_controller
from src.telemetry.logger import TelemetryLogger

K = 5
G2_TARGET = 0.80


def _recall(results, gold: list[str]) -> float | None:
    if not gold:
        return None
    cites = [h.chunk.citation for h in results[:K]]
    return sum(g in cites for g in gold) / len(gold)


async def evaluate_controller(
    scenarios: list[Scenario],
    settings: Settings | None = None,
    stack: RetrievalStack | None = None,
    controller: RetrievalController | None = None,
    telemetry: TelemetryLogger | None = None,
    engine: StreamingEngine | None = None,
) -> dict:
    """Controller-only by default (no decomposition), so G2 measures the controller alone.
    Pass a decomposing `engine` to measure G2 for the full Phase 3 pipeline."""
    s = settings or get_settings()
    stack = stack or build_retrieval(s)
    controller = controller or build_controller(stack, s)
    engine = engine or StreamingEngine(s, stack=stack, controller=controller, telemetry=telemetry, decompose=False)

    rows = []
    for scenario in scenarios:
        session_id = f"eval-{scenario.id}-{int(time.time() * 1000)}"
        prior: str | None = None
        for t in scenario.turns:
            t0 = time.perf_counter()
            result = await engine.run_turn(t.chunks, prior_output=prior,
                                           request_id=f"{scenario.id}-t{t.turn}", session_id=session_id)
            engine_ms = (time.perf_counter() - t0) * 1000
            first = result.first_retrieval_s
            rows.append({
                "scenario": scenario.id,
                "turn": t.turn,
                "category": scenario.category,
                "utterance": result.utterance,
                "retrieval_required": t.retrieval_required,
                "utterance_end_s": result.utterance_end_s,
                "first_retrieval_s": first,
                "retrieved": bool(result.retrievals),
                "early": result.retrieved_early,
                "seconds_gained": round(result.utterance_end_s - first, 3) if first is not None else None,
                "triggers": [e.trigger for e in result.retrieval_events],
                "queries": [e.query for e in result.retrieval_events],
                "decisions": [f"{d.ts}:{d.action}/{d.reason}" for d in result.decisions],
                "suppressed_reason": result.suppressed_reason,
                "first_recall_at_5": _recall(result.retrievals[0].results, t.gold_supporting) if result.retrievals else None,
                "final_recall_at_5": _recall(result.final_results, t.gold_supporting) if result.retrievals else None,
                "engine_ms": round(engine_ms, 3),
            })
            evidence = " ".join(h.chunk.text for h in result.final_results)
            prior = f"{result.utterance} {evidence}".strip()

    return {"summary": summarize(rows, engine.controller.name), "turns": rows}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def summarize(rows: list[dict], controller_name: str) -> dict:
    eligible = [r for r in rows if r["retrieval_required"]]
    no_retr = [r for r in rows if not r["retrieval_required"]]
    presentation = [r for r in no_retr if r["category"] == "presentation_only"]
    early = [r for r in eligible if r["early"]]
    rate = len(early) / len(eligible) if eligible else None
    return {
        "system": "controller_only",
        "controller": controller_name,
        "turns": len(rows),
        "g2_early_retrieval": {
            "eligible": len(eligible),
            "early": len(early),
            "rate": round(rate, 4) if rate is not None else None,
            "target": G2_TARGET,
            "pass": rate is not None and rate >= G2_TARGET,
            "missed_entirely": sum(not r["retrieved"] for r in eligible),
        },
        "false_trigger": {
            "no_retrieval_turns": len(no_retr),
            "retrieved": sum(r["retrieved"] for r in no_retr),
            "rate": round(sum(r["retrieved"] for r in no_retr) / len(no_retr), 4) if no_retr else None,
            "presentation_turns": len(presentation),
            "presentation_retrieved": sum(r["retrieved"] for r in presentation),
        },
        "seconds_gained_vs_baseline": {
            "mean_over_eligible": _mean([r["seconds_gained"] or 0.0 for r in eligible]),
            "mean_over_early": _mean([r["seconds_gained"] for r in early]),
        },
        "retrievals_per_eligible_turn": _mean([len(r["triggers"]) for r in eligible]),
        "first_retrieval_recall_at_5": _mean([r["first_recall_at_5"] for r in eligible]),
        "final_retrieval_recall_at_5": _mean([r["final_recall_at_5"] for r in eligible]),
        "engine_ms_per_turn": {"mean": _mean([r["engine_ms"] for r in rows]),
                               "max": max((r["engine_ms"] for r in rows), default=None)},
    }


def print_controller_summary(s: dict) -> None:
    g2, ft, sg = s["g2_early_retrieval"], s["false_trigger"], s["seconds_gained_vs_baseline"]
    print(f"\nSystem: controller_only ({s['controller']})   turns={s['turns']}")
    print("| metric | value |")
    print("|---|---|")
    print(f"| G2 early retrieval rate | {g2['early']}/{g2['eligible']} = {g2['rate']} "
          f"(target >= {g2['target']}: {'PASS' if g2['pass'] else 'FAIL'}) |")
    print(f"| eligible turns with no retrieval at all | {g2['missed_entirely']} |")
    print(f"| false-trigger rate on no-retrieval turns | {ft['retrieved']}/{ft['no_retrieval_turns']} = {ft['rate']} |")
    print(f"| presentation-only turns that retrieved | {ft['presentation_retrieved']}/{ft['presentation_turns']} |")
    print(f"| seconds gained vs baseline (mean over eligible / over early) | {sg['mean_over_eligible']} / {sg['mean_over_early']} |")
    print(f"| retrievals per eligible turn | {s['retrievals_per_eligible_turn']} |")
    print(f"| recall@5 of first / final retrieval | {s['first_retrieval_recall_at_5']} / {s['final_retrieval_recall_at_5']} |")
    print(f"| engine ms per turn (mean / max, simulated clock) | {s['engine_ms_per_turn']['mean']} / {s['engine_ms_per_turn']['max']} |")
