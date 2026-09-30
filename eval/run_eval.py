"""Run a system over the dev scenarios and report metrics (step 1.10).

    python eval/run_eval.py --system baseline [--out results/baseline_dev.json] [--verbose]
    python eval/run_eval.py --system controller_only --metrics g2 [--controller-mode rule_only]
    python eval/run_eval.py --system decompose_fusion --metrics g3
    python eval/run_eval.py --system full --metrics g4,g5
    python eval/run_eval.py --system full --metrics all     # full replay: gates G1-G6 (eval/replay.py)

Phase 2 metrics (controller_only): see eval/controller_eval.py (G2 early retrieval rate,
false-trigger rate, seconds gained vs baseline).
Phase 3 metrics (decompose_fusion): see eval/decompose_eval.py (G3 multi-intent identification,
over-split rate, per-intent evidence recall).
Phase 4 metrics (full): see eval/full_eval.py (G4 citation support + fabricated ids, G5 session
refinement continuity, presentation turns with zero retrieval).

Phase 1 metrics (baseline):
* recall@5            share of gold supporting sections found among the top-5 reranked chunks
                      (turns that need retrieval and have gold evidence)
* citation validity   share of citations in answers that resolve to a real corpus section;
                      `fabricated` counts the ones that do not
* latency             end-to-end ms per turn (p50 / p95 / mean)
Informational:
* uncertainty on no-evidence turns / false uncertainty on answerable turns
* retrieval on no-retrieval turns (the baseline has no controller, so it always retrieves)
* retrieval-only recall@5 for bm25 / dense / hybrid (input for later ablations)

The baseline is non-streaming and stateless: every turn is answered independently from its
full utterance. Dev scenarios are evaluation data only; nothing in src/ imports them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controller_eval import evaluate_controller, print_controller_summary  # noqa: E402
from eval.decompose_eval import build_engine, evaluate_decomposition, print_decompose_summary  # noqa: E402
from eval.full_eval import evaluate_full, print_full_summary  # noqa: E402
from eval.scenarios import SCENARIO_DIR, Scenario, load_scenarios  # noqa: E402
from src.baseline import BaselinePipeline  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.retrieval.factory import build_retrieval  # noqa: E402
from src.synthesis.citations import CitationIndex, extract_citations  # noqa: E402

K = 5
SYSTEMS = {"baseline": 1, "controller_only": 2, "decompose_fusion": 3, "full": 4}
IMPLEMENTED = {"baseline": {"phase1"}, "controller_only": {"g2"}, "decompose_fusion": {"g3"}, "full": {"g4,g5", "all"}}


def recall_at_k(retrieved_citations: list[str], gold: list[str]) -> float:
    if not gold:
        raise ValueError("recall is undefined without gold evidence")
    return sum(g in retrieved_citations for g in gold) / len(gold)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


async def evaluate_baseline(scenarios: list[Scenario], settings=None, telemetry=None, stack=None) -> dict:
    settings = settings or get_settings()
    stack = stack or build_retrieval(settings)
    pipeline = BaselinePipeline(settings, stack=stack, telemetry=telemetry)
    index = CitationIndex(stack.chunks)

    turns_out = []
    for scenario in scenarios:
        session_id = f"eval-{scenario.id}-{int(time.time() * 1000)}"
        for t in scenario.turns:
            utterance = t.utterance
            if not utterance:
                turns_out.append({"scenario": scenario.id, "turn": t.turn, "category": scenario.category,
                                  "skipped": "empty utterance"})
                continue
            result = await pipeline.answer(
                utterance, utterance_end_s=t.utterance_end_s,
                request_id=f"baseline-{scenario.id}-t{t.turn}", session_id=session_id,
            )
            top_citations = [h.chunk.citation for h in result.reranked[:K]]
            cited = extract_citations(result.record.answer)
            gold = t.gold_supporting
            row = {
                "scenario": scenario.id,
                "turn": t.turn,
                "category": scenario.category,
                "utterance": utterance,
                "retrieval_required": t.retrieval_required,
                "expect_uncertainty": t.expect_uncertainty,
                "gold_supporting": gold,
                "also_relevant": sorted({c for g in t.gold_sub_intents for c in g.also_relevant}),
                "top5": top_citations,
                "recall_at_5": recall_at_k(top_citations, gold) if (t.retrieval_required and gold) else None,
                "cited": cited,
                "invalid_citations": [c for c in cited if not index.is_valid(c)],
                "retrieved": bool(result.record.retrieval_events),
                "uncertainty": result.record.uncertainty,
                "latency_ms": round(result.latency_ms, 3),
                "record": result.record.model_dump(),
            }
            turns_out.append(row)

    # retrieval-only recall for each retriever (no rerank), same turns
    ablation = {}
    for name in ("bm25", "dense", "hybrid"):
        retriever = stack.by_name(name)
        vals = []
        for row in turns_out:
            if row.get("recall_at_5") is None:
                continue
            hits = retriever.search(row["utterance"], k=K)
            vals.append(recall_at_k([h.chunk.citation for h in hits], row["gold_supporting"]))
        ablation[name] = round(statistics.mean(vals), 4) if vals else None

    answered = [r for r in turns_out if "skipped" not in r]
    recalls = [r["recall_at_5"] for r in answered if r["recall_at_5"] is not None]
    all_cites = [c for r in answered for c in r["cited"]]
    invalid = [c for r in answered for c in r["invalid_citations"]]
    lat = [r["latency_ms"] for r in answered]
    no_ev = [r for r in answered if r["expect_uncertainty"]]
    answerable = [r for r in answered if r["retrieval_required"] and not r["expect_uncertainty"]]
    no_retr = [r for r in answered if not r["retrieval_required"]]

    summary = {
        "system": "baseline",
        "scenarios": len(scenarios),
        "turns": len(turns_out),
        "turns_skipped": len(turns_out) - len(answered),
        "recall_at_5": {"mean": round(statistics.mean(recalls), 4) if recalls else None, "n": len(recalls),
                        "perfect": sum(v == 1.0 for v in recalls), "zero": sum(v == 0.0 for v in recalls)},
        "citation_validity": {
            "citations": len(all_cites),
            "valid": len(all_cites) - len(invalid),
            "fabricated": len(invalid),
            "rate": round((len(all_cites) - len(invalid)) / len(all_cites), 4) if all_cites else None,
        },
        "latency_ms": {"p50": round(percentile(lat, 0.5), 3), "p95": round(percentile(lat, 0.95), 3),
                       "mean": round(statistics.mean(lat), 3) if lat else None, "n": len(lat)},
        "uncertainty": {
            "expected_turns": len(no_ev),
            "emitted_when_expected": sum(r["uncertainty"] is not None for r in no_ev),
            "answerable_turns": len(answerable),
            "false_uncertainty_on_answerable": sum(r["uncertainty"] is not None for r in answerable),
        },
        "no_retrieval_turns": {"n": len(no_retr), "retrieved_anyway": sum(r["retrieved"] for r in no_retr)},
        "retrieval_only_recall_at_5": ablation,
        "config": {
            "dense_backend": settings.dense_backend, "rerank_backend": settings.rerank_backend,
            "llm_provider": settings.llm_provider, "rrf_k": settings.rrf_k, "top_k": settings.top_k,
            "min_evidence_score": settings.min_evidence_score,
        },
    }
    return {"summary": summary, "turns": turns_out}


def print_summary(s: dict) -> None:
    r, c, lat, u, nr = s["recall_at_5"], s["citation_validity"], s["latency_ms"], s["uncertainty"], s["no_retrieval_turns"]
    print(f"\nSystem: {s['system']}   scenarios={s['scenarios']} turns={s['turns']} (skipped {s['turns_skipped']})")
    print("| metric | value |")
    print("|---|---|")
    print(f"| recall@5 (mean over {r['n']} turns) | {r['mean']} (perfect {r['perfect']}, zero {r['zero']}) |")
    print(f"| citation validity | {c['valid']}/{c['citations']} = {c['rate']} (fabricated {c['fabricated']}) |")
    print(f"| latency ms p50 / p95 / mean | {lat['p50']} / {lat['p95']} / {lat['mean']} |")
    print(f"| uncertainty emitted on no-evidence turns | {u['emitted_when_expected']}/{u['expected_turns']} |")
    print(f"| false uncertainty on answerable turns | {u['false_uncertainty_on_answerable']}/{u['answerable_turns']} |")
    print(f"| no-retrieval turns that retrieved anyway | {nr['retrieved_anyway']}/{nr['n']} |")
    ab = s["retrieval_only_recall_at_5"]
    print(f"| retrieval-only recall@5 bm25 / dense / hybrid | {ab['bm25']} / {ab['dense']} / {ab['hybrid']} |")
    print(f"config: {json.dumps(s['config'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="baseline", choices=sorted(SYSTEMS))
    parser.add_argument("--metrics", default=None, help="baseline: phase1 (default); controller_only: g2 (default).")
    parser.add_argument("--controller-mode", choices=["rule_only", "rule_stability"], default=None,
                        help="controller_only: override PRISM_CONTROLLER_MODE.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--out", type=Path, help="Write full per-turn results as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print one line per turn.")
    args = parser.parse_args()

    if args.system not in IMPLEMENTED:
        print(f"--system {args.system} is not implemented until Phase {SYSTEMS[args.system]}.", file=sys.stderr)
        return 2
    metrics = args.metrics or next(iter(IMPLEMENTED[args.system]))
    if args.system == "full":
        metrics = ",".join(sorted(m.strip() for m in metrics.split(",") if m.strip()))
        metrics = "g4,g5" if metrics in ("g4", "g5", "g4,g5") else metrics
        metrics = "all" if "all" in metrics.split(",") else metrics
    if metrics not in IMPLEMENTED[args.system]:
        print(f"--metrics {metrics} is not implemented for --system {args.system} "
              f"(available: {sorted(IMPLEMENTED[args.system])}).", file=sys.stderr)
        return 2

    scenarios = load_scenarios(args.scenarios)
    if args.system == "full" and metrics == "all":
        from eval.replay import run as run_replay
        from eval.replay import summary_markdown

        out = args.out or ROOT / "results" / "replay"
        res = asyncio.run(run_replay(out if out.suffix != ".json" else out.parent))
        print(summary_markdown(res))
        return 1 if any(g["pass"] is False for g in res["gates"].values()) else 0
    if args.system == "full":
        from src.session.pipeline import SessionPipeline

        results = asyncio.run(evaluate_full(scenarios, SessionPipeline()))
        if args.verbose:
            for row in results["turns"]:
                g5 = row.get("g5")
                flag = "" if g5 is None else (" G5-ok" if all(v for k, v in g5.items() if k != "touched_outside_affected") else " G5-FAIL")
                print(f"{row['scenario']:<12} t{row['turn']} v{row['version']} {row['kind']:<17} retr={row['retrieval_calls']} "
                      f"claims+{row['new_claims']} supported={row['new_claims_supported']}{flag} "
                      f"clauses={[(k, t) for k, t, _ in row['clauses']]}")
        print_full_summary(results["summary"])
        exit_code = 1 if results["summary"]["g4"]["fabricated_ids"] else 0
    elif args.system == "decompose_fusion":
        results = asyncio.run(evaluate_decomposition(scenarios, build_engine()))
        if args.verbose:
            for row in results["turns"]:
                preds = " || ".join(p["text"] for p in row["predicted"])
                print(f"{row['scenario']:<12} t{row['turn']} gold={row['gold_n']} correct={row['correct']} "
                      f"pred={len(row['predicted'])}: {preds}")
        print_decompose_summary(results["summary"])
        exit_code = 0
    elif args.system == "controller_only":
        from src.retrieval.factory import build_retrieval
        from src.stream.controller import build_controller

        settings = get_settings()
        stack = build_retrieval(settings)
        controller = build_controller(stack, settings, mode=args.controller_mode)
        results = asyncio.run(evaluate_controller(scenarios, settings, stack, controller))
        if args.verbose:
            for row in results["turns"]:
                print(f"{row['scenario']:<12} t{row['turn']} req={row['retrieval_required']!s:<5} "
                      f"early={row['early']!s:<5} first={row['first_retrieval_s']} end={row['utterance_end_s']} "
                      f"triggers={row['triggers']} suppressed={row['suppressed_reason']}")
        print_controller_summary(results["summary"])
        exit_code = 0
    else:
        results = asyncio.run(evaluate_baseline(scenarios))
        if args.verbose:
            for row in results["turns"]:
                if "skipped" in row:
                    print(f"{row['scenario']:<12} t{row['turn']} skipped: {row['skipped']}")
                    continue
                print(f"{row['scenario']:<12} t{row['turn']} recall@5={row['recall_at_5']} "
                      f"cited={row['cited']} uncertainty={'yes' if row['uncertainty'] else 'no'} "
                      f"{row['latency_ms']:.1f}ms")
        print_summary(results["summary"])
        exit_code = 1 if results["summary"]["citation_validity"]["fabricated"] else 0

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
