"""Controller ablation and threshold tuning (step 2.7).

    python eval/ablate_controller.py [--out reports/PHASE_2_CONTROLLER_ABLATION.md]

Compares
* rule_only       - gate + completeness/content rules; no stability probe
* rule_stability  - rule_only + BM25 top-k Jaccard stability (grid over its thresholds)
* llm             - one LLM call per chunk (only if PRISM_LLM_PROVIDER is configured;
                    otherwise reported as not run)

Selection rule for tuned defaults (fixed before looking at results): among configurations
with zero false triggers and zero presentation-turn retrievals, maximise the G2 early
retrieval rate; break ties by fewer retrievals per eligible turn, then higher recall@5 of
the first retrieval, then the more conservative setting (higher threshold, larger N, larger
content minimum). The tuning set is the 30 draft dev scenarios, so the numbers are
optimistic for held-out data.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controller_eval import evaluate_controller  # noqa: E402
from eval.scenarios import load_scenarios  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm.client import make_llm_client  # noqa: E402
from src.retrieval.factory import build_retrieval  # noqa: E402
from src.stream.controller import build_controller  # noqa: E402
from src.telemetry.logger import TelemetryLogger  # noqa: E402

GRID = {
    "stability_threshold": [0.2, 0.34, 0.5, 0.67, 1.0],
    "stable_chunks": [1, 2],
    "probe_k": [3, 5],
    "probe_min_content_terms": [1, 2, 3],
}
# Phase 2 starting values before any tuning (kept fixed so the table is reproducible).
START = {"stability_threshold": 0.5, "stable_chunks": 1, "probe_k": 3, "probe_min_content_terms": 2,
         "probe_min_tokens": 3}
RULE_ONLY_GRID = {
    "stable_chunks": [1, 2],
    "probe_min_content_terms": [1, 2, 3],
    "probe_min_tokens": [2, 3, 4],
}


async def run_config(scenarios, stack, settings, mode):
    controller = build_controller(stack, settings, mode=mode)
    out = await evaluate_controller(scenarios, settings, stack, controller, telemetry=TelemetryLogger(None))
    return out["summary"], out["turns"]


def row_of(label: str, params: dict, s: dict) -> dict:
    g2, ft, sg = s["g2_early_retrieval"], s["false_trigger"], s["seconds_gained_vs_baseline"]
    return {
        "label": label, **params,
        "g2": g2["rate"], "early": g2["early"], "eligible": g2["eligible"], "missed": g2["missed_entirely"],
        "false_trigger": ft["rate"], "presentation_retrieved": ft["presentation_retrieved"],
        "gain": sg["mean_over_eligible"], "retrievals": s["retrievals_per_eligible_turn"],
        "first_recall": s["first_retrieval_recall_at_5"], "engine_ms": s["engine_ms_per_turn"]["mean"],
    }


def select(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["false_trigger"] == 0 and r["presentation_retrieved"] == 0]
    return max(ok, key=lambda r: (r["g2"], -r["retrievals"], r["first_recall"] or 0,
                                  r.get("stability_threshold", 0), r["stable_chunks"], r["probe_min_content_terms"],
                                  r.get("probe_min_tokens", 0)))


def fmt(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)


async def main_async(out: Path | None) -> int:
    base = get_settings()
    scenarios = load_scenarios()
    stack = build_retrieval(base)

    base = dataclasses.replace(base, **START)
    grid_rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        summary, _ = await run_config(scenarios, stack, dataclasses.replace(base, **params), "rule_stability")
        grid_rows.append(row_of("rule_stability", params, summary))
    rule_rows = []
    for values in itertools.product(*RULE_ONLY_GRID.values()):
        params = dict(zip(RULE_ONLY_GRID, values))
        summary, _ = await run_config(scenarios, stack, dataclasses.replace(base, **params), "rule_only")
        rule_rows.append(row_of("rule_only", params, summary))
    best_stab, best_rule = select(grid_rows), select(rule_rows)
    tuned_stab = {k: best_stab[k] for k in GRID}
    tuned_rule = {k: best_rule[k] for k in RULE_ONLY_GRID}
    recommended = "rule_only" if best_rule["g2"] > best_stab["g2"] else "rule_stability"

    comparison = []
    untuned = dict(START)
    for label, mode, params in [
        ("rule_only (untuned start values)", "rule_only", untuned),
        ("rule_only (tuned)", "rule_only", tuned_rule),
        ("rule_stability (untuned start values)", "rule_stability", untuned),
        ("rule_stability (tuned)", "rule_stability", tuned_stab),
    ]:
        summary, _ = await run_config(scenarios, stack, dataclasses.replace(base, **params), mode)
        comparison.append(row_of(label, params, summary))
    tuned_params = {"rule_only": tuned_rule, "rule_stability": tuned_stab}

    llm_note = "not run: no LLM configured (PRISM_LLM_PROVIDER=none); see README teammate setup"
    if make_llm_client(base) is not None:  # pragma: no cover - needs a real LLM endpoint
        summary, _ = await run_config(scenarios, stack, base, "llm")
        comparison.append(row_of("llm (one call per chunk)", untuned, summary))
        llm_note = "run with the configured LLM"
    else:
        comparison.append({"label": "llm (one call per chunk)", "g2": llm_note})

    cols = ["label", "stability_threshold", "stable_chunks", "probe_k", "probe_min_content_terms", "probe_min_tokens",
            "g2", "early",
            "eligible", "missed", "false_trigger", "presentation_retrieved", "gain", "retrievals", "first_recall",
            "engine_ms"]
    order = lambda r: (-r["g2"], r["false_trigger"], r["retrievals"])  # noqa: E731
    text = (
        "# Phase 2 controller ablation\n\n"
        "Generated by `python eval/ablate_controller.py` on the 30 dev scenarios (40 turns, 34 eligible, "
        "6 no-retrieval of which 5 presentation-only). Simulated clock; chunks every 0.8 s. "
        "`gain` = mean seconds gained vs the baseline over eligible turns; `retrievals` = mean retrievals "
        "per eligible turn; `first_recall` = recall@5 of the first retrieval against gold sections; "
        "`engine_ms` = mean engine wall time per turn (CPU, simulated clock).\n\n"
        "## Controller comparison\n\n" + fmt(comparison, cols) + "\n\n"
        f"LLM controller: {llm_note}.\n\n"
        "## Selected thresholds\n\n"
        + "\n".join(f"- {mode}: " + ", ".join(f"`{k}` = {v}" for k, v in p.items()) for mode, p in tuned_params.items())
        + f"\n- recommended default mode (higher G2 under the constraints): `{recommended}`\n\n"
        "Selection rule (fixed in advance): zero false triggers and zero presentation retrievals, then max G2, "
        "then fewest retrievals per eligible turn, then highest first-retrieval recall, then the most "
        "conservative setting. Tuned on draft dev scenarios, so expect lower numbers on held-out data.\n\n"
        f"## rule_only grid ({len(rule_rows)} configurations)\n\n" + fmt(sorted(rule_rows, key=order), cols) + "\n\n"
        f"## rule_stability grid ({len(grid_rows)} configurations)\n\n" + fmt(sorted(grid_rows, key=order), cols) + "\n"
    )
    print(fmt(comparison, cols))
    print(f"\nselected: {tuned_params}\nrecommended default mode: {recommended}")
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args.out))


if __name__ == "__main__":
    sys.exit(main())
