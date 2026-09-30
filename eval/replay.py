"""Replay runner (step 5.2): the full replay suite, non-interactively, with gates G1-G6.

    python eval/replay.py [--out results/replay] [--scenario ID ...] [--clock simulated|wall] [--show]

Runs, over every dev scenario (one ephemeral session per scenario):
* the full streaming system (SessionPipeline: controller, decomposition, fusion, delta
  refinement, grounded synthesis) and
* the non-streaming baseline,
with the configured profile (PRISM_* settings; the intended profile uses a local 7-8B Ollama
model, the offline profile uses PRISM_LLM_PROVIDER=none). A configured LLM is health-checked
first; if it is unreachable or not pulled the run stops with an error instead of silently
falling back.

Writes to --out: `telemetry.jsonl` (this run only), `results.json` (gates, benchmark tables,
per-turn rows) and `summary.md` (G1-G6 table + streaming-vs-baseline table). Exit code 0 when
no gate fails; G1 is "not verified" unless the run happens inside the container
(PRISM_IN_CONTAINER=1, set by the Dockerfile).

Gates (thresholds from the problem statement, [Doc_01 §5]):
G1 reproducibility (container + single command + replay completes)   G2 early retrieval >= 0.80
G3 multi-intent identification >= 0.70                                G4 citation support >= 0.85, 0 fabricated
G5 session refinement continuity (all refinement turns)                G6 trace coverage = 100%
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.full_eval import evaluate_full  # noqa: E402
from eval.run_eval import evaluate_baseline  # noqa: E402
from eval.scenarios import load_scenarios  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm.client import LLMError, arequire_llm  # noqa: E402
from src.retrieval.factory import build_retrieval  # noqa: E402
from src.session.pipeline import SessionPipeline  # noqa: E402
from src.telemetry.coverage import check_file, load_events  # noqa: E402
from src.telemetry.logger import TelemetryLogger  # noqa: E402

TARGETS = {"G2": 0.80, "G3": 0.70, "G4": 0.85, "G6": 1.0}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - git is optional inside the container
        return None


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def _pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return round(values[min(len(values) - 1, max(0, round(q * (len(values) - 1))))], 3)


def latency_from_telemetry(events: list[dict]) -> dict:
    """Post-utterance latency (ms from the end of speech to the answer) per system."""
    by_req: dict[str, list[dict]] = {}
    for e in events:
        by_req.setdefault(e["request_id"], []).append(e)
    streaming, baseline = [], []
    for rid, evs in by_req.items():
        names = {e["event"]: e for e in evs}
        if rid.startswith("baseline-") and "request_completed" in names:
            baseline.append(names["request_completed"].get("stage_latency_ms"))
        elif "utterance_end" in names and "answer_emitted" in names:
            streaming.append((names["answer_emitted"]["wall_time"] - names["utterance_end"]["wall_time"]) * 1000)
    return {"streaming_post_utterance_ms": {"p50": _pct(streaming, .5), "p95": _pct(streaming, .95), "mean": _mean(streaming)},
            "baseline_ms": {"p50": _pct(baseline, .5), "p95": _pct(baseline, .95), "mean": _mean(baseline)}}


def tokens_from_telemetry(events: list[dict]) -> dict:
    """Total LLM tokens and estimated cost per system, from each request's answer_emitted event."""
    out = {"streaming": [0, 0, 0.0], "baseline": [0, 0, 0.0]}
    for e in events:
        if e["event"] == "answer_emitted":
            key = "baseline" if e["request_id"].startswith("baseline-") else "streaming"
            out[key][0] += e.get("tokens_in") or 0
            out[key][1] += e.get("tokens_out") or 0
            out[key][2] += e.get("est_cost_usd") or 0.0
    return out


def citation_accuracy(rows: list[dict], cited_key: str) -> tuple[float | None, float | None]:
    """Turn-level (recall, precision) of cited sections vs gold, over answerable turns.
    recall: gold supporting sections cited; precision: cited sections that are gold or also-relevant."""
    rec, prec = [], []
    for r in rows:
        if not r.get("retrieval_required") or not r.get("gold_supporting"):
            continue
        cited, gold = set(r.get(cited_key) or []), set(r["gold_supporting"])
        rec.append(len(cited & gold) / len(gold))
        if cited:
            prec.append(len(cited & (gold | set(r.get("also_relevant") or []))) / len(cited))
    return _mean(rec), _mean(prec)


def compute_gates(full: dict, coverage, in_container: bool, completed: bool) -> dict:
    rows = full["turns"]
    eligible = [r for r in rows if r["retrieval_required"]]
    no_retr = [r for r in rows if not r["retrieval_required"]]
    early = sum(r["early"] for r in eligible)
    g2 = early / len(eligible) if eligible else None
    compound = [r for r in rows if r["retrieval_required"] and r["gold_n"] >= 2]
    g3 = sum(r["g3_correct"] >= 2 for r in compound) / len(compound) if compound else None
    g4, g5 = full["summary"]["g4"], full["summary"]["g5"]
    false_triggers = sum(r["retrieved"] for r in no_retr)
    gates = {
        "G1": {"criterion": "Reproducibility", "target": "Pass/Fail",
               "value": "container run completed" if in_container and completed else "CLI runner completed",
               "pass": True if (in_container and completed) else None,
               "evidence": "replay completed inside the container (PRISM_IN_CONTAINER=1)" if in_container else
               "single-command CLI replay completed; the container itself was not exercised (requires Docker)"},
        "G2": {"criterion": "Early retrieval", "target": ">= 0.80 of eligible queries", "value": round(g2, 4) if g2 is not None else None,
               "pass": g2 is not None and g2 >= TARGETS["G2"],
               "evidence": f"{early}/{len(eligible)} eligible turns retrieved before utterance end; "
                           f"false triggers on no-retrieval turns {false_triggers}/{len(no_retr)}"},
        "G3": {"criterion": "Multi-intent identification", "target": ">= 0.70 of compound queries",
               "value": round(g3, 4) if g3 is not None else None, "pass": g3 is not None and g3 >= TARGETS["G3"],
               "evidence": f"{sum(r['g3_correct'] >= 2 for r in compound)}/{len(compound)} compound turns with >= 2 correct sub-intents"},
        "G4": {"criterion": "Factual grounding", "target": ">= 0.85 citation support, 0 fabricated ids",
               "value": g4["support_rate"], "pass": bool(g4["pass"]),
               "evidence": f"{g4['supported']}/{g4['claims']} claims literally supported by their cited chunk; "
                           f"fabricated ids {len(g4['fabricated_ids'])}; shuffled-citation control {g4['shuffled_control_support']}"},
        "G5": {"criterion": "Session refinement", "target": "verified state continuity",
               "value": f"{g5['passed']}/{g5['refinement_turns']}", "pass": bool(g5["pass"]),
               "evidence": f"refinement kinds {g5['kinds']}; presentation turns with zero retrieval "
                           f"{full['summary']['presentation']['zero_retrieval_and_no_new_ids']}/{full['summary']['presentation']['turns']}"},
        "G6": {"criterion": "Telemetry & observability", "target": "100% trace coverage",
               "value": round(coverage.coverage, 4), "pass": coverage.ok,
               "evidence": f"{coverage.complete}/{coverage.requests} requests with a complete trace"},
    }
    return gates


def summary_markdown(res: dict) -> str:
    g = res["gates"]
    status = lambda p: "PASS" if p is True else ("FAIL" if p is False else "NOT VERIFIED")  # noqa: E731
    lines = [f"# Replay summary ({res['run']['profile']} profile)", "",
             f"Run {res['run']['started']} · commit {res['run']['commit']} · LLM {res['run']['llm']['summary']}", "",
             "| Gate | Criterion | Target | Result | Status | Evidence |", "|---|---|---|---|---|---|"]
    for k in ("G1", "G2", "G3", "G4", "G5", "G6"):
        lines.append(f"| {k} | {g[k]['criterion']} | {g[k]['target']} | {g[k]['value']} | {status(g[k]['pass'])} | {g[k]['evidence']} |")
    b = res["benchmark"]
    lines += ["", "## Streaming system vs baseline", "", "| metric | streaming (full) | baseline |", "|---|---|---|"]
    for name, (sv, bv) in b.items():
        lines.append(f"| {name} | {sv} | {bv} |")
    return "\n".join(lines) + "\n"


async def run(out: Path, scenario_ids: list[str] | None = None, clock: str = "simulated", show: bool = False) -> dict:
    settings = get_settings()
    health = await arequire_llm(settings)  # raises LLMError when a configured model is unavailable
    scenarios = load_scenarios()
    if scenario_ids:
        scenarios = [s for s in scenarios if s.id in set(scenario_ids)]
        if not scenarios:
            raise SystemExit(f"no scenario matches {scenario_ids}")
    out.mkdir(parents=True, exist_ok=True)
    tel_path = out / "telemetry.jsonl"
    tel_path.unlink(missing_ok=True)

    def echo(e):
        if show and e.event in ("controller_decision", "retrieval_started", "decomposition", "cache_hit",
                                "answer_emitted", "utterance_end"):
            detail = {k: e.data.get(k) for k in ("stream_ts", "action", "reason", "query", "answer_version", "kind",
                                                  "citations") if k in e.data}
            print(f"  [{e.request_id}] {e.event} {e.trigger or ''} {json.dumps(detail, ensure_ascii=False)}", flush=True)

    telemetry = TelemetryLogger(tel_path, echo=echo)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.perf_counter()
    stack = build_retrieval(settings)
    pipeline = SessionPipeline(settings, stack=stack, telemetry=telemetry)

    def on_turn(row):
        if show:
            print(f"== {row['scenario']} t{row['turn']} -> v{row['version']} {row['kind']}\n{row['answer'] or '(no answer)'}"
                  f"\n   uncertainty: {row['uncertainty']}\n", flush=True)

    full = await evaluate_full(scenarios, pipeline, clock=clock, on_turn=on_turn)
    base = await evaluate_baseline(scenarios, settings=settings, telemetry=telemetry, stack=stack)
    wall_s = time.perf_counter() - t0
    coverage = check_file(tel_path)
    events = load_events(tel_path)
    in_container = os.environ.get("PRISM_IN_CONTAINER") == "1"
    gates = compute_gates(full, coverage, in_container, completed=True)

    rows, bs = full["turns"], base["summary"]
    lat = latency_from_telemetry(events)
    tok = tokens_from_telemetry(events)
    eligible = [r for r in rows if r["retrieval_required"]]
    gained = [(r["utterance_end_s"] - r["first_retrieval_s"]) if r["first_retrieval_s"] is not None else None for r in eligible]
    stage_keys = sorted({k for r in rows for k in r["stage_ms"]})
    s_rec, s_prec = citation_accuracy(rows, "turn_citations")
    b_rec, b_prec = citation_accuracy(base["turns"], "cited")
    benchmark = {
        "retrieval starts before utterance end (eligible turns)": (gates["G2"]["value"], 0.0),
        "mean seconds of retrieval head start vs utterance end": (_mean([g or 0.0 for g in gained]), 0.0),
        "post-utterance latency p50 / p95 ms": (f"{lat['streaming_post_utterance_ms']['p50']} / {lat['streaming_post_utterance_ms']['p95']}",
                                                 f"{lat['baseline_ms']['p50']} / {lat['baseline_ms']['p95']}"),
        "no-retrieval turns (presentation-only / incomplete) that searched the corpus":
            (f"{sum(r['retrieval_calls'] > 0 for r in rows if not r['retrieval_required'])}/{bs['no_retrieval_turns']['n']}",
             f"{bs['no_retrieval_turns']['retrieved_anyway']}/{bs['no_retrieval_turns']['n']}"),
        "answer cites the gold sections: recall / precision (answerable turns)": (f"{s_rec} / {s_prec}", f"{b_rec} / {b_prec}"),
        "fabricated citation ids": (len(full["summary"]["g4"]["fabricated_ids"]), bs["citation_validity"]["fabricated"]),
        "claims literally supported by cited chunk": (full["summary"]["g4"]["support_rate"], "n/a (not computed for baseline)"),
        "uncertainty on no-evidence turns": (f"{full['summary']['uncertainty']['emitted']}/{full['summary']['uncertainty']['expected_turns']}",
                                             f"{bs['uncertainty']['emitted_when_expected']}/{bs['uncertainty']['expected_turns']}"),
        "false uncertainty on answerable turns": (f"{full['summary']['uncertainty']['false_uncertainty']}/{full['summary']['uncertainty']['answerable_turns']}",
                                                  f"{bs['uncertainty']['false_uncertainty_on_answerable']}/{bs['uncertainty']['answerable_turns']}"),
        "late-constraint turns refined in place (answer version bumped, state kept)": (gates["G5"]["value"], "0 (stateless)"),
        "LLM tokens in / out (total)": (f"{tok['streaming'][0]} / {tok['streaming'][1]}",
                                        f"{tok['baseline'][0]} / {tok['baseline'][1]}"),
        "estimated cost USD (total)": (round(tok["streaming"][2], 6), round(tok["baseline"][2], 6)),
    }
    llm_summary = "none (offline extractive profile)" if health is None else (
        f"{health.provider}:{health.model} size={health.parameter_size_b}B intended_7_8b_class={health.in_intended_class}")
    res = {
        "run": {"started": started, "commit": _git_commit(), "wall_seconds": round(wall_s, 2),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "python": platform.python_version(), "in_container": in_container, "clock": clock,
                "profile": "offline" if health is None else "llm",
                "scenarios": len(scenarios), "turns": len(rows),
                "llm": {"summary": llm_summary, "health": None if health is None else health.__dict__},
                "settings": {k: getattr(settings, k) for k in ("llm_provider", "llm_model", "decomposer", "grounding_judge",
                                                               "controller_mode", "dense_backend", "rerank_backend")}},
        "gates": gates,
        "benchmark": benchmark,
        "stage_ms_mean": {k: _mean([r["stage_ms"].get(k) for r in rows]) for k in stage_keys},
        "full_summary": full["summary"], "baseline_summary": bs,
        "coverage_missing": coverage.missing,
        "turns": rows,
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(res), encoding="utf-8")
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "replay")
    parser.add_argument("--scenario", action="append", help="run only these scenario ids (repeatable)")
    parser.add_argument("--clock", choices=["simulated", "wall"], default="simulated")
    parser.add_argument("--show", action="store_true", help="print live telemetry and each answer (demo)")
    args = parser.parse_args()
    try:
        res = asyncio.run(run(args.out, args.scenario, args.clock, args.show))
    except LLMError as exc:
        print(f"LLM not available: {exc}", file=sys.stderr)
        return 2
    print(summary_markdown(res))
    print(f"wrote {args.out / 'results.json'} and {args.out / 'summary.md'}")
    return 1 if any(g["pass"] is False for g in res["gates"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
