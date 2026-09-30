"""Controller metrics (step 2.6)."""

import subprocess
import sys
from pathlib import Path

from eval.controller_eval import evaluate_controller, summarize
from eval.scenarios import load_scenarios
from src.config import Settings
from src.telemetry.logger import TelemetryLogger

ROOT = Path(__file__).resolve().parents[1]


def _row(req, early, retrieved, first, end, cat="single_intent", triggers=("provisional",)):
    return {"retrieval_required": req, "early": early, "retrieved": retrieved, "first_retrieval_s": first,
            "utterance_end_s": end, "seconds_gained": None if first is None else end - first, "category": cat,
            "triggers": list(triggers) if retrieved else [], "first_recall_at_5": None,
            "final_recall_at_5": None, "engine_ms": 1.0}


def test_summarize_math():
    rows = [
        _row(True, True, True, 0.8, 2.0),
        _row(True, False, True, 2.0, 2.0, triggers=("final",)),
        _row(True, False, False, None, 1.0, triggers=()),
        _row(False, False, False, None, 1.0, cat="presentation_only"),
        _row(False, False, True, 0.5, 1.0, cat="noisy_fragment"),
    ]
    s = summarize(rows, "x")
    assert s["g2_early_retrieval"]["rate"] == round(1 / 3, 4) and not s["g2_early_retrieval"]["pass"]
    assert s["g2_early_retrieval"]["missed_entirely"] == 1
    assert s["false_trigger"]["rate"] == 0.5 and s["false_trigger"]["presentation_retrieved"] == 0
    assert s["seconds_gained_vs_baseline"]["mean_over_eligible"] == 0.4
    assert s["seconds_gained_vs_baseline"]["mean_over_early"] == 1.2


async def test_evaluate_controller_on_subset(tmp_path):
    subset = [s for s in load_scenarios() if s.id in {"single_02", "present_04"}]
    out = await evaluate_controller(subset, Settings(index_dir=tmp_path / "idx"), telemetry=TelemetryLogger(None))
    s = out["summary"]
    assert s["turns"] == 3 and s["false_trigger"]["presentation_retrieved"] == 0
    row = next(r for r in out["turns"] if r["scenario"] == "present_04" and r["turn"] == 2)
    assert row["suppressed_reason"] == "presentation_restructure" and not row["retrieved"]


def test_cli_rejects_wrong_metric():
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--system", "controller_only", "--metrics", "g3"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "not implemented" in r.stderr
