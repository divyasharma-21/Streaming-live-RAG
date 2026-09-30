"""G3 metric and decomposition evaluation (step 3.7)."""

import subprocess
import sys
from pathlib import Path

from eval.decompose_eval import build_engine, evaluate_decomposition, match_subintents, summarize
from eval.scenarios import GoldSubIntent, load_scenarios
from src.config import Settings
from src.schemas import SubQuery
from src.telemetry.logger import TelemetryLogger

ROOT = Path(__file__).resolve().parents[1]


def g(i, text):
    return GoldSubIntent(id=i, text=text, supporting=[])


def test_matching_is_one_to_one_and_thresholded():
    gold = [g("s1", "reranker score calibration"), g("s2", "cache key design")]
    pred = [SubQuery(id="q1", text="how are reranker scores calibrated and cache keys designed")]
    m = match_subintents(gold, pred)
    assert len(m) == 1  # one sub-query cannot satisfy two gold intents
    pred2 = [SubQuery(id="q1", text="reranker score calibration"), SubQuery(id="q2", text="cache key design")]
    assert match_subintents(gold, pred2) == {"s1": ("q1", 1.0), "s2": ("q2", 1.0)}


def test_constraints_do_not_count_toward_matching():
    gold = [g("s1", "refund terms Chennai")]
    pred = [SubQuery(id="q1", text="the policy", constraints=["refund terms Chennai"])]
    assert match_subintents(gold, pred) == {}


def _row(gold_n, correct, predicted_n, req=True):
    return {"retrieval_required": req, "gold_n": gold_n, "correct": correct,
            "predicted": [{"text": "x"}] * predicted_n, "intents": [], "retrievals": 1, "cache_hits": 0,
            "early": True, "scenario": "s", "turn": 1}


def test_summary_math():
    rows = [_row(2, 2, 2), _row(2, 1, 2), _row(3, 2, 3), _row(1, 1, 2), _row(1, 1, 1), _row(0, 0, 0, req=False)]
    s = summarize(rows)
    assert s["g3"]["rate"] == round(2 / 3, 4) and s["g3"]["pass"] is False
    assert s["over_split"]["over_split"] == 1 and s["over_split"]["rate"] == 0.5


async def test_evaluate_on_subset(tmp_path):
    subset = [sc for sc in load_scenarios() if sc.id in {"single_05", "present_04"}]
    engine = build_engine(Settings(index_dir=tmp_path / "idx"), telemetry=TelemetryLogger(None))
    out = await evaluate_decomposition(subset, engine)
    assert out["summary"]["over_split"]["over_split"] == 0
    assert out["summary"]["no_retrieval_turns_retrieved"] == 0


def test_cli_g3():
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--system", "decompose_fusion", "--metrics", "g2"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "not implemented" in r.stderr
