"""G4 / G5 metrics (step 4.8)."""

import subprocess
import sys
from pathlib import Path

from eval.full_eval import claim_supported, evaluate_full, literally_supported, shuffled_control
from eval.scenarios import load_scenarios
from src.config import Settings
from src.corpus.chunker import load_chunks
from src.schemas import Claim
from src.session.pipeline import SessionPipeline
from src.synthesis.citations import CitationIndex
from src.telemetry.logger import TelemetryLogger

ROOT = Path(__file__).resolve().parents[1]


def test_literal_support_normalises_markdown_and_table_rows():
    chunk = "| Gate | Criterion |\n|---|---|\n| G1 | Reproducibility |\n\n**Bold** text with *italic* (`code`)."
    assert literally_supported("Gate: G1; Criterion: Reproducibility", chunk)
    assert literally_supported("Bold text with italic (code).", chunk)
    assert not literally_supported("Gate: G9", chunk)


def test_claim_check_and_shuffled_control_on_real_corpus():
    index = CitationIndex(load_chunks())
    good = Claim(id="a", text="Memory is strictly ephemeral and scoped to the active conversation session.",
                 subintent_id="s", chunk_ids=["Doc_01 §3#1"])
    assert claim_supported(good, index)
    assert not claim_supported(good.model_copy(update={"chunk_ids": ["Doc_01 §7#1"]}), index)
    assert not claim_supported(good.model_copy(update={"chunk_ids": ["Doc_999 §1#1"]}), index)
    assert shuffled_control([good], index) == 0.0


async def test_evaluate_full_on_subset(tmp_path):
    subset = [s for s in load_scenarios() if s.id in {"late_02", "present_04", "noisy_03"}]
    pipeline = SessionPipeline(Settings(index_dir=tmp_path / "idx", log_dir=tmp_path, llm_provider="none"),
                               telemetry=TelemetryLogger(None))
    out = await evaluate_full(subset, pipeline)
    s = out["summary"]
    assert s["g4"]["fabricated_ids"] == [] and s["g4"]["claims"] > 0
    assert s["g5"]["refinement_turns"] == 1 and s["g5"]["pass"]
    assert s["presentation"] == {"turns": 1, "zero_retrieval_and_no_new_ids": 1, "retrieval_calls": 0}
    assert len(pipeline.store) == 0  # every scenario's session was ended


def test_cli_full_rejects_other_metrics():
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--system", "full", "--metrics", "g3"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "not implemented" in r.stderr
