"""Replay runner and gates G1-G6 (step 5.2)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.replay import latency_from_telemetry, run, summary_markdown, tokens_from_telemetry
from src.llm.client import LLMError
from tests.fake_ollama import FakeOllama

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ["late_02", "multi_03", "present_04", "noevid_02"]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISM_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("PRISM_LOG_DIR", str(tmp_path / "logs"))
    for k in ("PRISM_LLM_PROVIDER", "PRISM_LLM_MODEL", "PRISM_DECOMPOSER", "PRISM_GROUNDING_JUDGE", "PRISM_IN_CONTAINER"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


async def test_offline_replay_writes_results_and_gates(env, tmp_path):
    res = await run(tmp_path / "out", SUBSET)
    gates = res["gates"]
    assert set(gates) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert gates["G1"]["pass"] is None and "requires Docker" in gates["G1"]["evidence"]
    assert gates["G6"]["pass"] and gates["G6"]["value"] == 1.0  # every request fully traced
    assert gates["G4"]["pass"] and gates["G5"]["pass"]
    assert res["run"]["profile"] == "offline" and res["run"]["turns"] == 6
    on_disk = json.loads((tmp_path / "out" / "results.json").read_text())
    assert on_disk["gates"]["G6"]["value"] == 1.0
    assert "| G6 | Telemetry" in (tmp_path / "out" / "summary.md").read_text()


async def test_g1_passes_only_inside_the_container(env, tmp_path):
    env.setenv("PRISM_IN_CONTAINER", "1")
    res = await run(tmp_path / "out", ["single_02"])
    assert res["gates"]["G1"]["pass"] is True and res["run"]["in_container"]


async def test_llm_profile_replay_over_fake_ollama(env, tmp_path):
    """Wiring check of the intended profile (Ollama + LLM decomposer + LLM judge) against the
    TEST-ONLY fake server. It proves the LLM path runs end to end; it says nothing about quality."""
    with FakeOllama("llama3.1:8b", "8.0B") as fake:
        env.setenv("PRISM_LLM_PROVIDER", "ollama")
        env.setenv("PRISM_LLM_MODEL", "llama3.1:8b")
        env.setenv("PRISM_LLM_BASE_URL", fake.url)
        env.setenv("PRISM_DECOMPOSER", "llm")
        env.setenv("PRISM_GROUNDING_JUDGE", "llm")
        res = await run(tmp_path / "out", SUBSET)
    assert res["run"]["profile"] == "llm"
    assert res["run"]["llm"]["health"]["in_intended_class"] is True
    assert res["gates"]["G6"]["pass"] and res["full_summary"]["g4"]["fabricated_ids"] == []
    tokens_in, _ = map(int, res["benchmark"]["LLM tokens in / out (total)"][0].split(" / "))
    assert tokens_in > 0


async def test_configured_but_unreachable_llm_stops_the_run(env, tmp_path):
    env.setenv("PRISM_LLM_PROVIDER", "ollama")
    env.setenv("PRISM_LLM_MODEL", "llama3.1:8b")
    env.setenv("PRISM_LLM_BASE_URL", "http://127.0.0.1:9")
    with pytest.raises(LLMError):
        await run(tmp_path / "out", ["single_02"])


def test_telemetry_helpers():
    events = [
        {"event": "utterance_end", "request_id": "a", "wall_time": 10.0},
        {"event": "answer_emitted", "request_id": "a", "wall_time": 10.02, "tokens_in": 5, "tokens_out": 2,
         "est_cost_usd": 0.0},
        {"event": "request_completed", "request_id": "baseline-a", "stage_latency_ms": 3.0},
        {"event": "answer_emitted", "request_id": "baseline-a", "wall_time": 1.0, "tokens_in": 1, "tokens_out": 1,
         "est_cost_usd": 0.0},
    ]
    lat = latency_from_telemetry(events)
    assert lat["streaming_post_utterance_ms"]["p50"] == pytest.approx(20.0, abs=0.01)
    assert lat["baseline_ms"]["p50"] == 3.0
    assert tokens_from_telemetry(events) == {"streaming": [5, 2, 0.0], "baseline": [1, 1, 0.0]}


def test_run_eval_all_cli(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PRISM_")}
    env["PRISM_INDEX_DIR"] = str(tmp_path / "idx")
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--system", "full", "--metrics", "all",
                        "--out", str(tmp_path / "all")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "| G2 | Early retrieval" in r.stdout and (tmp_path / "all" / "results.json").exists()


def test_summary_markdown_marks_unverified():
    res = {"run": {"profile": "offline", "started": "t", "commit": "c", "llm": {"summary": "none"}},
           "gates": {k: {"criterion": k, "target": "x", "value": 1, "pass": True, "evidence": "e"}
                     for k in ("G2", "G3", "G4", "G5", "G6")} | {"G1": {"criterion": "R", "target": "x", "value": "v",
                                                                    "pass": None, "evidence": "e"}},
           "benchmark": {"m": (1, 2)}}
    md = summary_markdown(res)
    assert "NOT VERIFIED" in md and md.count("PASS") == 5
