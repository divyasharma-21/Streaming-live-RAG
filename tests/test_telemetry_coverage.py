"""Telemetry completion and the G6 coverage checker (step 5.1)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.baseline import BaselinePipeline
from src.config import Settings
from src.retrieval.factory import build_retrieval
from src.session.pipeline import SessionPipeline
from src.stream.simulator import chunks_from_text
from src.telemetry.coverage import check_events, check_file, check_request, load_events
from src.telemetry.logger import TelemetryLogger

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


async def test_full_pipeline_session_is_fully_covered(stack, tmp_path):
    log = tmp_path / "t.jsonl"
    p = SessionPipeline(Settings(log_dir=tmp_path, llm_provider="none"), stack=stack, telemetry=TelemetryLogger(log))
    await p.handle_turn("s", chunks_from_text(["What are the hard engineering rules?"]))
    await p.handle_turn("s", chunks_from_text(["Only the grounding rule,", "and which gate measures it?"]))
    await p.handle_turn("s", chunks_from_text(["Put that as bullet points."]))
    await p.handle_turn("s", chunks_from_text(["so um", "can you"]))
    report = check_file(log)
    assert report.requests == 4 and report.ok, report.missing

    events = load_events(log)
    answers = [e for e in events if e["event"] == "answer_emitted"]
    assert [(a["data"]["answer_version_from"], a["data"]["answer_version"]) for a in answers] == [
        (0, 1), (1, 2), (2, 2), (2, 2)]
    first = answers[0]
    assert first["tokens_in"] == 0 and first["est_cost_usd"] == 0.0  # no LLM: counted as zero, not missing
    assert first["data"]["sources"] and all(first["data"]["sources"].values())
    assert {"controller", "planning", "retrieval", "fusion", "synthesis", "verification"} <= set(first["data"]["stage_ms"])
    # engine and pipeline events of one turn share one request id
    assert len({e["request_id"] for e in events}) == 4


async def test_baseline_requests_are_covered(stack, tmp_path):
    log = tmp_path / "b.jsonl"
    b = BaselinePipeline(Settings(log_dir=tmp_path, llm_provider="none"), stack=stack, telemetry=TelemetryLogger(log))
    await b.answer("What are the technical evaluation gates?")
    assert check_file(log).ok


def ev(name, rid="r", **data):
    base = {"event": name, "request_id": rid, "data": data}
    if name == "answer_emitted":
        base.update(tokens_in=0, tokens_out=0, est_cost_usd=0.0)
    return base


def complete_streaming(rid="r"):
    return [ev("request_started", rid, system="streaming", expects_answer=True), ev("controller_decision", rid),
            ev("utterance_end", rid), ev("retrieval_started", rid), ev("retrieval_completed", rid),
            ev("request_completed", rid), ev("citation_check", rid),
            ev("answer_emitted", rid, answer_version_from=0, answer_version=1, sub_queries=[], sources={}, stage_ms={})]


def test_checker_accepts_complete_and_flags_each_gap():
    assert check_request(complete_streaming()) == []
    assert "answer_emitted" in check_request([e for e in complete_streaming() if e["event"] != "answer_emitted"])
    unmatched = complete_streaming() + [ev("retrieval_started")]
    assert any(m.startswith("retrieval_completed") for m in check_request(unmatched))
    no_cost = complete_streaming()
    no_cost[-1]["est_cost_usd"] = None
    assert "answer_emitted.est_cost_usd" in check_request(no_cost)
    no_version = complete_streaming()
    del no_version[-1]["data"]["answer_version_from"]
    assert "answer_emitted.answer_version_from" in check_request(no_version)
    bad_llm = complete_streaming() + [{"event": "llm_call", "request_id": "r", "data": {}, "tokens_in": 3}]
    assert "llm_call token counts / cost" in check_request(bad_llm)


def test_engine_only_requests_need_no_answer_events():
    evs = [ev("request_started", system="streaming", expects_answer=False), ev("controller_decision"),
           ev("utterance_end"), ev("request_completed")]
    assert check_request(evs) == []


def test_report_and_cli(tmp_path):
    events = complete_streaming("a") + complete_streaming("b")[:-1]
    report = check_events(events)
    assert (report.requests, report.complete, list(report.missing)) == (2, 1, ["b"]) and not report.ok
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    r = subprocess.run([sys.executable, "-m", "src.telemetry.coverage", str(path)], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == 1 and "1/2" in r.stdout
