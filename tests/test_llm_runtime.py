"""Ollama runtime integration against a local FAKE Ollama server (tests/fake_ollama.py).

These tests exercise the real HTTP client, health check, retries and every LLM code path of
the full pipeline. They do not use or evaluate a real 7-8B model."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import Settings
from src.llm.client import LLMConnectionError, LLMError, OllamaClient, check_llm, require_llm
from src.retrieval.factory import build_retrieval
from src.session.pipeline import SessionPipeline
from src.stream.simulator import chunks_from_text
from src.synthesis.citations import CitationIndex
from src.telemetry.coverage import check_file, load_events
from src.telemetry.logger import TelemetryLogger
from tests.fake_ollama import FakeOllama

ROOT = Path(__file__).resolve().parents[1]
MODEL = "llama3.1:8b"


def llm_settings(url, **kw):
    base = dict(llm_provider="ollama", llm_model=MODEL, llm_base_url=url, llm_timeout_s=10, llm_retries=2)
    base.update(kw)
    return Settings(**base)


def test_health_reports_intended_7_8b_class():
    with FakeOllama(MODEL, "8.0B") as fake:
        h = check_llm(llm_settings(fake.url))
    assert h.ok and h.parameter_size_b == 8.0 and h.in_intended_class is True and h.quantization == "Q4_K_M"


def test_health_flags_a_small_model_instead_of_accepting_it():
    with FakeOllama("tiny:1b", "1.5B") as fake:
        h = check_llm(llm_settings(fake.url, llm_model="tiny:1b"))
    assert h.ok and h.in_intended_class is False and "outside the intended" in h.message


def test_health_missing_model_and_unreachable_server():
    with FakeOllama("other:8b") as fake:
        h = check_llm(llm_settings(fake.url))
    assert not h.ok and h.reachable and "ollama pull llama3.1:8b" in h.message
    h = check_llm(llm_settings("http://127.0.0.1:9"))
    assert not h.reachable and "not reachable" in h.message
    with pytest.raises(LLMError):
        require_llm(llm_settings("http://127.0.0.1:9"))
    assert require_llm(Settings(llm_provider="none")) is None


async def test_generate_sends_context_window_keep_alive_and_retries_transient_errors():
    with FakeOllama(MODEL, fail_first=2) as fake:
        client = OllamaClient(MODEL, fake.url, settings=llm_settings(fake.url, llm_num_ctx=8192, llm_keep_alive="3m"))
        resp = await client.generate("hello")
        body = fake.requests[-1][1]
    assert resp.text == "fake completion" and resp.tokens_in == 1 and not resp.tokens_estimated
    assert body["options"]["num_ctx"] == 8192 and body["keep_alive"] == "3m" and body["options"]["temperature"] == 0


async def test_retries_are_bounded():
    with FakeOllama(MODEL, fail_first=5) as fake:
        client = OllamaClient(MODEL, fake.url, settings=llm_settings(fake.url, llm_retries=1))
        with pytest.raises(LLMConnectionError):
            await client.generate("hello")


def test_check_cli(tmp_path):
    with FakeOllama(MODEL) as fake:
        env = {"PRISM_LLM_PROVIDER": "ollama", "PRISM_LLM_MODEL": MODEL, "PRISM_LLM_BASE_URL": fake.url,
               "PATH": "/usr/bin:/bin"}
        ok = subprocess.run([sys.executable, "-m", "src.llm.client", "--check"], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert ok.returncode == 0 and "intended_7_8b_class=True" in ok.stdout
    env["PRISM_LLM_BASE_URL"] = "http://127.0.0.1:9"
    bad = subprocess.run([sys.executable, "-m", "src.llm.client", "--check"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "not reachable" in bad.stdout


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


async def test_full_pipeline_runs_every_llm_path_over_http(stack, tmp_path):
    """LLM synthesis + LLM decomposer + LLM judge + LLM presentation, end to end via HTTP."""
    log = tmp_path / "t.jsonl"
    with FakeOllama(MODEL) as fake:
        s = llm_settings(fake.url, log_dir=tmp_path, decomposer="llm", grounding_judge="llm")
        p = SessionPipeline(s, stack=stack, telemetry=TelemetryLogger(log))
        first = await p.handle_turn("s", chunks_from_text(["What does the retrieval controller decide,",
                                                            "and which gate measures early retrieval?"]))
        refine = await p.handle_turn("s", chunks_from_text(["Only the provisional retrieval part."]))
        shown = await p.handle_turn("s", chunks_from_text(["Translate that into French please."]))
        schemas = [set((b.get("format") or {}).get("properties", {})) for path, b in fake.requests
                   if path == "/api/generate"]
    assert first.record.answer and first.invalid_citations == [] and first.grounding.fabricated_ids == []
    index = CitationIndex(stack.chunks)
    assert all(index.is_valid(c) for c in first.record.citations)
    assert refine.answer_version == 2 and shown.kind == "presentation_only" and shown.retrieval_calls == 0
    assert set(shown.record.citations) <= set(refine.record.citations)
    assert {"ops"} in schemas and {"claims", "uncertain"} in schemas and {"supported", "reason"} in schemas
    assert {"text"} in schemas  # translation went through the LLM
    assert first.tokens_in > 0 and first.tokens_out > 0
    events = load_events(log)
    llm_calls = [e for e in events if e["event"] == "llm_call"]
    assert llm_calls and all(e["tokens_in"] is not None and e["data"]["provider"] == "ollama" for e in llm_calls)
    assert check_file(log).ok


def test_fake_server_is_test_only():
    """Application code never imports the fake server (or anything under tests/)."""
    import ast

    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        mods += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not [m for m in mods if m.startswith("tests") or "fake_ollama" in m], path
