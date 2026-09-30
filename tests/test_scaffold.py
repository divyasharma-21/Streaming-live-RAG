"""Repository layout and the eval/ isolation rule (playbook standing rule 7)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "Dockerfile", "docker-compose.yml", "Makefile", "README.md", "pyproject.toml",
    "requirements.txt", ".gitignore", ".env.example",
    "uv.lock", "requirements-dev.txt", "requirements-optional.txt", "Dockerfile", ".dockerignore",
    "data/corpus", "data/corpus/MANIFEST.json", "data/PROVENANCE.md",
    "eval/dev_scenarios", "eval/run_eval.py", "eval/scenarios.py",
    "reports/PHASE_1_REPORT.md", "logs", "schemas/output_record.schema.json", "scripts",
    "src/config.py", "src/schemas.py", "src/llm/client.py",
    "src/corpus/loader.py", "src/corpus/chunker.py", "src/corpus/audit.py", "src/corpus/build_index.py",
    "src/retrieval/bm25.py", "src/retrieval/dense.py", "src/retrieval/hybrid.py",
    "src/retrieval/rrf.py", "src/retrieval/rerank.py", "src/retrieval/cache.py",
    "src/synthesis/generator.py", "src/synthesis/uncertainty.py", "src/synthesis/citations.py",
    "src/telemetry/events.py", "src/telemetry/logger.py", "src/baseline.py",
    "src/stream/simulator.py", "src/stream/controller.py", "src/stream/stability.py",
    "src/decompose/decomposer.py", "src/decompose/dedupe.py",
    "src/session/store.py", "src/session/ledger.py", "src/session/delta.py",
    "src/synthesis/grounding.py", "src/engine.py",
]


def test_required_layout_exists():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_src_never_imports_eval():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for mod in _imported_modules(path):
            if mod == "eval" or mod.startswith("eval."):
                offenders.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not offenders, offenders


def test_all_llm_calls_live_in_llm_client():
    """No module other than src/llm/client.py may talk HTTP or import an LLM SDK."""
    banned = ("urllib.request", "http.client", "requests", "httpx", "openai", "anthropic", "ollama")
    allowed = ROOT / "src" / "llm" / "client.py"
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path == allowed:
            continue
        for mod in _imported_modules(path):
            if mod in banned or any(mod.startswith(b + ".") for b in banned):
                offenders.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not offenders, offenders


def test_phase5_documents_exist_and_brief_fits_six_pages():
    brief = (ROOT / "reports" / "ARCHITECTURE_BRIEF.md").read_text(encoding="utf-8")
    assert len(brief.split()) <= 3300, "architecture brief must stay within ~6 pages"
    for section in ("Retrieval trigger logic", "Decomposition strategy", "Data provenance", "Trade-offs",
                    "Failure modes"):
        assert section in brief, section
    bench = (ROOT / "reports" / "BENCHMARK.md").read_text(encoding="utf-8")
    assert bench.count("**F") >= 3, "benchmark needs at least three analysed edge-case failures"
    assert "Architectural ablations" in bench


def test_demo_script_covers_required_items_and_files_exist():
    demo = (ROOT / "reports" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    for item in ("Early retrieval", "Multi-intent decomposition", "Late-detail refinement",
                 "Presentation suppression", "Citation traceability", "Live telemetry"):
        assert item in demo, item
    import re

    for name in set(re.findall(r"eval/dev_scenarios/(\w+\.json)", demo)) | {f"{s}.json" for s in
                                                                            re.findall(r"--scenario (\w+)", demo)}:
        assert (ROOT / "eval" / "dev_scenarios" / name).exists(), name


def test_one_command_runner_is_valid_bash():
    import os
    import subprocess

    script = ROOT / "scripts" / "run_all.sh"
    assert os.access(script, os.X_OK)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    text = script.read_text(encoding="utf-8")
    for step in ("requirements-dev.txt", "pytest", "build_index", "eval/replay.py"):
        assert step in text, step


def test_no_scenario_or_benchmark_text_is_hardcoded_in_src():
    """Standing rule 2: no prompts, queries or canned answers from the evaluation data in src/.
    Checks every dev-scenario chunk and gold label (>= 4 words) and the problem statement's own
    example utterances against every module under src/."""
    import json
    import re

    norm = lambda t: re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()  # noqa: E731
    src_text = " ".join(" ".join(norm(p.read_text(encoding="utf-8"))) for p in (ROOT / "src").rglob("*.py"))
    phrases = []
    for path in (ROOT / "eval" / "dev_scenarios").glob("*.json"):
        sc = json.loads(path.read_text(encoding="utf-8"))
        for turn in sc["turns"]:
            phrases += [c["text"] for c in turn["chunks"]]
            phrases += [g["text"] for g in turn["gold_sub_intents"]]
    phrases += ["I need to plan a customer workshop in", "Pune for 30 people", "Summarize the travel reimbursement rule",
                "The trip was international and the booking was made after travel",
                "Please repeat your last answer in two bullets", "Pune workshop venue capacity 30"]
    leaked = [p for p in phrases if len(norm(p)) >= 4 and " ".join(norm(p)) in src_text]
    assert not leaked, leaked
