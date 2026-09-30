"""Telemetry coverage checker (step 5.1, gate G6: 100% trace coverage).

    python -m src.telemetry.coverage logs/telemetry.jsonl      # exit 1 if any request is incomplete

Events are grouped by request id. A request's trace is complete when it has:
* every request: `request_started`, `request_completed`; every `retrieval_started` matched by
  a `retrieval_completed`; every `llm_call` carrying token counts and a cost estimate;
* baseline requests: `retrieval_started`, `citation_check`, `answer_emitted`;
* streaming requests: at least one `controller_decision` and one `utterance_end`;
* requests that expect an answer (baseline, or streaming with `expects_answer`):
  `citation_check` and an `answer_emitted` event that records the answer-version transition
  (`answer_version_from`, `answer_version`), the sub-queries, the claim -> source mapping,
  per-stage latencies (`stage_ms`), token counts and an estimated cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_FIELDS = ("answer_version_from", "answer_version", "sub_queries", "sources", "stage_ms")


@dataclass
class CoverageReport:
    requests: int
    complete: int
    missing: dict[str, list[str]] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.complete / self.requests if self.requests else 1.0

    @property
    def ok(self) -> bool:
        return self.requests > 0 and self.complete == self.requests


def check_request(events: list[dict]) -> list[str]:
    """Return what is missing from one request's events (empty list = complete)."""
    names = [e["event"] for e in events]
    missing: list[str] = []
    started = next((e for e in events if e["event"] == "request_started"), None)
    for required in ("request_started", "request_completed"):
        if required not in names:
            missing.append(required)
    if names.count("retrieval_started") != names.count("retrieval_completed"):
        missing.append(f"retrieval_completed ({names.count('retrieval_completed')}/{names.count('retrieval_started')})")
    for e in events:
        if e["event"] == "llm_call" and (e.get("tokens_in") is None or e.get("tokens_out") is None
                                         or e.get("est_cost_usd") is None):
            missing.append("llm_call token counts / cost")
    data = (started or {}).get("data", {})
    system = data.get("system")
    expects_answer = system == "baseline" or bool(data.get("expects_answer"))
    if system == "baseline" and "retrieval_started" not in names:
        missing.append("retrieval_started")
    if system == "streaming":
        for required in ("controller_decision", "utterance_end"):
            if required not in names:
                missing.append(required)
    if expects_answer:
        if "citation_check" not in names:
            missing.append("citation_check")
        answer = next((e for e in events if e["event"] == "answer_emitted"), None)
        if answer is None:
            missing.append("answer_emitted")
        else:
            missing += [f"answer_emitted.{f}" for f in ANSWER_FIELDS if f not in answer.get("data", {})]
            missing += [f"answer_emitted.{f}" for f in ("tokens_in", "tokens_out", "est_cost_usd") if answer.get(f) is None]
    return missing


def check_events(events: list[dict]) -> CoverageReport:
    by_request: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_request[e["request_id"]].append(e)
    missing = {rid: m for rid, evs in by_request.items() if (m := check_request(evs))}
    return CoverageReport(len(by_request), len(by_request) - len(missing), missing)


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_file(path: Path) -> CoverageReport:
    return check_events(load_events(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="telemetry JSONL file")
    args = parser.parse_args()
    report = check_file(args.path)
    print(f"trace coverage: {report.complete}/{report.requests} = {report.coverage:.4f}")
    for rid, m in list(report.missing.items())[:20]:
        print(f"  incomplete {rid}: missing {', '.join(m)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
