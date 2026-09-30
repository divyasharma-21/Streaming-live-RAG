"""Full-system metrics (step 4.8): G4 factual grounding and G5 session refinement.

Each scenario runs as one session through `SessionPipeline`; the session is ended after the
scenario, so nothing crosses scenarios.

G4 (target: support >= 0.85 and zero fabricated ids)
* support: share of emitted claims whose text is literally contained in their cited chunk
  (the claim is split at ';' and ': ' and every normalised part must occur in the chunk).
  This check is independent of the pipeline's own grounding judge (which already dropped
  claims it rejected, so re-using it would report 100% by construction).
* pre-verification support: share of generated claims the pipeline's judge accepted.
* fabricated ids: cited ids in answers or claims that do not exist in the corpus.
* shuffled control: the same check after re-citing every claim to a chunk from a different
  section; a meaningful check must drop sharply here.

G5 (target: every refinement turn keeps state continuity)
For every follow-up turn that is not presentation-only or unsearchable:
* state kept: the session still holds every earlier sub-intent, answer version = previous + 1;
* untouched claims kept: no claim outside the affected sub-intents was removed or changed;
* no full re-search: no earlier turn's retrieval query is issued again.
Presentation-only turns must make zero retrieval calls and add no citation.
"""

from __future__ import annotations

import random
import re
import statistics
import time

from eval.decompose_eval import match_subintents
from eval.scenarios import Scenario
from src.schemas import Claim
from src.session.pipeline import SessionPipeline
from src.synthesis.citations import CitationIndex

G4_TARGET = 0.85
_MARKUP_RE = re.compile(r"[\[\]`*_]")  # markdown markers: deleted (they sit next to words)


def _norm(text: str) -> str:
    text = _MARKUP_RE.sub("", text).replace("|", " ")  # table pipes separate cells
    return re.sub(r"\s+", " ", text).strip().lower()


def literally_supported(claim_text: str, chunk_text: str) -> bool:
    chunk = _norm(chunk_text)
    parts = [p for seg in claim_text.split(";") for p in seg.split(": ")]
    parts = [_norm(p) for p in parts if _norm(p)]
    return bool(parts) and all(p in chunk for p in parts)


def claim_supported(claim: Claim, index: CitationIndex) -> bool:
    texts = [index.chunks[c].text for c in claim.chunk_ids if c in index.chunks]
    return bool(texts) and literally_supported(claim.text, "\n".join(texts))


def shuffled_control(claims: list[Claim], index: CitationIndex, seed: int = 0) -> float | None:
    rng = random.Random(seed)
    ids = sorted(index.chunks)
    if not claims:
        return None
    ok = 0
    for c in claims:
        own = {index.chunks[x].section for x in c.chunk_ids if x in index.chunks}
        other = [x for x in ids if index.chunks[x].section not in own]
        ok += claim_supported(c.model_copy(update={"chunk_ids": [rng.choice(other)]}), index)
    return round(ok / len(claims), 4)


async def evaluate_full(scenarios: list[Scenario], pipeline: SessionPipeline, clock: str = "simulated",
                        speed: float = 1.0, on_turn=None) -> dict:
    index = pipeline.index
    rows, emitted, generated_checks = [], [], []
    for scenario in scenarios:
        sid = f"eval-{scenario.id}-{int(time.time() * 1000)}"
        previous_queries: set[str] = set()
        prev_version, prev_subintents = 0, set()
        for t in scenario.turns:
            out = await pipeline.handle_turn(sid, t.chunks, request_id=f"{scenario.id}-t{t.turn}", clock=clock,
                                             speed=speed)
            ledger = pipeline.store.get(sid).ledger
            queries = {e.query for e in out.record.retrieval_events}
            claims_now = ledger.active_claims()
            new_claims = [c for c in claims_now if c.id in set(out.ledger_diff.get("added", []))]
            emitted.extend(new_claims)
            if out.grounding is not None:
                generated_checks.extend(out.grounding.checks)
            affected = set(out.delta.affected) if out.delta else set()
            touched = set(out.ledger_diff.get("removed", [])) | set(out.ledger_diff.get("changed", []))
            touched_outside = sorted(cid for cid in touched if cid.rsplit(".c", 1)[0] not in affected)
            refinement = t.turn > 1 and out.kind in ("new_subintent", "parameter_update", "contradiction")
            row = {
                "scenario": scenario.id, "turn": t.turn, "category": scenario.category, "utterance": out.stream.utterance,
                "kind": out.kind,
                "clauses": [(c.kind, c.target, c.clause) for c in out.delta.clauses] if out.delta else [],
                "version": out.answer_version, "retrieval_calls": out.retrieval_calls,
                "answer": out.record.answer, "citations": out.record.citations, "uncertainty": out.record.uncertainty,
                "delta_citations": out.delta_citations, "ledger_diff": out.ledger_diff,
                "expect_uncertainty": t.expect_uncertainty, "retrieval_required": t.retrieval_required,
                "fabricated": sorted(set(out.invalid_citations) | set(out.grounding.fabricated_ids if out.grounding else [])),
                "new_claims": len(new_claims),
                "new_claims_supported": sum(claim_supported(c, index) for c in new_claims),
                # streaming fields for G2 / G3 (same run)
                "request_id": out.request_id,
                "retrieved": bool(out.stream.retrievals),
                "early": out.stream.retrieved_early,
                "first_retrieval_s": out.stream.first_retrieval_s,
                "utterance_end_s": out.stream.utterance_end_s,
                "gold_n": len(t.gold_sub_intents),
                "predicted": [q.text for q in out.stream.sub_queries],
                "g3_correct": len(match_subintents(t.gold_sub_intents, out.stream.sub_queries)),
                "turn_citations": sorted({index.chunks[c].citation for cl in new_claims for c in cl.chunk_ids
                                          if c in index.chunks}),
                "gold_supporting": t.gold_supporting,
                "also_relevant": sorted({c for g in t.gold_sub_intents for c in g.also_relevant}),
                "stage_ms": out.stage_ms, "tokens_in": out.tokens_in, "tokens_out": out.tokens_out,
                "est_cost_usd": out.est_cost_usd,
            }
            if refinement:
                row["g5"] = {
                    "state_kept": prev_subintents <= set(ledger.subintents) and out.answer_version == prev_version + 1,
                    "untouched_claims_kept": not touched_outside,
                    "touched_outside_affected": touched_outside,
                    "no_full_research": not (queries & previous_queries),
                }
            if out.kind == "presentation_only":
                row["presentation_ok"] = out.retrieval_calls == 0 and set(out.record.citations) <= set(
                    rows[-1]["citations"] if rows and rows[-1]["scenario"] == scenario.id else [])
            rows.append(row)
            if on_turn is not None:
                on_turn(row)
            previous_queries |= queries
            prev_version, prev_subintents = ledger.version, set(ledger.subintents)
        pipeline.end_session(sid)
    return {"summary": summarize(rows, emitted, generated_checks, index), "turns": rows}


def summarize(rows: list[dict], emitted: list[Claim], checks: list, index: CitationIndex) -> dict:
    supported = sum(claim_supported(c, index) for c in emitted)
    fabricated = sorted({f for r in rows for f in r["fabricated"]})
    rate = supported / len(emitted) if emitted else None
    refinement = [r for r in rows if "g5" in r]
    g5_ok = [r for r in refinement if all(v for k, v in r["g5"].items() if k != "touched_outside_affected")]
    presentation = [r for r in rows if r["kind"] == "presentation_only"]
    no_ev = [r for r in rows if r["expect_uncertainty"]]
    answerable = [r for r in rows if r["retrieval_required"] and not r["expect_uncertainty"]]
    return {
        "system": "full",
        "turns": len(rows),
        "g4": {
            "claims": len(emitted), "supported": supported, "support_rate": round(rate, 4) if rate is not None else None,
            "fabricated_ids": fabricated, "target": G4_TARGET,
            "pass": rate is not None and rate >= G4_TARGET and not fabricated,
            "pre_verification_support": round(sum(c.failure is None for c in checks) / len(checks), 4) if checks else None,
            "dropped_by_verifier": sum(c.failure is not None for c in checks),
            "shuffled_control_support": shuffled_control(emitted, index),
        },
        "g5": {
            "refinement_turns": len(refinement), "passed": len(g5_ok),
            "pass": bool(refinement) and len(g5_ok) == len(refinement),
            "failed": [f"{r['scenario']} t{r['turn']}" for r in refinement if r not in g5_ok],
            "kinds": {k: sum(r["kind"] == k for r in refinement) for k in ("new_subintent", "parameter_update", "contradiction")},
        },
        "presentation": {"turns": len(presentation), "zero_retrieval_and_no_new_ids": sum(bool(r.get("presentation_ok")) for r in presentation),
                         "retrieval_calls": sum(r["retrieval_calls"] for r in presentation)},
        "uncertainty": {"expected_turns": len(no_ev), "emitted": sum(r["uncertainty"] is not None for r in no_ev),
                        "answerable_turns": len(answerable),
                        "false_uncertainty": sum(r["uncertainty"] is not None and not r["answer"] for r in answerable)},
        "answer_kinds": {k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})},
        "mean_claims_per_answered_turn": round(statistics.mean([r["new_claims"] for r in rows if r["new_claims"]]), 2)
        if any(r["new_claims"] for r in rows) else None,
    }


def print_full_summary(s: dict) -> None:
    g4, g5, pr, u = s["g4"], s["g5"], s["presentation"], s["uncertainty"]
    print(f"\nSystem: full   turns={s['turns']}")
    print("| metric | value |")
    print("|---|---|")
    print(f"| G4 citation support (literal check) | {g4['supported']}/{g4['claims']} = {g4['support_rate']} "
          f"(target >= {g4['target']}, zero fabricated: {'PASS' if g4['pass'] else 'FAIL'}) |")
    print(f"| fabricated ids | {len(g4['fabricated_ids'])} {g4['fabricated_ids'] or ''} |")
    print(f"| pre-verification support (pipeline judge) / dropped | {g4['pre_verification_support']} / {g4['dropped_by_verifier']} |")
    print(f"| shuffled-citation control support | {g4['shuffled_control_support']} |")
    print(f"| G5 refinement turns with state continuity | {g5['passed']}/{g5['refinement_turns']} "
          f"({'PASS' if g5['pass'] else 'FAIL'}) kinds={g5['kinds']} |")
    print(f"| G5 failures | {', '.join(g5['failed']) or 'none'} |")
    print(f"| presentation turns: zero retrieval + no new ids | {pr['zero_retrieval_and_no_new_ids']}/{pr['turns']} "
          f"(retrieval calls {pr['retrieval_calls']}) |")
    print(f"| uncertainty on no-evidence turns / false uncertainty on answerable | {u['emitted']}/{u['expected_turns']} "
          f"/ {u['false_uncertainty']}/{u['answerable_turns']} |")
    print(f"| turn kinds | {s['answer_kinds']} |")
