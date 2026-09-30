"""Delta engine and session refinement (step 4.6). Utterances are generic, not dev-scenario text.

The pipeline tests compare ledger snapshots before and after a follow-up turn to prove that a
delta update touches only the affected claims (playbook pass criterion)."""

import pytest

from src.config import Settings
from src.retrieval.factory import build_retrieval
from src.schemas import Claim, SubQuery
from src.session.delta import DeltaClassifier, negated_terms
from src.session.ledger import ClaimLedger
from src.session.pipeline import SessionPipeline
from src.stream.simulator import chunks_from_text
from src.telemetry.logger import TelemetryLogger

# ------------------------------------------------------------------ classifier


@pytest.fixture
def ledger():
    lg = ClaimLedger()
    lg.upsert_subintent(SubQuery(id="t1.q1", text="What are the hotel booking rules?", constraints=["Pune"]))
    lg.upsert_subintent(SubQuery(id="t1.q2", text="What does the catering cost?"))
    lg.add_claims([Claim(id="t1.q1.c1", text="Hotels must be booked early.", subintent_id="t1.q1", chunk_ids=["x"]),
                   Claim(id="t1.q2.c1", text="Catering costs are fixed per guest.", subintent_id="t1.q2",
                         chunk_ids=["y"])])
    lg.commit_version(None)
    return lg


def kinds(decision):
    return [(c.kind, c.target) for c in decision.clauses]


def test_new_question_is_new_subintent(ledger):
    d = DeltaClassifier().classify([("How long does parking last?", "q1")], ledger)
    assert d.kind == "new_subintent" and kinds(d) == [("new_subintent", None)]


def test_constraint_statement_is_parameter_update_of_overlapping_subintent(ledger):
    d = DeltaClassifier().classify([("The hotel booking was made late.", "q1")], ledger)
    assert kinds(d) == [("parameter_update", "t1.q1")]


def test_back_reference_attaches_to_latest_subintent(ledger):
    d = DeltaClassifier().classify([("Also for vegetarians.", "q1")], ledger)
    assert kinds(d) == [("parameter_update", "t1.q2")]


def test_negation_of_a_stated_term_is_contradiction(ledger):
    d = DeltaClassifier().classify([("Use Mumbai instead of Pune.", "q1")], ledger)
    assert d.kind == "contradiction" and kinds(d) == [("contradiction", "t1.q1")]
    assert d.clauses[0].negated_terms == ["pune"]


def test_turn_kind_uses_priority(ledger):
    d = DeltaClassifier().classify([("How long does parking last?", "q1"), ("The hotel booking was late.", "q2")],
                                   ledger)
    assert d.kind == "parameter_update"


def test_presentation_only_and_empty_ledger():
    assert DeltaClassifier().classify([("x", "q1")], ClaimLedger(), presentation_only=True).kind == "presentation_only"
    assert DeltaClassifier().classify([("anything", "q1")], ClaimLedger()).kind == "new_subintent"


def test_negated_terms_exclude_replacement_words():
    assert negated_terms("not the old rule, the new rule") == ["old"]
    assert negated_terms("the trip was international") == []


# ------------------------------------------------------------------ pipeline: ledger diffs


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return build_retrieval(Settings(index_dir=tmp_path_factory.mktemp("idx")))


@pytest.fixture
def pipeline(stack, tmp_path):
    return SessionPipeline(Settings(log_dir=tmp_path, llm_provider="none"), stack=stack,
                           telemetry=TelemetryLogger(None))


async def turn(p, sid, *parts):
    return await p.handle_turn(sid, chunks_from_text(list(parts)))


async def test_parameter_update_keeps_existing_claims_and_adds_delta(pipeline):
    first = await turn(pipeline, "s", "What are the hard engineering rules?")
    before = pipeline.store.get("s").ledger.snapshot()
    assert first.answer_version == 1 and before
    second = await turn(pipeline, "s", "Only the grounding rule.")
    diff = second.ledger_diff
    assert second.kind == "parameter_update" and second.answer_version == 2
    assert diff["removed"] == [] and diff["changed"] == []
    assert set(diff["unchanged"]) == set(before)  # every earlier claim kept, same id/text/citations/version
    assert diff["added"] and all(cid.startswith("t1.q1.") for cid in diff["added"])  # only the affected sub-intent
    ledger = pipeline.store.get("s").ledger
    assert all(ledger.claims[c].constraints == ["Only the grounding rule."] for c in diff["added"])
    # targeted: the old sub-query was not searched again
    first_queries = {e.query for e in first.record.retrieval_events}
    assert not first_queries & {e.query for e in second.record.retrieval_events}


async def test_new_subintent_leaves_old_claims_untouched(pipeline):
    await turn(pipeline, "s", "What are the hard engineering rules?")
    before = pipeline.store.get("s").ledger.snapshot()
    out = await turn(pipeline, "s", "Which gate measures telemetry coverage?")
    assert out.kind == "new_subintent"
    assert set(out.ledger_diff["unchanged"]) == set(before) and out.ledger_diff["removed"] == []
    assert all(cid.startswith("t2.") for cid in out.ledger_diff["added"])


async def test_contradiction_invalidates_only_dependent_claims(pipeline):
    await turn(pipeline, "s", "What are the hard engineering rules?")
    base = set(pipeline.store.get("s").ledger.snapshot())
    upd = await turn(pipeline, "s", "Only the grounding rule.")
    dependent = set(upd.ledger_diff["added"])
    out = await turn(pipeline, "s", "Not the grounding rule, I meant the session-bound state rule.")
    assert out.kind == "contradiction" and out.answer_version == 3
    assert set(out.ledger_diff["removed"]) == dependent  # exactly the claims that depended on "grounding"
    assert base <= set(out.ledger_diff["unchanged"])  # the original answer is untouched
    constraints = pipeline.store.get("s").ledger.subintents["t1.q1"].subquery.constraints
    assert "Only the grounding rule." not in constraints


async def test_sessions_are_isolated_in_the_pipeline(pipeline):
    await turn(pipeline, "A", "What are the hard engineering rules?")
    b = await turn(pipeline, "B", "Only the grounding rule.")
    assert b.kind == "first_answer" and pipeline.store.get("B").ledger.version == 1
    assert pipeline.store.get("A").ledger.version == 1
    pipeline.end_session("A")
    assert "A" not in pipeline.store and "B" in pipeline.store


async def test_every_rendered_citation_exists(pipeline):
    out = await turn(pipeline, "s", "What does the retrieval controller decide, and which gate measures it?")
    assert out.invalid_citations == [] and out.record.citations
    assert out.grounding is not None and out.grounding.fabricated_ids == []


def test_rendered_answer_shows_each_statement_once():
    from src.session.pipeline import unique_by_text

    a = Claim(id="s1.c1", text="Same span.", subintent_id="s1", chunk_ids=["Doc_01 §6#1"])
    b = Claim(id="s2.c1", text="Same span.", subintent_id="s2", chunk_ids=["Doc_01 §6#1"])
    c = Claim(id="s2.c2", text="Other.", subintent_id="s2", chunk_ids=["Doc_01 §5#1"])
    assert [x.id for x in unique_by_text([a, b, c])] == ["s1.c1", "s2.c2"]
