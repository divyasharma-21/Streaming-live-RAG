"""Decomposer diff operations (step 3.1). Phrases are generic, not dev-scenario text."""

import json

import pytest

from src.decompose.decomposer import (
    DiffOp,
    IdGenerator,
    LLMDecomposer,
    RuleDecomposer,
    apply_ops,
    make_decomposer,
    split_clauses,
)
from src.llm.client import LLMError, MockLLMClient
from src.schemas import SubQuery


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What is the reranker, and how is telemetry logged?", ["What is the reranker", "how is telemetry logged?"]),
        ("Explain the gates and what the pitfalls are", ["Explain the gates", "what the pitfalls are"]),
        ("Tell me what the probe does, what the gate does, and what the cache does.",
         ["Tell me what the probe does", "what the gate does", "what the cache does."]),
        ("Which phase adds caching? And who owns telemetry?", ["Which phase adds caching?", "who owns telemetry?"]),
        ("What is RRF; also, how is rerank done?", ["What is RRF", "how is rerank done?"]),
    ],
)
def test_split_at_intent_boundaries(text, expected):
    assert split_clauses(text) == expected


@pytest.mark.parametrize(
    "text",
    ["For the demo video, what must it show?", "How does retrieval and fusion work?", "describe the cache"],
)
def test_single_intents_stay_whole(text):
    assert len(split_clauses(text)) == 1


def test_apply_ops_add_keep_merge_and_drop():
    live = [SubQuery(id="q1", text="a"), SubQuery(id="q2", text="b"), SubQuery(id="q3", text="c"),
            SubQuery(id="q4", text="d")]
    new_id = IdGenerator()
    for _ in range(4):
        new_id()
    ops = [
        DiffOp(op="keep", id="q1"),
        DiffOp(op="keep", id="q2", text="b refined"),
        DiffOp(op="merge", ids=["q3", "q4"], text="c and d"),
        DiffOp(op="add", text="e"),
        DiffOp(op="keep", id="nope"),
        DiffOp(op="add", text="  "),
    ]
    out = apply_ops(live, ops, new_id)
    assert [(q.id, q.text) for q in out] == [("q1", "a"), ("q2", "b refined"), ("q3", "c and d"), ("q5", "e")]


def test_unreferenced_live_subqueries_are_dropped():
    out = apply_ops([SubQuery(id="q1", text="a"), SubQuery(id="q2", text="b")], [DiffOp(op="keep", id="q2")],
                    IdGenerator())
    assert [q.id for q in out] == ["q2"]


async def test_rule_decomposer_diffs_incrementally():
    d, new_id = RuleDecomposer(), IdGenerator()
    live = apply_ops([], await d.decompose("What does the reranker do", []), new_id)
    assert [(q.id, q.text) for q in live] == [("q1", "What does the reranker do")]
    ops = await d.decompose("What does the reranker do, and how is the cache keyed?", live)
    assert [o.op for o in ops] == ["keep", "add"] and ops[0].id == "q1"
    live = apply_ops(live, ops, new_id)
    assert [q.id for q in live] == ["q1", "q2"]


async def test_rule_decomposer_merges_when_one_clause_covers_two():
    live = [SubQuery(id="q1", text="reranker scores"), SubQuery(id="q2", text="cache keys")]
    ops = await RuleDecomposer().decompose("Explain reranker scores with cache keys", live)
    assert [(o.op, o.ids) for o in ops] == [("merge", ["q1", "q2"])]


async def test_llm_decomposer_uses_json_schema_and_prompt_is_generic():
    reply = json.dumps({"ops": [{"op": "keep", "id": "q1", "text": "x"}, {"op": "add", "text": "y"}]})
    client = MockLLMClient([reply])
    ops = await LLMDecomposer(client).decompose("some transcript", [SubQuery(id="q1", text="x")])
    assert [o.op for o in ops] == ["keep", "add"]
    assert '"id": "q1"' in client.prompts[0] and "some transcript" in client.prompts[0]


async def test_llm_decomposer_falls_back_to_rules_on_error():
    def boom(prompt):
        raise LLMError("down")

    d = LLMDecomposer(MockLLMClient(boom))
    ops = await d.decompose("What is the gate, and what is the probe?", [])
    assert [o.op for o in ops] == ["add", "add"] and d.fallbacks == 1


def test_make_decomposer():
    assert make_decomposer("rules").name == "rules"
    with pytest.raises(ValueError):
        make_decomposer("llm", None)
    with pytest.raises(ValueError):
        make_decomposer("other")
