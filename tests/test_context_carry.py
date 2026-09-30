"""Context carry-over (step 3.3). Generic examples, not dev-scenario text."""

import pytest

from src.decompose.context import carry_context, extract_constraints, search_text
from src.schemas import SubQuery


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Book a room in New Delhi for 12 people next Friday", ["12 people", "next Friday", "New Delhi"]),
        ("What is the price of Plan B on 3 March 2025?", ["3 March 2025", "Plan B"]),
        ("due March 5, 2026 or tomorrow", ["March 5, 2026", "tomorrow"]),
        ("latency under 200 ms for 95% of calls", ["200 ms", "95%"]),
        ("What does the reranker do", []),
    ],
)
def test_extract_constraints(text, expected):
    assert extract_constraints(text) == expected


def test_constraints_propagate_forward_only():
    subs = [SubQuery(id="q1", text="the refund terms"),
            SubQuery(id="q2", text="a hall in Chennai for 40 guests"),
            SubQuery(id="q3", text="what catering options exist")]
    out = carry_context(subs)
    assert out[0].constraints == []
    assert out[1].constraints == []
    assert out[2].constraints == ["40 guests", "Chennai"]
    assert search_text(out[2]) == "what catering options exist 40 guests Chennai"


def test_constraint_already_mentioned_is_not_duplicated():
    subs = [SubQuery(id="q1", text="rooms in Chennai"), SubQuery(id="q2", text="parking in Chennai")]
    assert carry_context(subs)[1].constraints == []


def test_pronoun_reference_carries_previous_topic():
    subs = [SubQuery(id="q1", text="the speculative reuse cache"), SubQuery(id="q2", text="how is it invalidated")]
    out = carry_context(subs)
    assert out[1].constraints == ["speculative reuse cache"]


def test_input_is_not_mutated():
    subs = [SubQuery(id="q1", text="in Paris for 3 days"), SubQuery(id="q2", text="museums")]
    carry_context(subs)
    assert subs[1].constraints == []
