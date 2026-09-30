"""JSON-schema validation of the output record and the other event schemas (step 1.3)."""

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from src.schemas import (
    CITATION_PATTERN,
    ControllerDecision,
    CorpusChunk,
    OutputRecord,
    TelemetryEvent,
    TranscriptChunk,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "output_record.schema.json").read_text(encoding="utf-8"))

# Shape of the example record in the problem statement ([Doc_01 §4.1.2]). The "answer" and
# "uncertainty" strings are cut off at the page edge in the source PDF; they are used here
# exactly up to the truncation point and closed so the document is valid JSON.
PDF_EXAMPLE = {
    "retrieval_events": [
        {"timestamp_s": 0.8, "query": "Pune workshop venue capacity 30", "trigger": "provisional"},
        {"timestamp_s": 1.6, "query": "cancellation policy workshop venues Pune", "trigger": "multi_intent"},
        {"timestamp_s": 1.6, "query": "catering service options workshop Pune", "trigger": "multi_intent"},
    ],
    "sub_queries": [
        "venue capacity for 30 attendees in Pune",
        "cancellation terms and refund policies",
        "on-site and external catering options",
    ],
    "answer": "For a 30-person workshop in Pune, documented options include Venue A and Venue B. Venue A provide",
    "citations": ["Doc_12 §2", "Doc_31 §4", "Doc_09 §1"],
    "uncertainty": "Catering accommodation policies for Venue A could not be verified from the retrieved corpus.",
}


def test_committed_schema_files_are_current():
    result = subprocess.run(
        [sys.executable, "scripts/export_schemas.py", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


def test_pdf_example_validates_against_json_schema():
    jsonschema.validate(PDF_EXAMPLE, SCHEMA)
    assert OutputRecord.model_validate(PDF_EXAMPLE).citations[0] == "Doc_12 §2"


def test_uncertainty_may_be_null():
    record = dict(PDF_EXAMPLE, uncertainty=None)
    jsonschema.validate(record, SCHEMA)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("citations"),
        lambda r: r.update(extra_field=1),
        lambda r: r["retrieval_events"][0].update(trigger="eager"),
        lambda r: r.update(citations=["[Doc_12 §2]"]),  # brackets belong in answer text only
        lambda r: r.update(citations=["Doc_999"]),  # no section marker
    ],
)
def test_invalid_records_are_rejected(mutate):
    record = json.loads(json.dumps(PDF_EXAMPLE))
    mutate(record)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, SCHEMA)


def test_citation_pattern_accepts_nested_sections():
    import re

    assert re.match(CITATION_PATTERN, "Doc_01 §4.1.2")
    assert not re.match(CITATION_PATTERN, "Doc_01 §")


def test_model_constraints():
    with pytest.raises(ValidationError):
        ControllerDecision(action="search", reason="x")
    with pytest.raises(ValidationError):
        TranscriptChunk(ts=-1, text="x")
    with pytest.raises(ValidationError):
        TelemetryEvent(event="unknown", request_id="r")
    with pytest.raises(ValidationError):
        OutputRecord.model_validate(dict(PDF_EXAMPLE, citations=["Doc_999"]))
    chunk = CorpusChunk(chunk_id="Doc_01 §3#1", doc_id="Doc_01", section="3", text="t")
    assert chunk.citation == "Doc_01 §3"
