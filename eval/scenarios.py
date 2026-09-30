"""Dev scenario format and loader (step 1.9).

Each file in eval/dev_scenarios/ is one scenario with one or more turns. A turn is a stream
of timestamped transcript chunks ending in an utterance-end chunk (`is_final: true`,
empty text), plus gold labels:

* `retrieval_required`: false for presentation-only or incomplete turns.
* `gold_sub_intents`: the distinct intents; each lists the `supporting` sections needed to
  answer it (empty when the corpus has no evidence) and optional `also_relevant` sections.
* `expect_uncertainty`: an uncertainty note / clarification is the correct behaviour.

Gold ids are section-level citations (`Doc_01 §5`), so they stay valid if chunk sizes change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas import Citation, TranscriptChunk

SCENARIO_DIR = Path(__file__).resolve().parent / "dev_scenarios"

Category = Literal[
    "single_intent", "multi_intent", "late_constraint", "presentation_only", "noisy_fragment", "no_evidence"
]


class GoldSubIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str
    supporting: list[Citation]
    also_relevant: list[Citation] = Field(default_factory=list)


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn: int = Field(ge=1)
    chunks: list[TranscriptChunk] = Field(min_length=2)
    retrieval_required: bool
    gold_sub_intents: list[GoldSubIntent]
    expect_uncertainty: bool = False
    reuses_turn: int | None = None
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> Turn:
        ts = [c.ts for c in self.chunks]
        if ts != sorted(ts):
            raise ValueError("chunk timestamps must be non-decreasing")
        if not self.chunks[-1].is_final or any(c.is_final for c in self.chunks[:-1]):
            raise ValueError("exactly the last chunk must be the utterance end (is_final)")
        if not self.retrieval_required and self.gold_sub_intents:
            raise ValueError("a no-retrieval turn has no gold sub-intents")
        if self.retrieval_required and not self.gold_sub_intents:
            raise ValueError("a retrieval turn needs gold sub-intents")
        return self

    @property
    def utterance(self) -> str:
        return " ".join(c.text.strip() for c in self.chunks if c.text.strip())

    @property
    def utterance_end_s(self) -> float:
        return self.chunks[-1].ts

    @property
    def gold_supporting(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.gold_sub_intents:
            for c in s.supporting:
                seen.setdefault(c, None)
        return list(seen)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Category
    review_status: Literal["draft_pending_human_review", "approved"]
    description: str
    turns: list[Turn] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Scenario:
        if [t.turn for t in self.turns] != list(range(1, len(self.turns) + 1)):
            raise ValueError("turns must be numbered 1..n")
        return self


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    scenarios = []
    for path in sorted(directory.glob("*.json")):
        scenario = Scenario.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if scenario.id != path.stem:
            raise ValueError(f"{path.name}: id {scenario.id!r} does not match the file name")
        scenarios.append(scenario)
    return scenarios
