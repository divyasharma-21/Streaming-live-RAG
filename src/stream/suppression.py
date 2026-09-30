"""Suppression gate (step 2.2): detect presentation-only turns that operate on prior output
(reformat, shorten, repeat, translate, ...) so no corpus retrieval is issued.

Decision order:
1. No prior output in the session -> nothing to transform -> not presentation-only.
2. Generic patterns (decisive):
   * no transformation cue                                  -> retrieval required
   * cue + reference to prior output + no new content terms -> presentation_restructure
3. Anything else with a cue is ambiguous -> small classifier fallback: a logistic scorer
   over five generic features with hand-set weights (it is not trained on any data).

"New content" = content terms of the request that are not presentation vocabulary, not
already in the prior output, and (when a corpus vocabulary is supplied) searchable in the
corpus. All vocabularies below are generic English; none is taken from evaluation data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from src.retrieval.text import content_terms, tokenize

# Verbs and phrases that ask to transform existing output.
TRANSFORM_TERMS = frozenset(tokenize(
    "reformat format rephrase reword rewrite shorten condense compress summarize summarise simplify "
    "repeat restate translate paraphrase abbreviate trim tabulate recap reorganize reorganise "
    "restructure expand elaborate"
))
# Output-shape words (bullets, table, shorter, ...).
FORMAT_TERMS = frozenset(tokenize(
    "bullet bullets point points list table paragraph paragraphs sentence sentences line lines word "
    "words numbered format shorter simpler simply brief briefly concise concisely plain version summary "
    "tldr outline markdown json caps uppercase lowercase formal informal"
))
# Words that refer to the previous output, plus neutral request verbs.
REFERENCE_TERMS = frozenset(tokenize(
    "answer response reply message output text previous last earlier above said again version "
    "say put make write turn convert keep redo do same part bit piece"
))
PRESENTATION_VOCAB = TRANSFORM_TERMS | FORMAT_TERMS | REFERENCE_TERMS

_REFERENCE_RE = re.compile(
    r"\b(that|this|it|those|these|above|again|"
    r"(your|the) (last|previous|earlier|prior) (answer|response|reply|message|output|one)|"
    r"what you (just )?(said|wrote))\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(r"^\s*(what|which|who|whom|when|where|why|how|is|are|does|do|can|could)\b", re.I)
_POLITE_REQUEST_RE = re.compile(r"^\s*(can|could|would|will) you\b", re.I)


@dataclass(frozen=True)
class GateResult:
    presentation_only: bool
    retrieval_required: bool
    reason: str
    method: Literal["rules", "classifier"]
    score: float
    new_terms: tuple[str, ...] = ()


# Hand-set logistic weights for the fallback (bias, cue, reference, new terms, question, length).
DEFAULT_WEIGHTS = (-0.5, 1.5, 1.5, -2.0, -1.0, -0.15)


class PresentationGate:
    def __init__(
        self,
        corpus_vocabulary: frozenset[str] | set[str] | None = None,
        threshold: float = 0.5,
        weights: tuple[float, ...] = DEFAULT_WEIGHTS,
    ):
        self.vocabulary = frozenset(corpus_vocabulary) if corpus_vocabulary is not None else None
        self.threshold = threshold
        self.weights = weights

    def features(self, utterance: str, prior_output: str) -> dict[str, float]:
        terms = tokenize(utterance)
        cue = sum(t in TRANSFORM_TERMS or t in FORMAT_TERMS for t in terms)
        prior = content_terms(prior_output)
        new = [t for t in dict.fromkeys(terms) if t not in PRESENTATION_VOCAB and t not in prior]
        if self.vocabulary is not None:
            new = [t for t in new if t in self.vocabulary]
        question = bool(_QUESTION_START_RE.match(utterance)) and not _POLITE_REQUEST_RE.match(utterance)
        return {
            "cue": float(cue),
            "reference": float(bool(_REFERENCE_RE.search(utterance))),
            "new_terms": float(len(new)),
            "question": float(question),
            "length": float(len(utterance.split())),
            "_new": new,  # type: ignore[dict-item]
        }

    def classifier_score(self, f: dict[str, float]) -> float:
        b, w_cue, w_ref, w_new, w_q, w_len = self.weights
        z = b + w_cue * min(f["cue"], 2) + w_ref * f["reference"] + w_new * f["new_terms"] + w_q * f["question"] + w_len * f["length"]
        return 1.0 / (1.0 + math.exp(-z))

    def evaluate(self, utterance: str, prior_output: str | None) -> GateResult:
        if not prior_output or not prior_output.strip():
            return GateResult(False, True, "no_prior_output", "rules", 0.0)
        f = self.features(utterance, prior_output)
        new = tuple(f.pop("_new"))  # type: ignore[arg-type]
        if f["cue"] == 0:
            return GateResult(False, True, "no_presentation_cue", "rules", 0.0, new)
        if f["reference"] and f["new_terms"] == 0:
            return GateResult(True, False, "presentation_restructure", "rules", 1.0, new)
        score = self.classifier_score(f)
        if score >= self.threshold:
            return GateResult(True, False, "presentation_restructure", "classifier", round(score, 4), new)
        return GateResult(False, True, "new_content", "classifier", round(score, 4), new)
