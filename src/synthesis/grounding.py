"""Grounding verifier (step 4.4).

For each claim:
(a) every cited chunk id must exist in the corpus. Unknown ids are fabrications: they are
    removed, and a claim left with no valid id fails (`fabricated_id`);
(b) a support judge must confirm that the cited chunk text supports the claim
    (`unsupported` otherwise).
Failures are dropped (default) or kept and flagged (`mode="flag"`).

Support judges (one interface):
* `lexical` (CPU default, no model): share of the claim's content words present in the
  cited chunks, supported if >= `threshold`. A proxy for entailment, not entailment: it
  catches claims that talk about things the chunk never mentions, but not negations or
  wrong relations between words that do appear.
* `nli`: a cross-encoder NLI model (default `cross-encoder/nli-deberta-v3-small`) via
  sentence-transformers. Optional dependency; not installed or tested in this repo yet.
* `llm`: the configured LLM as judge through src/llm/client.py (JSON verdict). Tested with a
  mock client only; no real LLM has been run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel

from src.llm.client import LLMClient, LLMError
from src.retrieval.text import content_terms
from src.schemas import Claim
from src.synthesis.citations import CitationIndex
from src.telemetry.logger import RequestTrace


@dataclass(frozen=True)
class Verdict:
    supported: bool
    score: float
    method: str
    reason: str = ""


class SupportJudge(Protocol):
    name: str

    async def judge(self, claim: str, evidence: str, trace: RequestTrace | None = None) -> Verdict: ...


class LexicalSupportJudge:
    name = "lexical"

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    async def judge(self, claim: str, evidence: str, trace: RequestTrace | None = None) -> Verdict:
        terms = content_terms(claim)
        if not terms:
            return Verdict(False, 0.0, self.name, "claim has no content words")
        covered = terms & content_terms(evidence)
        score = len(covered) / len(terms)
        missing = sorted(terms - covered)
        reason = "" if score >= self.threshold else f"not in cited evidence: {', '.join(missing[:6])}"
        return Verdict(score >= self.threshold, round(score, 4), self.name, reason)


class NLISupportJudge:  # pragma: no cover - optional dependency, not installed in this repo
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("PRISM_GROUNDING_JUDGE=nli needs `pip install -r requirements-optional.txt`") from exc
        self._model = CrossEncoder(model_name, device="cpu")
        labels = {i: str(l).lower() for i, l in getattr(self._model.config, "id2label", {}).items()}
        self._entail = next((i for i, l in labels.items() if "entail" in l), None)
        if self._entail is None:
            raise RuntimeError(f"cannot find an entailment label in {model_name} ({labels})")
        self.threshold = threshold

    async def judge(self, claim: str, evidence: str, trace: RequestTrace | None = None) -> Verdict:
        import numpy as np

        logits = np.asarray(self._model.predict([(evidence, claim)]))[0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        p = float(probs[self._entail])
        return Verdict(p >= self.threshold, round(p, 4), self.name, "" if p >= self.threshold else "not entailed")


class JudgeVerdict(BaseModel):
    supported: bool
    reason: str = ""


JUDGE_SYSTEM = (
    "You check whether a claim is fully supported by the evidence text. Answer supported=true only "
    "if every part of the claim is stated or directly implied by the evidence. Use no other "
    "knowledge. Reply with JSON only."
)


class LLMSupportJudge:
    name = "llm"

    def __init__(self, client: LLMClient):
        self.client = client

    async def judge(self, claim: str, evidence: str, trace: RequestTrace | None = None) -> Verdict:
        prompt = (f"Evidence:\n{evidence}\n\nClaim: {claim}\n\n"
                  'Return JSON: {"supported": true | false, "reason": "<short reason>"}')
        try:
            resp = await self.client.generate(prompt, schema=JudgeVerdict, system=JUDGE_SYSTEM, trace=trace)
        except LLMError as exc:
            return Verdict(False, 0.0, self.name, f"judge error: {exc}")
        v: JudgeVerdict = resp.parsed
        return Verdict(v.supported, 1.0 if v.supported else 0.0, self.name, v.reason)


@dataclass
class ClaimCheck:
    claim: Claim
    verdict: Verdict | None
    fabricated_ids: list[str] = field(default_factory=list)
    failure: Literal["fabricated_id", "unsupported"] | None = None


@dataclass
class GroundingReport:
    kept: list[Claim]
    checks: list[ClaimCheck]

    @property
    def failed(self) -> list[ClaimCheck]:
        return [c for c in self.checks if c.failure is not None]

    @property
    def fabricated_ids(self) -> list[str]:
        return [i for c in self.checks for i in c.fabricated_ids]

    def failed_subintents(self) -> set[str]:
        return {c.claim.subintent_id for c in self.failed}


class GroundingVerifier:
    def __init__(self, index: CitationIndex, judge: SupportJudge, mode: Literal["drop", "flag"] = "drop"):
        if mode not in ("drop", "flag"):
            raise ValueError("mode must be 'drop' or 'flag'")
        self.index = index
        self.judge = judge
        self.mode = mode

    async def verify(self, claims: list[Claim], trace: RequestTrace | None = None) -> GroundingReport:
        kept, checks = [], []
        for claim in claims:
            valid = [cid for cid in claim.chunk_ids if self.index.is_valid_chunk(cid)]
            fabricated = [cid for cid in claim.chunk_ids if cid not in valid]
            if not valid:
                checks.append(ClaimCheck(claim, None, fabricated, "fabricated_id"))
                continue
            clean = claim if not fabricated else claim.model_copy(update={"chunk_ids": valid})
            evidence = "\n\n".join(self.index.chunks[cid].text for cid in valid)
            verdict = await self.judge.judge(clean.text, evidence, trace)
            check = ClaimCheck(clean, verdict, fabricated, None if verdict.supported else "unsupported")
            checks.append(check)
            if verdict.supported or self.mode == "flag":
                kept.append(clean)
        return GroundingReport(kept, checks)


def make_judge(kind: str, client: LLMClient | None = None, lexical_threshold: float = 0.8) -> SupportJudge:
    if kind == "lexical":
        return LexicalSupportJudge(lexical_threshold)
    if kind == "nli":
        return NLISupportJudge()
    if kind == "llm":
        if client is None:
            raise ValueError("grounding judge 'llm' needs an LLM (set PRISM_LLM_PROVIDER / PRISM_LLM_MODEL)")
        return LLMSupportJudge(client)
    raise ValueError(f"unknown grounding judge {kind!r} (expected lexical, nli or llm)")
