"""Retrieval controller (step 2.4): decide wait / retrieve / suppress on each transcript chunk.

Policy (mode `rule_stability`, the default):
* suppress  - the suppression gate classifies the turn as presentation-only
              (reason `presentation_restructure`), or at utterance end the transcript has no
              searchable content, or trails off unfinished below the content minimums
              (reason `insufficient_content`; the caller should ask for clarification), or
              is complete but contains no corpus term at all (reason `no_corpus_terms`; the
              caller should report that the corpus has no evidence).
* wait      - the fragment is incomplete, lacks content, or its stability score is below
              `stability_threshold`.
* retrieve  - trigger `provisional` once stability holds for `stable_chunks` consecutive
              chunks (at most `max_provisional` per utterance, to avoid thrashing);
              trigger `final` at utterance end if new content arrived since the last
              retrieval (or nothing was retrieved yet).

Mode `rule_only` drops the stability requirement: a complete, content-sufficient fragment
counts as stable. Both modes share the same gate and end-of-utterance rule so the ablation
isolates the effect of the stability probe. The `multi_intent` trigger belongs to Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.config import Settings, get_settings
from src.llm.client import LLMClient, LLMError, make_llm_client
from src.schemas import ControllerDecision, Trigger
from src.stream.stability import ProbeResult, StabilityProbe, is_incomplete
from src.stream.suppression import PresentationGate

Mode = Literal["rule_only", "rule_stability"]


@dataclass(frozen=True)
class ControllerConfig:
    mode: Mode = "rule_stability"
    stability_threshold: float = 0.5
    stable_chunks: int = 1
    max_provisional: int = 1

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> ControllerConfig:
        s = s or get_settings()
        return cls(
            mode=s.controller_mode,  # type: ignore[arg-type]
            stability_threshold=s.stability_threshold,
            stable_chunks=s.stable_chunks,
            max_provisional=s.max_provisional,
        )


class RetrievalController:
    def __init__(self, config: ControllerConfig, probe: StabilityProbe, gate: PresentationGate):
        if config.mode not in ("rule_only", "rule_stability"):
            raise ValueError(f"unknown controller mode {config.mode!r}")
        if config.stable_chunks < 1:
            raise ValueError("stable_chunks must be >= 1")
        self.config = config
        self.probe = probe
        self.gate = gate
        self.start_utterance(None)

    @property
    def name(self) -> str:
        return self.config.mode

    # ------------------------------------------------------------------ state

    def start_utterance(self, prior_output: str | None) -> None:
        """Reset per-utterance state. `prior_output` is the session's previous output, if any."""
        self.prior_output = prior_output
        self.probe.reset()
        self.streak = 0
        self.provisional_count = 0
        self.retrieved_terms: set[str] | None = None
        self.last_probe: ProbeResult | None = None

    def _terms(self, transcript: str) -> set[str]:
        slots, ents, _ = self.probe.content(transcript)
        return set(slots) | {e.lower() for e in ents}

    def _decision(self, action, reason, ts, trigger: Trigger | None = None) -> ControllerDecision:
        score = self.last_probe.stability_score if self.last_probe is not None else None
        return ControllerDecision(action=action, reason=reason, trigger=trigger, ts=ts, stability_score=score)

    def _retrieve(self, transcript: str, ts: float, trigger: Trigger, reason: str) -> ControllerDecision:
        self.retrieved_terms = self._terms(transcript)
        return self._decision("retrieve", reason, ts, trigger)

    # ------------------------------------------------------------------ events

    async def on_chunk(self, transcript: str, ts: float) -> ControllerDecision:
        gate = self.gate.evaluate(transcript, self.prior_output)
        if gate.presentation_only:
            self.streak = 0
            return self._decision("suppress", gate.reason, ts)

        p = self.last_probe = self.probe.probe(transcript, ts)
        if p.incomplete or not p.sufficient:
            self.streak = 0
            return self._decision("wait", "incomplete_fragment" if p.incomplete else "insufficient_content", ts)

        holds = True if self.config.mode == "rule_only" else p.stability_score >= self.config.stability_threshold
        self.streak = self.streak + 1 if holds else 0
        if not holds:
            return self._decision("wait", "low_stability", ts)
        if self.streak < self.config.stable_chunks:
            return self._decision("wait", "stabilizing", ts)
        if self.provisional_count >= self.config.max_provisional:
            return self._decision("wait", "provisional_limit", ts)
        self.provisional_count += 1
        return self._retrieve(transcript, ts, "provisional", "stable_intent")

    async def on_utterance_end(self, transcript: str, ts: float) -> ControllerDecision:
        gate = self.gate.evaluate(transcript, self.prior_output)
        if gate.presentation_only:
            return self._decision("suppress", gate.reason, ts)
        slots, ents, n_tokens = self.probe.content(transcript)
        n_content = len(slots) + len(ents)
        thin = n_content < self.probe.min_content_terms or n_tokens < self.probe.min_tokens
        incomplete = is_incomplete(transcript)
        if n_content == 0 and not incomplete and n_tokens >= self.probe.min_tokens:
            # a complete request, but none of its words occur in the corpus: nothing to search
            return self._decision("suppress", "no_corpus_terms", ts)
        if n_content == 0 or (incomplete and thin):
            return self._decision("suppress", "insufficient_content", ts)
        if self.retrieved_terms is None:
            return self._retrieve(transcript, ts, "final", "utterance_end")
        new = self._terms(transcript) - self.retrieved_terms
        if new:
            return self._retrieve(transcript, ts, "final", "new_content_since_last_retrieval")
        return self._decision("wait", "no_new_content", ts)


def build_controller(stack, settings: Settings | None = None, mode: str | None = None, **overrides):
    """Controller wired to the corpus BM25 index, configured from settings.

    mode: rule_stability | rule_only | llm (the last needs PRISM_LLM_PROVIDER configured)."""
    s = settings or get_settings()
    if (mode or s.controller_mode) == "llm":
        client = make_llm_client(s)
        if client is None:
            raise ValueError("controller mode 'llm' needs an LLM (set PRISM_LLM_PROVIDER / PRISM_LLM_MODEL)")
        return LLMController(client, max_provisional=s.max_provisional)
    cfg = ControllerConfig.from_settings(s)
    if mode is not None:
        overrides["mode"] = mode
    if overrides:
        cfg = ControllerConfig(**{**cfg.__dict__, **overrides})
    probe = StabilityProbe(stack.bm25, k=s.probe_k, min_content_terms=s.probe_min_content_terms,
                           min_tokens=s.probe_min_tokens)
    gate = PresentationGate(stack.bm25.vocabulary, threshold=s.gate_threshold)
    return RetrievalController(cfg, probe, gate)


# ---------------------------------------------------------------------- LLM controller (ablation)

class LLMControllerDecision(BaseModel):
    action: Literal["wait", "retrieve", "suppress"]
    reason: str = ""


LLM_CONTROLLER_SYSTEM = (
    "You control retrieval for a live voice assistant that answers from a document corpus. "
    "Given the user's transcript so far, choose one action: 'wait' if the request is still "
    "incomplete or unclear; 'retrieve' if the information need is clear enough to search now; "
    "'suppress' if the user only asks to reformat, shorten, repeat or translate the previous "
    "answer, or if there is nothing to search. Reply with JSON only."
)


def llm_controller_prompt(transcript: str, finished: bool, has_prior_output: bool) -> str:
    return (
        f"Transcript so far: {transcript!r}\n"
        f"Utterance finished: {'yes' if finished else 'no'}\n"
        f"A previous assistant answer exists in this session: {'yes' if has_prior_output else 'no'}\n"
        'Return JSON: {"action": "wait" | "retrieve" | "suppress", "reason": "<few words>"}'
    )


class LLMController:
    """Model-based controller for the 2.7 ablation: one LLM call per chunk (via src/llm/client.py).

    It keeps the same bookkeeping as the rule controller (at most `max_provisional`
    provisional retrievals; a final retrieval only if the transcript grew since the last
    one). On an LLM error it waits during the stream and retrieves at the utterance end.
    """

    name = "llm"

    def __init__(self, client: LLMClient, max_provisional: int = 1):
        self.client = client
        self.max_provisional = max_provisional
        self.calls = 0
        self.start_utterance(None)

    def start_utterance(self, prior_output: str | None) -> None:
        self.prior_output = prior_output
        self.provisional_count = 0
        self.retrieved_transcript: str | None = None

    async def _ask(self, transcript: str, finished: bool) -> LLMControllerDecision | None:
        self.calls += 1
        prompt = llm_controller_prompt(transcript, finished, bool(self.prior_output))
        try:
            resp = await self.client.generate(prompt, schema=LLMControllerDecision, system=LLM_CONTROLLER_SYSTEM)
        except LLMError:
            return None
        return resp.parsed

    async def on_chunk(self, transcript: str, ts: float) -> ControllerDecision:
        d = await self._ask(transcript, finished=False)
        if d is None:
            return ControllerDecision(action="wait", reason="llm_error", ts=ts)
        if d.action == "retrieve":
            if self.provisional_count >= self.max_provisional:
                return ControllerDecision(action="wait", reason="provisional_limit", ts=ts)
            self.provisional_count += 1
            self.retrieved_transcript = transcript
            return ControllerDecision(action="retrieve", reason=f"llm:{d.reason}"[:200], trigger="provisional", ts=ts)
        return ControllerDecision(action=d.action, reason=f"llm:{d.reason}"[:200], ts=ts)

    async def on_utterance_end(self, transcript: str, ts: float) -> ControllerDecision:
        d = await self._ask(transcript, finished=True)
        action = "retrieve" if d is None else d.action
        reason = "llm_error" if d is None else f"llm:{d.reason}"[:200]
        if action != "retrieve":
            return ControllerDecision(action=action if action == "suppress" else "wait", reason=reason, ts=ts)
        if self.retrieved_transcript is not None and len(transcript.split()) <= len(self.retrieved_transcript.split()):
            return ControllerDecision(action="wait", reason="no_new_content", ts=ts)
        self.retrieved_transcript = transcript
        return ControllerDecision(action="retrieve", reason=reason, trigger="final", ts=ts)
