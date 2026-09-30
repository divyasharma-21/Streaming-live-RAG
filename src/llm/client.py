"""The single LLM interface (step 1.7). Every LLM call in the project goes through here.

    client = make_llm_client(settings)            # None when PRISM_LLM_PROVIDER=none
    resp = await client.generate(prompt, schema=MyModel, trace=trace)
    resp.text, resp.parsed, resp.tokens_in, resp.tokens_out

Providers:
* `ollama`             POST {base_url}/api/generate               (local Ollama server; the
                       playbook's intended runtime is a 7-8B-class instruct model)
* `openai_compatible`  POST {base_url}/chat/completions           (any OpenAI-style endpoint)
* `mock`               scripted responses, for unit tests only (never selected by configuration)

Model, URL, context window, keep-alive, timeout, retries and cost rates all come from
PRISM_LLM_* settings; nothing machine-specific is hard-coded. Transient transport failures
(unreachable host, timeout, HTTP 5xx) are retried with exponential backoff.

`health()` / `check_llm()` / `require_llm()` verify before a run that the server is reachable
and the configured model is pulled, and report its parameter size so a run records whether
it used the intended 7-8B class (`python -m src.llm.client --check`). A configured but
unavailable model is an error, never a silent fallback to the offline path.

The HTTP providers follow the documented request/response shapes and are tested against local
fake HTTP servers (tests/fake_ollama.py). **They have not been run against a real Ollama
server with a 7-8B model in this repository's build environment.**

JSON-constrained output: pass `schema` as a pydantic model class or a JSON-schema dict.
The provider is asked for JSON; the reply is parsed (code fences tolerated) and validated.
On a parse/validation failure the call is retried once with a corrective instruction,
then `LLMOutputError` is raised.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from src.config import Settings, get_settings
from src.telemetry.events import LLM_CALL, estimate_cost_usd
from src.telemetry.logger import RequestTrace

Schema = type[BaseModel] | dict[str, Any] | None

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
JSON_RETRY_NOTE = "\n\nYour previous reply was not valid JSON for the required schema. Reply with JSON only."


class LLMError(RuntimeError):
    pass


class LLMOutputError(LLMError):
    pass


class LLMConnectionError(LLMError):
    """Transport failure (unreachable host, timeout, HTTP 5xx). Retried; `transient` marks it."""

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


@dataclass
class RawCompletion:
    text: str
    tokens_in: int | None
    tokens_out: int | None


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    provider: str
    model: str
    parsed: Any = None
    tokens_estimated: bool = False
    attempts: int = 1


def _estimate_tokens(text: str) -> int:
    """Rough fallback when a provider reports no usage: ~0.75 words per token."""
    return max(1, round(len(text.split()) / 0.75)) if text else 0


def _json_schema(schema: Schema) -> dict[str, Any] | None:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema()


def parse_json_output(text: str, schema: Schema) -> Any:
    """Parse a model reply as JSON and validate it against `schema`."""
    m = _FENCE_RE.match(text)
    body = m.group(1) if m else text.strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise LLMOutputError("reply is not JSON") from None
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"reply is not JSON: {exc}") from None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMOutputError(f"reply does not match schema: {exc.error_count()} errors") from None
    if isinstance(schema, dict):
        if schema.get("type") == "object" and not isinstance(data, dict):
            raise LLMOutputError("reply is not a JSON object")
        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            raise LLMOutputError(f"reply is missing required keys {missing}")
    return data


class LLMClient(ABC):
    provider = "abstract"

    def __init__(self, model: str, settings: Settings | None = None):
        self.model = model
        self.settings = settings or get_settings()

    @abstractmethod
    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion: ...

    async def generate(
        self,
        prompt: str,
        schema: Schema = None,
        system: str | None = None,
        trace: RequestTrace | None = None,
    ) -> LLMResponse:
        js = _json_schema(schema)
        t0 = time.perf_counter()
        tokens_in = tokens_out = 0
        estimated = False
        parsed: Any = None
        attempts = 0
        current_prompt = prompt
        error: LLMOutputError | None = None
        raw = RawCompletion("", 0, 0)
        for attempts in (1, 2):
            raw = await self._complete(current_prompt, system, js)
            if raw.tokens_in is None or raw.tokens_out is None:
                estimated = True
            tokens_in += raw.tokens_in if raw.tokens_in is not None else _estimate_tokens((system or "") + current_prompt)
            tokens_out += raw.tokens_out if raw.tokens_out is not None else _estimate_tokens(raw.text)
            if js is None:
                break
            try:
                parsed = parse_json_output(raw.text, schema)
                error = None
                break
            except LLMOutputError as exc:
                error = exc
                current_prompt = prompt + JSON_RETRY_NOTE
        latency_ms = (time.perf_counter() - t0) * 1000

        if trace is not None:
            trace.emit(
                LLM_CALL,
                stage_latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                est_cost_usd=estimate_cost_usd(
                    tokens_in, tokens_out, self.settings.cost_per_1k_input, self.settings.cost_per_1k_output
                ),
                provider=self.provider,
                model=self.model,
                json_mode=js is not None,
                attempts=attempts,
                tokens_estimated=estimated,
                ok=error is None,
            )
        if error is not None:
            raise error
        return LLMResponse(
            text=raw.text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            provider=self.provider,
            model=self.model,
            parsed=parsed,
            tokens_estimated=estimated,
            attempts=attempts,
        )


# ------------------------------------------------------------------ HTTP helpers


def _request_json(url: str, payload: dict | None, headers: dict[str, str], timeout: float) -> dict:
    """POST `payload` (or GET when None) and decode the JSON reply."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise LLMConnectionError(f"HTTP {exc.code} from {url}: {body}", transient=exc.code >= 500) from None
    except urllib.error.URLError as exc:
        raise LLMConnectionError(f"cannot reach {url}: {exc.reason}") from None
    except (TimeoutError, ConnectionError) as exc:
        raise LLMConnectionError(f"connection to {url} failed: {exc}") from None
    except json.JSONDecodeError as exc:
        raise LLMConnectionError(f"non-JSON reply from {url}: {exc}", transient=False) from None


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    return _request_json(url, payload, headers, timeout)


async def _with_retries(call, retries: int, backoff_s: float = 0.5):
    """Run a blocking HTTP call in a thread; retry transient failures with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return await asyncio.to_thread(call)
        except LLMConnectionError as exc:
            if not exc.transient or attempt == retries:
                raise
            await asyncio.sleep(backoff_s * (2 ** attempt))


# ------------------------------------------------------------------ model health


def parse_parameter_size(value: str | None) -> float | None:
    """'8.0B' -> 8.0, '567M' -> 0.567 (billions of parameters); None if unknown."""
    if not value:
        return None
    m = re.match(r"^\s*([\d.]+)\s*([BbMmKk])", str(value))
    if not m:
        return None
    scale = {"b": 1.0, "m": 1e-3, "k": 1e-6}[m.group(2).lower()]
    return float(m.group(1)) * scale


@dataclass
class ModelHealth:
    provider: str
    model: str
    base_url: str
    reachable: bool
    model_present: bool
    parameter_size_b: float | None = None
    quantization: str | None = None
    in_intended_class: bool | None = None  # None when the server does not report a size
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and self.model_present


def classify_size(size_b: float | None, settings: Settings) -> bool | None:
    if size_b is None:
        return None
    return settings.llm_intended_min_b <= size_b <= settings.llm_intended_max_b


class OllamaClient(LLMClient):
    """Local Ollama server (the playbook's intended runtime: a 7-8B-class instruct model).
    The model tag comes from PRISM_LLM_MODEL; nothing machine-specific is hard-coded."""

    provider = "ollama"

    def __init__(self, model: str, base_url: str, settings: Settings | None = None):
        super().__init__(model, settings)
        self.base_url = base_url.rstrip("/")

    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion:
        s = self.settings
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": s.llm_keep_alive,
            "options": {"temperature": 0, "num_predict": s.llm_max_tokens, "num_ctx": s.llm_num_ctx},
        }
        if system:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = json_schema
        url = f"{self.base_url}/api/generate"
        data = await _with_retries(lambda: _post_json(url, payload, {}, s.llm_timeout_s), s.llm_retries)
        return RawCompletion(data.get("response", ""), data.get("prompt_eval_count"), data.get("eval_count"))

    async def health(self) -> ModelHealth:
        h = ModelHealth(self.provider, self.model, self.base_url, reachable=False, model_present=False)
        try:
            tags = await _with_retries(
                lambda: _request_json(f"{self.base_url}/api/tags", None, {}, min(self.settings.llm_timeout_s, 10)), 0)
        except LLMError as exc:
            h.message = (f"Ollama is not reachable at {self.base_url} ({exc}). Start it (`ollama serve`, or "
                         f"`docker compose up ollama`) or set PRISM_LLM_BASE_URL.")
            return h
        h.reachable = True
        wanted = {self.model, self.model if ":" in self.model else f"{self.model}:latest"}
        entry = next((m for m in tags.get("models", []) if m.get("name") in wanted or m.get("model") in wanted), None)
        if entry is None:
            h.message = f"model {self.model!r} is not pulled on {self.base_url}; run `ollama pull {self.model}`."
            return h
        h.model_present = True
        details = entry.get("details") or {}
        h.parameter_size_b = parse_parameter_size(details.get("parameter_size"))
        h.quantization = details.get("quantization_level")
        h.in_intended_class = classify_size(h.parameter_size_b, self.settings)
        h.message = "ok"
        if h.in_intended_class is False:
            h.message = (f"ok, but {self.model} reports {details.get('parameter_size')} parameters, outside the "
                         f"intended {self.settings.llm_intended_min_b:g}-{self.settings.llm_intended_max_b:g}B class")
        return h


class OpenAICompatibleClient(LLMClient):
    provider = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "", settings: Settings | None = None):
        super().__init__(model, settings)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.settings.llm_max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"
        data = await _with_retries(lambda: _post_json(url, payload, headers, self.settings.llm_timeout_s),
                                   self.settings.llm_retries)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return RawCompletion(text, usage.get("prompt_tokens"), usage.get("completion_tokens"))

    async def health(self) -> ModelHealth:
        return await _openai_health(self)


async def _openai_health(client: OpenAICompatibleClient) -> ModelHealth:
    h = ModelHealth(client.provider, client.model, client.base_url, reachable=False, model_present=False)
    headers = {"Authorization": f"Bearer {client.api_key}"} if client.api_key else {}
    try:
        data = await asyncio.to_thread(
            _request_json, f"{client.base_url}/models", None, headers, min(client.settings.llm_timeout_s, 10))
    except LLMError as exc:
        h.message = f"endpoint {client.base_url} is not reachable ({exc}); check PRISM_LLM_BASE_URL / PRISM_LLM_API_KEY."
        return h
    h.reachable = True
    ids = {m.get("id") for m in data.get("data", []) if isinstance(m, dict)}
    h.model_present = client.model in ids or not ids  # some servers do not list models
    h.message = "ok (parameter size not reported by this API)" if h.model_present else \
        f"model {client.model!r} not offered by {client.base_url}"
    return h


class MockLLMClient(LLMClient):
    """Returns scripted replies. For tests only; never selected by default."""

    provider = "mock"

    def __init__(self, replies: list[str] | Callable[[str], str], settings: Settings | None = None):
        super().__init__("mock", settings)
        self._replies = replies
        self.prompts: list[str] = []

    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion:
        self.prompts.append(prompt)
        if callable(self._replies):
            return RawCompletion(self._replies(prompt), None, None)
        if not self._replies:
            raise LLMError("MockLLMClient ran out of scripted replies")
        return RawCompletion(self._replies.pop(0), None, None)


def make_llm_client(settings: Settings | None = None) -> LLMClient | None:
    """Build the configured client, or None when PRISM_LLM_PROVIDER=none."""
    s = settings or get_settings()
    provider = s.llm_provider.lower()
    if provider == "none":
        return None
    if not s.llm_model:
        raise LLMError(f"PRISM_LLM_PROVIDER={provider} requires PRISM_LLM_MODEL")
    if provider == "ollama":
        return OllamaClient(s.llm_model, s.llm_base_url, settings=s)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(s.llm_model, s.llm_base_url, s.llm_api_key, settings=s)
    raise LLMError(f"unknown PRISM_LLM_PROVIDER {provider!r} (expected none, ollama or openai_compatible)")


async def acheck_llm(settings: Settings | None = None) -> ModelHealth | None:
    """Health of the configured LLM, or None when PRISM_LLM_PROVIDER=none."""
    client = make_llm_client(settings)
    if client is None:
        return None
    return await client.health()


async def arequire_llm(settings: Settings | None = None) -> ModelHealth | None:
    """Fail fast before a run: a configured but unreachable / unpulled model is an error, never a
    silent fallback to the offline path."""
    health = await acheck_llm(settings)
    if health is not None and not health.ok:
        raise LLMError(health.message)
    return health


def check_llm(settings: Settings | None = None) -> ModelHealth | None:
    """Synchronous `acheck_llm` (for CLIs; do not call from inside an event loop)."""
    return asyncio.run(acheck_llm(settings))


def require_llm(settings: Settings | None = None) -> ModelHealth | None:
    """Synchronous `arequire_llm` (for CLIs; do not call from inside an event loop)."""
    return asyncio.run(arequire_llm(settings))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check the configured LLM (PRISM_LLM_* settings).")
    parser.add_argument("--check", action="store_true", help="contact the server and verify the model")
    args = parser.parse_args()
    s = get_settings()
    print(f"provider={s.llm_provider} model={s.llm_model or '-'} base_url={s.llm_base_url}")
    if s.llm_provider == "none":
        print("LLM disabled (PRISM_LLM_PROVIDER=none): the offline extractive path is used.")
        return 0
    if not args.check:
        return 0
    try:
        health = check_llm(s)
    except LLMError as exc:
        print(f"error: {exc}")
        return 1
    print(f"reachable={health.reachable} model_present={health.model_present} "
          f"parameter_size_b={health.parameter_size_b} quantization={health.quantization} "
          f"intended_7_8b_class={health.in_intended_class}")
    print(health.message)
    return 0 if health.ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
