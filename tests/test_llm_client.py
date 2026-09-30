"""LLM client (step 1.7). The HTTP providers are tested against a local stub server that
imitates the documented response shapes; no real Ollama or hosted model is contacted."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from src.config import Settings
from src.llm.client import (
    LLMError,
    LLMOutputError,
    MockLLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    make_llm_client,
    parse_json_output,
)
from src.telemetry.logger import TelemetryLogger


class Reply(BaseModel):
    answer: str
    n: int


async def test_plain_text_generation_with_estimated_tokens():
    client = MockLLMClient(["hello there world"])
    resp = await client.generate("say hi")
    assert resp.text == "hello there world"
    assert resp.tokens_estimated and resp.tokens_in > 0 and resp.tokens_out > 0
    assert resp.parsed is None


async def test_json_mode_with_pydantic_schema_and_code_fence():
    client = MockLLMClient(['```json\n{"answer": "ok", "n": 2}\n```'])
    resp = await client.generate("q", schema=Reply)
    assert resp.parsed == Reply(answer="ok", n=2)


async def test_json_mode_retries_once_then_succeeds():
    client = MockLLMClient(["not json", '{"answer": "ok", "n": 1}'])
    resp = await client.generate("q", schema=Reply)
    assert resp.attempts == 2 and resp.parsed.n == 1
    assert "Reply with JSON only" in client.prompts[1]


async def test_json_mode_raises_after_two_failures():
    client = MockLLMClient(["nope", '{"answer": "missing n"}'])
    with pytest.raises(LLMOutputError):
        await client.generate("q", schema=Reply)


def test_parse_json_output_with_dict_schema():
    schema = {"type": "object", "required": ["a"]}
    assert parse_json_output('text before {"a": 1} after', schema) == {"a": 1}
    with pytest.raises(LLMOutputError):
        parse_json_output('{"b": 1}', schema)


async def test_llm_call_is_logged_with_tokens(tmp_path):
    log = tmp_path / "t.jsonl"
    trace = TelemetryLogger(log).trace(request_id="r1")
    await MockLLMClient(['{"answer": "a", "n": 1}']).generate("p", schema=Reply, trace=trace)
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["event"] == "llm_call" and rec["request_id"] == "r1"
    assert rec["tokens_in"] > 0 and rec["tokens_out"] > 0
    assert rec["data"]["json_mode"] is True and rec["data"]["ok"] is True


def test_factory():
    assert make_llm_client(Settings(llm_provider="none")) is None
    assert isinstance(make_llm_client(Settings(llm_provider="ollama", llm_model="m")), OllamaClient)
    assert isinstance(
        make_llm_client(Settings(llm_provider="openai_compatible", llm_model="m")), OpenAICompatibleClient
    )
    with pytest.raises(LLMError):
        make_llm_client(Settings(llm_provider="ollama", llm_model=""))
    with pytest.raises(LLMError):
        make_llm_client(Settings(llm_provider="other", llm_model="m"))


# ------------------------------------------------------------------ stub HTTP server


class _Stub(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append((self.path, dict(self.headers), body))
        if self.path == "/api/generate":
            out = {"response": '{"answer": "x", "n": 3}', "prompt_eval_count": 11, "eval_count": 7}
        elif self.path == "/v1/chat/completions":
            out = {
                "choices": [{"message": {"content": '{"answer": "y", "n": 4}'}}],
                "usage": {"prompt_tokens": 13, "completion_tokens": 5},
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_url():
    _Stub.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


async def test_ollama_request_and_usage_parsing(stub_url):
    client = OllamaClient("some-model", stub_url, settings=Settings(llm_timeout_s=5))
    resp = await client.generate("prompt", schema=Reply, system="sys")
    assert resp.parsed.n == 3 and (resp.tokens_in, resp.tokens_out) == (11, 7) and not resp.tokens_estimated
    path, _, body = _Stub.requests[0]
    assert path == "/api/generate" and body["stream"] is False and body["system"] == "sys"
    assert body["format"]["required"] == ["answer", "n"]


async def test_openai_compatible_request_and_usage_parsing(stub_url):
    client = OpenAICompatibleClient("some-model", stub_url + "/v1", api_key="k", settings=Settings(llm_timeout_s=5))
    resp = await client.generate("prompt", schema=Reply)
    assert resp.parsed.n == 4 and (resp.tokens_in, resp.tokens_out) == (13, 5)
    path, headers, body = _Stub.requests[0]
    assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer k"
    assert body["response_format"] == {"type": "json_object"}


async def test_unreachable_server_raises_llm_error(stub_url):
    client = OllamaClient("m", stub_url + "/missing", settings=Settings(llm_timeout_s=5))
    with pytest.raises(LLMError):
        await client.generate("p")
