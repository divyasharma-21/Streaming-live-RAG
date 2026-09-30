"""A deterministic fake Ollama HTTP server, FOR TESTS AND WIRING CHECKS ONLY.

It implements the two endpoints the project uses (`GET /api/tags`, `POST /api/generate`) and
answers each JSON-schema request with a plausible, schema-valid reply derived mechanically
from the prompt (e.g. a synthesis claim = the first evidence sentence of each sub-question).
It exercises the real HTTP client, retries, token accounting and every LLM code path, but it
is not a language model: numbers produced with it measure wiring, never answer quality.

    python tests/fake_ollama.py --port 11500 --model llama3.1:8b --size 8.0B
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def _first_sentence(body: str) -> str:
    for line in body.splitlines():
        line = line.strip().strip("|").strip()
        if not line or line.startswith(("```", "---", "|---")) or set(line) <= set("-|: "):
            continue
        line = line.replace("**", "").replace("`", "")
        return re.split(r"(?<=[.!?])\s", line, maxsplit=1)[0][:300]
    return ""


def fake_reply(prompt: str, system: str, schema: dict | None) -> str:
    props = set((schema or {}).get("properties", {}))
    if "ops" in props:  # decomposer: split the transcript at '?' / ', and', keep matching live ids
        transcript = re.search(r"Transcript so far: '(.*?)'\n", prompt, re.S)
        text = transcript.group(1) if transcript else ""
        live = json.loads(re.search(r"Current sub-queries: (\[.*?\])\n", prompt, re.S).group(1)) if "Current sub-queries" in prompt else []
        clauses = [c.strip(" ,") for c in re.split(r"\?\s+|,\s*and\s+(?=what|which|how|who|why|when|where)", text) if c.strip(" ,")]
        ops, used = [], set()
        for clause in clauses:
            match = next((q for q in live if q["id"] not in used and _terms(q["text"]) <= _terms(clause)), None)
            if match:
                used.add(match["id"])
                ops.append({"op": "keep", "id": match["id"], "text": clause})
            else:
                ops.append({"op": "add", "text": clause})
        return json.dumps({"ops": ops})
    if "claims" in props and "uncertain" in props:  # per-sub-intent synthesis
        claims, uncertain = [], []
        for sid, block in re.findall(r'<subquestion id="([^"]+)">(.*?)</subquestion>', prompt, re.S):
            chunk = re.search(r'<chunk id="([^"]+)">\n(.*?)\n</chunk>', block, re.S)
            if chunk is None:
                uncertain.append({"subintent_id": sid, "note": "no evidence supplied"})
                continue
            body = chunk.group(2).split("\n\n", 1)[-1]
            claims.append({"subintent_id": sid, "text": _first_sentence(body), "chunk_ids": [chunk.group(1)]})
        return json.dumps({"claims": claims, "uncertain": uncertain})
    if "claims" in props:  # single-request synthesis (baseline)
        chunk = re.search(r'<chunk id="([^"]+)">\n(.*?)\n</chunk>', prompt, re.S)
        claims = [{"text": _first_sentence(chunk.group(2).split("\n\n", 1)[-1]), "chunk_ids": [chunk.group(1)]}] if chunk else []
        return json.dumps({"claims": claims, "uncertainty": None if claims else "no evidence"})
    if "supported" in props:  # grounding judge: lexical containment
        evidence = prompt.split("Evidence:", 1)[-1].split("Claim:", 1)[0]
        claim = prompt.split("Claim:", 1)[-1].split("Return JSON", 1)[0]
        ct = _terms(claim)
        ok = bool(ct) and len(ct & _terms(evidence)) / len(ct) >= 0.8
        return json.dumps({"supported": ok, "reason": "terms found in evidence" if ok else "terms missing"})
    if "action" in props:  # controller
        finished = "Utterance finished: yes" in prompt
        return json.dumps({"action": "retrieve" if finished else "wait", "reason": "fake"})
    if "text" in props:  # presentation transform: return the answer unchanged (citations intact)
        return json.dumps({"text": prompt.split("Answer:\n", 1)[-1].split("\n\nReturn JSON", 1)[0]})
    return "fake completion"


class FakeOllama:
    def __init__(self, model: str = "llama3.1:8b", size: str = "8.0B", fail_first: int = 0, port: int = 0):
        self.model, self.size, self.fail_first = model, size, fail_first
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, obj: dict) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                outer.requests.append((self.path, {}))
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": outer.model, "model": outer.model,
                                                 "details": {"parameter_size": outer.size,
                                                             "quantization_level": "Q4_K_M"}}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self._send(500, {"error": "injected failure"})
                    return
                if self.path != "/api/generate":
                    self._send(404, {"error": "not found"})
                    return
                if body.get("model") != outer.model:
                    self._send(404, {"error": f"model '{body.get('model')}' not found"})
                    return
                schema = body.get("format") if isinstance(body.get("format"), dict) else None
                reply = fake_reply(body.get("prompt", ""), body.get("system", ""), schema)
                self._send(200, {"model": outer.model, "response": reply, "done": True,
                                 "prompt_eval_count": len(body.get("prompt", "").split()),
                                 "eval_count": len(reply.split())})

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FakeOllama:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=11500)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--size", default="8.0B")
    a = parser.parse_args()
    fake = FakeOllama(a.model, a.size, port=a.port)
    print(f"fake Ollama (TEST ONLY) serving {a.model} ({a.size}) at {fake.url}", flush=True)
    fake.server.serve_forever()
