"""Dense retrieval interface (step 1.6).

Two embedder backends behind one `Embedder` protocol:

* `HashingEmbedder` (default): deterministic feature hashing of word unigrams, bigrams and
  character trigrams into a fixed-size vector. numpy only, no download, a few MB of RAM.
  It is **not** a semantic model: it captures lexical and sub-word overlap, not meaning.
  It exists so the pipeline runs anywhere and the interface is exercised by tests.
* `SentenceTransformerEmbedder`: a real embedding model (default `BAAI/bge-small-en-v1.5`).
  Requires `requirements-optional.txt`; not installed or tested in Phase 1.

Search is exact inner product over L2-normalised vectors (cosine). With `use_faiss=True`
and `faiss-cpu` installed, a FAISS `IndexFlatIP` is used instead (same results, not tested
in Phase 1). Embeddings can be cached on disk keyed by chunk ids + texts.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from src.schemas import CorpusChunk, ScoredChunk

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feat in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))  # stable across processes, unlike hash()
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, v in counts.items():
                mat[row, idx] = np.sign(v) * np.log1p(abs(v))
        return _normalise(mat)


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency, not installed in Phase 1
    def __init__(self, model_name: str, batch_size: int = 16):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "PRISM_DENSE_BACKEND=sentence_transformers needs `pip install -r requirements-optional.txt`"
            ) from exc
        self.name = f"st-{model_name}"
        self._model = SentenceTransformer(model_name, device="cpu")
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=self._batch_size, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(backend: str, model_name: str = "") -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown dense backend {backend!r} (expected 'hashing' or 'sentence_transformers')")


def _cache_key(embedder: Embedder, chunks: list[CorpusChunk]) -> str:
    h = hashlib.sha256(embedder.name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        chunks: list[CorpusChunk],
        embedder: Embedder,
        cache_dir: Path | None = None,
        use_faiss: bool = False,
    ):
        if not chunks:
            raise ValueError("DenseRetriever needs at least one chunk")
        self.chunks = chunks
        self.embedder = embedder
        self.cache_hit = False
        self.matrix = self._load_or_embed(cache_dir)
        self._faiss = None
        if use_faiss:  # pragma: no cover - optional dependency
            import faiss

            self._faiss = faiss.IndexFlatIP(self.matrix.shape[1])
            self._faiss.add(self.matrix)

    def _load_or_embed(self, cache_dir: Path | None) -> np.ndarray:
        if cache_dir is None:
            return self.embedder.embed([c.text for c in self.chunks])
        key = _cache_key(self.embedder, self.chunks)
        npy, meta = cache_dir / f"dense_{key}.npy", cache_dir / f"dense_{key}.json"
        if npy.exists() and meta.exists():
            self.cache_hit = True
            return np.load(npy)
        mat = self.embedder.embed([c.text for c in self.chunks])
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy, mat)
        meta.write_text(json.dumps({"embedder": self.embedder.name, "chunk_ids": [c.chunk_id for c in self.chunks]}))
        return mat

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not query.strip():
            return []
        q = self.embedder.embed([query])
        k = min(k, len(self.chunks))
        if self._faiss is not None:  # pragma: no cover
            scores, idx = self._faiss.search(q, k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = (self.matrix @ q[0]).tolist()
            order = sorted(range(len(sims)), key=lambda i: (-sims[i], self.chunks[i].chunk_id))[:k]
            pairs = [(i, sims[i]) for i in order]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(s), rank=r, source=self.name)
            for r, (i, s) in enumerate(pairs, start=1)
            if s > 0
        ]
