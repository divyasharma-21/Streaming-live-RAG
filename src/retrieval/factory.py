"""Build all retrievers from settings. Indexes live in memory and are rebuilt at start-up
from data/corpus/ (dense embeddings are cached in PRISM_INDEX_DIR)."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, get_settings
from src.corpus.chunker import load_chunks
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, make_embedder
from src.retrieval.hybrid import HybridRetriever
from src.schemas import CorpusChunk


@dataclass
class RetrievalStack:
    chunks: list[CorpusChunk]
    bm25: BM25Retriever
    dense: DenseRetriever
    hybrid: HybridRetriever

    def by_name(self, name: str):
        return {"bm25": self.bm25, "dense": self.dense, "hybrid": self.hybrid}[name]


def build_retrieval(settings: Settings | None = None, chunks: list[CorpusChunk] | None = None) -> RetrievalStack:
    s = settings or get_settings()
    chunks = chunks if chunks is not None else load_chunks(s)
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, make_embedder(s.dense_backend, s.dense_model), cache_dir=s.index_dir)
    hybrid = HybridRetriever([bm25, dense], rrf_k=s.rrf_k, candidates=s.candidates)
    return RetrievalStack(chunks=chunks, bm25=bm25, dense=dense, hybrid=hybrid)
