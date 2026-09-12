"""Dense, BM25 and hybrid retrieval over the chunk index, with an optional cross-encoder rerank.

The index is plain files (chunks.jsonl + dense.npy) so a run is reproducible and nothing
needs a database. Embeddings and the reranker run locally through fastembed (ONNX).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import Chunk, read_chunks, save_chunks

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def passage(c: Chunk) -> str:
    return f"{c.title}\n{c.text}"


def build_index(chunks: list[Chunk], out: Path, model: str = EMBED_MODEL) -> None:
    from fastembed import TextEmbedding

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    emb = TextEmbedding(model_name=model)
    vectors = np.array(list(emb.embed([passage(c) for c in chunks], batch_size=64)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(out / "dense.npy", vectors)
    save_chunks(chunks, out / "chunks.jsonl")
    (out / "model.txt").write_text(model, encoding="utf-8")


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: each list votes 1/(k+rank) for its items."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, index_dir: Path, mode: str = "dense", k: int = 5, rerank: bool = False, candidates: int = 20):
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown retrieval mode: {mode}")
        index_dir = Path(index_dir)
        self.chunks = read_chunks(index_dir / "chunks.jsonl")
        self.vectors = np.load(index_dir / "dense.npy")
        self.mode, self.k, self.rerank, self.candidates = mode, k, rerank, candidates
        self._embedder = None
        self._reranker = None
        self._bm25 = None
        if mode in ("bm25", "hybrid"):
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([tokenize(passage(c)) for c in self.chunks])
        model_file = index_dir / "model.txt"
        self.embed_model = model_file.read_text(encoding="utf-8").strip() if model_file.exists() else EMBED_MODEL

    @property
    def name(self) -> str:
        return f"{self.mode}{'+rerank' if self.rerank else ''}@{self.k}"

    def _dense(self, query: str, n: int) -> list[int]:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self.embed_model)
        q = np.array(next(iter(self._embedder.query_embed([query]))), dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self.vectors @ q
        return list(np.argsort(-sims)[:n])

    def _lexical(self, query: str, n: int) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def search(self, query: str) -> list[Hit]:
        n = self.candidates if self.rerank else self.k
        if self.mode == "dense":
            ids = self._dense(query, n)
        elif self.mode == "bm25":
            ids = self._lexical(query, n)
        else:
            ids = rrf([self._dense(query, self.candidates), self._lexical(query, self.candidates)])[:n]
        if not self.rerank:
            return [Hit(self.chunks[i], 1.0 / (r + 1)) for r, i in enumerate(ids[: self.k])]
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(model_name=RERANK_MODEL)
        scores = list(self._reranker.rerank(query, [passage(self.chunks[i]) for i in ids]))
        order = sorted(range(len(ids)), key=lambda j: -scores[j])[: self.k]
        return [Hit(self.chunks[ids[j]], float(scores[j])) for j in order]
