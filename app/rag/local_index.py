"""
Local hybrid retriever — no vector DB, no API key, no network.

Three rankings fused with Reciprocal Rank Fusion (same k=60 as the Weaviate path):

  1. BM25 over chunk content  — issue text vs. code
  2. BM25 over symbols        — file path + function/class name only. Issues often
                                name the function that breaks; this ranking lets an
                                exact symbol hit outrank a long file that merely
                                mentions the same words.
  3. BM25 over content, queried with HyDE code (a hypothetical fix written by the
     LLM) when available — code-shaped queries match code-shaped documents.

Identifiers are split on snake_case and camelCase *and* kept whole, so
"merge_environment_settings" matches both the exact name and "environment settings".
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.rag.chunker import CodeChunk
from app.rag.types import RetrievedChunk, rrf

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with", "as",
    "be", "this", "that", "are", "was", "not", "but", "if", "when", "by", "at", "from",
    "self", "def", "return", "none", "true", "false", "import", "class", "pass", "should",
    "would", "can", "we", "i", "you", "my", "me", "have", "has", "do", "does",
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text or ""):
        low = word.lower()
        parts = [p.lower() for piece in word.split("_") if piece for p in _CAMEL.findall(piece)]
        if len(parts) > 1 and low not in _STOP:
            out.append(low)  # keep the whole identifier as a strong token
        out.extend(p for p in parts if len(p) > 1 and p not in _STOP)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = set(query)
        res = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            res.append(s)
        return res

    def rank(self, query: list[str], limit: int) -> list[int]:
        sc = self.scores(query)
        order = sorted((i for i, s in enumerate(sc) if s > 0), key=lambda i: sc[i], reverse=True)
        return order[:limit]


class LocalIndex:
    def __init__(self, chunks: list[CodeChunk]):
        self.chunks = chunks
        self.content = BM25([tokenize(f"{c.file_path} {c.name}\n{c.content}") for c in chunks])
        self.symbols = BM25([tokenize(f"{c.file_path.replace('/', ' ')} {c.name}") for c in chunks])

    def search(self, query: str, top_k: int = 10, hyde_code: str = "") -> list[RetrievedChunk]:
        q = tokenize(query)
        rankings = [self.content.rank(q, top_k * 3), self.symbols.rank(q, top_k * 3)]
        if hyde_code:
            rankings.append(self.content.rank(tokenize(hyde_code), top_k * 3))

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + rrf(rank)

        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for idx, score in best:
            c = self.chunks[idx]
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id, file_path=c.file_path, name=c.name,
                chunk_type=c.chunk_type, content=c.content,
                start_line=c.start_line, end_line=c.end_line, rrf_score=score,
            ))
        return out


_REGISTRY: dict[str, LocalIndex] = {}


def build_local_index(chunks: list[CodeChunk], repo_name: str) -> int:
    _REGISTRY[repo_name] = LocalIndex(chunks)
    return len(chunks)


def get_local_index(repo_name: str) -> LocalIndex | None:
    return _REGISTRY.get(repo_name)


def drop_local_index(repo_name: str) -> None:
    _REGISTRY.pop(repo_name, None)
