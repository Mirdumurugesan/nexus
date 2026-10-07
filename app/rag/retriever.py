"""
Hybrid retrieval front door.

    RETRIEVER=local     in-memory BM25 (content + symbols) + HyDE, fused with RRF.
                        Zero infrastructure; what the tests, CLI and eval use.
    RETRIEVER=weaviate  Weaviate BM25 + OpenAI-embedding vector search + HyDE, RRF.

Both paths return the same RetrievedChunk list, so agents don't care which ran.
HyDE is best-effort: if the LLM call fails, retrieval continues without it.
"""
from __future__ import annotations
import logging

from app.core import llm
from app.core.config import get_settings
from app.rag.chunker import CodeChunk
from app.rag.types import RetrievedChunk, rrf

logger = logging.getLogger(__name__)

HYDE_SYSTEM = "You are an expert Python developer. Reply with code only, no explanation."


def generate_hyde(issue_title: str, issue_body: str) -> str:
    """HyDE: ask for a hypothetical fix; embed/search with that instead of prose."""
    prompt = (
        "Write a SHORT hypothetical Python function (5-15 lines) that would fix or "
        f"relate to this GitHub issue.\n\nTitle: {issue_title}\nBody: {issue_body[:500]}\n"
    )
    try:
        return llm.call("hyde", HYDE_SYSTEM, prompt, cheap=True) or ""
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[retriever] HyDE skipped: {type(e).__name__}")
        return ""


def index_repository(chunks: list[CodeChunk], repo_name: str) -> int:
    if get_settings().retriever == "weaviate":
        from app.rag.embedder import index_chunks
        return index_chunks(chunks, repo_name=repo_name)
    from app.rag.local_index import build_local_index
    return build_local_index(chunks, repo_name)


def hybrid_retrieve(
    issue_title: str,
    issue_body: str,
    repo_name: str,
    top_k: int = 10,
    use_hyde: bool = True,
) -> list[RetrievedChunk]:
    hyde = generate_hyde(issue_title, issue_body) if use_hyde else ""
    query = f"{issue_title}\n{issue_body[:1500]}"

    if get_settings().retriever == "weaviate":
        return _weaviate_retrieve(query, repo_name, top_k, hyde)

    from app.rag.local_index import get_local_index
    index = get_local_index(repo_name)
    if index is None:
        raise RuntimeError(f"Repository {repo_name!r} has not been indexed")
    return index.search(query, top_k=top_k, hyde_code=hyde)


def _weaviate_retrieve(query: str, repo_name: str, top_k: int, hyde: str) -> list[RetrievedChunk]:
    import weaviate.classes as wvc
    from app.rag.embedder import COLLECTION_NAME, embed_texts, get_weaviate_client

    props = ["chunk_id", "file_path", "name", "chunk_type", "content", "start_line", "end_line"]
    client = get_weaviate_client()
    try:
        col = client.collections.get(COLLECTION_NAME)
        flt = wvc.query.Filter.by_property("repo_name").equal(repo_name)
        bm25 = col.query.bm25(query=query[:600], limit=top_k * 2, filters=flt, return_properties=props)
        vec = embed_texts([hyde or query])[0]
        dense = col.query.near_vector(near_vector=vec, limit=top_k * 2, filters=flt, return_properties=props)

        scores: dict[str, float] = {}
        data: dict[str, dict] = {}
        for ranking in (bm25.objects, dense.objects):
            for rank, obj in enumerate(ranking):
                cid = obj.properties["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + rrf(rank)
                data[cid] = obj.properties

        out = []
        for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            p = data[cid]
            out.append(RetrievedChunk(
                chunk_id=cid, file_path=p.get("file_path", ""), name=p.get("name", ""),
                chunk_type=p.get("chunk_type", ""), content=p.get("content", ""),
                start_line=int(p.get("start_line", 0)), end_line=int(p.get("end_line", 0)),
                rrf_score=score,
            ))
        return out
    finally:
        client.close()


def format_context_for_llm(chunks: list[RetrievedChunk], max_tokens: int = 6000) -> str:
    """Format retrieved chunks for the LLM, with real line numbers so diffs anchor correctly."""
    parts: list[str] = []
    total = 0
    limit = max_tokens * 4  # ~4 chars per token
    for c in chunks:
        numbered = "\n".join(
            f"{c.start_line + i:>5} | {line}" for i, line in enumerate(c.content.split("\n"))
        )
        section = (
            f"### {c.file_path} :: {c.chunk_type} {c.name} (lines {c.start_line}-{c.end_line})\n"
            f"```python\n{numbered}\n```\n"
        )
        if total + len(section) > limit:
            break
        parts.append(section)
        total += len(section)
    return "\n".join(parts)
