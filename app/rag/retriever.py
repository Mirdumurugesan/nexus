"""
Hybrid RAG retriever:
  1. BM25 keyword search  (great for exact function names, variable names)
  2. Vector semantic search (great for conceptual similarity)
  3. RRF score fusion      (combines both rankings)
  4. HyDE query expansion  (generate hypothetical code, then search with that)
"""
import weaviate
import weaviate.classes as wvc
from openai import OpenAI
from dataclasses import dataclass
from app.core.config import get_settings
from app.rag.embedder import get_weaviate_client, embed_texts, COLLECTION_NAME

settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)


@dataclass
class RetrievedChunk:
    chunk_id: str
    file_path: str
    name: str
    chunk_type: str
    content: str
    start_line: int
    end_line: int
    rrf_score: float


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a result at position `rank`."""
    return 1.0 / (k + rank)


def _generate_hyde_query(issue_title: str, issue_body: str) -> str:
    """
    HyDE: Generate a hypothetical Python function that would fix this issue.
    Embed this instead of the raw question — closer to actual code in embedding space.
    """
    prompt = f"""You are an expert Python developer.
Given this GitHub issue, write a SHORT hypothetical Python function 
(5-15 lines) that would fix or relate to this issue.
Do NOT add explanation — only code.

Issue Title: {issue_title}
Issue Body: {issue_body[:500]}

Hypothetical fix:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def hybrid_retrieve(
    issue_title: str,
    issue_body: str,
    repo_name: str,
    top_k: int = 10,
    use_hyde: bool = True,
) -> list[RetrievedChunk]:
    """
    Main retrieval function.
    Combines BM25 + vector search with RRF fusion.
    Optionally uses HyDE for better vector query.
    """
    client = get_weaviate_client()

    try:
        collection = client.collections.get(COLLECTION_NAME)
        repo_filter = wvc.query.Filter.by_property("repo_name").equal(repo_name)

        # ── Step 1: BM25 keyword search ──────────────────────────────
        bm25_query = f"{issue_title} {issue_body[:300]}"
        bm25_results = collection.query.bm25(
            query=bm25_query,
            limit=top_k * 2,
            filters=repo_filter,
            return_properties=["chunk_id", "file_path", "name", "chunk_type",
                               "content", "start_line", "end_line"],
        )

        # ── Step 2: Vector semantic search ───────────────────────────
        if use_hyde:
            hyde_code = _generate_hyde_query(issue_title, issue_body)
            query_for_embedding = hyde_code
        else:
            query_for_embedding = f"{issue_title}\n{issue_body[:400]}"

        query_vector = embed_texts([query_for_embedding])[0]

        vector_results = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k * 2,
            filters=repo_filter,
            return_properties=["chunk_id", "file_path", "name", "chunk_type",
                               "content", "start_line", "end_line"],
        )

        # ── Step 3: RRF Fusion ────────────────────────────────────────
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for rank, obj in enumerate(bm25_results.objects):
            cid = obj.properties["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank)
            chunk_data[cid] = obj.properties

        for rank, obj in enumerate(vector_results.objects):
            cid = obj.properties["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank)
            chunk_data[cid] = obj.properties

        # ── Step 4: Sort by RRF score and return top_k ────────────────
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            props = chunk_data[chunk_id]
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                file_path=props.get("file_path", ""),
                name=props.get("name", ""),
                chunk_type=props.get("chunk_type", ""),
                content=props.get("content", ""),
                start_line=int(props.get("start_line", 0)),
                end_line=int(props.get("end_line", 0)),
                rrf_score=score,
            ))

        return results

    finally:
        client.close()


def format_context_for_llm(chunks: list[RetrievedChunk], max_tokens: int = 6000) -> str:
    """Format retrieved chunks into a clean context string for the LLM."""
    context_parts = []
    total_chars = 0
    char_limit = max_tokens * 4  # rough chars-to-tokens ratio

    for chunk in chunks:
        section = (
            f"### File: {chunk.file_path} | {chunk.chunk_type}: {chunk.name} "
            f"(lines {chunk.start_line}–{chunk.end_line})\n"
            f"```python\n{chunk.content}\n```\n"
        )
        if total_chars + len(section) > char_limit:
            break
        context_parts.append(section)
        total_chars += len(section)

    return "\n".join(context_parts)