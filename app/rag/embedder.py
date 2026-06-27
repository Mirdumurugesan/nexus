"""
Embedding + Weaviate indexing pipeline.
Uses OpenAI text-embedding-3-small for cost efficiency in Phase 1.
Weaviate stores both the vector and the raw content for BM25 hybrid search.
"""
import weaviate
import weaviate.classes as wvc
from openai import OpenAI
from app.core.config import get_settings
from app.rag.chunker import CodeChunk

settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)

COLLECTION_NAME = "CodeChunk"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_weaviate_client() -> weaviate.WeaviateClient:
    from weaviate.auth import AuthApiKey
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.weaviate_url,
        auth_credentials=AuthApiKey(settings.weaviate_api_key),
    )


def create_collection_if_not_exists(client: weaviate.WeaviateClient):
    """Create Weaviate collection with hybrid search enabled."""
    if client.collections.exists(COLLECTION_NAME):
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # we supply vectors ourselves
        properties=[
            wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True, index_filterable=True),
            wvc.config.Property(name="file_path", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True, index_filterable=True),
            wvc.config.Property(name="repo_name", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True, index_filterable=True),
            wvc.config.Property(name="language", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True),
            wvc.config.Property(name="chunk_type", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True),
            wvc.config.Property(name="name", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="start_line", data_type=wvc.config.DataType.INT,
                                skip_vectorization=True),
            wvc.config.Property(name="end_line", data_type=wvc.config.DataType.INT,
                                skip_vectorization=True),
            wvc.config.Property(name="imports", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True),
            wvc.config.Property(name="called_functions", data_type=wvc.config.DataType.TEXT,
                                skip_vectorization=True),
        ],
        inverted_index_config=wvc.config.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
        ),
    )
    print(f"[embedder] Created Weaviate collection: {COLLECTION_NAME}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using OpenAI. Max 2048 texts per call."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def index_chunks(chunks: list[CodeChunk], repo_name: str) -> int:
    """
    Embed and upload all chunks to Weaviate.
    Returns: number of chunks successfully indexed.
    """
    client = get_weaviate_client()

    try:
        create_collection_if_not_exists(client)
        collection = client.collections.get(COLLECTION_NAME)

        # Delete existing chunks for this repo (re-indexing)
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("repo_name").equal(repo_name)
        )

        # Batch embed (process in groups of 100 for API rate limits)
        BATCH_SIZE = 100
        total_indexed = 0

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts_to_embed = [
                f"File: {c.file_path}\nFunction: {c.name}\n\n{c.content}"
                for c in batch
            ]

            vectors = embed_texts(texts_to_embed)

            with collection.batch.dynamic() as batch_writer:
                for chunk, vector in zip(batch, vectors):
                    obj = chunk.to_weaviate_object()
                    obj["repo_name"] = repo_name
                    batch_writer.add_object(properties=obj, vector=vector)

            total_indexed += len(batch)
            print(f"[embedder] Indexed {total_indexed}/{len(chunks)} chunks")

        return total_indexed

    finally:
        client.close()