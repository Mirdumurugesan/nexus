from dataclasses import dataclass


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


def rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion contribution for a 0-based rank."""
    return 1.0 / (k + rank + 1)
