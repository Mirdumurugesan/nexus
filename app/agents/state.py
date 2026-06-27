"""
NEXUS Agent State — the shared data structure that flows through LangGraph.
Every node reads from and writes to this TypedDict.
"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class SubTask(TypedDict):
    id: str
    description: str
    file_hint: str       # which file likely needs changing
    status: str          # pending | done | failed


class NexusState(TypedDict):
    # Input
    task_id: str
    issue_title: str
    issue_body: str
    repo_name: str
    repo_url: str

    # Planning
    plan: list[SubTask]
    plan_reasoning: str

    # RAG
    retrieved_context: str          # formatted string of retrieved chunks

    # Patch
    patch: str
    patch_explanation: str
    files_modified: list[str]
    confidence: float
    root_cause: str

    # Review
    review_score: float             # 0.0–1.0
    review_feedback: str
    review_passed: bool

    # Control flow
    reflection_count: int           # how many times we've reflected
    error: str
    status: str                     # planning | engineering | reviewing | reflecting | done | failed
