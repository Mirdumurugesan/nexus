"""
NEXUS Agent State — the shared data structure that flows through LangGraph.
Every node reads from and writes to this TypedDict.
"""
from typing import TypedDict


class SubTask(TypedDict):
    id: str
    description: str
    file_hint: str       # which file likely needs changing
    status: str          # pending | done | failed


class Attempt(TypedDict):
    round: int           # 0 = engineer, 1..n = reflector rounds
    agent: str
    gate_passed: bool
    gate_summary: str
    review_score: float
    passed: bool


class NexusState(TypedDict, total=False):
    # Input
    task_id: str
    issue_title: str
    issue_body: str
    repo_name: str
    repo_url: str
    repo_path: str                  # local checkout the patch gate applies against
    index_key: str                  # retrieval index id (defaults to repo_name)
    use_hyde: bool

    # Planning
    plan: list[SubTask]
    plan_reasoning: str

    # RAG
    retrieved_context: str
    retrieved_files: list[str]

    # Current attempt: search/replace edits -> NEXUS-built unified diff
    edits: list[dict]
    edit_errors: list[str]
    patch: str
    patch_explanation: str
    files_modified: list[str]
    confidence: float
    root_cause: str

    # Review (deterministic gate + LLM)
    gate: dict
    review_score: float             # 0.0–1.0, forced to 0 when the gate fails
    review_feedback: str
    review_issues: list[str]
    review_passed: bool

    # Best attempt so far — a reflection that makes things worse never wins
    best_patch: str
    best_score: float
    best_gate_passed: bool
    best_explanation: str
    best_files: list[str]

    # Control flow / trace
    reflection_count: int
    history: list[Attempt]
    error: str
    status: str                     # planning | engineering | reviewing | reflecting | done | failed


def initial_state(**kw) -> NexusState:
    base: NexusState = {
        "task_id": "", "issue_title": "", "issue_body": "", "repo_name": "",
        "repo_url": "", "repo_path": "", "index_key": "", "use_hyde": True,
        "plan": [], "plan_reasoning": "",
        "retrieved_context": "", "retrieved_files": [],
        "edits": [], "edit_errors": [],
        "patch": "", "patch_explanation": "", "files_modified": [],
        "confidence": 0.0, "root_cause": "",
        "gate": {}, "review_score": 0.0, "review_feedback": "", "review_issues": [],
        "review_passed": False,
        "best_patch": "", "best_score": -1.0, "best_gate_passed": False,
        "best_explanation": "", "best_files": [],
        "reflection_count": 0, "history": [], "error": "", "status": "planning",
    }
    base.update(kw)
    return base
