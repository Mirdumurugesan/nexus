"""
NEXUS LangGraph — the multi-agent orchestration graph.

Flow:
  planner → engineer → reviewer ──(pass)──→ [END]
                           └──(fail, <2x)──→ reflector → reviewer → ...

This is the core of NEXUS Phase 2-4.
"""
from langgraph.graph import StateGraph, END
from app.agents.state import NexusState
from app.agents.planner import run_planner
from app.agents.engineer import run_engineer
from app.agents.reviewer import run_reviewer
from app.agents.reflector import run_reflector, MAX_REFLECTIONS


def should_reflect(state: NexusState) -> str:
    """
    Conditional edge: after reviewing, decide if we're done or need reflection.
    """
    if state.get("review_passed", False):
        return "done"
    if state.get("reflection_count", 0) >= MAX_REFLECTIONS:
        return "done"  # exhausted reflections, accept best result
    return "reflect"


def build_nexus_graph() -> StateGraph:
    """Build and compile the NEXUS multi-agent LangGraph."""
    graph = StateGraph(NexusState)

    # Add nodes
    graph.add_node("planner", run_planner)
    graph.add_node("engineer", run_engineer)
    graph.add_node("reviewer", run_reviewer)
    graph.add_node("reflector", run_reflector)

    # Entry point
    graph.set_entry_point("planner")

    # Edges
    graph.add_edge("planner", "engineer")
    graph.add_edge("engineer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        should_reflect,
        {
            "done": END,
            "reflect": "reflector",
        }
    )
    graph.add_edge("reflector", "reviewer")  # loop back to reviewer

    return graph.compile()


# Singleton compiled graph
_nexus_graph = None


def get_nexus_graph():
    global _nexus_graph
    if _nexus_graph is None:
        _nexus_graph = build_nexus_graph()
    return _nexus_graph


async def run_nexus_pipeline(
    task_id: str,
    issue_title: str,
    issue_body: str,
    repo_name: str,
    repo_url: str,
) -> NexusState:
    """
    Run the full NEXUS multi-agent pipeline.
    Returns the final state with patch, confidence, review score, etc.
    """
    initial_state: NexusState = {
        "task_id": task_id,
        "issue_title": issue_title,
        "issue_body": issue_body,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "plan": [],
        "plan_reasoning": "",
        "retrieved_context": "",
        "patch": "",
        "patch_explanation": "",
        "files_modified": [],
        "confidence": 0.0,
        "root_cause": "",
        "review_score": 0.0,
        "review_feedback": "",
        "review_passed": False,
        "reflection_count": 0,
        "error": "",
        "status": "planning",
    }

    graph = get_nexus_graph()
    final_state = await graph.ainvoke(initial_state)
    return final_state
