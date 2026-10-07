"""
NEXUS LangGraph — the multi-agent control loop.

    planner -> engineer -> reviewer --(passed)--------------------> finalize -> END
                              ^   |--(failed, rounds left)--> reflector
                              |___________________________________|
                                  |--(failed, no rounds left)--> finalize -> END

`finalize` promotes the best attempt seen (ranked by gate-passed, then score),
not simply the last one.
"""
import asyncio
import logging
from langgraph.graph import END, StateGraph

from app.agents.engineer import run_engineer
from app.agents.planner import run_planner
from app.agents.reflector import run_reflector
from app.agents.reviewer import run_reviewer
from app.agents.state import NexusState, initial_state
from app.core import llm
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def budget_exhausted() -> bool:
    usage = llm.current_usage()
    return usage is not None and usage.total_tokens >= get_settings().max_tokens_per_task


def should_reflect(state: NexusState) -> str:
    """Stop on: review pass, reflection cap, or token budget kill-switch."""
    if state.get("review_passed", False):
        return "done"
    if state.get("reflection_count", 0) >= get_settings().max_reflections:
        return "done"
    if budget_exhausted():
        logger.warning("[graph] token budget exhausted; stopping with best patch")
        return "done"
    return "reflect"


def after_reflect(state: NexusState) -> str:
    """A reflector that couldn't reach any model ends the run with the best attempt."""
    return "done" if state.get("status") == "done" else "review"


def finalize(state: NexusState) -> NexusState:
    if state.get("best_patch") and not state.get("review_passed"):
        state = {
            **state,
            "patch": state["best_patch"],
            "files_modified": state.get("best_files", state.get("files_modified", [])),
            "review_score": state.get("best_score", 0.0),
        }
    return {**state, "status": "done"}


def build_nexus_graph():
    g = StateGraph(NexusState)
    g.add_node("planner", run_planner)
    g.add_node("engineer", run_engineer)
    g.add_node("reviewer", run_reviewer)
    g.add_node("reflector", run_reflector)
    g.add_node("finalize", finalize)

    g.set_entry_point("planner")
    g.add_edge("planner", "engineer")
    g.add_edge("engineer", "reviewer")
    g.add_conditional_edges("reviewer", should_reflect, {"done": "finalize", "reflect": "reflector"})
    g.add_conditional_edges("reflector", after_reflect, {"review": "reviewer", "done": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def get_nexus_graph():
    global _graph
    if _graph is None:
        _graph = build_nexus_graph()
    return _graph


def run_graph(**inputs) -> NexusState:
    """Synchronous run (CLI, eval, background worker thread)."""
    return get_nexus_graph().invoke(initial_state(**inputs), {"recursion_limit": 25})


async def run_nexus_pipeline(**inputs) -> NexusState:
    """Async wrapper for FastAPI: run the blocking graph off the event loop."""
    return await asyncio.to_thread(run_graph, **inputs)
