"""
Planner Agent — decomposes a GitHub issue into 2-4 ordered subtasks.
"""
import logging
import uuid

from pydantic import BaseModel, Field

from app.core import llm
from app.agents.state import NexusState, SubTask

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a senior software engineering planner.
Given a GitHub issue, decompose the fix into 2-4 concrete subtasks.

Each subtask must:
- Be a single, atomic code change
- Reference the likely file to modify
- Be ordered by dependency (do task 1 before task 2)"""


class PlanItem(BaseModel):
    description: str = Field(description="One atomic code change")
    file_hint: str = Field(default="unknown", description="Most likely file path to modify")


class PlannerOutput(BaseModel):
    reasoning: str = Field(description="One sentence: what is the core problem?")
    subtasks: list[PlanItem] = Field(description="2-4 ordered subtasks")


def run_planner(state: NexusState) -> NexusState:
    logger.info(f"[planner] Planning fix for: {state['issue_title']}")
    result: PlannerOutput = llm.call(
        "planner",
        PLANNER_SYSTEM,
        f"Issue Title: {state['issue_title']}\n"
        f"Issue Body: {state['issue_body'][:1500]}\n"
        f"Repository: {state['repo_name']}\n\nPlan the fix:",
        schema=PlannerOutput,
    )

    plan: list[SubTask] = []
    for st in result.subtasks[:4]:
        item = st if isinstance(st, PlanItem) else PlanItem(**st)
        plan.append(SubTask(
            id=uuid.uuid4().hex[:8],
            description=item.description,
            file_hint=item.file_hint or "unknown",
            status="pending",
        ))

    logger.info(f"[planner] {len(plan)} subtasks")
    return {**state, "plan": plan, "plan_reasoning": result.reasoning, "status": "engineering"}
