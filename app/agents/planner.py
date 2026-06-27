"""
Planner Agent — decomposes a GitHub issue into ordered subtasks.
Uses GPT-4o with structured output to produce a deterministic plan.
"""
import uuid
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.agents.state import NexusState, SubTask

settings = get_settings()

PLANNER_SYSTEM = """You are a senior software engineering planner.
Given a GitHub issue, decompose the fix into 2-4 concrete subtasks.

Each subtask must:
- Be a single, atomic code change
- Reference the likely file to modify
- Be ordered by dependency (do task 1 before task 2)

Return ONLY a JSON object — no extra text."""


class PlannerOutput(BaseModel):
    reasoning: str = Field(description="One sentence: what is the core problem?")
    subtasks: list[dict] = Field(description="List of {description, file_hint} dicts, 2-4 items")


def run_planner(state: NexusState) -> NexusState:
    """LangGraph node: plan the fix for the issue."""
    print(f"[planner] Planning fix for: {state['issue_title']}")

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=settings.openai_api_key,
        temperature=0.2,
    ).with_structured_output(PlannerOutput)

    result = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"""
Issue Title: {state['issue_title']}
Issue Body: {state['issue_body'][:1000]}
Repository: {state['repo_name']}

Plan the fix:"""),
    ])

    subtasks: list[SubTask] = [
        SubTask(
            id=str(uuid.uuid4())[:8],
            description=st["description"],
            file_hint=st.get("file_hint", "unknown"),
            status="pending",
        )
        for st in result.subtasks
    ]

    print(f"[planner] Created {len(subtasks)} subtasks")
    return {
        **state,
        "plan": subtasks,
        "plan_reasoning": result.reasoning,
        "status": "engineering",
    }
