"""
Engineer Agent — generates the actual code patch using retrieved context.
Runs after the Planner has created a plan and RAG has retrieved context.
"""
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.agents.state import NexusState
from app.rag.retriever import hybrid_retrieve, format_context_for_llm

settings = get_settings()

ENGINEER_SYSTEM = """You are an expert software engineer fixing a GitHub issue.

You have:
1. A detailed plan with subtasks
2. Relevant code context from the repository

Generate a complete, working patch in unified diff format.
Be minimal — only change what is necessary to fix the issue.
Preserve existing code style exactly."""


class EngineerOutput(BaseModel):
    root_cause: str = Field(description="One sentence: the root cause of the bug")
    approach: str = Field(description="One sentence: how the patch fixes it")
    patch: str = Field(description="Complete patch in unified diff format (--- a/file +++ b/file)")
    files_modified: list[str] = Field(description="List of file paths modified")
    confidence: float = Field(description="0.0-1.0 confidence in the fix", ge=0.0, le=1.0)
    test_hint: str = Field(description="What to test to verify the fix works")


def run_engineer(state: NexusState) -> NexusState:
    """LangGraph node: generate the patch using RAG context + plan."""
    print(f"[engineer] Generating patch for: {state['issue_title']}")

    # RAG retrieval
    retrieved = hybrid_retrieve(
        issue_title=state["issue_title"],
        issue_body=state["issue_body"],
        repo_name=state["repo_name"],
        top_k=12,
        use_hyde=True,
    )
    context = format_context_for_llm(retrieved, max_tokens=6000)

    # Format plan for LLM
    plan_text = "\n".join(
        f"{i+1}. [{st['file_hint']}] {st['description']}"
        for i, st in enumerate(state.get("plan", []))
    )

    user_message = f"""## GitHub Issue
Title: {state['issue_title']}
Body: {state['issue_body'][:1200]}

## Plan
{plan_text}

## Relevant Code Context
{context}

Generate the unified diff patch:"""

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=0.1,
        ).with_structured_output(EngineerOutput)
        result = llm.invoke([
            SystemMessage(content=ENGINEER_SYSTEM),
            HumanMessage(content=user_message),
        ])
    except Exception as e:
        print(f"[engineer] GPT-4o failed: {e}. Falling back to Groq.")
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.1,
        ).with_structured_output(EngineerOutput)
        result = llm.invoke([
            SystemMessage(content=ENGINEER_SYSTEM),
            HumanMessage(content=user_message),
        ])

    print(f"[engineer] Patch generated. Confidence: {result.confidence:.2f}")
    return {
        **state,
        "retrieved_context": context,
        "patch": result.patch,
        "patch_explanation": f"Root cause: {result.root_cause}\nApproach: {result.approach}\nTest: {result.test_hint}",
        "files_modified": result.files_modified,
        "confidence": result.confidence,
        "root_cause": result.root_cause,
        "status": "reviewing",
    }
