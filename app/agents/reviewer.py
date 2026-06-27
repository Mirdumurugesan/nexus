"""
Reviewer Agent — scores the generated patch and decides if it needs reflection.
This is Phase 3: quality gate before finalizing.
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.agents.state import NexusState

settings = get_settings()

REVIEWER_SYSTEM = """You are a senior code reviewer.
Review the generated patch for a GitHub issue.

Score the patch on:
1. Correctness (does it actually fix the issue?)
2. Completeness (are all cases handled?)
3. Safety (no regressions or side effects?)
4. Style (matches existing code conventions?)

Be strict. A score below 0.7 means the patch needs improvement."""


class ReviewOutput(BaseModel):
    score: float = Field(description="Overall quality score 0.0-1.0", ge=0.0, le=1.0)
    passed: bool = Field(description="True if patch is good enough to ship (score >= 0.7)")
    feedback: str = Field(description="Specific, actionable feedback for improvement if score < 0.7")
    issues_found: list[str] = Field(description="List of specific problems found (empty if passed)")


def run_reviewer(state: NexusState) -> NexusState:
    """LangGraph node: review the patch quality."""
    print(f"[reviewer] Reviewing patch (confidence was {state.get('confidence', 0):.2f})")

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=settings.openai_api_key,
        temperature=0.1,
    ).with_structured_output(ReviewOutput)

    result = llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"""
## GitHub Issue
Title: {state['issue_title']}
Body: {state['issue_body'][:800]}

## Generated Patch
{state.get('patch', 'No patch generated')}

## Root Cause Analysis
{state.get('root_cause', 'Not provided')}

## Files Modified
{', '.join(state.get('files_modified', []))}

Review this patch:"""),
    ])

    print(f"[reviewer] Score: {result.score:.2f} | Passed: {result.passed}")
    return {
        **state,
        "review_score": result.score,
        "review_feedback": result.feedback,
        "review_passed": result.passed,
        "status": "done" if result.passed else "reflecting",
    }
