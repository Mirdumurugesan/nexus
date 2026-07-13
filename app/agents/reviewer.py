"""
Reviewer Agent — scores the generated patch and decides if it needs reflection.
This is Phase 3: quality gate before finalizing.
"""
import logging
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.core.llm import unpack_structured
from app.agents.state import NexusState

logger = logging.getLogger(__name__)
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
    logger.info("[reviewer] Reviewing patch (confidence was %.2f)", state.get("confidence", 0))

    llm = ChatOpenAI(
        model=settings.primary_llm,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_retries=2,
    ).with_structured_output(ReviewOutput, include_raw=True)

    raw_result = llm.invoke([
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
    result, in_tok, out_tok = unpack_structured(raw_result)

    logger.info("[reviewer] Score: %.2f | Passed: %s", result.score, result.passed)
    return {
        **state,
        "review_score": result.score,
        "review_feedback": result.feedback,
        "review_passed": result.passed,
        "review_issues_found": list(result.issues_found or []),
        "prompt_tokens": state.get("prompt_tokens", 0) + in_tok,
        "completion_tokens": state.get("completion_tokens", 0) + out_tok,
        "status": "done" if result.passed else "reflecting",
    }
