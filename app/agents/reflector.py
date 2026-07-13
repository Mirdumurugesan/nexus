"""
Reflector Agent — Phase 4: self-improvement loop.
If the reviewer rejects the patch, the reflector analyzes the feedback
and generates an improved patch. Max 2 reflection rounds.
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

MAX_REFLECTIONS = 2

REFLECTOR_SYSTEM = """You are an expert software engineer improving a rejected code patch.

You will receive:
1. The original GitHub issue
2. The rejected patch
3. Specific reviewer feedback on what was wrong

Your job: generate an improved patch that addresses ALL the reviewer's concerns.
Be precise and minimal."""


class ReflectorOutput(BaseModel):
    improved_patch: str = Field(description="The improved patch in unified diff format")
    changes_made: str = Field(description="What you changed vs the previous patch and why")
    new_confidence: float = Field(description="Your confidence in the improved patch 0.0-1.0", ge=0.0, le=1.0)


def run_reflector(state: NexusState) -> NexusState:
    """LangGraph node: improve the patch based on reviewer feedback."""
    reflection_count = state.get("reflection_count", 0) + 1
    logger.info("[reflector] Reflection round %d/%d", reflection_count, MAX_REFLECTIONS)

    if reflection_count > MAX_REFLECTIONS:
        # Give up after max reflections — use best patch we have
        logger.info("[reflector] Max reflections reached. Using current patch.")
        return {
            **state,
            "reflection_count": reflection_count,
            "status": "done",
        }

    llm = ChatOpenAI(
        model=settings.primary_llm,
        api_key=settings.openai_api_key,
        temperature=0.2,
        max_retries=2,
    ).with_structured_output(ReflectorOutput, include_raw=True)

    raw_result = llm.invoke([
        SystemMessage(content=REFLECTOR_SYSTEM),
        HumanMessage(content=f"""
## GitHub Issue
Title: {state['issue_title']}
Body: {state['issue_body'][:800]}

## Previous (Rejected) Patch
{state.get('patch', 'No patch')}

## Reviewer Feedback (why it was rejected)
Score: {state.get('review_score', 0):.2f}/1.0
Feedback: {state.get('review_feedback', 'No feedback')}
Issues: {', '.join(state.get('review_issues_found', []))}

## Code Context (relevant files)
{state.get('retrieved_context', '')[:3000]}

Generate an improved patch:"""),
    ])
    result, in_tok, out_tok = unpack_structured(raw_result)

    logger.info("[reflector] Improved patch generated. New confidence: %.2f", result.new_confidence)
    return {
        **state,
        "patch": result.improved_patch,
        "patch_explanation": state.get("patch_explanation", "") + f"\n[Reflection {reflection_count}]: {result.changes_made}",
        "confidence": result.new_confidence,
        "reflection_count": reflection_count,
        "prompt_tokens": state.get("prompt_tokens", 0) + in_tok,
        "completion_tokens": state.get("completion_tokens", 0) + out_tok,
        "status": "reviewing",  # go back to reviewer
    }
