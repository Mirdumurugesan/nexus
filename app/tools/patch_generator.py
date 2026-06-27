"""
Patch Generator — the LLM call that produces the actual code fix.
Uses structured output (Pydantic) to enforce a consistent response shape.
Primary model: GPT-4o. Fallback: Groq LLaMA 3.
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.rag.retriever import RetrievedChunk, format_context_for_llm

settings = get_settings()


class GeneratedPatch(BaseModel):
    """Structured output from the patch generator."""
    root_cause: str = Field(description="One sentence explaining the root cause of the bug")
    approach: str = Field(description="One sentence explaining the fix approach")
    patch: str = Field(description="The complete code fix as a unified diff or full file content")
    files_modified: list[str] = Field(description="List of file paths that need to be changed")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    test_hint: str = Field(description="Suggestion for what to test to verify the fix")


SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a GitHub issue.

You will receive:
1. The issue title and description
2. Relevant code context retrieved from the repository

Your job:
1. Identify the root cause
2. Generate a precise, minimal code fix
3. Only change what is necessary — do not refactor unrelated code
4. Return a unified diff format patch when possible

Rules:
- Be precise and conservative — minimal changes only
- Preserve existing code style (indentation, naming conventions)
- If you cannot confidently fix the issue, set confidence < 0.5
- Do not hallucinate function names or imports that don't exist
"""


def generate_patch(
    issue_title: str,
    issue_body: str,
    retrieved_chunks: list[RetrievedChunk],
) -> GeneratedPatch:
    """
    Generate a code patch for the given issue using retrieved context.
    Falls back to Groq if OpenAI fails.
    """
    context = format_context_for_llm(retrieved_chunks, max_tokens=6000)

    user_message = f"""## GitHub Issue

**Title:** {issue_title}

**Description:**
{issue_body[:1500]}

---

## Relevant Code Context

{context}

---

Based on the issue and code context above, generate a fix.
"""

    # Try primary LLM (GPT-4o)
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(GeneratedPatch)
        result = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        return result

    except Exception as e:
        print(f"[patch_generator] Primary LLM failed: {e}. Falling back to Groq.")

    # Fallback LLM (Groq LLaMA 3)
    llm_fallback = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0.1,
    )
    structured_llm_fallback = llm_fallback.with_structured_output(GeneratedPatch)
    return structured_llm_fallback.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])