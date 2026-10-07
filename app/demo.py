"""
Offline demo: NEXUS fixes a real bug in a small wealth-planning library.

The bug (demo/fixture/wealth/sip.py): SIP future value compounds the *annual*
rate every month, overstating a 10-year, 12% SIP by ~3x. Two tests fail.

What is real here:   git checkout, AST chunking, BM25 hybrid retrieval, the
                     LangGraph loop, `git apply --check`, the compile check and
                     the pytest run inside the patch gate.
What is scripted:    the LLM replies (unless --live). The first engineer edit
                     deliberately hallucinates the code it replaces (a variable
                     named `rate` that doesn't exist), the most common way LLM
                     edits fail. Watch the gate reject it with the closest real
                     line and the reflector repair it.

    python -m app.cli demo           # scripted LLM, no keys, ~2 seconds
    python -m app.cli demo --live    # same repo + gate, real OpenAI/Groq calls
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from app.core import llm
from app.core.config import get_settings
from app.pipeline import SolveResult, solve_issue

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "demo", "fixture")

ISSUE_TITLE = "SIP future value is wildly overstated"
ISSUE_BODY = (
    "future_value(10000, 12, 10) returns about 75 lakh, but a 10,000/month SIP at 12% "
    "for 10 years should be roughly 23.2 lakh. required_monthly_sip() is wrong for the "
    "same reason. Looks like the yearly rate is being applied every month."
)

HALLUCINATED_PATCH = '''--- a/wealth/sip.py
+++ b/wealth/sip.py
@@ -9,5 +9,5 @@
 def future_value(monthly_amount: float, annual_rate_pct: float, years: int) -> float:
     """Future value of a monthly SIP."""
-    rate = annual_rate_pct / 100
+    rate = annual_rate_pct / 12 / 100
     n = years * 12
     if rate == 0:
'''

CORRECT_PATCH = '''--- a/wealth/sip.py
+++ b/wealth/sip.py
@@ -9,6 +9,6 @@
 def future_value(monthly_amount: float, annual_rate_pct: float, years: int) -> float:
     """Future value of a monthly SIP with contributions at the start of each month."""
-    r = annual_rate_pct / 100
+    r = monthly_rate(annual_rate_pct)
     n = years * 12
     if r == 0:
         return round(monthly_amount * n, 2)
'''


# What the agents actually return: search/replace edits. NEXUS builds the diff.
HALLUCINATED_EDIT = {"file": "wealth/sip.py",
                     "search": "    rate = annual_rate_pct / 100",
                     "replace": "    rate = annual_rate_pct / 12 / 100"}
CORRECT_EDIT = {"file": "wealth/sip.py",
                "search": "    r = annual_rate_pct / 100",
                "replace": "    r = monthly_rate(annual_rate_pct)"}


def scripted_llm(role, schema, system, user):
    """Deterministic stand-in for the model, keyed by agent role."""
    if role == "hyde":
        return "def future_value(p, annual_rate_pct, years):\n    r = monthly_rate(annual_rate_pct)\n    n = years * 12"
    if role == "planner":
        return schema(
            reasoning="future_value compounds the annual rate monthly instead of the monthly rate.",
            subtasks=[
                {"description": "Use monthly_rate() for the per-period rate in future_value",
                 "file_hint": "wealth/sip.py"},
                {"description": "Confirm required_monthly_sip inherits the fix (it calls future_value)",
                 "file_hint": "wealth/sip.py"},
            ],
        )
    if role == "engineer":
        return schema(
            root_cause="future_value uses annual_rate_pct/100 as a monthly rate.",
            approach="Derive the per-month rate via monthly_rate().",
            edits=[HALLUCINATED_EDIT],
            confidence=0.9,
            test_hint="future_value(10000, 12, 10) ≈ 23.23 lakh",
        )
    if role == "reflector":
        return schema(
            edits=[CORRECT_EDIT],
            changes_made="Search block now copied verbatim from wealth/sip.py (the variable is `r`).",
            new_confidence=0.95,
        )
    if role == "reviewer":
        return schema(
            score=0.93,
            feedback="Minimal fix at the root cause; required_monthly_sip is fixed transitively.",
            issues_found=[],
        )
    raise ValueError(f"no scripted reply for role {role!r}")


def _make_repo(dest: str) -> str:
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=demo@nexus", "-c", "user.name=nexus", "commit", "-qm", "fixture"]):
        subprocess.run(["git", *cmd], cwd=dest, check=True)
    return dest


def run_demo(live: bool = False) -> SolveResult:
    s = get_settings()
    previous = s.gate_test_command
    s.gate_test_command = f'"{sys.executable}" -m pytest -q -p no:cacheprovider'
    tmp = tempfile.mkdtemp(prefix="nexus-demo-")
    try:
        repo = _make_repo(os.path.join(tmp, "wealth-lib"))
        kwargs = dict(issue_title=ISSUE_TITLE, issue_body=ISSUE_BODY,
                      repo_name="demo/wealth-lib", repo_path=repo)
        if live:
            if not llm.available():
                raise SystemExit("--live needs OPENAI_API_KEY or GROQ_API_KEY in .env")
            return solve_issue(**kwargs)
        with llm.use_llm(scripted_llm):
            return solve_issue(**kwargs)
    finally:
        s.gate_test_command = previous
        shutil.rmtree(tmp, ignore_errors=True)
