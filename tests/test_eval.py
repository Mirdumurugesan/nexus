"""The evaluator's scoring and its full loop on a local 'SWE-bench-shaped' instance."""
import json

from app.core import llm
from app.demo import CORRECT_PATCH, scripted_llm
from evals import swebench_eval as ev
from tests.conftest import git


def test_split_problem():
    assert ev.split_problem("Title here\nBody line 1\nline 2") == ("Title here", "Body line 1\nline 2")
    assert ev.split_problem("") == ("(untitled issue)", "")


def test_score_instance():
    s = ev.score_instance(CORRECT_PATCH, CORRECT_PATCH, ["tests/x.py", "wealth/sip.py"])
    assert s["file_hit"] and s["file_exact"] and s["retrieval_hit"]
    miss = ev.score_instance("", CORRECT_PATCH, ["wealth/goals.py"])
    assert not miss["file_hit"] and not miss["retrieval_hit"]


def test_full_loop_writes_harness_predictions(wealth_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE", tmp_path / "cache")
    sha = git(wealth_repo, "rev-parse", "HEAD").strip()
    inst = {"instance_id": "demo__wealth-1", "repo": "demo/wealth-lib", "repo_url": wealth_repo,
            "base_commit": sha, "patch": CORRECT_PATCH,
            "problem_statement": "SIP future value is wildly overstated\nannual rate applied monthly"}
    with llm.use_llm(scripted_llm):
        summary = ev.run([inst], str(tmp_path / "out"))

    assert summary["gate_pass_pct"] == 100.0 and summary["file_hit_pct"] == 100.0
    pred = json.loads((tmp_path / "out" / "predictions.jsonl").read_text().strip())
    assert set(pred) == {"instance_id", "model_name_or_path", "model_patch"}
    assert pred["model_patch"].strip() == CORRECT_PATCH.strip()


def test_crash_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE", tmp_path / "cache")
    inst = {"instance_id": "x__y-1", "repo": "x/y", "repo_url": str(tmp_path / "missing"),
            "base_commit": "deadbeef", "problem_statement": "t"}
    summary = ev.run([inst], str(tmp_path / "out"))
    assert summary["completed_pct"] == 0.0 and "error" in summary["rows"][0]
