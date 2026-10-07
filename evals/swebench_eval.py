"""
NEXUS × SWE-bench Lite
──────────────────────
Runs the real pipeline (solve_issue, not the HTTP API) on SWE-bench Lite
instances, each checked out at its exact `base_commit`.

What this script measures itself (cheap, no Docker):
  completed        pipeline ran without crashing
  gate_pass        final patch applies at base_commit and compiles
  file_hit         patch edits at least one file the gold patch edits
  retrieval_hit    a gold file was among the retrieved context files

What it does NOT claim: "% resolved". That needs each repo's test environment.
Instead it writes predictions in the official format, so you can score them
with the SWE-bench harness:

    python evals/swebench_eval.py --limit 10
    python -m swebench.harness.run_evaluation \\
        --dataset_name princeton-nlp/SWE-bench_Lite \\
        --predictions_path evals/out/predictions.jsonl \\
        --max_workers 4 --run_id nexus

Instances come from HuggingFace (`pip install datasets`) or a local JSONL via
--instances, with the same fields as the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.patch_gate import files_in_patch  # noqa: E402

CACHE = Path(os.environ.get("NEXUS_REPO_CACHE", Path.home() / ".cache" / "nexus" / "repos"))


# ── data ──────────────────────────────────────────────────────────────────────

def load_instances(limit: int, path: str | None = None, ids: list[str] | None = None) -> list[dict]:
    if path:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset  # optional dependency
        rows = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
    if ids:
        rows = [r for r in rows if r["instance_id"] in set(ids)]
    return rows[:limit]


def split_problem(problem: str) -> tuple[str, str]:
    """SWE-bench problem_statement = issue title + body in one string."""
    lines = (problem or "").strip().split("\n", 1)
    title = lines[0].strip()[:300] or "(untitled issue)"
    body = lines[1].strip() if len(lines) > 1 else ""
    return title, body


# ── scoring ───────────────────────────────────────────────────────────────────

def score_instance(pred_patch: str, gold_patch: str, retrieved_files: list[str]) -> dict:
    pred = set(files_in_patch(pred_patch or ""))
    gold = set(files_in_patch(gold_patch or ""))
    return {
        "pred_files": sorted(pred),
        "gold_files": sorted(gold),
        "file_hit": bool(pred & gold),
        "file_exact": bool(gold) and pred == gold,
        "retrieval_hit": bool(gold & set(retrieved_files)),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    pct = lambda k: round(100 * sum(1 for r in rows if r.get(k)) / n, 1)  # noqa: E731
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    return {
        "instances": len(rows),
        "completed_pct": pct("completed"),
        "gate_pass_pct": pct("gate_passed"),
        "verified_pct": pct("passed"),
        "file_hit_pct": pct("file_hit"),
        "retrieval_hit_pct": pct("retrieval_hit"),
        "avg_reflections": round(sum(r.get("reflections", 0) for r in rows) / n, 2),
        "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in rows) / n, 1),
        "median_seconds": sorted(secs)[len(secs) // 2] if secs else None,
    }


# ── checkout ──────────────────────────────────────────────────────────────────

def checkout_at(repo: str, base_commit: str, dest: str, repo_url: str | None = None) -> str:
    """Full clone cached once per repo, then a cheap --shared clone per instance."""
    url = repo_url or f"https://github.com/{repo}.git"
    mirror = CACHE / repo.replace("/", "__")
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", url, str(mirror)], check=True)
    subprocess.run(["git", "clone", "-q", "--shared", str(mirror), dest], check=True)
    r = subprocess.run(["git", "checkout", "-q", base_commit], cwd=dest)
    if r.returncode != 0:  # commit newer than the cache
        subprocess.run(["git", "fetch", "-q", "origin"], cwd=str(mirror), check=True)
        subprocess.run(["git", "fetch", "-q", "origin"], cwd=dest, check=True)
        subprocess.run(["git", "checkout", "-q", base_commit], cwd=dest, check=True)
    return dest


# ── main loop ─────────────────────────────────────────────────────────────────

def run(instances: list[dict], out_dir: str, model_name: str = "nexus", use_hyde: bool = True) -> dict:
    import shutil
    import tempfile

    from app.pipeline import solve_issue

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path, rows = out / "predictions.jsonl", []

    with open(pred_path, "w", encoding="utf-8") as preds:
        for i, inst in enumerate(instances, 1):
            iid = inst["instance_id"]
            print(f"\n[{i}/{len(instances)}] {iid}")
            title, body = split_problem(inst["problem_statement"])
            row: dict = {"instance_id": iid, "repo": inst["repo"], "completed": False}
            tmp = tempfile.mkdtemp(prefix="nexus-eval-")
            t0 = time.time()
            try:
                path = checkout_at(inst["repo"], inst["base_commit"], os.path.join(tmp, "repo"),
                                   inst.get("repo_url"))
                r = solve_issue(title, body, inst["repo"], repo_path=path, use_hyde=use_hyde, task_id=iid)
                row.update(
                    completed=True, passed=r.passed, gate_passed=r.gate_passed,
                    review_score=r.review_score, reflections=r.reflections,
                    llm_calls=r.llm_calls, seconds=r.seconds,
                    **score_instance(r.patch, inst.get("patch", ""), r.retrieved_files),
                )
                patch = r.patch
            except Exception as e:  # noqa: BLE001
                row.update(error=f"{type(e).__name__}: {e}"[:500], seconds=round(time.time() - t0, 1))
                patch = ""
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

            preds.write(json.dumps({"instance_id": iid, "model_name_or_path": model_name,
                                    "model_patch": patch}) + "\n")
            preds.flush()
            rows.append(row)
            print(f"  gate={row.get('gate_passed')} file_hit={row.get('file_hit')} "
                  f"retrieval_hit={row.get('retrieval_hit')} {row.get('error', '')}")

    summary = {"run_at": datetime.now(timezone.utc).isoformat(), "model": model_name,
               **summarize(rows), "rows": rows}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"\nPredictions for the official harness: {pred_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--instances", help="local JSONL instead of HuggingFace")
    ap.add_argument("--ids", nargs="*", help="only these instance_ids")
    ap.add_argument("--out", default="evals/out")
    ap.add_argument("--model-name", default="nexus")
    ap.add_argument("--no-hyde", action="store_true")
    a = ap.parse_args()
    run(load_instances(a.limit, a.instances, a.ids), a.out, a.model_name, use_hyde=not a.no_hyde)


if __name__ == "__main__":
    main()
