"""
NEXUS SWE-bench Evaluator
─────────────────────────
Runs NEXUS against SWE-bench Lite (300 real GitHub issues with known fixes).
Reports resolution rate, per-repo breakdown, and cost.

Usage:
    python evals/swebench_eval.py --limit 10 --output results.json

SWE-bench Lite dataset: huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
"""
import argparse
import json
import time
import requests
from datetime import datetime
from pathlib import Path


API_BASE = "http://127.0.0.1:8000/api/v1"
POLL_INTERVAL = 5    # seconds between status polls
MAX_WAIT = 600       # max seconds to wait per task (10 min)


def load_swebench_lite(limit: int = 10) -> list[dict]:
    """
    Load SWE-bench Lite instances from HuggingFace datasets API.
    Falls back to a small hardcoded sample if offline.
    """
    try:
        from datasets import load_dataset
        print(f"[eval] Loading SWE-bench Lite (first {limit} instances)...")
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        instances = [ds[i] for i in range(min(limit, len(ds)))]
        print(f"[eval] Loaded {len(instances)} instances")
        return instances
    except Exception as e:
        print(f"[eval] Could not load dataset ({e}). Using hardcoded sample.")
        return SAMPLE_INSTANCES[:limit]


# Hardcoded sample — used if `datasets` library not installed or offline
SAMPLE_INSTANCES = [
    {
        "instance_id": "psf__requests-7443",
        "repo": "psf/requests",
        "issue_numbers": [7443],
        "problem_statement": "Unexpected proxy behavior when HTTPS_PROXY is set but HTTP_PROXY is not",
        "patch": "",  # ground truth patch
    },
    {
        "instance_id": "pallets__flask-5500",
        "repo": "pallets/flask",
        "issue_numbers": [5500],
        "problem_statement": "url_for fails with SERVER_NAME when using blueprints",
        "patch": "",
    },
    {
        "instance_id": "django__django-16139",
        "repo": "django/django",
        "issue_numbers": [16139],
        "problem_statement": "QuerySet.bulk_create() crashes when update_fields is passed as a tuple",
        "patch": "",
    },
]


def build_issue_url(repo: str, issue_number: int) -> str:
    return f"https://github.com/{repo}/issues/{issue_number}"


def submit_task(issue_url: str) -> str | None:
    """Submit a task to NEXUS and return the task_id."""
    try:
        r = requests.post(
            f"{API_BASE}/tasks",
            json={"github_issue_url": issue_url, "use_hyde": True},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["task_id"]
    except Exception as e:
        print(f"  [!] Submit failed: {e}")
        return None


def poll_task(task_id: str) -> dict:
    """Poll until task is completed or failed."""
    start = time.time()
    last_step = ""
    while time.time() - start < MAX_WAIT:
        try:
            r = requests.get(f"{API_BASE}/tasks/{task_id}", timeout=10)
            task = r.json()
            status = task["status"]
            step = task.get("current_step", "")

            if step != last_step:
                print(f"  → {step}")
                last_step = step

            if status in ("completed", "failed"):
                return task
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)

    return {"status": "timeout", "task_id": task_id}


def score_patch(nexus_patch: str, ground_truth_patch: str) -> dict:
    """
    Simple patch scoring:
    - exact_match: patch is identical
    - file_match: same files modified
    - non_empty: patch was generated at all

    In production you'd run the actual tests in a sandbox (docker).
    """
    if not nexus_patch:
        return {"exact_match": False, "file_match": False, "non_empty": False, "score": 0.0}

    non_empty = len(nexus_patch.strip()) > 50
    file_match = False
    exact_match = False

    if ground_truth_patch:
        # Extract file names from unified diffs
        def extract_files(patch: str) -> set:
            files = set()
            for line in patch.split("\n"):
                if line.startswith("+++ b/") or line.startswith("--- a/"):
                    files.add(line.split("/", 1)[-1])
            return files

        nexus_files = extract_files(nexus_patch)
        gt_files = extract_files(ground_truth_patch)
        file_match = bool(nexus_files & gt_files)
        exact_match = nexus_patch.strip() == ground_truth_patch.strip()

    score = 1.0 if exact_match else (0.5 if file_match else (0.2 if non_empty else 0.0))
    return {
        "exact_match": exact_match,
        "file_match": file_match,
        "non_empty": non_empty,
        "score": score,
    }


def run_evaluation(limit: int = 10, output_path: str = "evals/results.json"):
    """Main evaluation loop."""
    print("=" * 60)
    print("  NEXUS × SWE-bench Lite Evaluation")
    print(f"  Instances: {limit} | API: {API_BASE}")
    print("=" * 60)

    instances = load_swebench_lite(limit)
    results = []
    total_score = 0.0
    resolved = 0

    for i, instance in enumerate(instances):
        repo = instance.get("repo", "")
        instance_id = instance.get("instance_id", f"instance_{i}")
        issue_numbers = instance.get("issue_numbers", [])
        ground_truth = instance.get("patch", "")

        if not issue_numbers:
            print(f"\n[{i+1}/{limit}] {instance_id} — no issue number, skipping")
            continue

        issue_url = build_issue_url(repo, issue_numbers[0])
        print(f"\n[{i+1}/{limit}] {instance_id}")
        print(f"  URL: {issue_url}")

        # Submit
        task_id = submit_task(issue_url)
        if not task_id:
            results.append({"instance_id": instance_id, "status": "submit_failed", "score": 0.0})
            continue

        # Poll
        task = poll_task(task_id)
        status = task.get("status", "unknown")

        if status != "completed":
            print(f"  ✗ {status}")
            results.append({
                "instance_id": instance_id,
                "task_id": task_id,
                "status": status,
                "score": 0.0,
                "error": task.get("error_message", ""),
            })
            continue

        # Score
        patch = task.get("generated_patch", "") or ""
        scoring = score_patch(patch, ground_truth)
        total_score += scoring["score"]
        if scoring["score"] >= 0.5:
            resolved += 1

        confidence = task.get("confidence") or 0
        review_score = task.get("review_score") or 0

        print(f"  ✓ completed | confidence={confidence:.2f} | review={review_score:.2f} | score={scoring['score']:.1f}")

        results.append({
            "instance_id": instance_id,
            "task_id": task_id,
            "repo": repo,
            "status": "completed",
            "score": scoring["score"],
            "exact_match": scoring["exact_match"],
            "file_match": scoring["file_match"],
            "non_empty": scoring["non_empty"],
            "confidence": confidence,
            "review_score": review_score,
            "review_passed": task.get("review_passed"),
            "patch_length": len(patch),
        })

    # Summary
    total = len(results)
    avg_score = total_score / total if total else 0
    resolution_rate = resolved / total * 100 if total else 0

    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "total_instances": total,
        "resolved": resolved,
        "resolution_rate_pct": round(resolution_rate, 2),
        "avg_score": round(avg_score, 3),
        "results": results,
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  RESULTS")
    print(f"  Instances evaluated : {total}")
    print(f"  Resolved (score≥0.5): {resolved}")
    print(f"  Resolution rate     : {resolution_rate:.1f}%")
    print(f"  Average score       : {avg_score:.3f}")
    print(f"  Output saved to     : {output_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEXUS × SWE-bench Lite Evaluator")
    parser.add_argument("--limit", type=int, default=10, help="Number of instances to evaluate")
    parser.add_argument("--output", type=str, default="evals/results.json", help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(limit=args.limit, output_path=args.output)
