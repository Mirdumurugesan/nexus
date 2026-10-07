"""
NEXUS command line.

    python -m app.cli demo [--live]                       offline demo on a bundled repo
    python -m app.cli solve <issue-url> [--out fix.diff]  real issue, real repo
"""
from __future__ import annotations

import argparse
import json
import sys

BOLD, DIM, GREEN, RED, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m"


def _report(r, title: str) -> None:
    print(f"\n{BOLD}NEXUS · {title}{RESET}")
    print(f"{DIM}{'─' * 64}{RESET}")
    print(f"indexed {r.chunks_indexed} code chunks · retrieved from: {', '.join(r.retrieved_files[:4])}")
    print(f"{BOLD}plan{RESET}")
    for i, p in enumerate(r.plan, 1):
        print(f"  {i}. [{p.get('file_hint')}] {p.get('description')}")
    print(f"{BOLD}attempts{RESET}")
    for h in r.history:
        mark = f"{GREEN}✓{RESET}" if h["passed"] else f"{RED}✗{RESET}"
        print(f"  {mark} round {h['round']} ({h['agent']}): {h['gate_summary'][:150]} · review {h['review_score']:.2f}")
    verdict = f"{GREEN}VERIFIED{RESET}" if r.passed else f"{RED}NOT VERIFIED{RESET} (best attempt shown)"
    print(f"{BOLD}result{RESET}   {verdict} · score {r.review_score:.2f} · {r.reflections} reflection(s) · "
          f"{r.llm_calls} LLM calls · {r.seconds}s")
    if r.llm_failovers:
        print(f"{DIM}provider failovers: {len(r.llm_failovers)}{RESET}")
    print(f"{BOLD}patch{RESET}{CYAN}\n{r.patch}{RESET}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nexus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="fix the bundled wealth-lib bug (no keys needed)")
    d.add_argument("--live", action="store_true", help="use real LLM providers instead of the scripted replay")
    d.add_argument("--json", action="store_true")

    s = sub.add_parser("solve", help="solve a real GitHub issue")
    s.add_argument("issue_url")
    s.add_argument("--out", help="write the final patch here")
    s.add_argument("--no-hyde", action="store_true")
    s.add_argument("--test-cmd", default="", help="run this in the patched checkout as part of the gate")
    s.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "demo":
        from app.demo import run_demo
        r = run_demo(live=args.live)
        title = "demo · " + ("live LLM" if args.live else "scripted LLM replay, real gate")
    else:
        from app.core.config import get_settings
        from app.pipeline import solve_issue
        from app.tools.github_parser import fetch_github_issue
        if args.test_cmd:
            get_settings().gate_test_command = args.test_cmd
        issue = fetch_github_issue(args.issue_url)
        r = solve_issue(issue.issue_title, issue.issue_body, issue.repo_full_name,
                        repo_url=issue.repo_url, use_hyde=not args.no_hyde)
        title = f"{issue.repo_full_name}#{issue.issue_number}"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(r.patch)

    if args.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        _report(r, title)
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
