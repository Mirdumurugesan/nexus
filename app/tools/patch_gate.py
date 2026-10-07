"""
Patch Gate — the deterministic half of review.

An LLM reviewer can be talked into a 0.9 for a diff that doesn't even apply.
So before any model sees the patch, NEXUS checks the things that are facts,
not opinions:

  1. The diff is non-empty and only touches paths inside the repo.
  2. `git apply --check` accepts it against the actual checkout.
  3. Every modified .py file still compiles after applying it.
  4. (optional) A test command passes on the patched tree.

A patch that fails the gate can never be marked as passed, whatever score the
LLM reviewer gives it. The exact git / compiler error is fed to the Reflector,
which is far more actionable than "the patch seems incomplete".

The checkout is always restored afterwards, so the gate is safe to run in a loop.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass, field

_FENCE = re.compile(r"^```[a-zA-Z]*\s*$")


@dataclass
class GateResult:
    passed: bool
    applies: bool = False
    syntax_ok: bool = False
    tests_ran: bool = False
    tests_passed: bool | None = None
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    test_output_tail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.passed:
            extra = ", tests pass" if self.tests_ran else ""
            return f"gate OK: applies cleanly, {len(self.files)} file(s) compile{extra}"
        return "gate FAILED: " + " ".join("; ".join(self.errors[:3]).split())


def clean_patch(text: str) -> str:
    """Normalise what LLMs actually return: code fences, CRLF, prose before the diff."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    lines = [ln for ln in text.split("\n") if not _FENCE.match(ln.strip())]
    # Drop any explanation before the first diff header.
    for i, ln in enumerate(lines):
        if ln.startswith("diff --git") or ln.startswith("--- "):
            lines = lines[i:]
            break
    else:
        return ""
    out = "\n".join(lines).rstrip("\n") + "\n"
    return out


def files_in_patch(patch: str) -> list[str]:
    files: list[str] = []
    for ln in patch.split("\n"):
        if ln.startswith("+++ ") or ln.startswith("--- "):
            path = ln[4:].split("\t")[0].strip()
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path and path not in files:
                files.append(path)
    return files


def _unsafe_path(path: str) -> bool:
    norm = os.path.normpath(path)
    return os.path.isabs(path) or norm.startswith("..") or norm.startswith(".git")


def _git(repo: str, *args: str, stdin: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, input=stdin, capture_output=True, text=True, timeout=timeout,
    )


def _restore(repo: str) -> None:
    _git(repo, "checkout", "--", ".")
    _git(repo, "clean", "-fdq")


def check_patch(repo_path: str, patch: str, test_command: str = "", test_timeout_s: int = 300) -> GateResult:
    patch = clean_patch(patch)
    if not patch.strip():
        return GateResult(passed=False, errors=["empty patch: no unified diff found in model output"])

    files = files_in_patch(patch)
    result = GateResult(passed=False, files=files)
    if not files:
        result.errors.append("diff has no file headers (--- a/... / +++ b/...)")
        return result
    bad = [f for f in files if _unsafe_path(f)]
    if bad:
        result.errors.append(f"patch touches paths outside the repository: {bad}")
        return result

    check = _git(repo_path, "apply", "--check", "--whitespace=nowarn", "-", stdin=patch)
    if check.returncode != 0:
        result.errors.append("git apply --check: " + (check.stderr.strip() or "rejected")[:600])
        return result
    result.applies = True

    try:
        applied = _git(repo_path, "apply", "--whitespace=nowarn", "-", stdin=patch)
        if applied.returncode != 0:  # pragma: no cover — --check passed, so this is unexpected
            result.applies = False
            result.errors.append("git apply: " + applied.stderr.strip()[:600])
            return result

        syntax_errors = []
        for f in files:
            full = os.path.join(repo_path, f)
            if f.endswith(".py") and os.path.exists(full):
                try:
                    with open(full, encoding="utf-8", errors="replace") as fh:
                        compile(fh.read(), f, "exec")
                except SyntaxError as e:
                    syntax_errors.append(f"{f}:{e.lineno}: SyntaxError: {e.msg}")
        result.syntax_ok = not syntax_errors
        result.errors.extend(syntax_errors)

        if result.syntax_ok and test_command:
            result.tests_ran = True
            try:
                proc = subprocess.run(
                    test_command, cwd=repo_path, shell=True, capture_output=True,
                    text=True, timeout=test_timeout_s,
                )
                result.tests_passed = proc.returncode == 0
                result.test_output_tail = (proc.stdout + proc.stderr)[-1500:]
                if not result.tests_passed:
                    result.errors.append(f"tests failed (`{test_command}`):\n{result.test_output_tail[-800:]}")
            except subprocess.TimeoutExpired:
                result.tests_passed = False
                result.errors.append(f"tests timed out after {test_timeout_s}s")
    finally:
        _restore(repo_path)

    result.passed = result.applies and result.syntax_ok and result.tests_passed is not False
    return result
