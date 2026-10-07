"""The patch gate is the safety net: test it against a real git checkout."""
import sys

from app.demo import CORRECT_PATCH, HALLUCINATED_PATCH
from app.tools.patch_gate import check_patch, clean_patch, files_in_patch
from tests.conftest import git

PYTEST = f"\"{sys.executable}\" -m pytest -q -p no:cacheprovider"


def _pristine(repo):
    return git(repo, "status", "--porcelain").strip() == ""


def test_correct_patch_passes_and_tests_go_green(wealth_repo):
    r = check_patch(wealth_repo, CORRECT_PATCH, test_command=PYTEST)
    assert r.passed and r.applies and r.syntax_ok and r.tests_passed
    assert r.files == ["wealth/sip.py"]
    assert _pristine(wealth_repo), "gate must restore the checkout"


def test_hallucinated_context_is_rejected_with_git_error(wealth_repo):
    r = check_patch(wealth_repo, HALLUCINATED_PATCH)
    assert not r.passed and not r.applies
    assert "patch does not apply" in r.errors[0]
    assert _pristine(wealth_repo)


def test_patch_that_breaks_syntax_is_rejected(wealth_repo):
    bad = CORRECT_PATCH.replace("+    r = monthly_rate(annual_rate_pct)", "+    r = monthly_rate(annual_rate_pct")
    r = check_patch(wealth_repo, bad)
    assert r.applies and not r.syntax_ok and not r.passed
    assert "SyntaxError" in r.errors[0]
    assert _pristine(wealth_repo)


def test_patch_that_applies_but_fails_tests_is_rejected(wealth_repo):
    wrong = CORRECT_PATCH.replace("monthly_rate(annual_rate_pct)", "annual_rate_pct / 1200 * 2")
    r = check_patch(wealth_repo, wrong, test_command=PYTEST)
    assert r.applies and r.syntax_ok and r.tests_ran and r.tests_passed is False
    assert not r.passed
    assert _pristine(wealth_repo)


def test_bug_is_real_without_patch(wealth_repo):
    import subprocess
    proc = subprocess.run(PYTEST, cwd=wealth_repo, shell=True, capture_output=True, text=True)
    assert proc.returncode != 0 and "2 failed" in proc.stdout


def test_path_traversal_rejected(wealth_repo):
    evil = "--- a/../outside.py\n+++ b/../outside.py\n@@ -0,0 +1 @@\n+x = 1\n"
    r = check_patch(wealth_repo, evil)
    assert not r.passed and "outside the repository" in r.errors[0]


def test_empty_and_prose_only_output_rejected(wealth_repo):
    assert "empty patch" in check_patch(wealth_repo, "").errors[0]
    assert "empty patch" in check_patch(wealth_repo, "I think the bug is in sip.py").errors[0]


def test_clean_patch_strips_fences_prose_and_crlf():
    raw = "Here is the fix:\r\n```diff\r\n" + CORRECT_PATCH.replace("\n", "\r\n") + "```\r\n"
    cleaned = clean_patch(raw)
    assert cleaned.startswith("--- a/wealth/sip.py") and "\r" not in cleaned and "```" not in cleaned


def test_files_in_patch_handles_new_and_deleted_files():
    p = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    assert files_in_patch(p) == ["new.py", "old.py"]


def test_fenced_correct_patch_still_passes(wealth_repo):
    assert check_patch(wealth_repo, f"```diff\n{CORRECT_PATCH}```").passed
