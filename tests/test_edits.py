"""Search/replace edits -> unified diff that git really applies."""
import subprocess

from app.tools.edits import build_patch
from app.tools.patch_gate import check_patch


def _write(repo, path, text, crlf=False):
    import os
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", newline="") as f:
        f.write(text.replace("\n", "\r\n") if crlf else text)


def test_exact_edit_builds_a_diff_the_gate_accepts(wealth_repo):
    patch, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "    r = annual_rate_pct / 100",
                                             "replace": "    r = monthly_rate(annual_rate_pct)"}])
    assert errs == [] and patch.startswith("--- a/wealth/sip.py\n+++ b/wealth/sip.py\n@@")
    assert check_patch(wealth_repo, patch).passed


def test_not_found_names_the_closest_real_line(wealth_repo):
    patch, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "    rate = annual_rate_pct / 100",
                                             "replace": "x"}])
    assert patch == "" and "not found" in errs[0] and "r = annual_rate_pct / 100" in errs[0]


def test_ambiguous_search_is_rejected(wealth_repo):
    _, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "    n = years * 12", "replace": "x"}])
    assert errs == [] or "matches" in errs[0]  # unique in fixture -> fine
    _, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "return", "replace": "x"}])
    assert "matches" in errs[0]


def test_tolerates_wrong_indentation_and_reindents(wealth_repo):
    patch, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "r = annual_rate_pct / 100\nn = years * 12",
                                             "replace": "r = monthly_rate(annual_rate_pct)\nn = years * 12"}])
    assert errs == []
    assert "+    r = monthly_rate(annual_rate_pct)" in patch
    assert check_patch(wealth_repo, patch).passed


def test_strips_copied_line_number_prefixes(wealth_repo):
    patch, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "   11 |     r = annual_rate_pct / 100",
                                             "replace": "   11 |     r = monthly_rate(annual_rate_pct)"}])
    assert errs == [] and "+    r = monthly_rate(annual_rate_pct)" in patch


def test_new_file_and_multiple_edits(wealth_repo):
    patch, errs = build_patch(wealth_repo, [
        {"file": "wealth/sip.py", "search": "    r = annual_rate_pct / 100", "replace": "    r = monthly_rate(annual_rate_pct)"},
        {"file": "wealth/sip.py", "search": "    n = years * 12", "replace": "    n = int(years * 12)"},
        {"file": "wealth/new_mod.py", "search": "", "replace": "X = 1"},
    ])
    assert errs == []
    assert "--- /dev/null\n+++ b/wealth/new_mod.py" in patch
    gate = check_patch(wealth_repo, patch)
    assert gate.passed and set(gate.files) == {"wealth/sip.py", "wealth/new_mod.py"}


def test_crlf_files_are_handled(tmp_path):
    repo = str(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _write(repo, "m.py", "def f():\n    return 1\n", crlf=True)
    patch, errs = build_patch(repo, [{"file": "m.py", "search": "    return 1", "replace": "    return 2"}])
    assert errs == [] and "+    return 2\n" in patch and "\r" not in patch


def test_unsafe_paths_and_bad_edits_rejected(wealth_repo):
    _, errs = build_patch(wealth_repo, [{"file": "../evil.py", "search": "", "replace": "x"}])
    assert "outside the repository" in errs[0]
    _, errs = build_patch(wealth_repo, [{"file": "missing.py", "search": "a", "replace": "b"}])
    assert "does not exist" in errs[0]
    _, errs = build_patch(wealth_repo, [{"file": "wealth/sip.py", "search": "    n = years * 12",
                                         "replace": "    n = years * 12"}])
    assert "no change" in errs[0]
