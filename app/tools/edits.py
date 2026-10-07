"""
Search/replace edits -> unified diff.

Models are unreliable at writing unified diffs (wrong line numbers, made-up
context, or their own patch dialects: gpt-oss emits "*** Begin Patch").
They are much better at "replace this exact snippet with that one". So the
agents return edit blocks, and NEXUS builds the diff itself with difflib.
The diff then goes through the same patch gate as before.

Matching, in order:
  1. exact substring (must be unique in the file)
  2. line-by-line ignoring trailing whitespace
  3. line-by-line ignoring all leading/trailing whitespace, re-indented to the file
A search block that matches nothing, or matches more than once, becomes a
concrete error for the reflector, including the closest real lines.
"""
from __future__ import annotations

import difflib
import os

from pydantic import BaseModel, Field


class Edit(BaseModel):
    file: str = Field(description="Repo-relative path, e.g. pkg/module.py")
    search: str = Field(description="Exact existing lines to replace, copied from the code WITHOUT the "
                                    "'  N | ' line-number prefix. Empty only when creating a new file.")
    replace: str = Field(description="The new lines that replace `search`")


def _unsafe(path: str) -> bool:
    norm = os.path.normpath(path)
    return not path or os.path.isabs(path) or norm.startswith("..") or norm.startswith(".git")


def _strip_prefixes(text: str) -> str:
    """Remove '  12 | ' prefixes if the model copied them from the context."""
    import re
    lines = text.split("\n")
    pat = re.compile(r"^\s*\d+ \| ?")
    if lines and sum(bool(pat.match(ln)) for ln in lines if ln.strip()) >= max(1, len([ln for ln in lines if ln.strip()])):
        return "\n".join(pat.sub("", ln, count=1) for ln in lines)
    return text


def _find_lines(content: str, search: str, normalize) -> list[int]:
    hay = content.split("\n")
    needle = [normalize(x) for x in search.strip("\n").split("\n")]
    n = len(needle)
    return [i for i in range(len(hay) - n + 1) if [normalize(x) for x in hay[i:i + n]] == needle]


def _reindent(replace: str, file_line: str, search_first: str) -> str:
    """When matched ignoring indentation, shift `replace` by the indentation difference."""
    have = len(file_line) - len(file_line.lstrip())
    got = len(search_first) - len(search_first.lstrip())
    delta = have - got
    if delta == 0:
        return replace
    out = []
    for ln in replace.split("\n"):
        if delta > 0:
            out.append(" " * delta + ln if ln.strip() else ln)
        else:
            cut = min(-delta, len(ln) - len(ln.lstrip()))
            out.append(ln[cut:])
    return "\n".join(out)


def _apply_one(content: str, search: str, replace: str, path: str) -> tuple[str, str | None]:
    if search in content:
        count = content.count(search)
        if count > 1:
            return content, f"{path}: search block matches {count} places; include more surrounding lines"
        return content.replace(search, replace, 1), None

    lines = content.split("\n")
    s_lines = search.strip("\n").split("\n")
    for normalize, reindent in ((str.rstrip, False), (str.strip, True)):
        hits = _find_lines(content, search, normalize)
        if len(hits) == 1:
            i = hits[0]
            rep = replace.strip("\n")
            if reindent:
                rep = _reindent(rep, lines[i], s_lines[0])
            new = lines[:i] + rep.split("\n") + lines[i + len(s_lines):]
            return "\n".join(new), None
        if len(hits) > 1:
            return content, f"{path}: search block matches {len(hits)} places; include more surrounding lines"

    first = next((ln for ln in s_lines if ln.strip()), "")
    close = difflib.get_close_matches(first, lines, n=3, cutoff=0.5)
    hint = (" Closest real lines: " + " | ".join(repr(c) for c in close)) if close else ""
    return content, f"{path}: search block not found (first line {first!r}).{hint}"


def build_patch(repo_path: str, edits: list) -> tuple[str, list[str]]:
    """Apply edits in memory to the pristine checkout and return (unified diff, errors)."""
    errors: list[str] = []
    originals: dict[str, str | None] = {}
    current: dict[str, str] = {}

    for e in edits:
        e = e if isinstance(e, Edit) else Edit(**e)
        path = e.file.strip().replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        if _unsafe(path):
            errors.append(f"{e.file!r}: path is outside the repository")
            continue
        if path not in current:
            full = os.path.join(repo_path, path)
            if os.path.exists(full):
                with open(full, encoding="utf-8", errors="replace", newline="") as fh:
                    text = fh.read().replace("\r\n", "\n")
                originals[path] = text
                current[path] = text
            else:
                originals[path] = None
                current[path] = ""
        search = _strip_prefixes(e.search)
        replace = _strip_prefixes(e.replace)
        if originals[path] is None and current[path] == "":
            if search.strip():
                errors.append(f"{path}: file does not exist (use an empty search to create it)")
                continue
            current[path] = replace if replace.endswith("\n") else replace + "\n"
            continue
        if not search.strip():
            errors.append(f"{path}: empty search block for an existing file")
            continue
        current[path], err = _apply_one(current[path], search, replace, path)
        if err:
            errors.append(err)

    if errors:
        return "", errors

    chunks = []
    for path, new in current.items():
        old = originals[path]
        if old == new:
            continue
        a = [] if old is None else (old if old.endswith("\n") else old + "\n").splitlines(keepends=True)
        b = (new if new.endswith("\n") else new + "\n").splitlines(keepends=True)
        chunks.append("".join(difflib.unified_diff(
            a, b, fromfile="/dev/null" if old is None else f"a/{path}", tofile=f"b/{path}")))
    patch = "".join(chunks)
    if not patch:
        return "", ["edits made no change to any file"]
    return patch, []
