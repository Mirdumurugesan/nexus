import re
from dataclasses import dataclass
from github import Github
from app.core.config import get_settings

settings = get_settings()


@dataclass
class ParsedIssue:
    repo_url: str
    repo_full_name: str      # e.g. "owner/repo"
    issue_number: int
    issue_title: str
    issue_body: str
    labels: list[str]
    language: str | None     # detected primary language


def parse_github_issue_url(url: str) -> tuple[str, str, int]:
    """
    Input:  https://github.com/owner/repo/issues/123
    Output: (owner, repo, 123)
    """
    pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Invalid GitHub issue URL: {url}")
    owner, repo, issue_num = match.groups()
    return owner, repo, int(issue_num)


def fetch_github_issue(issue_url: str) -> ParsedIssue:
    """Fetch full issue details from GitHub API."""
    g = Github(settings.github_token)

    owner, repo_name, issue_number = parse_github_issue_url(issue_url)
    repo = g.get_repo(f"{owner}/{repo_name}")
    issue = repo.get_issue(number=issue_number)

    # Detect primary language
    try:
        language = repo.language
    except Exception:
        language = None

    return ParsedIssue(
        repo_url=repo.clone_url,
        repo_full_name=repo.full_name,
        issue_number=issue_number,
        issue_title=issue.title,
        issue_body=issue.body or "",
        labels=[label.name for label in issue.labels],
        language=language,
    )