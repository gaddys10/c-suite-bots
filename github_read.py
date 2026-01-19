# github_read.py (replace entire file with this)
import os
from github import Github
from github import GithubException

_token = (os.getenv("GITHUB_TOKEN") or "").strip()
if not _token:
    raise RuntimeError("GITHUB_TOKEN missing")

gh = Github(_token, per_page=100)

def _repo(full_name: str):
    return gh.get_repo(full_name)

def latest_commit_sha(full_name: str) -> str:
    r = _repo(full_name)
    try:
        commits = r.get_commits()
        first = commits[0] if commits.totalCount else None
        return first.sha if first else ""
    except GithubException as e:
        if e.status == 409:
            return ""
        raise

def latest_open_pr(repo_full_name: str):
    repo = _repo(repo_full_name)
    prs = repo.get_pulls(state="open", sort="updated", direction="desc")
    pr = prs[0] if prs.totalCount else None
    if not pr:
        return None
    return {
        "number": pr.number,
        "title": pr.title,
        "body": (pr.body or "")[:2000],
        "user": pr.user.login if pr.user else None,
        "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
        "html_url": pr.html_url,
    }

def recent_open_prs(repo_full_name: str, limit: int = 10) -> list[dict]:
    repo = _repo(repo_full_name)
    prs = repo.get_pulls(state="open", sort="updated", direction="desc")
    out = []
    for i, pr in enumerate(prs):
        if i >= limit:
            break
        out.append({
            "number": pr.number,
            "title": pr.title,
            "body": (pr.body or "")[:2000],
            "user": pr.user.login if pr.user else None,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
            "html_url": pr.html_url,
        })
    return out

def recent_commit_shas(full_name: str, limit: int = 10) -> list[str]:
    r = _repo(full_name)
    try:
        commits = r.get_commits()
    except GithubException as e:
        if e.status == 409:
            return []
        raise
    out = []
    for i, c in enumerate(commits):
        if i >= limit:
            break
        out.append(c.sha)
    return out

def commit_summary(repo_full_name: str, sha: str) -> dict:
    repo = _repo(repo_full_name)
    c = repo.get_commit(sha)
    files = []
    for f in (c.files or []):
        files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
        })

    author = None
    if c.author:
        author = c.author.login
    elif c.commit and c.commit.author:
        author = c.commit.author.name

    msg = c.commit.message if c.commit else ""

    return {
        "sha": c.sha,
        "message": msg.split("\n")[0][:200],
        "author": author,
        "date": c.commit.author.date.isoformat() if c.commit and c.commit.author else None,
        "stats": getattr(c, "stats", None) and {
            "additions": c.stats.additions,
            "deletions": c.stats.deletions,
            "total": c.stats.total,
        },
        "files": files[:30],
        "html_url": c.html_url,
    }

def compare_commits(repo_full_name: str, base_sha: str, head_sha: str) -> dict:
    repo = _repo(repo_full_name)
    comp = repo.compare(base_sha, head_sha)

    commits = []
    for c in (comp.commits or [])[:10]:
        commits.append({
            "sha": c.sha,
            "message": (c.commit.message.split("\n")[0] if c.commit else "")[:200],
            "author": (c.author.login if c.author else None),
        })

    files = []
    for f in (comp.files or [])[:30]:
        files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
        })

    return {
        "total_commits": comp.total_commits,
        "commits": commits,
        "files": files,
    }
