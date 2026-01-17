import os
from github import Github

gh = Github(os.getenv("GITHUB_TOKEN"))

def latest_commit_sha(full_name: str) -> str:
    r = gh.get_repo(full_name)
    return r.get_commits()[0].sha

def latest_open_pr(full_name: str):
    r = gh.get_repo(full_name)
    prs = r.get_pulls(state="open", sort="updated", direction="desc")
    pr = prs[0] if prs.totalCount else None
    if not pr:
        return None
    return {"number": pr.number, "title": pr.title, "updated_at": pr.updated_at.isoformat()}
