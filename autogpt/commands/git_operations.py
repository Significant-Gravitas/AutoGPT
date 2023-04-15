import git
from config import Config

cfg = Config()


def clone_repository(repo_url, clone_path):
    """Clone a github repository locally"""
    split_url = repo_url.split("//")
    auth_repo_url = f"//{cfg.github_username}:{cfg.github_api_key}@".join(split_url)
    git.Repo.clone_from(auth_repo_url, clone_path)
    result = f"""Cloned {repo_url} to {clone_path}"""

    return result
