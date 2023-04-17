"""Git operations for autogpt"""
from git.repo import Repo

from autogpt.commands.command import command
from autogpt.config import Config

CFG = Config()


@command(
    "clone_repository",
    "Clone Repositoryy",
    '"repository_url": "<url>", "clone_path": "<directory>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
def clone_repository(repo_url: str, clone_path: str) -> str:
    """Clone a github repository locally

    Args:
        repo_url (str): The URL of the repository to clone
        clone_path (str): The path to clone the repository to

    Returns:
        str: The result of the clone operation"""
    split_url = repo_url.split("//")
    auth_repo_url = f"//{CFG.github_username}:{CFG.github_api_key}@".join(split_url)
    try:
        Repo.clone_from(auth_repo_url, clone_path)
        return f"""Cloned {repo_url} to {clone_path}"""
    except Exception as e:
        return f"Error: {str(e)}"
