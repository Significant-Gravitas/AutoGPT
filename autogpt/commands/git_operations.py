"""Git operations for autogpt"""
import git
from autogpt.config import Config

CFG = Config()


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
        git.Repo.clone_from(auth_repo_url, clone_path)
        return f"""Cloned {repo_url} to {clone_path}"""
    except Exception as e:
        return f"Error: {str(e)}"
