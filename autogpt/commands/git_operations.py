"""Git operations for autogpt"""

from git.repo import Repo

from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.url_utils.validators import validate_url

from .decorators import sanitize_path_arg


@command(
    "clone_repository",
    "Clones a Repository",
    {
        "url": {
            "type": "string",
            "description": "The URL of the repository to clone",
            "required": True,
        },
        "clone_path": {
            "type": "string",
            "description": "The path to clone the repository to",
            "required": True,
        },
    },
    lambda config: bool(config.github_username and config.github_api_key),
    "Configure github_username and github_api_key.",
)
@sanitize_path_arg("clone_path")
@validate_url
def clone_repository(url: str, clone_path: str, agent: Agent) -> str:
    """Clone a GitHub repository locally.

    Args:
        url (str): The URL of the repository to clone.
        clone_path (str): The path to clone the repository to.

    Returns:
        str: The result of the clone operation.
    """
    split_url = url.split("//")
    auth_repo_url = (
        f"//{agent.config.github_username}:{agent.config.github_api_key}@".join(
            split_url
        )
    )
    try:
        Repo.clone_from(url=auth_repo_url, to_path=clone_path)
        return f"""Cloned {url} to {clone_path}"""
    except Exception as e:
        return f"Error: {str(e)}"
