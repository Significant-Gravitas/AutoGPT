"""Commands to perform Git operations"""

from pathlib import Path

from git.repo import Repo

from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import CommandExecutionError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.url_utils.validators import validate_url

from .decorators import sanitize_path_arg

COMMAND_CATEGORY = "git_operations"
COMMAND_CATEGORY_TITLE = "Git Operations"


@command(
    "clone_repository",
    "Clones a Repository",
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The URL of the repository to clone",
            required=True,
        ),
        "clone_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path to clone the repository to",
            required=True,
        ),
    },
    lambda config: bool(config.github_username and config.github_api_key),
    "Configure github_username and github_api_key.",
)
@sanitize_path_arg("clone_path")
@validate_url
def clone_repository(url: str, clone_path: Path, agent: Agent) -> str:
    """Clone a GitHub repository locally.

    Args:
        url (str): The URL of the repository to clone.
        clone_path (Path): The path to clone the repository to.

    Returns:
        str: The result of the clone operation.
    """
    split_url = url.split("//")
    auth_repo_url = f"//{agent.legacy_config.github_username}:{agent.legacy_config.github_api_key}@".join(  # noqa: E501
        split_url
    )
    try:
        Repo.clone_from(url=auth_repo_url, to_path=clone_path)
    except Exception as e:
        raise CommandExecutionError(f"Could not clone repo: {e}")

    return f"""Cloned {url} to {clone_path}"""
