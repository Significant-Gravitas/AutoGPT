from pathlib import Path
from typing import Iterator, Optional

from git.repo import Repo
from pydantic import BaseModel, SecretStr

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError
from forge.utils.url_validator import validate_url


class GitOperationsConfiguration(BaseModel):
    github_username: Optional[str] = UserConfigurable(None, from_env="GITHUB_USERNAME")
    github_api_key: Optional[SecretStr] = UserConfigurable(
        None, from_env="GITHUB_API_KEY", exclude=True
    )


class GitOperationsComponent(
    CommandProvider, ConfigurableComponent[GitOperationsConfiguration]
):
    """Provides commands to perform Git operations."""

    config_class = GitOperationsConfiguration

    def __init__(self, config: Optional[GitOperationsConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        self._enabled = bool(self.config.github_username and self.config.github_api_key)
        self._disabled_reason = "Configure github_username and github_api_key."

    def get_commands(self) -> Iterator[Command]:
        yield self.clone_repository

    @command(
        parameters={
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
    )
    @validate_url
    def clone_repository(self, url: str, clone_path: Path) -> str:
        """Clone a GitHub repository locally.

        Args:
            url (str): The URL of the repository to clone.
            clone_path (Path): The path to clone the repository to.

        Returns:
            str: The result of the clone operation.
        """
        split_url = url.split("//")
        api_key = (
            self.config.github_api_key.get_secret_value()
            if self.config.github_api_key
            else None
        )
        auth_repo_url = f"//{self.config.github_username}:" f"{api_key}@".join(
            split_url
        )
        try:
            Repo.clone_from(url=auth_repo_url, to_path=clone_path)
        except Exception as e:
            raise CommandExecutionError(f"Could not clone repo: {e}")

        return f"""Cloned {url} to {clone_path}"""
