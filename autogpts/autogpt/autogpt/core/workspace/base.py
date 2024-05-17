from __future__ import annotations

import abc
import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from autogpt.core.agent.simple import AgentConfiguration


class Workspace(abc.ABC):
    """The workspace is the root directory for all generated files.

    The workspace is responsible for creating the root directory and
    providing a method for getting the full path to an item in the
    workspace.

    """

    @property
    @abc.abstractmethod
    def root(self) -> Path:
        """The root directory of the workspace."""
        ...

    @property
    @abc.abstractmethod
    def restrict_to_workspace(self) -> bool:
        """Whether to restrict generated paths to the workspace."""
        ...

    @staticmethod
    @abc.abstractmethod
    def setup_workspace(
        configuration: AgentConfiguration, logger: logging.Logger
    ) -> Path:
        """Create the workspace root directory and set up all initial content.

        Parameters
        ----------
        configuration
            The Agent's configuration.
        logger
            The Agent's logger.

        Returns
        -------
        Path
            The path to the workspace root directory.

        """
        ...

    @abc.abstractmethod
    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the workspace.

        Parameters
        ----------
        relative_path
            The path to the item relative to the workspace root.

        Returns
        -------
        Path
            The full path to the item.

        """
        ...
