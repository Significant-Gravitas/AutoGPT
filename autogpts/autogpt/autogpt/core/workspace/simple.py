import json
import logging
import typing
from pathlib import Path

from forge.models.config import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from pydantic import SecretField

from autogpt.core.workspace.base import Workspace

if typing.TYPE_CHECKING:
    # Cyclic import
    from autogpt.core.agent.simple import AgentSettings


class WorkspaceConfiguration(SystemConfiguration):
    root: str
    parent: str = UserConfigurable()
    restrict_to_workspace: bool = UserConfigurable()


class WorkspaceSettings(SystemSettings):
    configuration: WorkspaceConfiguration


class SimpleWorkspace(Configurable, Workspace):
    default_settings = WorkspaceSettings(
        name="workspace",
        description="The workspace is the root directory for all agent activity.",
        configuration=WorkspaceConfiguration(
            root="",
            parent="~/auto-gpt/agents",
            restrict_to_workspace=True,
        ),
    )

    NULL_BYTES = ["\0", "\000", "\x00", "\u0000", "%00"]

    def __init__(
        self,
        settings: WorkspaceSettings,
        logger: logging.Logger,
    ):
        self._configuration = settings.configuration
        self._logger = logger.getChild("workspace")

    @property
    def root(self) -> Path:
        return Path(self._configuration.root)

    @property
    def debug_log_path(self) -> Path:
        return self.root / "logs" / "debug.log"

    @property
    def cycle_log_path(self) -> Path:
        return self.root / "logs" / "cycle.log"

    @property
    def configuration_path(self) -> Path:
        return self.root / "configuration.yml"

    @property
    def restrict_to_workspace(self) -> bool:
        return self._configuration.restrict_to_workspace

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the workspace.

        Parameters
        ----------
        relative_path
            The relative path to resolve in the workspace.

        Returns
        -------
        Path
            The resolved path relative to the workspace.

        """
        return self._sanitize_path(
            relative_path,
            root=self.root,
            restrict_to_root=self.restrict_to_workspace,
        )

    def _sanitize_path(
        self,
        relative_path: str | Path,
        root: str | Path = None,
        restrict_to_root: bool = True,
    ) -> Path:
        """Resolve the relative path within the given root if possible.

        Parameters
        ----------
        relative_path
            The relative path to resolve.
        root
            The root path to resolve the relative path within.
        restrict_to_root
            Whether to restrict the path to the root.

        Returns
        -------
        Path
            The resolved path.

        Raises
        ------
        ValueError
            If the path is absolute and a root is provided.
        ValueError
            If the path is outside the root and the root is restricted.

        """

        # Posix systems disallow null bytes in paths. Windows is agnostic about it.
        # Do an explicit check here for all sorts of null byte representations.

        for null_byte in self.NULL_BYTES:
            if null_byte in str(relative_path) or null_byte in str(root):
                raise ValueError("embedded null byte")

        if root is None:
            return Path(relative_path).resolve()

        self._logger.debug(f"Resolving path '{relative_path}' in workspace '{root}'")
        root, relative_path = Path(root).resolve(), Path(relative_path)
        self._logger.debug(f"Resolved root as '{root}'")

        if relative_path.is_absolute():
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' "
                f"in workspace '{root}'."
            )
        full_path = root.joinpath(relative_path).resolve()

        self._logger.debug(f"Joined paths as '{full_path}'")

        if restrict_to_root and not full_path.is_relative_to(root):
            raise ValueError(
                f"Attempted to access path '{full_path}' outside of workspace '{root}'."
            )

        return full_path

    ###################################
    # Factory methods for agent setup #
    ###################################

    @staticmethod
    def setup_workspace(settings: "AgentSettings", logger: logging.Logger) -> Path:
        workspace_parent = settings.workspace.configuration.parent
        workspace_parent = Path(workspace_parent).expanduser().resolve()
        workspace_parent.mkdir(parents=True, exist_ok=True)

        agent_name = settings.agent.name

        workspace_root = workspace_parent / agent_name
        workspace_root.mkdir(parents=True, exist_ok=True)

        settings.workspace.configuration.root = str(workspace_root)

        with (workspace_root / "agent_settings.json").open("w") as f:
            settings_json = settings.json(
                encoder=lambda x: x.get_secret_value()
                if isinstance(x, SecretField)
                else x,
            )
            f.write(settings_json)

        # TODO: What are all the kinds of logs we want here?
        log_path = workspace_root / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "debug.log").touch()
        (log_path / "cycle.log").touch()

        return workspace_root

    @staticmethod
    def load_agent_settings(workspace_root: Path) -> "AgentSettings":
        # Cyclic import
        from autogpt.core.agent.simple import AgentSettings

        with (workspace_root / "agent_settings.json").open("r") as f:
            agent_settings = json.load(f)

        return AgentSettings.parse_obj(agent_settings)
