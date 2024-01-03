"""
The LocalFileWorkspace class implements a AbstractFileWorkspace that works with local files.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from AFAAS.configs.schema import Configurable, SystemConfiguration, UserConfigurable
from AFAAS.interfaces.workspace import (
    AbstractFileWorkspace,
    AbstractFileWorkspaceConfiguration,
)

if TYPE_CHECKING:
    # Cyclic import
    pass


class AbstractFileWorkspaceConfiguration(SystemConfiguration):
    root: str = ""
    app_workspace: str = UserConfigurable(default="~/AFAAS/agents")
    restrict_to_agent_workspace: bool = UserConfigurable(default=True)


class LocalFileWorkspace(AbstractFileWorkspace):
    class SystemSettings(AbstractFileWorkspace.SystemSettings):
        name = "workspace"
        description = "The workspace is the root directory for all agent activity."
        configuration: AbstractFileWorkspaceConfiguration = (
            AbstractFileWorkspaceConfiguration()
        )

    NULL_BYTES = ["\0", "\000", "\x00", "\u0000", "%00"]

    def __init__(
        self,
        settings: LocalFileWorkspace.SystemSettings,
        agent_systems: list[Configurable],
    ):
        # self._configuration = settings.configuration
        # LOG = logger
        # LOG = LOG.getChild("workspace")
        self._root = self._sanitize_path(settings.configuration.root)
        self._restrict_to_agent_workspace = (
            settings.configuration.restrict_to_agent_workspace
        )
        super().__init__(settings)

    @property
    def root(self) -> Path:
        """The root directory of the file workspace."""
        return self._root

    # @property
    # def debug_log_path(self) -> Path:
    #     return self.root / "logs" / "debug.log"

    # @property
    # def cycle_log_path(self) -> Path:
    #     return self.root / "logs" / "cycle.log"

    # @property
    # def configuration_path(self) -> Path:
    #     return self.root / "configuration.yml"

    @property
    def restrict_to_agent_workspace(self):
        """Whether to restrict generated paths to the root."""
        return self._restrict_to_agent_workspace

    def _initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)

    def open_file(self, path: str | Path, mode: str = "r"):
        """Open a file in the workspace."""
        full_path = self.get_path(path)
        return open(full_path, mode)

    def read_file(self, path: str | Path, binary: bool = False):
        """Read a file in the workspace."""
        with self.open_file(path, "rb" if binary else "r") as file:
            return file.read()

    async def _write_file(self, path: str | Path, content: str | bytes):
        """Write to a file in the workspace."""
        with self.open_file(path, "wb" if type(content) is bytes else "w") as file:
            file.write(content)

    def list_files(self, path: str | Path = "."):
        """List all files in a directory in the workspace."""
        full_path = self.get_path(path)
        return [str(file) for file in full_path.glob("*") if file.is_file()]

    def delete_file(self, path: str | Path):
        """Delete a file in the workspace."""
        full_path = self.get_path(path)
        full_path.unlink()

    ###################################
    # Factory methods for agent setup #
    ###################################

    @classmethod
    def create_workspace(
        cls,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        settings: LocalFileWorkspace.SystemSettings,
    ) -> Path:
        workspace_root = cls.SystemSettings().configuration.app_workspace
        workspace_root = Path(workspace_root).expanduser().resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)

        # user_id = settings.user_id
        # agent_id = settings.agent_id
        agent_root = workspace_root / str(user_id) / str(agent_id)
        agent_root.mkdir(parents=True, exist_ok=True)

        cls.SystemSettings().configuration.root = str(agent_root)

        log_path = agent_root / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "debug.log").touch()
        (log_path / "cycle.log").touch()

        return agent_root
