"""
The LocalFileWorkspace class implements a AbstractFileWorkspace that works with local files.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import IO, Optional

from AFAAS.interfaces.workspace import (
    AbstractFileWorkspace,
    AbstractFileWorkspaceConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class AGPTLocalFileWorkspaceConfiguration(AbstractFileWorkspaceConfiguration):
    pass


class AGPTLocalFileWorkspace(AbstractFileWorkspace):
    """A class that represents a file workspace."""

    class SystemSettings(AbstractFileWorkspace.SystemSettings):
        configuration: AGPTLocalFileWorkspaceConfiguration = (
            AGPTLocalFileWorkspaceConfiguration()
        )

    def __init__(
        self,
        user_id: str,
        agent_id: str,
        config: AGPTLocalFileWorkspace = AGPTLocalFileWorkspaceConfiguration(),
    ):
        super().__init__(user_id=user_id, agent_id=agent_id, config=config)
        self._restrict_to_agent_workspace = config.restrict_to_agent_workspace

    @property
    def root(self) -> Path:
        """The root directory of the file workspace."""
        return self.agent_workspace

    @property
    def restrict_to_agent_workspace(self) -> bool:
        """Whether to restrict generated paths to the root."""
        return self._restrict_to_agent_workspace

    def _initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)

    def open_file(self, path: str | Path, binary: bool = False) -> IO:
        """Open a file in the workspace."""
        return self._open_file(path, "rb" if binary else "r")

    def _open_file(self, path: str | Path, mode: str = "r") -> IO:
        full_path = self.get_path(path)
        return open(full_path, mode)  # type: ignore

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the workspace."""
        with self._open_file(path, "rb" if binary else "r") as file:
            return file.read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the workspace."""
        with self._open_file(path, "wb" if type(content) is bytes else "w") as file:
            file.write(content)

        if self.on_write_file:
            path = Path(path)
            if path.is_absolute():
                path = path.relative_to(self.root)
            res = self.on_write_file(path)
            if inspect.isawaitable(res):
                await res

    def list(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the workspace."""
        path = self.get_path(path)
        return [file.relative_to(path) for file in path.rglob("*") if file.is_file()]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the workspace."""
        full_path = self.get_path(path)
        full_path.unlink()

    # @staticmethod
    def _sanitize_path(
        self,
        relative_path: str | Path,
        agent_workspace_path: Optional[str | Path] = None,
        restrict_to_agent_workspace: bool = True,
    ) -> Path:
        return super()._sanitize_path(
            relative_path=relative_path,
            agent_workspace_path=agent_workspace_path,
            restrict_to_agent_workspace=restrict_to_agent_workspace,
        )
