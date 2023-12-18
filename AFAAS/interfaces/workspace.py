"""
The AbstractFileWorkspace class provides an interface for interacting with a file workspace.
"""
from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from io import IOBase, TextIOBase
from pathlib import Path
from pydantic import Field
from typing import IO, Any, BinaryIO, Callable, Literal, Optional, TextIO, overload

from AFAAS.configs.schema import SystemConfiguration

from AFAAS.configs import Configurable, SystemConfiguration, UserConfigurable
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class AbstractFileWorkspaceConfiguration(SystemConfiguration):
    restrict_to_agent_workspace: bool = True
    app_workspace: Path = UserConfigurable(
        default=Path("~/auto-gpt/agents").expanduser().resolve()
    )
    agent_workspace: Path = Path("/")
    user_id: str = None
    agent_id: str = None

    @property
    def agent_workspace(self) -> Path:
        if not self.user_id or not self.agent_id:
            raise ValueError("user_id and agent_id must be set")
        return self.app_workspace / self.user_id / self.agent_id


class AbstractFileWorkspace(Configurable, ABC):
    """A class that represents a file workspace."""

    class SystemSettings(Configurable.SystemSettings):
        name = "workspace"
        description = "The workspace is the root directory for all agent activity."
        configuration: AbstractFileWorkspaceConfiguration

    def __init__(self, config: AbstractFileWorkspaceConfiguration, *args, **kwargs):
        self._config = config
        pass

    on_write_file: Callable[[Path], Any] | None = None
    """
    Event hook, executed after writing a file.

    Params:
        Path: The path of the file that was written, relative to the workspace root.
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """The root path of the file workspace."""

    @property
    @abstractmethod
    def restrict_to_root(self) -> bool:
        """Whether to restrict file access to within the workspace's root path."""

    def initialize(self) -> None:
        """
        Calling `initialize()` should bring the workspace to a ready-to-use state.
        For example, it can create the resource in which files will be stored, if it
        doesn't exist yet. E.g. a folder on disk, or an S3 Bucket.
        """
        self._agent_workspace = self._sanitize_path(self._config.agent_workspace)
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        ...

    @overload
    @abstractmethod
    def open_file(
        self, path: str | Path, binary: Literal[False] = False
    ) -> TextIO | TextIOBase:
        """Returns a readable text file-like object representing the file."""

    @overload
    @abstractmethod
    def open_file(
        self, path: str | Path, binary: Literal[True] = True
    ) -> BinaryIO | IOBase:
        """Returns a readable binary file-like object representing the file."""

    @abstractmethod
    def open_file(self, path: str | Path, binary: bool = False) -> IO | IOBase:
        """Returns a readable file-like object representing the file."""

    @overload
    @abstractmethod
    def read_file(self, path: str | Path, binary: Literal[False] = False) -> str:
        """Read a file in the workspace as text."""
        ...

    @overload
    @abstractmethod
    def read_file(self, path: str | Path, binary: Literal[True] = True) -> bytes:
        """Read a file in the workspace as binary."""
        ...

    @abstractmethod
    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the workspace."""

    # async def write_file(self, path: str | Path, content: str | bytes) -> None:
    #     self._write_file(path, content)

    #     if self.on_write_file:
    #         path = Path(path)
    #         if path.is_absolute():
    #             path = path.relative_to(self.root)
    #         res = self.on_write_file(path)
    #         if inspect.isawaitable(res):
    #             await res

    # @abstractmethod
    # async def _write_file(self, path: str | Path, content: str | bytes) -> None:
    #     """Write to a file in the workspace."""

    @abstractmethod
    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the workspace."""

    @abstractmethod
    def list(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the workspace."""

    @abstractmethod
    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the workspace."""

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the workspace.

        Parameters:
            relative_path: The relative path to resolve in the workspace.

        Returns:
            Path: The resolved path relative to the workspace.
        """
        return self._sanitize_path(relative_path, self.root)

    @staticmethod
    @abstractmethod
    def _sanitize_path(
        relative_path: str | Path,
        agent_workspace_path: Optional[str | Path] = None,
        restrict_to_root: bool = True,
    ) -> Path:
        """Resolve the relative path within the given root if possible.

        Parameters:
            relative_path: The relative path to resolve.
            root: The root path to resolve the relative path within.
            restrict_to_root: Whether to restrict the path to the root.

        Returns:
            Path: The resolved path.

        Raises:
            ValueError: If the path is absolute and a root is provided.
            ValueError: If the path is outside the root and the root is restricted.
        """

        # Posix systems disallow null bytes in paths. Windows is agnostic about it.
        # Do an explicit check here for all sorts of null byte representations.

        if "\0" in str(relative_path) or "\0" in str(agent_workspace_path):
            raise ValueError("embedded null byte")

        if agent_workspace_path is None:
            return Path(relative_path).resolve()

        LOG.debug(
            f"Resolving path '{relative_path}' in workspace '{agent_workspace_path}'"
        )

        agent_workspace_path, relative_path = Path(
            agent_workspace_path
        ).resolve(), Path(relative_path)

        LOG.debug(f"Resolved root as '{agent_workspace_path}'")

        # Allow absolute paths if they are contained in the workspace.
        if (
            relative_path.is_absolute()
            and restrict_to_root
            and not relative_path.is_relative_to(agent_workspace_path)
        ):
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' "
                f"in workspace '{agent_workspace_path}'."
            )

        full_path = agent_workspace_path.joinpath(relative_path).resolve()

        LOG.debug(f"Joined paths as '{full_path}'")

        if restrict_to_root and not full_path.is_relative_to(agent_workspace_path):
            raise ValueError(
                f"Attempted to access path '{full_path}' outside of workspace '{agent_workspace_path}'."
            )

        return full_path

    @classmethod
    def create_workspace(
        cls,
        user_id: str,
        agent_id: str,
        settings: AbstractFileWorkspace.SystemSettings,
    ) -> Path:
        workspace = cls()
        workspace.initialize(config=settings.configuration)

        # log_path = workspace.root / "logs"
        # #FIXME: create a log file for each agent
        # log_path.mkdir(parents=True, exist_ok=True)
        # (log_path / "debug.log").touch()
        # (log_path / "cycle.log").touch()

        return workspace
