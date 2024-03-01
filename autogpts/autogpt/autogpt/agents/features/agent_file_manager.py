from __future__ import annotations

import logging

from autogpt.file_storage.base import FileStorage

from ..base import BaseAgent, BaseAgentSettings

logger = logging.getLogger(__name__)


class AgentFileManagerMixin:
    """Mixin that adds file manager (e.g. Agent state)
    and workspace manager (e.g. Agent output files) support."""

    files: FileStorage = None
    """Agent-related files, e.g. state, logs.
    Use `workspace` to access the agent's workspace files."""

    workspace: FileStorage = None
    """Workspace that the agent has access to, e.g. for reading/writing files.
    Use `files` to access agent-related files, e.g. state, logs."""

    STATE_FILE = "state.json"
    """The name of the file where the agent's state is stored."""

    LOGS_FILE = "file_logger.log"
    """The name of the file where the agent's logs are stored."""

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(AgentFileManagerMixin, self).__init__(**kwargs)

        if not isinstance(self, BaseAgent):
            raise NotImplementedError(
                f"{__class__.__name__} can only be applied to BaseAgent derivatives"
            )

        if "file_storage" not in kwargs:
            raise ValueError(
                "AgentFileManagerMixin requires a file_storage in the constructor."
            )

        state: BaseAgentSettings = getattr(self, "state")
        if not state.agent_id:
            raise ValueError("Agent must have an ID.")

        file_storage: FileStorage = kwargs["file_storage"]
        self.files = file_storage.clone_with_subroot(f"agents/{state.agent_id}/")
        self.workspace = file_storage.clone_with_subroot(
            f"agents/{state.agent_id}/workspace"
        )
        self._file_storage = file_storage
        # Read and cache logs
        self._file_logs_cache = []
        if self.files.exists(self.LOGS_FILE):
            self._file_logs_cache = self.files.read_file(self.LOGS_FILE).split("\n")

    async def log_file_operation(self, content: str) -> None:
        """Log a file operation to the agent's log file."""
        logger.debug(f"Logging operation: {content}")
        self._file_logs_cache.append(content)
        await self.files.write_file(
            self.LOGS_FILE, "\n".join(self._file_logs_cache) + "\n"
        )

    def get_file_operation_lines(self) -> list[str]:
        """Get the agent's file operation logs as list of strings."""
        return self._file_logs_cache

    async def save_state(self) -> None:
        """Save the agent's state to the state file."""
        state: BaseAgentSettings = getattr(self, "state")
        await self.files.write_file(self.files.root / self.STATE_FILE, state.json())

    def change_agent_id(self, new_id: str):
        """Change the agent's ID and update the file storage accordingly."""
        state: BaseAgentSettings = getattr(self, "state")
        # Rename the agent's files and workspace
        self._file_storage.rename(f"agents/{state.agent_id}", f"agents/{new_id}")
        # Update the file storage objects
        self.files = self._file_storage.clone_with_subroot(f"agents/{new_id}/")
        self.workspace = self._file_storage.clone_with_subroot(
            f"agents/{new_id}/workspace"
        )
        state.agent_id = new_id
