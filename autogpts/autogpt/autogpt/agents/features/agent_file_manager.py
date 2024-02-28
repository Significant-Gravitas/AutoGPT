from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseAgent

import logging

from autogpt.agents.utils.file_manager import FileManager

from ..base import BaseAgentSettings

logger = logging.getLogger(__name__)


class AgentFileManagerMixin:
    """Mixin that adds file manager (e.g. Agent state)
    and workspace manager (e.g. Agent output files) support."""

    files: FileManager = None
    """Agent-related files, e.g. state, logs.
    Use `workspace` to access the agent's workspace files."""

    workspace: FileManager = None
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
        if not state.agent_id or not state.agent_data_dir:
            raise ValueError("Agent must have an ID and a data directory.")

        file_storage = kwargs["file_storage"]
        self.files = FileManager(file_storage, f"agents/{state.agent_id}/")
        self.workspace = FileManager(file_storage, f"agents/{state.agent_id}/workspace")
        # Read and cache logs
        self._logs_cache = []
        if self.files.exists(self.LOGS_FILE):
            self._logs_cache = self.files.read_file(self.LOGS_FILE).split("\n")

    async def log_operation(self, content: str) -> None:
        """Log an operation to the agent's log file."""
        logger.debug(f"Logging operation: {content}")
        self._logs_cache.append(content)
        await self.files.write_file(self.LOGS_FILE, "\n".join(self._logs_cache) + "\n")

    def get_logs(self) -> list[str]:
        """Get the agent's logs."""
        return self._logs_cache

    async def save_state(self) -> None:
        """Save the agent's state to the state file."""
        state: BaseAgentSettings = getattr(self, "state")
        await self.files.write_file(self.files.root / self.STATE_FILE, state.json())
