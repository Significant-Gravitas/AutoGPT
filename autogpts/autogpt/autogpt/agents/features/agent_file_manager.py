from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..base import BaseAgent, Config

import logging
from pathlib import Path

from autogpt.agents.utils.file_manager import FileManager

from ..base import BaseAgent, BaseAgentSettings

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
                "AgentFileManagerMixin requires a file_storage argument in the constructor."
            )

        state: BaseAgentSettings = getattr(self, "state")
        if not state.agent_id or not state.agent_data_dir:
            raise ValueError("Agent must have an ID and a data directory.")

        file_storage = kwargs["file_storage"]
        self.files = FileManager(file_storage, f"agents/{state.agent_id}/")
        self.workspace = FileManager(file_storage, f"agents/{state.agent_id}/workspace")

    async def log_operation(self, content: str) -> None:
        logger.debug(f"Logging operation: {content}")
        logs = ""
        # TODO kcze maybe instead of reading each time just cache the logs
        if self.files.exists(self.LOGS_FILE):
            logs = self.files.read_file(self.LOGS_FILE) + "\n"
        await self.files.write_file(self.LOGS_FILE, logs + content)

    def get_logs(self) -> list[str]:
        if not self.files.exists(self.LOGS_FILE):
            return []
        return self.files.read_file(self.LOGS_FILE).split("\n")

    async def save_state(self) -> None:
        state: BaseAgentSettings = getattr(self, "state")
        await self.files.write_file(self.files.root / self.STATE_FILE, state.json())
