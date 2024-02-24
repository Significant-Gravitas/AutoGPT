from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..base import BaseAgent, Config

from ..base import BaseAgentSettings
import logging
from pathlib import Path

from autogpt.agents.utils.file_manager import FileManager

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

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(AgentFileManagerMixin, self).__init__(**kwargs)

        if "file_storage" not in kwargs:
            raise ValueError(
                "AgentFileManagerMixin requires a file_storage argument in the constructor."
            )

        state: BaseAgentSettings = getattr(self, "state")
        if not state.agent_id or not state.agent_data_dir:
            raise ValueError("Agent must have an ID and a data directory.")

        self._file_storage = kwargs["file_storage"]
        self.files = FileManager(self._file_storage, f"agents/{state.agent_id}/")
        self.workspace = FileManager(
            self._file_storage, f"agents/{state.agent_id}/workspace/"
        )
        self.init_file_ops_log()

    @property
    def _file_ops_log_path(self) -> Path:
        return self.files.root / "file_logger.log"

    def init_file_ops_log(self) -> None:
        if not self.files.exists(self._file_ops_log_path):
            self.workspace.write_file(self._file_ops_log_path, "")

    def log_operation(self, content: str) -> None:
        # TODO kcze - better would be to append
        logs = self.workspace.read_file(self._file_ops_log_path)
        logs = f"{logs}\n{content}"
        self.workspace.write_file(self._file_ops_log_path, logs)

    def save_state(self) -> None:
        state: BaseAgentSettings = getattr(self, "state")
        self.files.write_file(self.files.root / "state.json", state.model_dump_json())
