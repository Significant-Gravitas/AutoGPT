from __future__ import annotations

import logging
from pathlib import Path

from autogpt.file_workspace.base import FileWorkspace

logger = logging.getLogger(__name__)


class AgentFileManager:
    """A class that represents a workspace for an AutoGPT agent."""

    def __init__(self, agent_data_dir: Path, file_workspace: FileWorkspace):
        self._root = agent_data_dir.resolve()
        self.workspace = file_workspace
        self.workspace.make_dir(self.root)
        self.init_file_ops_log()

    @property
    def root(self) -> Path:
        """The root directory of the workspace."""
        return self._root

    @property
    def state_file_path(self) -> Path:
        return self.root / "state.json"

    @property
    def file_ops_log_path(self) -> Path:
        return self.root / "file_logger.log"

    def init_file_ops_log(self) -> None:
        if not self.workspace.exists(self.file_ops_log_path):
            self.workspace.write_file(self.file_ops_log_path, "")

    def log_operation(self, content: str) -> None:
        logs = self.workspace.read_file(self.file_ops_log_path)
        logs += content
        self.workspace.write_file(self.file_ops_log_path, logs)

    @staticmethod
    def get_state_file_path(agent_data_dir: Path) -> Path:
        return agent_data_dir / "state.json"
