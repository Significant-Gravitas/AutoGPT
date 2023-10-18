from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentFileManager:
    """A class that represents a workspace for an AutoGPT agent."""

    def __init__(self, agent_data_dir: Path):
        self._root = agent_data_dir.resolve()

    @property
    def root(self) -> Path:
        """The root directory of the workspace."""
        return self._root

    def initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)
        self.init_file_ops_log(self.file_ops_log_path)

    @property
    def state_file_path(self) -> Path:
        return self.root / "state.json"

    @property
    def file_ops_log_path(self) -> Path:
        return self.root / "file_logger.log"

    @staticmethod
    def init_file_ops_log(file_logger_path: Path) -> Path:
        if not file_logger_path.exists():
            with file_logger_path.open(mode="w", encoding="utf-8") as f:
                f.write("")
        return file_logger_path
