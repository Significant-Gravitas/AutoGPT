"""
=========
Workspace
=========

The workspace is a directory containing configuration and working files for an AutoGPT
agent.

"""
from __future__ import annotations
from pathlib import Path
from autogpt.config import Config
from autogpt.core.workspace.simple import SimpleWorkspace


class Workspace(SimpleWorkspace):
    """A class that represents a workspace for an AutoGPT agent."""
    @staticmethod
    def build_file_logger_path(config: Config, workspace_directory: Path):
        file_logger_path = workspace_directory / "file_logger.txt"
        if not file_logger_path.exists():
            with file_logger_path.open(mode="w", encoding="utf-8") as f:
                f.write("File Operation Logger ")
        config.file_logger_path = str(file_logger_path)
