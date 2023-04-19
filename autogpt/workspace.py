from __future__ import annotations

import os
from pathlib import Path

from autogpt.config import Config

CFG = Config()

# Set a dedicated folder for file I/O
WORKSPACE_PATH = Path(os.getcwd()) / "auto_gpt_workspace"

# Create the directory if it doesn't exist
if not os.path.exists(WORKSPACE_PATH):
    os.makedirs(WORKSPACE_PATH)


def path_in_workspace(relative_path: str | Path) -> Path:
    """Get full path for item in workspace

    Parameters:
        relative_path (str | Path): Path to translate into the workspace

    Returns:
        Path: Absolute path for the given path in the workspace
    """
    return WORKSPACE_PATH.joinpath(relative_path)
