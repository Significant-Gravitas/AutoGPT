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
    return safe_path_join(WORKSPACE_PATH, relative_path)


def safe_path_join(base: Path, *paths: str | Path) -> Path:
    """Join one or more path components, asserting the resulting path is within the workspace.

    Args:
        base (Path): The base path
        *paths (str): The paths to join to the base path

    Returns:
        Path: The joined path
    """
    base = base.resolve()
    joined_path = base.joinpath(*paths).resolve()

    if CFG.restrict_to_workspace and not joined_path.is_relative_to(base):
        raise ValueError(
            f"Attempted to access path '{joined_path}' outside of workspace '{base}'."
        )

    return joined_path
