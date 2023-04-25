from __future__ import annotations

import os
from pathlib import Path

from autogpt.config import Config

CFG = Config()

# Set a dedicated folder for file I/O
WORKSPACE_PATH = Path(os.getcwd()) / "auto_gpt_workspace"

# Create the directory if it doesn't exist
WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)

def path_in_workspace(relative_path: str | Path) -> Path:
    """Get full path for item in workspace.

    Args:
        relative_path (str | Path): Path to translate into the workspace.

    Returns:
        Path: Absolute path for the given path in the workspace.
    """
    return safe_path_join(WORKSPACE_PATH, relative_path)


def safe_path_join(base: Path, *paths: str | Path) -> Path:
    """Join one or more path components, asserting the resulting path is within the workspace.

    Args:
        base (Path): The base path.
        *paths (str | Path): The paths to join to the base path.

    Returns:
        Path: The joined path.
    """
    base = base.resolve()
    joined_path = base.joinpath(*paths).resolve()

    if CFG.restrict_to_workspace and not is_path_within_workspace(base, joined_path):
        raise ValueError(
            f"Attempted to access path '{joined_path}' outside of workspace '{base}'."
        )

    return joined_path


def is_path_within_workspace(base: Path, path: Path) -> bool:
    """Check if a given path is within the workspace.

    Args:
        base (Path): The base path (workspace path).
        path (Path): The path to check.

    Returns:
        bool: True if the path is within the workspace, False otherwise.
    """
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
