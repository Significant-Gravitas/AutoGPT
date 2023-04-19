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
    sanitized_paths = [sanitize_path(path) for path in paths]
    joined_path = base.joinpath(*sanitized_paths).resolve(strict=False)
    real_joined_path = Path(os.path.realpath(joined_path))

    if CFG.restrict_to_workspace and not real_joined_path.is_relative_to(base):
        raise ValueError(
            f"Attempted to access path '{real_joined_path}' outside of workspace '{base}'."
        )

    return real_joined_path

def sanitize_path(path: str | Path) -> str:
    """Sanitize user-provided path to prevent path traversal.

    Args:
        path (str | Path): The path to sanitize

    Returns:
        str: The sanitized path
    """
    if '\x00' in str(path):
        raise ValueError(f"Path contains NULL bytes: '{path}'")

    sanitized_path = os.path.normpath(str(path))
    if '..' in sanitized_path or os.path.isabs(sanitized_path):
        raise ValueError(f"Invalid relative path '{sanitized_path}' detected.")

    # Check for symbolic links in the joined path
    try:
        # Join the input path to the current working directory before calling os.path.realpath()
        real_path = os.path.realpath(os.path.join(os.getcwd(), sanitized_path))
        if os.path.islink(real_path):
            raise ValueError(f"Symbolic link detected in path '{sanitized_path}'")
    except OSError:
        pass

    # Ensure the path does not contain symbolic links
    path_parts = Path(sanitized_path).parts
    if any(os.path.islink(part) for part in path_parts):
        raise ValueError(f"Symbolic link detected in path '{sanitized_path}'")

    return sanitized_path
