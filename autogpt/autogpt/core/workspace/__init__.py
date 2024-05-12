"""The workspace is the central hub for the Agent's on disk resources."""
from autogpt.core.workspace.base import Workspace
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings

__all__ = [
    "SimpleWorkspace",
    "Workspace",
    "WorkspaceSettings",
]
