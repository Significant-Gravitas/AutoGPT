"""The workspace is the central hub for the Agent's on disk resources."""
from autogpts.autogpt.autogpt.core.workspace.base import AbstractWorkspace
from autogpts.autogpt.autogpt.core.workspace.simple import SimpleWorkspace


__all__ = [
    "SimpleWorkspace",
    "Workspace",
    "WorkspaceSettings",
]
