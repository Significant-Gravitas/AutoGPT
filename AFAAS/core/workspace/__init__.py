"""The workspace is the central hub for the Agent's on disk resources."""
from AFAAS.core.workspace.base import AbstractWorkspace
from AFAAS.core.workspace.simple import SimpleWorkspace


__all__ = [
    "SimpleWorkspace",
    "Workspace",
    "WorkspaceSettings",
]
