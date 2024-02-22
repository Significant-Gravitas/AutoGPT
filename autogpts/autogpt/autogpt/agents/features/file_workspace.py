from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..base import BaseAgent, Config

from autogpt.file_workspace import (
    FileWorkspace,
    FileWorkspaceBackendName,
    get_workspace,
)

from ..base import AgentFileManager, BaseAgentSettings


class FileWorkspaceMixin:
    """Mixin that adds workspace support to a class"""

    workspace: FileWorkspace = None
    """Workspace that the agent has access to, e.g. for reading/writing files."""

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(FileWorkspaceMixin, self).__init__(**kwargs)

        state: BaseAgentSettings = getattr(self, "state")
        if not state.agent_id or not state.agent_data_dir:
            return

        self._setup_workspace()

    def attach_fs(self, agent_dir: Path):
        res = super(FileWorkspaceMixin, self).attach_fs(agent_dir)

        self._setup_workspace()

        return res

    def _setup_workspace(self) -> None:
        state: BaseAgentSettings = getattr(self, "state")
        assert state.agent_id, "Cannot attach workspace to anonymous agent"
        app_config: Config = getattr(self, "legacy_config")

        ws_backend = app_config.workspace_backend
        local = ws_backend == FileWorkspaceBackendName.LOCAL
        workspace = get_workspace(
            backend=ws_backend,
            id=state.agent_id if not local else "",
            root_path=state.agent_data_dir / "workspace" if local else None,
        )
        if local and state.config.allow_fs_access:
            workspace._restrict_to_root = False  # type: ignore
        workspace.initialize()
        self.workspace = workspace
