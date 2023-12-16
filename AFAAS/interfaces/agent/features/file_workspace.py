from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from AFAAS.interfaces.agent import BaseAgent, Config

from autogpt.file_workspace import (
    AbstractFileWorkspace,
    AbstractFileWorkspaceBackendName,
    get_workspace,
)

from ..base import AgentFileManager, BaseAgentSettings


class AbstractFileWorkspaceMixin:
    """Mixin that adds workspace support to a class"""

    workspace: AbstractFileWorkspace = None
    """Workspace that the agent has access to, e.g. for reading/writing files."""

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(AbstractFileWorkspaceMixin, self).__init__(**kwargs)

        file_manager: AgentFileManager = getattr(self, "file_manager")
        if not file_manager:
            return

        self._setup_workspace()

    def attach_fs(self, agent_dir: Path):
        res = super(AbstractFileWorkspaceMixin, self).attach_fs(agent_dir)

        self._setup_workspace()

        return res

    def _setup_workspace(self) -> None:
        settings: BaseAgentSettings = getattr(self, "state")
        assert settings.agent_id, "Cannot attach workspace to anonymous agent"
        app_config: Config = getattr(self, "legacy_config")
        file_manager: AgentFileManager = getattr(self, "file_manager")

        ws_backend = app_config.workspace_backend
        local = ws_backend == AbstractFileWorkspaceBackendName.LOCAL
        workspace = get_workspace(
            backend=ws_backend,
            id=settings.agent_id if not local else "",
            root_path=file_manager.root / "workspace" if local else None,
        )
        if local and settings.config.allow_fs_access:
            workspace._restrict_to_root = False  # type: ignore
        workspace.initialize()
        self.workspace = workspace


def get_agent_workspace(agent: BaseAgent) -> AbstractFileWorkspace | None:
    if isinstance(agent, AbstractFileWorkspaceMixin):
        return agent.workspace

    return None
