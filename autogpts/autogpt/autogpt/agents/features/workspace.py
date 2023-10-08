from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..base import BaseAgent

from autogpt.workspace import Workspace

from ..base import AgentFileManager, BaseAgentConfiguration


class WorkspaceMixin:
    """Mixin that adds workspace support to a class"""

    workspace: Workspace | None
    """Workspace that the agent has access to, e.g. for reading/writing files."""

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(WorkspaceMixin, self).__init__(**kwargs)

        config: BaseAgentConfiguration = getattr(self, "config")
        if not isinstance(config, BaseAgentConfiguration):
            raise ValueError(
                "Cannot initialize Workspace for Agent without compatible .config"
            )
        file_manager: AgentFileManager = getattr(self, "file_manager")
        if not file_manager:
            return

        self.workspace = _setup_workspace(file_manager, config)

    def attach_fs(self, agent_dir: Path):
        res = super(WorkspaceMixin, self).attach_fs(agent_dir)

        self.workspace = _setup_workspace(self.file_manager, self.config)

        return res


def _setup_workspace(file_manager: AgentFileManager, config: BaseAgentConfiguration):
    workspace = Workspace(
        file_manager.root / "workspace",
        restrict_to_workspace=not config.allow_fs_access,
    )
    workspace.initialize()
    return workspace


def get_agent_workspace(agent: BaseAgent) -> Workspace | None:
    if isinstance(agent, WorkspaceMixin):
        return agent.workspace

    return None
