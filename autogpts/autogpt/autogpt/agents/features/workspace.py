from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseAgent

from autogpt.config import Config
from autogpt.workspace import Workspace


class WorkspaceMixin:
    """Mixin that adds workspace support to a class"""

    workspace: Workspace
    """Workspace that the agent has access to, e.g. for reading/writing files."""

    def __init__(self, **kwargs):
        # Initialize other bases first, because we need the config from BaseAgent
        super(WorkspaceMixin, self).__init__(**kwargs)

        legacy_config: Config = getattr(self, "legacy_config")
        if not isinstance(legacy_config, Config):
            raise ValueError(f"Cannot initialize Workspace for Agent without Config")
        if not legacy_config.workspace_path:
            raise ValueError(
                f"Cannot set up Workspace: no WORKSPACE_PATH in legacy_config"
            )

        self.workspace = Workspace(
            legacy_config.workspace_path, legacy_config.restrict_to_workspace
        )


def get_agent_workspace(agent: BaseAgent) -> Workspace | None:
    if isinstance(agent, WorkspaceMixin):
        return agent.workspace

    return None
