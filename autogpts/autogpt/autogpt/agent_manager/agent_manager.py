from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from autogpt.config.config import Config
from autogpt.file_workspace import get_workspace
from autogpt.file_workspace.base import FileWorkspace
from autogpt.agents.agent import AgentSettings

if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings

from autogpt.agents.utils.agent_file_manager import AgentFileManager


class AgentManager:
    def __init__(self, legacy_config: Config):
        self.agents_dir = legacy_config.app_data_dir / "agents"
        self.workspace: FileWorkspace = None
        self._setup_workspace(legacy_config)

    def _setup_workspace(self, config: Config) -> None:
        fm_backend = config.file_manager_backend
        workspace = get_workspace(
            backend=fm_backend,
            root_path=config.app_data_dir,
        )

        workspace.initialize()
        workspace.make_dir(self.agents_dir)
        self.workspace = workspace

    @staticmethod
    def generate_id(agent_name: str) -> str:
        unique_id = str(uuid.uuid4())[:8]
        return f"{agent_name}-{unique_id}"

    def list_agents(self) -> list[str]:
        return [
            dir.name
            for dir in self.agents_dir.iterdir()
            if dir.is_dir() and AgentFileManager.get_state_file_path(dir).exists()
        ]

    def get_agent_dir(self, agent_id: str, must_exist: bool = False) -> Path:
        assert len(agent_id) > 0
        agent_dir = self.agents_dir / agent_id
        if must_exist and not agent_dir.exists():
            raise FileNotFoundError(f"No agent with ID '{agent_id}'")
        return agent_dir

    def load_agent_state(self, agent_id: str) -> AgentSettings:
        agent_dir = self.get_agent_dir(agent_id, True)
        state_file_path = AgentFileManager.get_state_file_path(agent_dir)
        if not state_file_path.exists():
            raise FileNotFoundError(f"Agent with ID '{agent_id}' has no state.json")
        
        text = self.workspace.read_file(state_file_path)
        return AgentSettings.from_json(text)
        

