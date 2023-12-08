from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings

from autogpt.agents.utils.agent_file_manager import AgentFileManager


class AgentManager:
    def __init__(self, app_data_dir: Path):
        self.agents_dir = app_data_dir / "agents"
        if not self.agents_dir.exists():
            self.agents_dir.mkdir()

    @staticmethod
    def generate_id(agent_name: str) -> str:
        unique_id = str(uuid.uuid4())[:8]
        return f"{agent_name}-{unique_id}"

    def list_agents(self) -> list[str]:
        return [
            dir.name
            for dir in self.agents_dir.iterdir()
            if dir.is_dir() and AgentFileManager(dir).state_file_path.exists()
        ]

    def get_agent_dir(self, agent_id: str, must_exist: bool = False) -> Path:
        assert len(agent_id) > 0
        agent_dir = self.agents_dir / agent_id
        if must_exist and not agent_dir.exists():
            raise FileNotFoundError(f"No agent with ID '{agent_id}'")
        return agent_dir

    def retrieve_state(self, agent_id: str) -> AgentSettings:
        from autogpt.agents.agent import AgentSettings

        agent_dir = self.get_agent_dir(agent_id, True)
        state_file = AgentFileManager(agent_dir).state_file_path
        if not state_file.exists():
            raise FileNotFoundError(f"Agent with ID '{agent_id}' has no state.json")

        state = AgentSettings.load_from_json_file(state_file)
        state.agent_data_dir = agent_dir
        return state
