from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from autogpt.agents.agent import AgentSettings
from autogpt.agents.utils.file_manager import FileManager
from autogpt.file_storage.base import FileStorage

if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings


class AgentManager:
    def __init__(self, file_storage: FileStorage):
        self.file_manager = FileManager(file_storage, "agents")

    @staticmethod
    def generate_id(agent_name: str) -> str:
        unique_id = str(uuid.uuid4())[:8]
        return f"{agent_name}-{unique_id}"

    def list_agents(self) -> list[str]:
        agent_dirs: list[str] = []
        for dir in self.file_manager.list_folders():
            if dir.is_dir() and self.file_manager.exists(dir / "state.json"):
                agent_dirs.append(dir.name)
        return agent_dirs

    def get_agent_dir(self, agent_id: str) -> Path:
        assert len(agent_id) > 0
        agent_dir: Path | None = None
        if self.file_manager.exists(agent_id):
            agent_dir = self.file_manager.root / agent_id
        else:
            raise FileNotFoundError(f"No agent with ID '{agent_id}'")
        return agent_dir

    def load_agent_state(self, agent_id: str) -> AgentSettings:
        state_file_path = Path(agent_id) / "state.json"
        if not self.file_manager.exists(state_file_path):
            raise FileNotFoundError(f"Agent with ID '{agent_id}' has no state.json")

        text = self.file_manager.read_file(state_file_path)
        return AgentSettings.parse_raw(text)
