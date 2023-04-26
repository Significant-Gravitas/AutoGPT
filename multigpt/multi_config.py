import os
from autogpt.config.config import Config
from multigpt.agent_selection import AgentSelection


class MultiConfig(Config):
    def __init__(self):
        super().__init__()
        next_agent_selection_str = os.getenv("NEXTAGENTSELECTION", "SMART_SELECTION")
        self.next_agent_selection = {
            "ROUND_ROBIN": AgentSelection.ROUND_ROBIN,
            "RANDOM": AgentSelection.RANDOM,
        }.get(next_agent_selection_str, AgentSelection.SMART_SELECTION)

        self.min_experts = os.getenv("MIN_EXPERTS", 2)
        self.max_experts = os.getenv("MAX_EXPERTS", 5)
        self.chat_only_mode = os.getenv("CHAT_ONLY_MODE", "True") == "True"
