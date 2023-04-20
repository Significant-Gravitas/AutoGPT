import os
from autogpt.config.config import Config
from multigpt.agent_selection import AgentSelection


class MultiConfig(Config):
    def __init__(self):
        super().__init__()
        # TODO: find a more elegant solution
        next_agent_selection_str = os.getenv("NEXTAGENTSELECTION", "ROUND_ROBIN")
        if next_agent_selection_str == "SMART_SELECTION":
            self.next_agent_selection = AgentSelection.SMART_SELECTION
        elif next_agent_selection_str == "RANDOM":
            self.next_agent_selection = AgentSelection.RANDOM
        else:
            self.next_agent_selection = AgentSelection.ROUND_ROBIN

        self.min_experts = os.getenv("MIN_EXPERTS", 2)
        self.max_experts = os.getenv("MAX_EXPERTS", 5)
        self.chat_only_mode = (
            os.getenv("CHAT_ONLY_MODE", "True") == "True"
        )