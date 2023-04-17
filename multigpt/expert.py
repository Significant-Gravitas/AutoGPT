import yaml
import os
from typing import List
from autogpt.config.ai_config import AIConfig
from multigpt.multi_agent_manager import MULTIAGENTMANAGER

class Expert(AIConfig):
    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)

    def to_string(self) -> str:
        return f"Name: {self.name}, Role: {self.role}, Goals: {self.goals}"

    def construct_full_prompt(self, experts) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
              including the ai_name, ai_role and ai_goals.
        """

        prompt_start = (
            "Your decisions must always be made independently but you are allowed to collaborate, discuss and disagree"
            " with your team members. Play to your strengths as ChatGPT and pursue"
            " simple strategies with no legal complications."
            f"\n\nYour team consists of: {MULTIAGENTMANAGER.experts_to_list(experts)}"
            "\n\nFoster critical discussions but avoid conforming to others' ideas within team collaboration."
            "Critically question other's opinions and disagree. Disagree frequently!"
        )

        from multigpt.prompt import get_prompt

        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt()}"
        return full_prompt