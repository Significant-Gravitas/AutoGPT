"""
AIPlanning class

Create a plane of what steps to do and what abilities to use for each task
as a system prompt for completion of the task
"""

import os

from .forge_log import ForgeLogger

from . import PromptEngine
from . import chat_completion_request
from . import Workspace

class AIPlanning:
    def __init__(
        self, 
        task: str,
        task_id: str,
        abilities: str,
        workspace: Workspace,
        model: str = os.getenv("OPENAI_MODEL")):
        
        self.task = task
        self.task_id = task_id
        self.abilities = abilities
        self.workspace = workspace
        self.model = model

        self.logger = ForgeLogger(__name__)

        self.prompt_engine = PromptEngine(os.getenv("OPENAI_MODEL"))

    async def create_steps(self) -> str:
        step_format_prompt = self.prompt_engine.load_prompt(
            "step-format"
        )

        # add abilities prompt
        abilities_prompt = self.prompt_engine.load_prompt(
            "abilities-list",
            **{"abilities": self.abilities}
        )
        
        step_prompt = self.prompt_engine.load_prompt(
            "get-steps",
            **{
                "task": self.task
            }
        )

        chat_list = [
            {
                "role": "system",
                "content": step_format_prompt
            },
            {
                "role": "system",
                "content": abilities_prompt
            },
            {
                "role": "system",
                "content": "You are a professional Project Planner."
            },
            {
                "role": "user", 
                "content": step_prompt
            }
        ]

        self.logger.info(f"🤔 AIPlanner\n")
        for chat in chat_list:
            self.logger.info(f"role: {chat['role']}\ncontent: {chat['content']}")

        chat_completion_parms = {
            "messages": chat_list,
            "model": self.model,
            "temperature": 0.8
        }
        
        response = await chat_completion_request(
            **chat_completion_parms)
            
        return response["choices"][0]["message"]["content"]