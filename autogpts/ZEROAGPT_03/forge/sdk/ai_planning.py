"""
AIPlanning class

Create a plane of what steps to do and what abilities to use for each task
as a system prompt for completion of the task
"""

import openai
import os
import json

from .forge_log import ForgeLogger

from . import PromptEngine
from . import chat_completion_request

class AIPlanning:
    def __init__(
        self, 
        task: str,
        abilities: str):
        
        self.task = task
        self.abilities = abilities

        self.logger = ForgeLogger(__name__)

        self.prompt_engine = PromptEngine(os.getenv("OPENAI_MODEL"))

    async def create_steps(self) -> str:
        step_prompt = self.prompt_engine.load_prompt(
            "get-steps",
            **{
                "task": self.task,
                "abilities": self.abilities
            }
        )

        chat_list = [
            {
                "role": "system",
                "content": "You are a professional Project Planner."
            },
            {
                "role": "user", 
                "content": step_prompt
            }
        ]

        self.logger.info(f"[AIPlanner] chat: {chat_list}")

        chat_completion_parms = {
            "messages": chat_list,
            "model": os.getenv("OPENAI_MODEL"),
            "temperature": 0.0
        }
        
        response = await chat_completion_request(
            **chat_completion_parms)
            
        return response["choices"][0]["message"]["content"]