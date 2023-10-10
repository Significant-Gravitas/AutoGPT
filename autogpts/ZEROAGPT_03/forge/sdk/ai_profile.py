"""
PROFILE CONCEPT:

The profile generator is used to intiliase and configure an ai agent. 
It came from the obsivation that if an llm is provided with a profile such as:
```
Expert: 

```
Then it's performance at a task can impove. Here we use the profile to generate
a system prompt for the agent to use. However, it can be used to configure other
aspects of the agent such as memory, planning, and actions available.

The possibilities are limited just by your imagination.
"""

import openai
import os
import json

from forge.sdk import PromptEngine
from . import chat_completion_request


class ProfileGenerator:
    def __init__(
        self, 
        task: str,
        model: str = os.getenv("OPENAI_MODEL")):
        """
        Initialize the profile generator with the task to be performed.
        """
        self.task = task
        self.model = model
        
        self.prompt_engine = PromptEngine(self.model)

    async def role_find(self) -> str:
        """
        Ask LLM what role this task would fit
        Return role
        """
        role_prompt = self.prompt_engine.load_prompt(
            "role-selection",
            **{"task": self.task.input}
        )

        chat_list = [
            {
                "role": "system",
                "content": """
                Respond in JSON format:
                {
                    "name": "the expert's name",
                    "expertise": "specify the area in which the expert specializes"
                }
                Only return the JSON object          
                """
            },
            {
                "role": "system",
                "content": "You are a professional HR Specialist"
            },
            {
                "role": "user", 
                "content": role_prompt
            }
        ]

        chat_completion_parms = {
            "messages": chat_list,
            "model": self.model,
            "temperature": 0.6
        }

        response = await chat_completion_request(
            **chat_completion_parms)
        
        return response["choices"][0]["message"]["content"]


