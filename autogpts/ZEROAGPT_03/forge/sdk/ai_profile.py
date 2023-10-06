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
    def __init__(self, task: str, prompt_engine: PromptEngine):
        """
        Initialize the profile generator with the task to be performed.
        """
        self.task = task
        self.prompt_engine = prompt_engine

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
                Reply only in JSON exactly in the following format:
                {
                    "name": "the expert's name",
                    "expertise": "specify the area in which the expert specializes"
                }
                                    
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
            "model": os.getenv("OPENAI_MODEL"),
            "temperature": 0.9
        }

        response = await chat_completion_request(
            **chat_completion_parms)

        json_resp = "{}"

        try:
            json_resp = json.loads(
                response["choices"][0]["message"]["content"])
        except Exception as err:
            self.logger.error(f"JSON loads failed: {err}")
            print(
                response["choices"][0]["message"]["content"])
            
        return json_resp


