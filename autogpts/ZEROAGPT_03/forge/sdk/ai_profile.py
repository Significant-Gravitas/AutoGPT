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

from forge.sdk import PromptEngine


class ProfileGenerator:
    def __init__(self, task: str, prompt_engine: PromptEngine):
        """
        Initialize the profile generator with the task to be performed.
        """
        self.task = task
        self.prompt_engine = prompt_engine

    def role_find(self) -> str:
        """
        Ask LLM what role this task would fit
        Return role
        """
        role_prompt = self.prompt_engine.load_prompt(
            "role-selection",
            **{"task": self.task.input}
        )

        model = "text-davinci-003"
        response = openai.Completion.create(engine=model, prompt=role_prompt, max_tokens=50)

        return response.choices[0].text.strip()


