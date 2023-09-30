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


class ProfileGenerator:
    def __init__(self, task: str):
        """
        Initialize the profile generator with the task to be performed.
        """
        self.task = task

    def role_find(self) -> str:
        """
        Ask LLM what role this task would fit
        Return role
        """

        prompt = f"Give me the type of expert role this task would work great for. Return only the type of expert name and nothing else. {self.task.input}"

        model = "text-davinci-003"
        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50)

        return response.choices[0].text.strip()


