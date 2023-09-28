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

from forge.sdk import PromptEngine


class ProfileGenerator:
    def __init__(self, task: str, PromptEngine: PromptEngine):
        """
        Initialize the profile generator with the task to be performed.
        """
        self.task = task
