from colorama import Fore

from autogpt.logs import logger
from autogpt.promptgenerator import PromptGenerator
from autogpt.setup import prompt_user
from autogpt.utils import clean_input


class MultiPromptGenerator(PromptGenerator):

    def __init__(self, cfg) -> None:
        """
        Initialize the MultiPromptGenerator object with empty lists of constraints,
            commands, resources, and performance evaluations.
            This is identical to the PromptGeneratorClass, except that text/speak are different for these agents.
        """
        super().__init__()
        self.response_format = {
            "thoughts": {
                "text": "Thoughts you keep to yourself. They are NOT shared with your team.",
                "reasoning": "reasoning",
                "plan": "- short bulleted\n- list that conveys\n- long-term plan",
                "criticism": "constructive self-criticism",
                "speak": "Thoughts you say out loud. They ARE shared with your team.",
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }
        self.cfg = cfg

    def construct_prompt(self) -> str:
        # TODO MAKE USE OF THIS INSTEAD OF construct_prompt IN prompt.py
        pass
