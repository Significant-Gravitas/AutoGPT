from typing import Iterator

from autogpt.agents.protocols import CommandProvider
from autogpt.app.utils import clean_input
from autogpt.command_decorator import command
from autogpt.config.config import Config
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command
from autogpt.utils.utils import DEFAULT_ASK_COMMAND


class UserInteractionComponent(CommandProvider):
    """Provides commands to interact with the user."""

    def __init__(self, config: Config):
        self.config = config
        self._enabled = not config.noninteractive_mode

    def get_commands(self) -> Iterator[Command]:
        yield self.ask_user

    @command(
        names=[DEFAULT_ASK_COMMAND],
        parameters={
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The question or prompt to the user",
                required=True,
            )
        },
    )
    def ask_user(self, question: str) -> str:
        """If you need more details or information regarding the given task,
        you can ask the user for input."""
        print(f"\nQ: {question}")
        resp = clean_input(self.config, "A:")
        return f"The user's answer: '{resp}'"
