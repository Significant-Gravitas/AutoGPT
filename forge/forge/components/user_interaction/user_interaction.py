from typing import Iterator

import click

from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.config.config import Config
from forge.json.model import JSONSchema
from forge.utils.const import ASK_COMMAND


class UserInteractionComponent(CommandProvider):
    """Provides commands to interact with the user."""

    def __init__(self, config: Config):
        self._enabled = not config.noninteractive_mode

    def get_commands(self) -> Iterator[Command]:
        yield self.ask_user

    @command(
        names=[ASK_COMMAND],
        parameters={
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The question or prompt to the user",
                required=True,
            )
        },
    )
    def ask_user(self, question: str) -> str:
        """If you need more details or information regarding the given goals,
        you can ask the user for input."""
        print(f"\nQ: {question}")
        resp = click.prompt("A:")
        return f"The user's answer: '{resp}'"
