from __future__ import annotations
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    
from autogpt.app.utils import clean_input
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.agents.components import Component
from autogpt.agents.protocols import CommandProvider
from autogpt.config.config import Config
from autogpt.models.command import Command
from autogpt.models.command_parameter import CommandParameter


def ask_user(question: str, agent: Agent) -> str:
    print(f"\nQ: {question}")
    resp = clean_input(agent.legacy_config, "A:")
    return f"The user's answer: '{resp}'"


class UserInteractionComponent(Component, CommandProvider):
    """Provides commands to interact with the user"""

    def __init__(self, config: Config):
        self.enabled = not config.noninteractive_mode

    def get_commands(self) -> Iterator[Command]:
        yield Command(
            "ask_user",
            "If you need more details or information regarding the given goals,"
            " you can ask the user for input",
            ask_user,
            [
                CommandParameter(
                    "question",
                    JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description="The question or prompt to the user",
                        required=True,
                    ),
                )
            ],
        )