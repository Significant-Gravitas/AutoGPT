from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from autogpt.agents import Agent

from autogpt.agents.utils.exceptions import (
    AgentFinished,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command_parameter import CommandParameter
from autogpt.agents.components import (
    Component,
)
from autogpt.agents.protocols import CommandProvider
from autogpt.models.command import Command


class SystemComponent(Component, CommandProvider):
    def get_commands(self) -> Iterator[Command]:
        yield Command(
            "finish",
            "Use this to shut down once you have completed your task,"
            " or when there are insurmountable problems that make it impossible"
            " for you to finish your task.",
            self.finish,
            [
                CommandParameter(
                    "reason",
                    JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description="A summary to the user of how the goals were accomplished",
                        required=True,
                    ),
                ),
            ],
        )

    def finish(self, reason: str, agent):
        raise AgentFinished(reason)
