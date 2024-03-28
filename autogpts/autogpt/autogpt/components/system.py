from typing import Iterator

from autogpt.command_decorator import command

from autogpt.agents.utils.exceptions import (
    AgentFinished,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.agents.components import (
    Component,
)
from autogpt.agents.protocols import CommandProvider
from autogpt.models.command import Command


class SystemComponent(Component, CommandProvider):
    def get_commands(self) -> Iterator[Command]:
        yield self.finish.command

    @command(
        parameters={
            "reason": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A summary to the user of how the goals were accomplished",
                required=True,
            ),
        }
    )
    def finish(self, reason: str):
        """Use this to shut down once you have completed your task,
        or when there are insurmountable problems that make it impossible
        for you to finish your task."""
        raise AgentFinished(reason)
