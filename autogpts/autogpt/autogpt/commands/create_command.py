from typing import Iterator

from autogpt.agents.protocols import CommandProvider, DirectiveProvider
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command


class CreateCommandComponent(DirectiveProvider, CommandProvider):
    """
    Writes a new command for the agent using the AutoGPT Codex API.
    Once the command is created, it is added to the agent's command list.
    """

    def __init__(self, codex_base_url: str = "http://127.0.0.1:8080/api/v1") -> None:
        self.codex_base_url = codex_base_url
        super().__init__()

    def get_resources(self) -> Iterator[str]:
        yield "Ability to create a new commands to use."

    def get_commands(self) -> Iterator["Command"]:  # type: ignore
        yield self.execute_create_command

    @command(
        ["execute_create_command"],
        "Writes a new command for the agent and adds it to the agent's command list.",
        {
            "command_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the new command",
                required=True,
            ),
            "description": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The description of what the command needs to do",
                required=True,
            ),
        },
    )
    def execute_create_command(self, command_name: str, description: str) -> str:
        return ""
