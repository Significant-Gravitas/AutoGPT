from typing import Iterator

from autogpt.agents.protocols import CommandProvider, DirectiveProvider
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command
from pydantic import BaseModel
import aiohttp
import logging
import asyncio

logger = logging.getLogger(__name__)

class FunctionSpecRequest(BaseModel):
    """
    A request to generate a correctly formated function spec
    """

    name: str
    description: str
    inputs: str
    outputs: str


class FunctionResponse(BaseModel):
    id: str
    name: str
    requirements: list[str]
    code: str


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
            "inputs": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The description of what inputs the command needs",
                required=True,
            ),
            "output": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The description of what the command needs to output",
                required=True,
            ),
        },
    )
    
    def execute_create_command(
        self, command_name: str, description: str, inputs: str, outputs: str
    ) -> str:
        """
        Executes the create command with the given parameters.

        Args:
            command_name (str): The name of the command.
            description (str): The description of the command.
            inputs (str): The inputs of the command.
            outputs (str): The outputs of the command.

        Returns:
            str: The generated code.

        Raises:
            Exception: If there is an error when calling Codex.
        """
        req = FunctionSpecRequest(
            name=command_name, description=description, inputs=inputs, outputs=outputs
        )
        try:
            func_date = self.run_generate_command(req)
            code = func_date.code
        except Exception as e:
            return f"Error when calling Codex: {e}"
        
        return code
    
    def create_command(self, func: FunctionResponse) -> str:
        return ""
    
    def run_generate_command(self, req: FunctionSpecRequest) -> FunctionResponse:
        loop = asyncio.new_event_loop()
        func = loop.run_until_complete(self._write_function(req))
        return func

    
    async def _write_function(self, req: FunctionSpecRequest) -> FunctionResponse: # type: ignore
        headers: dict[str, str] = {"accept": "application/json"}

        url = f"{self.codex_base_url}/function/spec/"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=req.json()
                ) as response:
                    response.raise_for_status()

                    data = await response.json()
                    return FunctionResponse(**data)

        except aiohttp.ClientError as e:
            logger.exception(f"Error getting user: {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unknown Error when write function: {e}")
            raise e
        
