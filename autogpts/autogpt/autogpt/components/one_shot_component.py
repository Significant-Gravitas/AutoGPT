import json
import logging
from typing import TYPE_CHECKING

from autogpt.agents.protocols import ParseResponse, Single
from autogpt.agents.components import Single

from autogpt.config.config import Config
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatModelInfo,
)
from autogpt.agents.base import ThoughtProcessOutput
from autogpt.agents.components import Component
from autogpts.autogpt.autogpt.agents.agent import DEFAULT_RESPONSE_SCHEMA
from autogpts.autogpt.autogpt.core.utils.json_utils import extract_dict_from_json, json_loads
from autogpts.forge.forge.sdk.errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings


logger = logging.getLogger(__name__)


class OneShotComponent(Component, ParseResponse):
    def __init__(
        self,
        settings: "AgentSettings",
        legacy_config: Config,
        llm_provider,
        send_token_limit: int,
        llm: ChatModelInfo,
    ):
        # TODO kcze temp
        self.settings = settings
        self.legacy_config = legacy_config
        self.llm_provider = llm_provider
        self.send_token_limit = send_token_limit
        self.llm = llm

    def parse_response(
        self, result: ThoughtProcessOutput, response: AssistantChatMessage
    ) -> Single[ThoughtProcessOutput]:
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        logger.debug(
            "LLM response content:"
            + (
                f"\n{response.content}"
                if "\n" in response.content
                else f" '{response.content}'"
            )
        )
        assistant_reply_dict = extract_dict_from_json(response.content)
        logger.debug(
            "Validating object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )

        _, errors = DEFAULT_RESPONSE_SCHEMA.validate_object(
            object=assistant_reply_dict,
            logger=logger,
        )
        if errors:
            raise InvalidAgentResponseError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        # Get command name and arguments
        command_name, arguments = extract_command(
            assistant_reply_dict, response, self.settings.config.use_functions_api
        )

        # TODO kcze overwrite pipeline result for now
        result.command_name = command_name
        result.command_args = arguments
        result.thoughts = assistant_reply_dict
        return Single(result)

    
def extract_command(
    assistant_reply_json: dict,
    assistant_reply: AssistantChatMessage,
    use_openai_functions_api: bool,
) -> tuple[str, dict[str, str]]:
    """Parse the response and return the command name and arguments

    Args:
        assistant_reply_json (dict): The response object from the AI
        assistant_reply (AssistantChatMessage): The model response from the AI
        config (Config): The config object

    Returns:
        tuple: The command name and arguments

    Raises:
        json.decoder.JSONDecodeError: If the response is not valid JSON

        Exception: If any other error occurs
    """
    if use_openai_functions_api:
        if not assistant_reply.tool_calls:
            raise InvalidAgentResponseError("No 'tool_calls' in assistant reply")
        assistant_reply_json["command"] = {
            "name": assistant_reply.tool_calls[0].function.name,
            "args": json_loads(assistant_reply.tool_calls[0].function.arguments),
        }
    try:
        if not isinstance(assistant_reply_json, dict):
            raise InvalidAgentResponseError(
                f"The previous message sent was not a dictionary {assistant_reply_json}"
            )

        if "command" not in assistant_reply_json:
            raise InvalidAgentResponseError("Missing 'command' object in JSON")

        command = assistant_reply_json["command"]
        if not isinstance(command, dict):
            raise InvalidAgentResponseError("'command' object is not a dictionary")

        if "name" not in command:
            raise InvalidAgentResponseError("Missing 'name' field in 'command' object")

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        return command_name, arguments

    except json.decoder.JSONDecodeError:
        raise InvalidAgentResponseError("Invalid JSON")

    except Exception as e:
        raise InvalidAgentResponseError(str(e))
