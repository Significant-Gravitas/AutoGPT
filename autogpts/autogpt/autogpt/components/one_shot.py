import json
import logging
import platform
import re

import distro

from autogpt.agents.base import ThoughtProcessOutput
from autogpt.agents.protocols import PromptStrategy
from autogpt.utils.exceptions import InvalidAgentResponseError
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.config.config import Config
from autogpt.core.prompting.schema import ChatPrompt
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    ChatModelInfo,
)
from autogpt.core.utils.json_utils import extract_dict_from_json, json_loads
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.command import Command
from autogpt.prompts.utils import format_numbered_list
from autogpt.utils.schema import DEFAULT_RESPONSE_SCHEMA

logger = logging.getLogger(__name__)


class OneShotStrategy(PromptStrategy):
    """Component for one-shot Agents. Builds prompt and parses response."""

    def __init__(
        self,
        legacy_config: Config,
        llm_provider,
        send_token_limit: int,
        llm: ChatModelInfo,
    ):
        self.legacy_config = legacy_config
        self.llm_provider = llm_provider
        self.send_token_limit = send_token_limit
        self.llm = llm

    def build_prompt(
        self,
        messages: list[ChatMessage],
        commands: list[Command],
        task: str,
        profile: AIProfile,
        directives: AIDirectives,
    ) -> ChatPrompt:
        use_functions_api = self.legacy_config.openai_functions

        messages.insert(
            0,
            ChatMessage.system(
                f"You are {profile.ai_name}, {profile.ai_role.rstrip('.')}."
                "Your decisions must always be made independently without seeking "
                "user assistance. Play to your strengths as an LLM and pursue "
                "simple strategies with no legal complications."
                f"{self._generate_os_info()}\n"
                "## Constraints\n"
                "You operate within the following constraints:\n"
                f"{format_numbered_list(directives.constraints)}\n"
                "## Resources\n"
                "You can leverage access to the following resources:\n"
                f"{format_numbered_list(directives.resources)}\n"
                "## Commands\n"
                "These are the ONLY commands you can use."
                " Any action you perform must be possible"
                " through one of these commands:\n"
                f"{format_numbered_list([str(cmd) for cmd in commands])}\n"
                "## Best practices\n"
                f"{format_numbered_list(directives.best_practices)}\n"
                "## Your Task\n"
                "The user will specify a task for you to execute, in triple quotes,"
                " in the next message. Your job is to complete the task while following"
                " your directives as given above, and terminate when your task is done."
            ),
        )
        messages.insert(1, ChatMessage.user(f'"""{task}"""'))
        messages.insert(
            2, ChatMessage.system(self._response_format_instruction(use_functions_api))
        )

        messages.append(
            ChatMessage.user(
                "Determine exactly one command to use next based on the given goals "
                "and the progress you have made so far, "
                "and respond using the JSON schema specified previously."
            )
        )

        prompt = ChatPrompt(
            messages=messages,
            functions=get_openai_command_specs(commands),
        )

        return prompt

    def parse_response(
        self, response: AssistantChatMessage
    ) -> ThoughtProcessOutput:
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
            assistant_reply_dict, response, self.legacy_config.openai_functions
        )

        return ThoughtProcessOutput(command_name, arguments, assistant_reply_dict)

    def _generate_os_info(self) -> str:
        """Generates the OS information part of the prompt."""
        if not self.legacy_config.execute_local_commands:
            return ""

        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return f"The OS you are running on is: {os_info}"

    def _response_format_instruction(self, use_functions_api: bool) -> str:
        response_schema = DEFAULT_RESPONSE_SCHEMA.copy(deep=True)
        if (
            use_functions_api
            and response_schema.properties
            and "command" in response_schema.properties
        ):
            del response_schema.properties["command"]

        # Unindent for performance
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface("Response"),
        )

        instruction = (
            "Respond with pure JSON containing your thoughts, " "and invoke a tool."
            if use_functions_api
            else "Respond with pure JSON."
        )

        return (
            f"{instruction} "
            "The JSON object should be compatible with the TypeScript type `Response` "
            f"from the following:\n{response_format}"
        )


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
