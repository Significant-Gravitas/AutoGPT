from __future__ import annotations

import json
import platform
import re
from logging import Logger
from typing import TYPE_CHECKING, Callable, Optional

import distro

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.models.action_history import Episode

from autogpt.agents.utils.exceptions import InvalidAgentResponseError
from autogpt.config import AIDirectives, AIProfile
from autogpt.core.configuration.schema import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import (
    ChatPrompt,
    LanguageModelClassification,
    PromptStrategy,
)
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.json_utils.utilities import extract_dict_from_response
from autogpt.prompts.utils import format_numbered_list, indent


class OneShotAgentPromptConfiguration(SystemConfiguration):
    DEFAULT_BODY_TEMPLATE: str = (
        "## Constraints\n"
        "You operate within the following constraints:\n"
        "{constraints}\n"
        "\n"
        "## Resources\n"
        "You can leverage access to the following resources:\n"
        "{resources}\n"
        "\n"
        "## Commands\n"
        "These are the ONLY commands you can use."
        " Any action you perform must be possible through one of these commands:\n"
        "{commands}\n"
        "\n"
        "## Best practices\n"
        "{best_practices}"
    )

    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Determine exactly one command to use next based on the given goals "
        "and the progress you have made so far, "
        "and respond using the JSON schema specified previously:"
    )

    DEFAULT_RESPONSE_SCHEMA = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "thoughts": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                required=True,
                properties={
                    "observations": JSONSchema(
                        description=(
                            "Relevant observations from your last action (if any)"
                        ),
                        type=JSONSchema.Type.STRING,
                        required=False,
                    ),
                    "text": JSONSchema(
                        description="Thoughts",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "reasoning": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "self_criticism": JSONSchema(
                        description="Constructive self-criticism",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "plan": JSONSchema(
                        description=(
                            "Short markdown-style bullet list that conveys the "
                            "long-term plan"
                        ),
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "speak": JSONSchema(
                        description="Summary of thoughts, to say to user",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                },
            ),
            "command": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                required=True,
                properties={
                    "name": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "args": JSONSchema(
                        type=JSONSchema.Type.OBJECT,
                        required=True,
                    ),
                },
            ),
        },
    )

    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
    response_schema: dict = UserConfigurable(
        default_factory=DEFAULT_RESPONSE_SCHEMA.to_dict
    )
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    use_functions_api: bool = UserConfigurable(default=False)

    #########
    # State #
    #########
    # progress_summaries: dict[tuple[int, int], str] = Field(
    #     default_factory=lambda: {(0, 0): ""}
    # )


class OneShotAgentPromptStrategy(PromptStrategy):
    default_configuration: OneShotAgentPromptConfiguration = (
        OneShotAgentPromptConfiguration()
    )

    def __init__(
        self,
        configuration: OneShotAgentPromptConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.response_schema = JSONSchema.from_dict(configuration.response_schema)
        self.logger = logger

    @property
    def model_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.FAST_MODEL  # FIXME: dynamic switching

    def build_prompt(
        self,
        *,
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        event_history: list[Episode],
        include_os_info: bool,
        max_prompt_tokens: int,
        count_tokens: Callable[[str], int],
        count_message_tokens: Callable[[ChatMessage | list[ChatMessage]], int],
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary
            as needed
        3. `cycle_instruction`
        """
        if not extra_messages:
            extra_messages = []

        system_prompt = self.build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )
        system_prompt_tlength = count_message_tokens(ChatMessage.system(system_prompt))

        user_task = f'"""{task}"""'
        user_task_tlength = count_message_tokens(ChatMessage.user(user_task))

        response_format_instr = self.response_format_instruction(
            self.config.use_functions_api
        )
        extra_messages.append(ChatMessage.system(response_format_instr))

        final_instruction_msg = ChatMessage.user(self.config.choose_action_instruction)
        final_instruction_tlength = count_message_tokens(final_instruction_msg)

        if event_history:
            progress = self.compile_progress(
                event_history,
                count_tokens=count_tokens,
                max_tokens=(
                    max_prompt_tokens
                    - system_prompt_tlength
                    - user_task_tlength
                    - final_instruction_tlength
                    - count_message_tokens(extra_messages)
                ),
            )
            extra_messages.insert(
                0,
                ChatMessage.system(f"## Progress\n\n{progress}"),
            )

        prompt = ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(user_task),
                *extra_messages,
                final_instruction_msg,
            ],
        )

        return prompt

    def build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> str:
        system_prompt_parts = (
            self._generate_intro_prompt(ai_profile)
            + (self._generate_os_info() if include_os_info else [])
            + [
                self.config.body_template.format(
                    constraints=format_numbered_list(
                        ai_directives.constraints
                        + self._generate_budget_constraint(ai_profile.api_budget)
                    ),
                    resources=format_numbered_list(ai_directives.resources),
                    commands=self._generate_commands_list(commands),
                    best_practices=format_numbered_list(ai_directives.best_practices),
                )
            ]
            + [
                "## Your Task\n"
                "The user will specify a task for you to execute, in triple quotes,"
                " in the next message. Your job is to complete the task while following"
                " your directives as given above, and terminate when your task is done."
            ]
        )

        # Join non-empty parts together into paragraph format
        return "\n\n".join(filter(None, system_prompt_parts)).strip("\n")

    def compile_progress(
        self,
        episode_history: list[Episode],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        # start: int = len(episode_history)

        for i, c in reversed(list(enumerate(episode_history))):
            step = f"### Step {i+1}: Executed `{c.action.format_call()}`\n"
            step += f'- **Reasoning:** "{c.action.reasoning}"\n'
            step += (
                f"- **Status:** `{c.result.status if c.result else 'did_not_finish'}`\n"
            )
            if c.result:
                if c.result.status == "success":
                    result = str(c.result)
                    result = "\n" + indent(result) if "\n" in result else result
                    step += f"- **Output:** {result}"
                elif c.result.status == "error":
                    step += f"- **Reason:** {c.result.reason}\n"
                    if c.result.error:
                        step += f"- **Error:** {c.result.error}\n"
                elif c.result.status == "interrupted_by_human":
                    step += f"- **Feedback:** {c.result.feedback}\n"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)
        #     start = i

        # # TODO: summarize remaining
        # part = slice(0, start)

        return "\n\n".join(steps)

    def response_format_instruction(self, use_functions_api: bool) -> str:
        response_schema = self.response_schema.copy(deep=True)
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

    def _generate_intro_prompt(self, ai_profile: AIProfile) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return [
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    def _generate_os_info(self) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return [f"The OS you are running on is: {os_info}"]

    def _generate_budget_constraint(self, api_budget: float) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        if api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${api_budget:.3f}"
            ]
        return []

    def _generate_commands_list(self, commands: list[CompletionModelFunction]) -> str:
        """Lists the commands available to the agent.

        Params:
            agent: The agent for which the commands are being listed.

        Returns:
            str: A string containing a numbered list of commands.
        """
        try:
            return format_numbered_list([cmd.fmt_line() for cmd in commands])
        except AttributeError:
            self.logger.warning(f"Formatting commands failed. {commands}")
            raise

    def parse_response_content(
        self,
        response: AssistantChatMessageDict,
    ) -> Agent.ThoughtProcessOutput:
        if "content" not in response:
            raise InvalidAgentResponseError("Assistant response has no text content")

        assistant_reply_dict = extract_dict_from_response(response["content"])

        _, errors = self.response_schema.validate_object(
            object=assistant_reply_dict,
            logger=self.logger,
        )
        if errors:
            raise InvalidAgentResponseError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        # Get command name and arguments
        command_name, arguments = extract_command(
            assistant_reply_dict, response, self.config.use_functions_api
        )
        return command_name, arguments, assistant_reply_dict


#############
# Utilities #
#############


def extract_command(
    assistant_reply_json: dict,
    assistant_reply: AssistantChatMessageDict,
    use_openai_functions_api: bool,
) -> tuple[str, dict[str, str]]:
    """Parse the response and return the command name and arguments

    Args:
        assistant_reply_json (dict): The response object from the AI
        assistant_reply (ChatModelResponse): The model response from the AI
        config (Config): The config object

    Returns:
        tuple: The command name and arguments

    Raises:
        json.decoder.JSONDecodeError: If the response is not valid JSON

        Exception: If any other error occurs
    """
    if use_openai_functions_api:
        if not assistant_reply.get("tool_calls"):
            raise InvalidAgentResponseError("No 'tool_calls' in assistant reply")
        assistant_reply_json["command"] = {
            "name": assistant_reply["tool_calls"][0]["function"]["name"],
            "args": json.loads(
                assistant_reply["tool_calls"][0]["function"]["arguments"]
            ),
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
