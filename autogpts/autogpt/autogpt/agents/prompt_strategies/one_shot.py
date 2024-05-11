from __future__ import annotations

import json
import platform
import re
from logging import Logger

import distro
from forge.components.action_history import ActionProposal
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.json.model import JSONSchema
from forge.json.parsing import extract_dict_from_json
from forge.llm.prompting import ChatPrompt, LanguageModelClassification, PromptStrategy
from forge.llm.prompting.utils import format_numbered_list
from forge.llm.providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.config import SystemConfiguration, UserConfigurable
from forge.models.utils import ModelWithSummary
from forge.utils.exceptions import InvalidAgentResponseError
from pydantic import Field

_RESPONSE_INTERFACE_NAME = "AssistantResponse"


class AssistantThoughts(ModelWithSummary):
    observations: str = Field(
        ..., description="Relevant observations from your last action (if any)"
    )
    text: str = Field(..., description="Thoughts")
    reasoning: str = Field(..., description="Reasoning behind the thoughts")
    self_criticism: str = Field(..., description="Constructive self-criticism")
    plan: list[str] = Field(
        ..., description="Short list that conveys the long-term plan"
    )
    speak: str = Field(..., description="Summary of thoughts, to say to user")

    def summary(self) -> str:
        return self.text


class OneShotAgentActionProposal(ActionProposal):
    thoughts: AssistantThoughts


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

    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
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
        self.response_schema = JSONSchema.from_dict(OneShotAgentActionProposal.schema())
        self.logger = logger

    @property
    def model_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.FAST_MODEL  # FIXME: dynamic switching

    def build_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
        **extras,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        3. `cycle_instruction`
        """
        system_prompt, response_prefill = self.build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        final_instruction_msg = ChatMessage.user(self.config.choose_action_instruction)

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'"""{task}"""'),
                *messages,
                final_instruction_msg,
            ],
            prefill_response=response_prefill,
            functions=commands if self.config.use_functions_api else [],
        )

    def build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> tuple[str, str]:
        """
        Builds the system prompt.

        Returns:
            str: The system prompt body
            str: The desired start for the LLM's response; used to steer the output
        """
        response_fmt_instruction, response_prefill = self.response_format_instruction(
            self.config.use_functions_api
        )
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
            + ["## RESPONSE FORMAT\n" + response_fmt_instruction]
        )

        # Join non-empty parts together into paragraph format
        return (
            "\n\n".join(filter(None, system_prompt_parts)).strip("\n"),
            response_prefill,
        )

    def response_format_instruction(self, use_functions_api: bool) -> tuple[str, str]:
        response_schema = self.response_schema.copy(deep=True)
        if (
            use_functions_api
            and response_schema.properties
            and "use_tool" in response_schema.properties
        ):
            del response_schema.properties["use_tool"]

        # Unindent for performance
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface(_RESPONSE_INTERFACE_NAME),
        )
        response_prefill = f'{{\n    "{list(response_schema.properties.keys())[0]}":'

        return (
            (
                f"YOU MUST ALWAYS RESPOND WITH A JSON OBJECT OF THE FOLLOWING TYPE:\n"
                f"{response_format}"
                + ("\n\nYOU MUST ALSO INVOKE A TOOL!" if use_functions_api else "")
            ),
            response_prefill,
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
        response: AssistantChatMessage,
    ) -> OneShotAgentActionProposal:
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        self.logger.debug(
            "LLM response content:"
            + (
                f"\n{response.content}"
                if "\n" in response.content
                else f" '{response.content}'"
            )
        )
        assistant_reply_dict = extract_dict_from_json(response.content)
        self.logger.debug(
            "Parsing object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )

        parsed_response = OneShotAgentActionProposal.parse_obj(assistant_reply_dict)
        if self.config.use_functions_api:
            if not response.tool_calls:
                raise InvalidAgentResponseError("Assistant did not use a tool")
            parsed_response.use_tool = response.tool_calls[0].function
        return parsed_response
