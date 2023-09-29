from __future__ import annotations
import abc

from pydantic import validator
from typing import TYPE_CHECKING, Union
from autogpt.core.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from autogpt.core.agents.simple.main import SimpleAgent
    from autogpt.core.agents.base.main import BaseAgent

from autogpt.core.prompting.utils import json_loads, to_numbered_list
from autogpt.core.configuration import SystemConfiguration
from autogpt.core.prompting.schema import (
    LanguageModelClassification,
    ChatPrompt,
    CompletionModelFunction,
)

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
    AssistantChatMessageDict,
    ChatMessage,
)


class PromptStrategiesConfiguration(SystemConfiguration):
    pass

class PlanningPromptStrategiesConfiguration(PromptStrategiesConfiguration):

    RESPONSE_SCHEMA = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "thoughts": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                required=True,
                properties={
                    "text": JSONSchema(
                        description="Thoughts",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "reasoning": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "plan": JSONSchema(
                        description="Short markdown-style bullet list that conveys the long-term plan",
                        type=JSONSchema.Type.STRING,
                        required=True,
                    ),
                    "criticism": JSONSchema(
                        description="Constructive self-criticism",
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


class PromptStrategy(abc.ABC):
    STRATEGY_NAME: str
    default_configuration: SystemConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessageDict):
        ...


class BasePromptStrategy(PromptStrategy):
    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # TODO : This implementation is shit :)
    def get_functions(self) -> list[CompletionModelFunction]:
        """
        Returns a list of functions related to refining user context.

        Returns:
            list: A list of CompletionModelFunction objects detailing each function's purpose and parameters.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> functions = strategy.get_functions()
            >>> print(functions[0].name)
            'refine_requirements'
        """
        return self._functions

    # TODO : This implementation is shit :)
    def get_functions_names(self) -> list[str]:
        """
        Returns a list of names of functions related to refining user context.

        Returns:
            list: A list of strings, each representing the name of a function.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> function_names = strategy.get_functions_names()
            >>> print(function_names)
            ['refine_requirements']
        """
        return [item.name for item in self._functions]

class PlanningPromptStrategy(BasePromptStrategy):


    def __init__(self, **kwargs) -> None:
        self._prepend_messages: list[ChatMessage] = []
        self._append_messages: list[ChatMessage] = []

    # NOTE : Legacy Autogpt and it's dodgy architecture :)
    def construct_base_prompt(
        self,
        agent: SimpleAgent,
        thought_process_id: str,
        prepend_messages: list[ChatMessage] = [],
        append_messages: list[ChatMessage] = [],
        reserve_tokens: int = 0,
    ) -> list[ChatMessage]:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. `prepend_messages`
        3. `append_messages`

        Params:
            prepend_messages: Messages to insert between the system prompt and message history
            append_messages: Messages to insert after the message history
            reserve_tokens: Number of tokens to reserve for content that is added later
        """

        # NOTE : PLANCE HOLDER
        # if agent.event_history:
        #     self._prepend_messages.insert(
        #         0,
        #         ChatMessage.system(
        #             "## Progress\n\n" f"{agent.event_history.fmt_paragraph()}"
        #         ),
        #     )

        messages: list[ChatMessage] = [
            ChatMessage.system(self._construct_system_prompt(agent=agent))
        ]
        if self._prepend_messages:
            messages.append(self._prepend_messages)
        if self._append_messages:
            messages.append(self._append_messages)

        return messages

    # FIXME : Uncompleted migration from AutoGPT Agent
    def _construct_system_prompt(self, agent: SimpleAgent) -> str:
        """Constructs a system prompt containing the most important information for the AI.

        Params:
            agent: The agent for which the system prompt is being constructed.

        Returns:
            str: The constructed system prompt.
        """

        # for plugin in agent.config.plugins:
        #     if not plugin.can_handle_post_prompt():
        #         continue
        #     plugin.post_prompt(self)

        # Construct full prompt
        from autogpt.core.planning.simple import get_os_info

        full_prompt_parts: list[str] = (
            self._generate_intro_prompt(agent=agent)
            + [
                f"The OS you are running on is: {get_os_info()}"
            ]  # NOTE : Should now be KWARG
            + self._generate_body(
                agent=agent,
            )  # additional_constraints=self._generate_budget_info(),
            + self._generate_goals_info(agent=agent)
        )

        # Join non-empty parts together into paragraph format
        return "\n\n".join(filter(None, full_prompt_parts)).strip("\n")

    def _generate_intro_prompt(self, agent: Union[SimpleAgent, BaseAgent]) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return [
            f"You are {agent.agent_name}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    # FIXME :)
    def _generate_budget_info(self, agent: Union[SimpleAgent, BaseAgent]) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        if self.ai_config.api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${agent.ai_config.api_budget:.3f}"
            ]
        return []

    def _generate_goals_info(self, agent: SimpleAgent) -> list[str]:
        """Generates the goals information part of the prompt.

        Returns:
            str: The goals information part of the prompt.
        """
        if agent.agent_goals:
            return [
                "\n".join(
                    [
                        "## Goals",
                        "For your task, you must fulfill the following goals:",
                        to_numbered_list(agent.agent_goals),
                    ]
                )
            ]
        return []

    def _generate_body(
        self,
        agent: Union[SimpleAgent, BaseAgent],
        *,
        additional_constraints: list[str] = [],
        additional_resources: list[str] = [],
        additional_best_practices: list[str] = [],
    ) -> list[str]:
        """
        Generates a prompt section containing the constraints, commands, resources,
        and best practices.

        Params:
            agent: The agent for which the prompt string is being generated.
            additional_constraints: Additional constraints to be included in the prompt string.
            additional_resources: Additional resources to be included in the prompt string.
            additional_best_practices: Additional best practices to be included in the prompt string.

        Returns:
            str: The generated prompt section.
        """
        body: list[str] = []

        # NOTE : if agent.constraints :
        #     body.append("## Constraints\n"
        #     "You operate within the following constraints:\n"
        #     f"{to_numbered_list(agent.constraints + additional_constraints)}")

        # NOTE : PLANCE HOLDER
        # if agent.resources :
        #     body.append("## Resources\n"
        #     "You can leverage access to the following resources:\n"
        #     f"{to_numbered_list(agent.resources + additional_resources)}")

        body.append(
            "## Commands\n"
            "You have access to the following commands:\n"
            f"{self._list_commands(agent)}"
        )

        # NOTE : PLANCE HOLDER
        # if agent.best_practices :
        #     body.append("## Best practices\n"
        #     f"{to_numbered_list(agent.best_practices + additional_best_practices)}")

        return body

    def _list_commands(self, agent: Union[SimpleAgent, BaseAgent]) -> str:
        """Lists the commands available to the agent.

        Params:
            agent: The agent for which the commands are being listed.

        Returns:
            str: A string containing a numbered list of commands.
        """
        if agent._tool_registry:
            return to_numbered_list(agent._tool_registry.list_tools_descriptions())

        return []
    

    # NOTE : based on autogpt agent.py
    # This can be expanded to support multiple types of (inter)actions within an agent
    def response_format_instruction(
        self, agent: SimpleAgent, thought_process_id: str, model_name: str
    ) -> str:
        # FIXME : Remove ?
        if thought_process_id != "one-shot":
            raise NotImplementedError(f"Unknown thought process '{thought_process_id}'")

        RESPONSE_FORMAT_WITH_COMMAND = """```ts
        interface Response {
            thoughts: {
                // Thoughts
                text: string;
                reasoning: string;
                // Short markdown-style bullet list that conveys the long-term plan
                plan: string;
                // Constructive self-criticism
                criticism: string;
                // Summary of thoughts to say to the user
                speak: string;
            };
            command: {
                name: string;
                args: Record<string, any>;
            };
        }
        ```"""

        RESPONSE_FORMAT_WITHOUT_COMMAND = """```ts
        interface Response {
            thoughts: {
                // Thoughts
                text: string;
                reasoning: string;
                // Short markdown-style bullet list that conveys the long-term plan
                plan: string;
                // Constructive self-criticism
                criticism: string;
                // Summary of thoughts to say to the user
                speak: string;
            };
        }
        ```"""

        import re

        # use_functions : bool  = agent._openai_provider.has_function_call_api(model_name = self._model_classification)
        use_functions: bool = agent._openai_provider.has_function_call_api(
            model_name=model_name
        )
        response_format: str = re.sub(
            r"\n\s+",
            "\n",
            RESPONSE_FORMAT_WITHOUT_COMMAND
            if use_functions
            else RESPONSE_FORMAT_WITH_COMMAND,
        )

        return (
            f"Respond strictly with JSON{', and also specify a command to use through a function_call' if use_functions else ''}. "
            "The JSON should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )
