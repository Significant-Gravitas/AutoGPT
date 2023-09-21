from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Optional

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from pydantic import Field, validator

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.core.resource.model_providers.schema import (
        ChatModelInfo,
        ChatModelProvider,
        ChatModelResponse,
    )
    from autogpt.models.command_registry import CommandRegistry

from autogpt.config.ai_config import AIConfig
from autogpt.config.ai_directives import AIDirectives
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.prompting.schema import ChatMessage, ChatPrompt
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIModelName,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.action_history import ActionResult, EpisodicActionHistory
from autogpt.prompts.generator import PromptGenerator
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

logger = logging.getLogger(__name__)

CommandName = str
CommandArgs = dict[str, str]
AgentThoughts = dict[str, Any]


class BaseAgentConfiguration(SystemConfiguration):
    fast_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT3_16k)
    smart_llm: OpenAIModelName = UserConfigurable(default=OpenAIModelName.GPT4)
    use_functions_api: bool = UserConfigurable(default=False)

    default_cycle_instruction: str = DEFAULT_TRIGGERING_PROMPT
    """The default instruction passed to the AI for a thinking cycle."""

    big_brain: bool = UserConfigurable(default=True)
    """
    Whether this agent uses the configured smart LLM (default) to think,
    as opposed to the configured fast LLM. Enabling this disables hybrid mode.
    """

    cycle_budget: Optional[int] = 1
    """
    The number of cycles that the agent is allowed to run unsupervised.

    `None` for unlimited continuous execution,
    `1` to require user approval for every step,
    `0` to stop the agent.
    """

    cycles_remaining = cycle_budget
    """The number of cycles remaining within the `cycle_budget`."""

    cycle_count = 0
    """The number of cycles that the agent has run since its initialization."""

    send_token_limit: Optional[int] = None
    """
    The token limit for prompt construction. Should leave room for the completion;
    defaults to 75% of `llm.max_tokens`.
    """

    summary_max_tlength: Optional[
        int
    ] = None  # TODO: move to ActionHistoryConfiguration

    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)

    class Config:
        arbitrary_types_allowed = True  # Necessary for plugins

    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    @validator("use_functions_api")
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            fast_llm = values["fast_llm"]
            assert all(
                [
                    not any(s in name for s in {"-0301", "-0314"})
                    for name in {smart_llm, fast_llm}
                ]
            ), (
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )


class BaseAgentSettings(SystemSettings):
    ai_config: AIConfig
    """The AIConfig or "personality" object associated with this agent."""

    config: BaseAgentConfiguration
    """The configuration for this BaseAgent subsystem instance."""

    history: EpisodicActionHistory
    """(STATE) The action history of the agent."""


class BaseAgent(Configurable[BaseAgentSettings], ABC):
    """Base class for all AutoGPT agent classes."""

    ThoughtProcessID = Literal["one-shot"]
    ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts]

    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__,
        ai_config=AIConfig(),
        config=BaseAgentConfiguration(),
        history=EpisodicActionHistory(),
    )

    def __init__(
        self,
        settings: BaseAgentSettings,
        llm_provider: ChatModelProvider,
        command_registry: CommandRegistry,
        legacy_config: Config,
    ):
        self.ai_config = settings.ai_config

        self.llm_provider = llm_provider

        self.command_registry = command_registry
        """The registry containing all commands available to the agent."""

        self.llm_provider = llm_provider

        self.prompt_generator = PromptGenerator(
            ai_config=settings.ai_config,
            ai_directives=AIDirectives.from_file(legacy_config.prompt_settings_file),
            command_registry=command_registry,
        )
        """The prompt generator used for generating the system prompt."""

        self.legacy_config = legacy_config
        self.config = settings.config
        """The applicable application configuration."""

        self.event_history = settings.history

        # Support multi-inheritance and mixins for subclasses
        super(BaseAgent, self).__init__()

    @property
    def system_prompt(self) -> str:
        """
        The system prompt sets up the AI's personality and explains its goals,
        available resources, and restrictions.
        """
        return self.prompt_generator.construct_system_prompt(self)

    @property
    def llm(self) -> ChatModelInfo:
        """The LLM that the agent uses to think."""
        llm_name = (
            self.config.smart_llm if self.config.big_brain else self.config.fast_llm
        )
        return OPEN_AI_CHAT_MODELS[llm_name]

    @property
    def send_token_limit(self) -> int:
        return self.config.send_token_limit or self.llm.max_tokens * 3 // 4

    async def think(
        self,
        instruction: Optional[str] = None,
        thought_process_id: ThoughtProcessID = "one-shot",
    ) -> ThoughtProcessOutput:
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """

        instruction = instruction or self.config.default_cycle_instruction

        prompt: ChatPrompt = self.construct_prompt(instruction, thought_process_id)
        prompt = self.on_before_think(prompt, thought_process_id, instruction)

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        raw_response = await self.llm_provider.create_chat_completion(
            prompt.messages,
            functions=get_openai_command_specs(self.command_registry)
            if self.config.use_functions_api
            else [],
            model_name=self.llm.name,
        )
        self.config.cycle_count += 1

        return self.on_response(raw_response, thought_process_id, prompt, instruction)

    @abstractmethod
    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        """Executes the given command, if any, and returns the agent's response.

        Params:
            command_name: The name of the command to execute, if any.
            command_args: The arguments to pass to the command, if any.
            user_input: The user's input, if any.

        Returns:
            The results of the command.
        """
        ...

    def construct_base_prompt(
        self,
        thought_process_id: ThoughtProcessID,
        prepend_messages: list[ChatMessage] = [],
        append_messages: list[ChatMessage] = [],
        reserve_tokens: int = 0,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. `prepend_messages`
        3. `append_messages`

        Params:
            prepend_messages: Messages to insert between the system prompt and message history
            append_messages: Messages to insert after the message history
            reserve_tokens: Number of tokens to reserve for content that is added later
        """

        if self.event_history:
            prepend_messages.insert(
                0,
                ChatMessage.system(
                    "## Progress\n\n" f"{self.event_history.fmt_paragraph()}"
                ),
            )

        prompt = ChatPrompt(
            messages=[
                ChatMessage.system(self.system_prompt),
            ]
            + prepend_messages
            + (append_messages or []),
        )

        return prompt

    def construct_prompt(
        self,
        cycle_instruction: str,
        thought_process_id: ThoughtProcessID,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary as needed
        3. `cycle_instruction`

        Params:
            cycle_instruction: The final instruction for a thinking cycle
        """

        if not cycle_instruction:
            raise ValueError("No instruction given")

        cycle_instruction_msg = ChatMessage.user(cycle_instruction)
        cycle_instruction_tlength = self.llm_provider.count_message_tokens(
            cycle_instruction_msg, self.llm.name
        )

        append_messages: list[ChatMessage] = []

        response_format_instr = self.response_format_instruction(thought_process_id)
        if response_format_instr:
            append_messages.append(ChatMessage.system(response_format_instr))

        prompt = self.construct_base_prompt(
            thought_process_id,
            append_messages=append_messages,
            reserve_tokens=cycle_instruction_tlength,
        )

        # ADD user input message ("triggering prompt")
        prompt.messages.append(cycle_instruction_msg)

        return prompt

    # This can be expanded to support multiple types of (inter)actions within an agent
    def response_format_instruction(self, thought_process_id: ThoughtProcessID) -> str:
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

        response_format = re.sub(
            r"\n\s+",
            "\n",
            RESPONSE_FORMAT_WITHOUT_COMMAND
            if self.config.use_functions_api
            else RESPONSE_FORMAT_WITH_COMMAND,
        )

        use_functions = self.config.use_functions_api and self.command_registry.commands
        return (
            f"Respond strictly with JSON{', and also specify a command to use through a function_call' if use_functions else ''}. "
            "The JSON should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )

    def on_before_think(
        self,
        prompt: ChatPrompt,
        thought_process_id: ThoughtProcessID,
        instruction: str,
    ) -> ChatPrompt:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The prompt to execute
        """
        current_tokens_used = self.llm_provider.count_message_tokens(
            prompt.messages, self.llm.name
        )
        plugin_count = len(self.config.plugins)
        for i, plugin in enumerate(self.config.plugins):
            if not plugin.can_handle_on_planning():
                continue
            plugin_response = plugin.on_planning(self.prompt_generator, prompt.raw())
            if not plugin_response or plugin_response == "":
                continue
            message_to_add = ChatMessage.system(plugin_response)
            tokens_to_add = self.llm_provider.count_message_tokens(
                message_to_add, self.llm.name
            )
            if current_tokens_used + tokens_to_add > self.send_token_limit:
                logger.debug(f"Plugin response too long, skipping: {plugin_response}")
                logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
                break
            prompt.messages.insert(
                -1, message_to_add
            )  # HACK: assumes cycle instruction to be at the end
            current_tokens_used += tokens_to_add
        return prompt

    def on_response(
        self,
        llm_response: ChatModelResponse,
        thought_process_id: ThoughtProcessID,
        prompt: ChatPrompt,
        instruction: str,
    ) -> ThoughtProcessOutput:
        """Called upon receiving a response from the chat model.

        Adds the last/newest message in the prompt and the response to `history`,
        and calls `self.parse_and_process_response()` to do the rest.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        return self.parse_and_process_response(
            llm_response, thought_process_id, prompt, instruction
        )

        # TODO: update memory/context

    @abstractmethod
    def parse_and_process_response(
        self,
        llm_response: ChatModelResponse,
        thought_process_id: ThoughtProcessID,
        prompt: ChatPrompt,
        instruction: str,
    ) -> ThoughtProcessOutput:
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """
        pass
