from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from pydantic import Field, validator

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.core.prompting.base import PromptStrategy
    from autogpt.core.resource.model_providers.schema import (
        AssistantChatMessageDict,
        ChatModelInfo,
        ChatModelProvider,
        ChatModelResponse,
    )
    from autogpt.models.command_registry import CommandRegistry

from autogpt.agents.utils.prompt_scratchpad import PromptScratchpad
from autogpt.config import ConfigBuilder
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.prompting.schema import (
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIModelName,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.action_history import ActionResult, EpisodicActionHistory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

from .utils.agent_file_manager import AgentFileManager

logger = logging.getLogger(__name__)

CommandName = str
CommandArgs = dict[str, str]
AgentThoughts = dict[str, Any]


class BaseAgentConfiguration(SystemConfiguration):
    allow_fs_access: bool = UserConfigurable(default=False)

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

    summary_max_tlength: Optional[int] = None
    # TODO: move to ActionHistoryConfiguration

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
        return v


class BaseAgentSettings(SystemSettings):
    agent_id: str = ""
    agent_data_dir: Optional[Path] = None

    ai_profile: AIProfile = Field(default_factory=lambda: AIProfile(ai_name="AutoGPT"))
    """The AI profile or "personality" of the agent."""

    directives: AIDirectives = Field(
        default_factory=lambda: AIDirectives.from_file(
            ConfigBuilder.default_settings.prompt_settings_file
        )
    )
    """Directives (general instructional guidelines) for the agent."""

    task: str = "Terminate immediately"  # FIXME: placeholder for forge.sdk.schema.Task
    """The user-given task that the agent is working on."""

    config: BaseAgentConfiguration = Field(default_factory=BaseAgentConfiguration)
    """The configuration for this BaseAgent subsystem instance."""

    history: EpisodicActionHistory = Field(default_factory=EpisodicActionHistory)
    """(STATE) The action history of the agent."""

    def save_to_json_file(self, file_path: Path) -> None:
        with file_path.open("w") as f:
            f.write(self.json())

    @classmethod
    def load_from_json_file(cls, file_path: Path):
        return cls.parse_file(file_path)


class BaseAgent(Configurable[BaseAgentSettings], ABC):
    """Base class for all AutoGPT agent classes."""

    ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts]

    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__,
    )

    def __init__(
        self,
        settings: BaseAgentSettings,
        llm_provider: ChatModelProvider,
        prompt_strategy: PromptStrategy,
        command_registry: CommandRegistry,
        legacy_config: Config,
    ):
        self.state = settings
        self.config = settings.config
        self.ai_profile = settings.ai_profile
        self.directives = settings.directives
        self.event_history = settings.history

        self.legacy_config = legacy_config
        """LEGACY: Monolithic application configuration."""

        self.file_manager: AgentFileManager = (
            AgentFileManager(settings.agent_data_dir)
            if settings.agent_data_dir
            else None
        )  # type: ignore

        self.llm_provider = llm_provider

        self.prompt_strategy = prompt_strategy

        self.command_registry = command_registry
        """The registry containing all commands available to the agent."""

        self._prompt_scratchpad: PromptScratchpad | None = None

        # Support multi-inheritance and mixins for subclasses
        super(BaseAgent, self).__init__()

        logger.debug(f"Created {__class__} '{self.ai_profile.ai_name}'")

    def set_id(self, new_id: str, new_agent_dir: Optional[Path] = None):
        self.state.agent_id = new_id
        if self.state.agent_data_dir:
            if not new_agent_dir:
                raise ValueError(
                    "new_agent_dir must be specified if one is currently configured"
                )
            self.attach_fs(new_agent_dir)

    def attach_fs(self, agent_dir: Path) -> AgentFileManager:
        self.file_manager = AgentFileManager(agent_dir)
        self.file_manager.initialize()
        self.state.agent_data_dir = agent_dir
        return self.file_manager

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

    async def propose_action(self) -> ThoughtProcessOutput:
        """Proposes the next action to execute, based on the task and current state.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        assert self.file_manager, (
            f"Agent has no FileManager: call {__class__.__name__}.attach_fs()"
            " before trying to run the agent."
        )

        # Scratchpad as surrogate PromptGenerator for plugin hooks
        self._prompt_scratchpad = PromptScratchpad()

        prompt: ChatPrompt = self.build_prompt(scratchpad=self._prompt_scratchpad)
        prompt = self.on_before_think(prompt, scratchpad=self._prompt_scratchpad)

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        response = await self.llm_provider.create_chat_completion(
            prompt.messages,
            functions=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + list(self._prompt_scratchpad.commands.values())
            if self.config.use_functions_api
            else [],
            model_name=self.llm.name,
            completion_parser=lambda r: self.parse_and_process_response(
                r,
                prompt,
                scratchpad=self._prompt_scratchpad,
            ),
        )
        self.config.cycle_count += 1

        return self.on_response(
            llm_response=response,
            prompt=prompt,
            scratchpad=self._prompt_scratchpad,
        )

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
            ActionResult: An object representing the result(s) of the command.
        """
        ...

    def build_prompt(
        self,
        scratchpad: PromptScratchpad,
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs a prompt using `self.prompt_strategy`.

        Params:
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)
            extra_commands: Additional commands that the agent has access to.
            extra_messages: Additional messages to include in the prompt.
        """
        if not extra_commands:
            extra_commands = []
        if not extra_messages:
            extra_messages = []

        # Apply additions from plugins
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(scratchpad)
        ai_directives = self.directives.copy(deep=True)
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        extra_commands += list(scratchpad.commands.values())

        prompt = self.prompt_strategy.build_prompt(
            task=self.state.task,
            ai_profile=self.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + extra_commands,
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt

    def on_before_think(
        self,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ChatPrompt:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            prompt: The prompt that is about to be executed.
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)

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
            plugin_response = plugin.on_planning(scratchpad, prompt.raw())
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
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Called upon receiving a response from the chat model.

        Calls `self.parse_and_process_response()`.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        return llm_response.parsed_result

        # TODO: update memory/context

    @abstractmethod
    def parse_and_process_response(
        self,
        llm_response: AssistantChatMessageDict,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """
        pass
