from __future__ import annotations

import copy
import logging
from abc import ABCMeta
from typing import TYPE_CHECKING, Any, Optional

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from pydantic import Field, validator


if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.core.prompting.base import PromptStrategy
    from autogpt.core.resource.model_providers.schema import (
        AssistantChatMessage,
        ChatModelInfo,
        ChatModelProvider,
        ChatModelResponse,
    )
    from autogpt.models.command_registry import CommandRegistry

from autogpt.agents.components import (
    Component,
    ComponentError,
    PipelineError,
)
from autogpt.config import ConfigBuilder
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAIModelName,
)
from autogpt.models.action_history import EpisodicActionHistory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

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


class AgentMeta(type):
    def __call__(cls, *args, **kwargs):
        # Create instance of the class (Agent or BaseAgent)
        instance = super().__call__(*args, **kwargs)
        # Automatically collect modules after the instance is created
        instance._collect_components(instance)
        return instance


class CombinedMeta(ABCMeta, AgentMeta):
    def __new__(cls, name, bases, namespace, **kwargs):
        return super().__new__(cls, name, bases, namespace, **kwargs)


class ComponentAgent(Configurable[BaseAgentSettings], metaclass=CombinedMeta):
    # TODO kcze change to named tuple?
    ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts]

    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__ if __doc__ else "",
    )

    def __init__(
        self,
        settings: BaseAgentSettings,
        llm_provider: ChatModelProvider,
    ):
        self.state = settings
        self.components = []
        self.llm_provider = llm_provider#TODO move to SimpleAgent
        self.config = settings.config

        logger.debug(f"Created {__class__} '{self.state.ai_profile.ai_name}'")

    def _collect_components(self):
        # Skip collecting and sort if ordering is explicit
        if self.components:
            return
        
        self.components = [
            getattr(self, attr)
            for attr in dir(self)
            if isinstance(getattr(self, attr), Component)
        ]
        self.components = self._topological_sort()

    def _topological_sort(self) -> list[Component]:
        visited = set()
        stack = []

        def visit(node: Component):
            if node in visited:
                return
            visited.add(node)
            for neighbor_class in node.__class__.get_dependencies():
                # Find the instance of neighbor_class in self.components
                neighbor = next(
                    (m for m in self.components if isinstance(m, neighbor_class)), None
                )
                if neighbor:
                    visit(neighbor)
            stack.append(node)

        for component in self.components:
            visit(component)

        return stack

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

    #TODO method_name type unsafe!
    def foreach_components(self, method_name: str, *args, retry_limit: int = 3):
        original_args = copy.deepcopy(args)  # Clone parameters to revert on failure
        pipeline_attempts = 0

        while pipeline_attempts < retry_limit:
            try:
                for component in self.components:
                    component_attempts = 0
                    method = getattr(component, method_name, None)
                    if not callable(method):
                        continue
                    while component_attempts < retry_limit:
                        try:
                            new_args = copy.deepcopy(args)
                            result = method(*new_args)
                            if result is not None:
                                args += result
                            else:
                                args = new_args
                        except ComponentError:
                            # Retry the same component on ComponentError
                            component_attempts += 1
                            continue
                        break
                # If execution reaches here without errors, break the loop
                break
            except PipelineError:
                # Restart from the beginning on PipelineError
                args = copy.deepcopy(original_args)  # Revert to original parameters
                pipeline_attempts += 1
                continue  # Start the loop over
            except Exception as e:
                #TODO pass pipeline info to exception
                raise e
