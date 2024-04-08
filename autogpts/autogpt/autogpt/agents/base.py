from __future__ import annotations

import copy
import inspect
import logging
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore
from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from autogpt.core.resource.model_providers.schema import (
        ChatModelInfo,
        ChatModelProvider,
    )

from autogpt.agents.components import AgentComponent, ComponentError, ProtocolError, Single
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
        instance._collect_components()
        return instance


class CombinedMeta(ABCMeta, AgentMeta):
    def __new__(cls, name, bases, namespace, **kwargs):
        return super().__new__(cls, name, bases, namespace, **kwargs)


@dataclass
class ThoughtProcessOutput:
    command_name: str = ""
    command_args: dict[str, str] = field(default_factory=dict)
    thoughts: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield from (self.command_name, self.command_args, self.thoughts)


class BaseAgent(Configurable[BaseAgentSettings], metaclass=CombinedMeta):
    T = TypeVar("T", bound=AgentComponent)

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
        self.components: list[AgentComponent] = []
        self.llm_provider = llm_provider  # TODO kcze move to SimpleAgent
        self.config = settings.config
        self.ai_profile = settings.ai_profile
        self.directives = settings.directives
        # Execution data for debugging
        self._trace: list[str] = []

        logger.debug(f"Created {__class__} '{self.state.ai_profile.ai_name}'")

    def _collect_components(self):
        components = [
            getattr(self, attr)
            for attr in dir(self)
            if isinstance(getattr(self, attr), AgentComponent)
        ]

        if self.components:
            # Check if any coponent is missed (added to Agent but not to components)
            for component in components:
                if component not in self.components:
                    logger.warning(
                        f"Component {component.__class__.__name__} "
                        "is attached to an agent but not added to components list"
                    )
            # Skip collecting anf sorting and sort if ordering is explicit
            return
        self.components = self._topological_sort(components)

    def _topological_sort(self, components: list[AgentComponent]) -> list[AgentComponent]:
        visited = set()
        stack = []

        def visit(node: AgentComponent):
            if node in visited:
                return
            visited.add(node)
            for neighbor_class in node.__class__.run_after:
                # Find the instance of neighbor_class in components
                neighbor = next(
                    (m for m in components if isinstance(m, neighbor_class)), None
                )
                if neighbor:
                    visit(neighbor)
            stack.append(node)

        for component in components:
            visit(component)

        return stack

    def reset_trace(self):
        self._trace = []

    # Just collect when running foreach_components
    @property
    def trace(self) -> list[str]:
        return self._trace

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

    # Generic function to get components of a specific type
    def get_component(self, component_type: type[T]) -> Optional[T]:
        for component in self.components:
            if isinstance(component, component_type):
                return component
        return None

    def _selective_copy(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        copied_args = []
        for item in args:
            if isinstance(item, list):
                # Shallow copy for lists
                copied_item = item[:]
            elif isinstance(item, dict):
                # Shallow copy for dicts
                copied_item = item.copy()
            elif isinstance(item, BaseModel):
                # Deep copy for Pydantic models (deep=True to also copy nested models)
                copied_item = item.copy(deep=True)
            else:
                # Deep copy for other objects
                copied_item = copy.deepcopy(item)
            copied_args.append(copied_item)
        return tuple(copied_args)

    def is_enabled(self, component: AgentComponent) -> bool:
        if callable(component.enabled):
            return component.enabled()
        return component.enabled

    async def foreach_components(
        self, method_name: str, *args, retry_limit: int = 3
    ) -> Any:
        # Clone parameters to revert on failure
        original_args = self._selective_copy(args)
        pipeline_attempts = 0
        method_result: list[Any] | Any = []
        self._trace.append(f"⬇️  {Fore.BLUE}{method_name}{Fore.RESET}")

        while pipeline_attempts < retry_limit:
            try:
                for component in self.components:
                    # Skip disabled components
                    component_attempts = 0
                    method = getattr(component, method_name, None)
                    if not callable(method):
                        continue
                    if not self.is_enabled(component):
                        self._trace.append(
                            f"   {Fore.LIGHTBLACK_EX}"
                            f"{component.__class__.__name__}{Fore.RESET}"
                        )
                        continue
                    while component_attempts < retry_limit:
                        try:
                            component_args = self._selective_copy(args)
                            if inspect.iscoroutinefunction(method):
                                result = await method(*component_args)
                            else:
                                result = method(*component_args)
                            if result is not None:
                                if isinstance(result, Single):
                                    method_result = result.value
                                else:
                                    method_result.extend(result)
                            args = component_args
                            self._trace.append(f"✅ {component.__class__.__name__}")

                        except ComponentError:
                            self._trace.append(
                                f"❌ {Fore.YELLOW}{component.__class__.__name__}: "
                                f"ComponentError{Fore.RESET}"
                            )
                            # Retry the same component on ComponentError
                            component_attempts += 1
                            continue
                        # Successful component execution
                        break
                # Successful pipeline execution
                break
            except ProtocolError:
                self._trace.append(
                    f"❌ {Fore.LIGHTRED_EX}{component.__class__.__name__}: "
                    f"ProtocolError{Fore.RESET}"
                )
                # Restart from the beginning on ProtocolError
                # Revert to original parameters
                args = self._selective_copy(original_args)
                pipeline_attempts += 1
                continue  # Start the loop over
            except Exception as e:
                raise e
        return method_result
