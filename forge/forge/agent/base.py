from __future__ import annotations

import copy
import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Iterator,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

from colorama import Fore
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_core import from_json, to_json

from forge.agent import protocols
from forge.agent.components import (
    AgentComponent,
    ComponentEndpointError,
    ConfigurableComponent,
    EndpointPipelineError,
)
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.llm.providers import CHAT_MODELS, ModelName, OpenAIModelName
from forge.llm.providers.schema import ChatModelInfo
from forge.models.action import ActionResult, AnyProposal
from forge.models.config import SystemConfiguration, SystemSettings, UserConfigurable

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

DEFAULT_TRIGGERING_PROMPT = (
    "Determine exactly one command to use next based on the given goals "
    "and the progress you have made so far, "
    "and respond using the JSON schema specified previously:"
)


class BaseAgentConfiguration(SystemConfiguration):
    allow_fs_access: bool = UserConfigurable(default=False)

    fast_llm: ModelName = UserConfigurable(default=OpenAIModelName.GPT3_16k)
    smart_llm: ModelName = UserConfigurable(default=OpenAIModelName.GPT4)
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

    cycles_remaining: int = cycle_budget
    """The number of cycles remaining within the `cycle_budget`."""

    cycle_count: int = 0
    """The number of cycles that the agent has run since its initialization."""

    send_token_limit: Optional[int] = None
    """
    The token limit for prompt construction. Should leave room for the completion;
    defaults to 75% of `llm.max_tokens`.
    """

    @field_validator("use_functions_api")
    def validate_openai_functions(cls, value: bool, info: ValidationInfo):
        if value:
            smart_llm = info.data["smart_llm"]
            fast_llm = info.data["fast_llm"]
            assert all(
                [
                    not any(s in name for s in {"-0301", "-0314"})
                    for name in {smart_llm, fast_llm}
                ]
            ), (
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return value


class BaseAgentSettings(SystemSettings):
    agent_id: str = ""

    ai_profile: AIProfile = Field(default_factory=lambda: AIProfile(ai_name="AutoGPT"))
    """The AI profile or "personality" of the agent."""

    directives: AIDirectives = Field(default_factory=AIDirectives)
    """Directives (general instructional guidelines) for the agent."""

    task: str = "Terminate immediately"  # FIXME: placeholder for forge.sdk.schema.Task
    """The user-given task that the agent is working on."""

    config: BaseAgentConfiguration = Field(default_factory=BaseAgentConfiguration)
    """The configuration for this BaseAgent subsystem instance."""


class AgentMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # Create instance of the class (Agent or BaseAgent)
        instance = super().__call__(*args, **kwargs)
        # Automatically collect modules after the instance is created
        instance._collect_components()
        return instance


class BaseAgent(Generic[AnyProposal], metaclass=AgentMeta):
    def __init__(
        self,
        settings: BaseAgentSettings,
    ):
        self.state = settings
        self.components: list[AgentComponent] = []
        self.config = settings.config
        # Execution data for debugging
        self._trace: list[str] = []

        logger.debug(f"Created {__class__} '{self.state.ai_profile.ai_name}'")

    @property
    def trace(self) -> list[str]:
        return self._trace

    @property
    def llm(self) -> ChatModelInfo:
        """The LLM that the agent uses to think."""
        llm_name = (
            self.config.smart_llm if self.config.big_brain else self.config.fast_llm
        )
        return CHAT_MODELS[llm_name]

    @property
    def send_token_limit(self) -> int:
        return self.config.send_token_limit or self.llm.max_tokens * 3 // 4

    @abstractmethod
    async def propose_action(self) -> AnyProposal:
        ...

    @abstractmethod
    async def execute(
        self,
        proposal: AnyProposal,
        user_feedback: str = "",
    ) -> ActionResult:
        ...

    @abstractmethod
    async def do_not_execute(
        self,
        denied_proposal: AnyProposal,
        user_feedback: str,
    ) -> ActionResult:
        ...

    def reset_trace(self):
        self._trace = []

    @overload
    async def run_pipeline(
        self, protocol_method: Callable[P, Iterator[T]], *args, retry_limit: int = 3
    ) -> list[T]:
        ...

    @overload
    async def run_pipeline(
        self,
        protocol_method: Callable[P, None | Awaitable[None]],
        *args,
        retry_limit: int = 3,
    ) -> list[None]:
        ...

    async def run_pipeline(
        self,
        protocol_method: Callable[P, Iterator[T] | None | Awaitable[None]],
        *args,
        retry_limit: int = 3,
    ) -> list[T] | list[None]:
        method_name = protocol_method.__name__
        protocol_name = protocol_method.__qualname__.split(".")[0]
        protocol_class = getattr(protocols, protocol_name)
        if not issubclass(protocol_class, AgentComponent):
            raise TypeError(f"{repr(protocol_method)} is not a protocol method")

        # Clone parameters to revert on failure
        original_args = self._selective_copy(args)
        pipeline_attempts = 0
        method_result: list[T] = []
        self._trace.append(f"⬇️  {Fore.BLUE}{method_name}{Fore.RESET}")

        while pipeline_attempts < retry_limit:
            try:
                for component in self.components:
                    # Skip other protocols
                    if not isinstance(component, protocol_class):
                        continue

                    # Skip disabled components
                    if not component.enabled:
                        self._trace.append(
                            f"   {Fore.LIGHTBLACK_EX}"
                            f"{component.__class__.__name__}{Fore.RESET}"
                        )
                        continue

                    method = cast(
                        Callable[..., Iterator[T] | None | Awaitable[None]] | None,
                        getattr(component, method_name, None),
                    )
                    if not callable(method):
                        continue

                    component_attempts = 0
                    while component_attempts < retry_limit:
                        try:
                            component_args = self._selective_copy(args)
                            result = method(*component_args)
                            if inspect.isawaitable(result):
                                result = await result
                            if result is not None:
                                method_result.extend(result)
                            args = component_args
                            self._trace.append(f"✅ {component.__class__.__name__}")

                        except ComponentEndpointError:
                            self._trace.append(
                                f"❌ {Fore.YELLOW}{component.__class__.__name__}: "
                                f"ComponentEndpointError{Fore.RESET}"
                            )
                            # Retry the same component on ComponentEndpointError
                            component_attempts += 1
                            continue
                        # Successful component execution
                        break
                # Successful pipeline execution
                break
            except EndpointPipelineError as e:
                self._trace.append(
                    f"❌ {Fore.LIGHTRED_EX}{e.triggerer.__class__.__name__}: "
                    f"EndpointPipelineError{Fore.RESET}"
                )
                # Restart from the beginning on EndpointPipelineError
                # Revert to original parameters
                args = self._selective_copy(original_args)
                pipeline_attempts += 1
                continue  # Start the loop over
            except Exception as e:
                raise e
        return method_result

    def dump_component_configs(self) -> str:
        configs: dict[str, Any] = {}
        for component in self.components:
            if isinstance(component, ConfigurableComponent):
                config_type_name = component.config.__class__.__name__
                configs[config_type_name] = component.config
        return to_json(configs).decode()

    def load_component_configs(self, serialized_configs: str):
        configs_dict: dict[str, dict[str, Any]] = from_json(serialized_configs)

        for component in self.components:
            if not isinstance(component, ConfigurableComponent):
                continue
            config_type = type(component.config)
            config_type_name = config_type.__name__
            if config_type_name in configs_dict:
                # Parse the serialized data and update the existing config
                updated_data = configs_dict[config_type_name]
                data = {**component.config.model_dump(), **updated_data}
                component.config = component.config.__class__(**data)

    def _collect_components(self):
        components = [
            getattr(self, attr)
            for attr in dir(self)
            if isinstance(getattr(self, attr), AgentComponent)
        ]

        if self.components:
            # Check if any component is missing (added to Agent but not to components)
            for component in components:
                if component not in self.components:
                    logger.warning(
                        f"Component {component.__class__.__name__} "
                        "is attached to an agent but not added to components list"
                    )
            # Skip collecting and sorting and sort if ordering is explicit
            return
        self.components = self._topological_sort(components)

    def _topological_sort(
        self, components: list[AgentComponent]
    ) -> list[AgentComponent]:
        visited = set()
        stack = []

        def visit(node: AgentComponent):
            if node in visited:
                return
            visited.add(node)
            for neighbor_class in node._run_after:
                neighbor = next(
                    (m for m in components if isinstance(m, neighbor_class)), None
                )
                if neighbor and neighbor not in visited:
                    visit(neighbor)
            stack.append(node)

        for component in components:
            visit(component)

        return stack

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
                copied_item = item.model_copy(deep=True)
            else:
                # Deep copy for other objects
                copied_item = copy.deepcopy(item)
            copied_args.append(copied_item)
        return tuple(copied_args)
