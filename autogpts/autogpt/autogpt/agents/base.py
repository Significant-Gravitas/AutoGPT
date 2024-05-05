from __future__ import annotations

import copy
import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    ParamSpec,
    TypeVar,
    overload,
)

from colorama import Fore
from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from autogpt.core.resource.model_providers.schema import (
        ChatModelInfo,
    )
    from autogpt.models.action_history import ActionResult

from autogpt.agents import protocols as _protocols
from autogpt.agents.components import (
    AgentComponent,
    ComponentEndpointError,
    EndpointPipelineError,
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
from autogpt.core.resource.model_providers import (
    CHAT_MODELS,
    AssistantFunctionCall,
    ModelName,
)
from autogpt.core.resource.model_providers.openai import OpenAIModelName
from autogpt.models.utils import ModelWithSummary
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


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


class AgentMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # Create instance of the class (Agent or BaseAgent)
        instance = super().__call__(*args, **kwargs)
        # Automatically collect modules after the instance is created
        instance._collect_components()
        return instance


class BaseAgentActionProposal(BaseModel):
    thoughts: str | ModelWithSummary
    use_tool: AssistantFunctionCall = None


class BaseAgent(Configurable[BaseAgentSettings], metaclass=AgentMeta):
    C = TypeVar("C", bound=AgentComponent)

    default_settings = BaseAgentSettings(
        name="BaseAgent",
        description=__doc__ if __doc__ else "",
    )

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
    async def propose_action(self) -> BaseAgentActionProposal:
        ...

    @abstractmethod
    async def execute(
        self,
        proposal: BaseAgentActionProposal,
        user_feedback: str = "",
    ) -> ActionResult:
        ...

    @abstractmethod
    async def do_not_execute(
        self,
        denied_proposal: BaseAgentActionProposal,
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
        self, protocol_method: Callable[P, None], *args, retry_limit: int = 3
    ) -> list[None]:
        ...

    async def run_pipeline(
        self,
        protocol_method: Callable[P, Iterator[T] | None],
        *args,
        retry_limit: int = 3,
    ) -> list[T] | list[None]:
        method_name = protocol_method.__name__
        protocol_name = protocol_method.__qualname__.split(".")[0]
        protocol_class = getattr(_protocols, protocol_name)
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

                    method = getattr(component, method_name, None)
                    if not callable(method):
                        continue

                    component_attempts = 0
                    while component_attempts < retry_limit:
                        try:
                            component_args = self._selective_copy(args)
                            if inspect.iscoroutinefunction(method):
                                result = await method(*component_args)
                            else:
                                result = method(*component_args)
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
            except EndpointPipelineError:
                self._trace.append(
                    f"❌ {Fore.LIGHTRED_EX}{component.__class__.__name__}: "
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

    def _collect_components(self):
        components = [
            getattr(self, attr)
            for attr in dir(self)
            if isinstance(getattr(self, attr), AgentComponent)
        ]

        if self.components:
            # Check if any component is missed (added to Agent but not to components)
            for component in components:
                if component not in self.components:
                    logger.warning(
                        f"Component {component.__class__.__name__} "
                        "is attached to an agent but not added to components list"
                    )
            # Skip collecting anf sorting and sort if ordering is explicit
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
