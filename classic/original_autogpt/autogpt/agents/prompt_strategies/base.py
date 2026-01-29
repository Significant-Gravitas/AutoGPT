"""Base classes and utilities for prompt strategies.

This module provides shared infrastructure for different prompt strategy
implementations including ReWOO, Plan-and-Execute, Reflexion, and Tree of Thoughts.
"""

from __future__ import annotations

import asyncio
import enum
import platform
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar

import distro
from pydantic import BaseModel, ConfigDict, Field

from forge.agent.execution_context import (
    ExecutionContext,
    SubAgentHandle,
    SubAgentStatus,
    generate_sub_agent_id,
)
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.llm.prompting import ChatPrompt, LanguageModelClassification
from forge.llm.prompting.utils import format_numbered_list
from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    CompletionModelFunction,
)
from forge.models.action import ActionProposal
from forge.models.config import SystemConfiguration, UserConfigurable
from forge.models.utils import ModelWithSummary

if TYPE_CHECKING:
    pass


class PromptStrategyType(str, enum.Enum):
    """Available prompt strategy types."""

    ONE_SHOT = "one_shot"
    REWOO = "rewoo"
    PLAN_EXECUTE = "plan_execute"
    REFLEXION = "reflexion"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    LATS = "lats"  # Language Agent Tree Search (sub-agent based)
    MULTI_AGENT_DEBATE = "multi_agent_debate"  # Multi-agent debate (sub-agent based)


class PlannedStep(BaseModel):
    """A single step in a multi-step plan.

    Used by ReWOO and Plan-and-Execute strategies.
    """

    thought: str = Field(description="Reasoning for this step")
    tool_name: str = Field(description="Name of the tool to call")
    tool_arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool"
    )
    variable_name: str = Field(
        default="", description="Variable name for output (e.g., #E1)"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="Variable dependencies"
    )
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Optional[str] = None

    def to_function_call(self) -> AssistantFunctionCall:
        """Convert to AssistantFunctionCall."""
        return AssistantFunctionCall(
            name=self.tool_name,
            arguments=self.tool_arguments,
        )


class Reflection(BaseModel):
    """A stored reflection from past action execution.

    Used by the Reflexion strategy for episodic memory.
    Supports both structured and verbal (free-form) reflection formats.
    """

    action_name: str = Field(description="Name of the action taken")
    action_arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments used"
    )
    result_summary: str = Field(description="Summary of the result")
    what_went_wrong: str = Field(
        default="", description="Analysis of what went wrong (if anything)"
    )
    what_to_do_differently: str = Field(
        default="", description="Lessons for future attempts"
    )
    success: bool = Field(default=True, description="Whether the action succeeded")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Verbal reflection support (from Reflexion paper)
    verbal_reflection: str = Field(
        default="", description="Free-form verbal reflection text"
    )
    reflection_format: Literal["structured", "verbal"] = Field(
        default="structured", description="Format of this reflection"
    )
    evaluation_score: Optional[float] = Field(
        default=None, description="Evaluator score (0-1) if available"
    )

    def to_prompt_text(self) -> str:
        """Format reflection for inclusion in prompts."""
        # If verbal format, return the verbal reflection directly
        if self.reflection_format == "verbal" and self.verbal_reflection:
            score_text = (
                f" [score: {self.evaluation_score:.2f}]"
                if self.evaluation_score is not None
                else ""
            )
            return f"Reflection{score_text}: {self.verbal_reflection}"

        # Structured format
        status = "succeeded" if self.success else "failed"
        text = f"Action '{self.action_name}' {status}: {self.result_summary}"
        if self.what_went_wrong:
            text += f"\n  - Issue: {self.what_went_wrong}"
        if self.what_to_do_differently:
            text += f"\n  - Lesson: {self.what_to_do_differently}"
        if self.evaluation_score is not None:
            text += f"\n  - Score: {self.evaluation_score:.2f}"
        return text


class ReflexionMemory(BaseModel):
    """Episodic memory of reflections for the Reflexion strategy."""

    reflections: list[Reflection] = Field(default_factory=list)
    max_reflections: int = Field(default=10, description="Maximum reflections to keep")

    def add_reflection(self, reflection: Reflection) -> None:
        """Add a reflection, maintaining max size."""
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_reflections:
            self.reflections = self.reflections[-self.max_reflections :]

    def get_relevant_reflections(
        self, action_name: Optional[str] = None, limit: int = 5
    ) -> list[Reflection]:
        """Get relevant reflections, optionally filtered by action name."""
        if action_name:
            relevant = [r for r in self.reflections if r.action_name == action_name]
        else:
            relevant = list(self.reflections)
        return relevant[-limit:]

    def get_failed_reflections(self, limit: int = 5) -> list[Reflection]:
        """Get reflections from failed actions."""
        failed = [r for r in self.reflections if not r.success]
        return failed[-limit:]


class WorkerExecution(BaseModel):
    """Worker execution record for ReWOO strategy.

    Tracks the execution of each planned step by the Worker module,
    including variable substitutions and raw outputs (per the ReWOO paper).
    """

    step: PlannedStep = Field(description="The planned step that was executed")
    input_substituted: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments after variable substitution (e.g., #E1 -> actual value)",
    )
    raw_output: str = Field(default="", description="Raw output from tool execution")
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )


class Thought(BaseModel):
    """A node in the Tree of Thoughts.

    Represents a single thought/reasoning step that can branch into children.
    Includes parent pointer for proper backtracking (per ToT paper).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str = Field(description="The thought content")
    score: float = Field(default=0.0, description="Self-evaluation score (0-10)")
    depth: int = Field(default=0, description="Depth in the tree")
    children: list["Thought"] = Field(default_factory=list)
    is_terminal: bool = Field(
        default=False, description="Whether this thought leads to an action"
    )
    action: Optional[AssistantFunctionCall] = Field(
        default=None, description="Action if terminal"
    )

    # Parent pointer for backtracking (excluded from serialization)
    parent: Optional["Thought"] = Field(default=None, exclude=True)

    # Categorical evaluation support (from ToT paper)
    categorical_evaluation: Optional[Literal["sure", "maybe", "impossible"]] = Field(
        default=None, description="Categorical evaluation: sure/maybe/impossible"
    )
    evaluation_votes: dict[str, int] = Field(
        default_factory=dict, description="Vote counts for multi-sample evaluation"
    )

    def add_child(self, thought: "Thought") -> None:
        """Add a child thought and set its parent pointer."""
        thought.depth = self.depth + 1
        thought.parent = self  # Set parent pointer for backtracking
        self.children.append(thought)

    def best_child(self) -> Optional["Thought"]:
        """Return the highest-scoring child."""
        if not self.children:
            return None
        return max(self.children, key=lambda t: t.score)

    def get_path_to_root(self) -> list["Thought"]:
        """Traverse parent pointers to get path from this node to root."""
        path = []
        node: Optional[Thought] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class BasePromptStrategyConfiguration(SystemConfiguration):
    """Base configuration for all prompt strategies."""

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

    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
    use_prefill: bool = True

    # Sub-agent configuration
    enable_sub_agents: bool = UserConfigurable(default=True)
    """Enable sub-agent spawning for this strategy."""

    max_sub_agents: int = UserConfigurable(default=5)
    """Maximum number of sub-agents that can be spawned."""

    sub_agent_timeout_seconds: int = UserConfigurable(default=300)
    """Timeout for sub-agent execution in seconds."""

    sub_agent_max_cycles: int = UserConfigurable(default=25)
    """Maximum execution cycles per sub-agent."""


class BaseMultiStepPromptStrategy(ABC):
    """Base class for multi-step prompt strategies.

    Provides common utilities for strategies that involve multiple phases
    like planning, execution, synthesis, or reflection.

    Also provides sub-agent spawning capabilities when enabled via config.
    """

    def __init__(
        self,
        configuration: BasePromptStrategyConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.logger = logger
        self._execution_context: Optional[ExecutionContext] = None

    # ===== Sub-Agent Support Methods =====

    def set_execution_context(self, context: ExecutionContext) -> None:
        """Inject execution context. Called by Agent after creation.

        This provides the strategy with access to shared resources needed
        for sub-agent spawning (LLM provider, file storage, agent factory).

        Args:
            context: The execution context from the parent agent.
        """
        self._execution_context = context
        self.logger.debug(
            f"ExecutionContext set (depth={context.depth}, "
            f"sub_agents_enabled={self.config.enable_sub_agents})"
        )

    def can_spawn_sub_agent(self) -> bool:
        """Check if sub-agent spawning is available and allowed.

        Returns:
            True if sub-agents can be spawned, False otherwise.
        """
        if not self.config.enable_sub_agents:
            return False
        if self._execution_context is None:
            return False
        return self._execution_context.can_spawn_sub_agent()

    async def spawn_sub_agent(
        self,
        task: str,
        ai_profile: Optional[AIProfile] = None,
        directives: Optional[AIDirectives] = None,
        strategy: Optional[str] = None,
    ) -> SubAgentHandle:
        """Spawn a sub-agent to handle a subtask.

        The sub-agent runs with its own execution context (reduced budget,
        restricted file storage) and can be run synchronously or in background.

        Args:
            task: The task for the sub-agent to accomplish.
            ai_profile: Optional AI profile override.
            directives: Optional directives override.
            strategy: Optional strategy name override (e.g., "one_shot").

        Returns:
            A SubAgentHandle for tracking and interacting with the sub-agent.

        Raises:
            RuntimeError: If sub-agent spawning is not available.
        """
        if not self.can_spawn_sub_agent():
            raise RuntimeError(
                "Cannot spawn sub-agent: "
                + ("not enabled" if not self.config.enable_sub_agents else "no context")
            )

        assert self._execution_context is not None

        # Generate unique ID for sub-agent
        parent_id = self._execution_context.parent_agent_id
        agent_id = generate_sub_agent_id(parent_id)

        # Create handle
        handle = SubAgentHandle(
            agent_id=agent_id,
            task=task,
            status=SubAgentStatus.PENDING,
        )

        # Create child context with restricted resources
        child_context = self._execution_context.create_child_context(agent_id)

        # Create the sub-agent via factory
        factory = self._execution_context.agent_factory
        if factory is None:
            raise RuntimeError("No agent factory available")

        try:
            agent = factory.create_agent(
                agent_id=agent_id,
                task=task,
                context=child_context,
                ai_profile=ai_profile,
                directives=directives,
                strategy=strategy,
            )
            handle._agent = agent
            handle.status = SubAgentStatus.PENDING

            # Register with parent context
            self._execution_context.register_sub_agent(handle)

            self.logger.info(f"Spawned sub-agent {agent_id} for task: {task[:100]}...")

        except Exception as e:
            handle.status = SubAgentStatus.FAILED
            handle.error = str(e)
            self.logger.error(f"Failed to spawn sub-agent: {e}")

        return handle

    async def run_sub_agent(
        self,
        handle: SubAgentHandle,
        max_cycles: Optional[int] = None,
    ) -> Any:
        """Run a sub-agent until completion.

        Executes the sub-agent's action loop until it finishes or hits
        the cycle limit.

        Args:
            handle: The sub-agent handle from spawn_sub_agent().
            max_cycles: Maximum cycles to run (default from config).

        Returns:
            The result from the sub-agent (typically the finish command output).

        Raises:
            RuntimeError: If the sub-agent is not in a runnable state.
        """
        if handle._agent is None:
            raise RuntimeError(f"Sub-agent {handle.agent_id} has no agent instance")

        if handle.status not in (SubAgentStatus.PENDING, SubAgentStatus.RUNNING):
            raise RuntimeError(
                f"Sub-agent {handle.agent_id} is not runnable "
                f"(status={handle.status})"
            )

        max_cycles = max_cycles or self.config.sub_agent_max_cycles
        timeout = self.config.sub_agent_timeout_seconds

        handle.status = SubAgentStatus.RUNNING
        agent = handle._agent

        try:
            result = await asyncio.wait_for(
                self._run_agent_loop(agent, max_cycles, handle),
                timeout=timeout,
            )
            handle.result = result
            handle.status = SubAgentStatus.COMPLETED
            return result

        except asyncio.TimeoutError:
            handle.status = SubAgentStatus.FAILED
            handle.error = f"Timed out after {timeout}s"
            self.logger.warning(f"Sub-agent {handle.agent_id} timed out")
            return None

        except asyncio.CancelledError:
            handle.status = SubAgentStatus.CANCELLED
            self.logger.info(f"Sub-agent {handle.agent_id} was cancelled")
            raise

        except Exception as e:
            handle.status = SubAgentStatus.FAILED
            handle.error = str(e)
            self.logger.error(f"Sub-agent {handle.agent_id} failed: {e}")
            return None

    async def _run_agent_loop(
        self,
        agent: Any,
        max_cycles: int,
        handle: SubAgentHandle,
    ) -> Any:
        """Run the agent's propose/execute loop.

        Args:
            agent: The agent instance.
            max_cycles: Maximum cycles to run.
            handle: The sub-agent handle for status tracking.

        Returns:
            The result from the finish command, or None if max cycles reached.
        """
        for cycle in range(max_cycles):
            # Check for cancellation
            if self._execution_context and self._execution_context.cancelled:
                handle.status = SubAgentStatus.CANCELLED
                return None

            # Propose next action
            proposal = await agent.propose_action()

            # Check for finish command
            if proposal.use_tool.name == "finish":
                # Extract result from finish arguments
                result = proposal.use_tool.arguments.get("reason", "")
                handle.summary = result[:200] if result else "Task completed"
                return result

            # Execute the action
            result = await agent.execute(proposal)

            # Log progress
            self.logger.debug(
                f"Sub-agent {handle.agent_id} cycle {cycle + 1}: "
                f"{proposal.use_tool.name}"
            )

        # Hit max cycles
        handle.summary = f"Reached max cycles ({max_cycles})"
        return None

    async def spawn_and_run(
        self,
        task: str,
        ai_profile: Optional[AIProfile] = None,
        directives: Optional[AIDirectives] = None,
        strategy: Optional[str] = None,
        max_cycles: Optional[int] = None,
    ) -> Any:
        """Convenience method: spawn and immediately run a sub-agent.

        This is the most common pattern for sub-agent usage.

        Args:
            task: The task for the sub-agent.
            ai_profile: Optional AI profile override.
            directives: Optional directives override.
            strategy: Optional strategy name override.
            max_cycles: Maximum cycles to run.

        Returns:
            The result from the sub-agent.
        """
        handle = await self.spawn_sub_agent(
            task=task,
            ai_profile=ai_profile,
            directives=directives,
            strategy=strategy,
        )

        if handle.status == SubAgentStatus.FAILED:
            self.logger.error(f"Failed to spawn sub-agent: {handle.error}")
            return None

        return await self.run_sub_agent(handle, max_cycles=max_cycles)

    async def run_parallel(
        self,
        tasks: list[str],
        strategy: Optional[str] = None,
        max_cycles: Optional[int] = None,
    ) -> list[Any]:
        """Run multiple sub-agents in parallel.

        Useful for patterns like multi-agent debate or parallel exploration.

        Args:
            tasks: List of tasks for sub-agents.
            strategy: Optional strategy name for all sub-agents.
            max_cycles: Maximum cycles per sub-agent.

        Returns:
            List of results from all sub-agents (in same order as tasks).
        """
        # Spawn all sub-agents
        handles = []
        for task in tasks:
            handle = await self.spawn_sub_agent(task=task, strategy=strategy)
            handles.append(handle)

        # Run all in parallel
        async def run_one(h: SubAgentHandle) -> Any:
            if h.status == SubAgentStatus.FAILED:
                return None
            return await self.run_sub_agent(h, max_cycles=max_cycles)

        results = await asyncio.gather(*[run_one(h) for h in handles])
        return list(results)

    def get_sub_agent_results(self) -> dict[str, Any]:
        """Get results from all completed sub-agents.

        Returns:
            Dictionary mapping agent_id to result.
        """
        if self._execution_context is None:
            return {}

        return {
            agent_id: handle.result
            for agent_id, handle in self._execution_context.sub_agents.items()
            if handle.status == SubAgentStatus.COMPLETED
        }

    @property
    @abstractmethod
    def llm_classification(self) -> LanguageModelClassification:
        """Declare whether this strategy needs a fast or smart model."""
        ...

    @abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        """Build the prompt for the current phase."""
        ...

    @abstractmethod
    def parse_response_content(self, response: AssistantChatMessage) -> ActionProposal:
        """Parse the LLM response into an action proposal."""
        ...

    def generate_intro_prompt(self, ai_profile: AIProfile) -> list[str]:
        """Generate the introduction part of the prompt."""
        return [
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    def generate_os_info(self) -> list[str]:
        """Generate OS information for the prompt."""
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return [f"The OS you are running on is: {os_info}"]

    def generate_commands_list(self, commands: list[CompletionModelFunction]) -> str:
        """Generate a formatted list of available commands."""
        try:
            return format_numbered_list([cmd.fmt_line() for cmd in commands])
        except AttributeError:
            self.logger.warning(f"Formatting commands failed. {commands}")
            raise

    def build_body(
        self,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
    ) -> str:
        """Build the body section of the prompt."""
        return self.config.body_template.format(
            constraints=format_numbered_list(ai_directives.constraints),
            resources=format_numbered_list(ai_directives.resources),
            commands=self.generate_commands_list(commands),
            best_practices=format_numbered_list(ai_directives.best_practices),
        )


class MultiStepThoughts(ModelWithSummary):
    """Extended thoughts model for multi-step strategies.

    Includes additional fields for reflections and planning.
    """

    observations: str = Field(
        description="Relevant observations from your last action (if any)"
    )
    text: str = Field(description="Thoughts")
    reasoning: str = Field(description="Reasoning behind the thoughts")
    self_criticism: str = Field(description="Constructive self-criticism")
    plan: list[str] = Field(description="Short list that conveys the long-term plan")
    reflections_used: list[str] = Field(
        default_factory=list, description="Lessons applied from past reflections"
    )

    def summary(self) -> str:
        return self.text


# Type variable for action proposals
T = TypeVar("T", bound=ActionProposal)


def get_strategy_class(
    strategy_type: PromptStrategyType,
) -> type[BaseMultiStepPromptStrategy]:
    """Get the strategy class for a given strategy type.

    This is a registry function that returns the appropriate strategy class.
    Import is done lazily to avoid circular imports.
    """
    from .one_shot import OneShotAgentPromptStrategy

    strategy_map: dict[PromptStrategyType, type] = {
        PromptStrategyType.ONE_SHOT: OneShotAgentPromptStrategy,
    }

    # Lazy import for new strategies to avoid circular imports
    if strategy_type == PromptStrategyType.REWOO:
        from .rewoo import ReWOOPromptStrategy

        return ReWOOPromptStrategy
    elif strategy_type == PromptStrategyType.PLAN_EXECUTE:
        from .plan_execute import PlanExecutePromptStrategy

        return PlanExecutePromptStrategy
    elif strategy_type == PromptStrategyType.REFLEXION:
        from .reflexion import ReflexionPromptStrategy

        return ReflexionPromptStrategy
    elif strategy_type == PromptStrategyType.TREE_OF_THOUGHTS:
        from .tree_of_thoughts import TreeOfThoughtsPromptStrategy

        return TreeOfThoughtsPromptStrategy
    elif strategy_type == PromptStrategyType.LATS:
        from .lats import LATSPromptStrategy

        return LATSPromptStrategy
    elif strategy_type == PromptStrategyType.MULTI_AGENT_DEBATE:
        from .multi_agent_debate import MultiAgentDebateStrategy

        return MultiAgentDebateStrategy

    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategy_map[strategy_type]
