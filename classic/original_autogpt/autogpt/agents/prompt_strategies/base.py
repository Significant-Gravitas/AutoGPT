"""Base classes and utilities for prompt strategies.

This module provides shared infrastructure for different prompt strategy
implementations including ReWOO, Plan-and-Execute, Reflexion, and Tree of Thoughts.
"""

from __future__ import annotations

import enum
import platform
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar

import distro
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
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class PromptStrategyType(str, enum.Enum):
    """Available prompt strategy types."""

    ONE_SHOT = "one_shot"
    REWOO = "rewoo"
    PLAN_EXECUTE = "plan_execute"
    REFLEXION = "reflexion"
    TREE_OF_THOUGHTS = "tree_of_thoughts"


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

    def to_prompt_text(self) -> str:
        """Format reflection for inclusion in prompts."""
        status = "succeeded" if self.success else "failed"
        text = f"Action '{self.action_name}' {status}: {self.result_summary}"
        if self.what_went_wrong:
            text += f"\n  - Issue: {self.what_went_wrong}"
        if self.what_to_do_differently:
            text += f"\n  - Lesson: {self.what_to_do_differently}"
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


class Thought(BaseModel):
    """A node in the Tree of Thoughts.

    Represents a single thought/reasoning step that can branch into children.
    """

    content: str = Field(description="The thought content")
    score: float = Field(default=0.0, description="Self-evaluation score (0-10)")
    depth: int = Field(default=0, description="Depth in the tree")
    children: list[Thought] = Field(default_factory=list)
    is_terminal: bool = Field(
        default=False, description="Whether this thought leads to an action"
    )
    action: Optional[AssistantFunctionCall] = Field(
        default=None, description="Action if terminal"
    )

    def add_child(self, thought: Thought) -> None:
        """Add a child thought."""
        thought.depth = self.depth + 1
        self.children.append(thought)

    def best_child(self) -> Optional[Thought]:
        """Return the highest-scoring child."""
        if not self.children:
            return None
        return max(self.children, key=lambda t: t.score)


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


class BaseMultiStepPromptStrategy(ABC):
    """Base class for multi-step prompt strategies.

    Provides common utilities for strategies that involve multiple phases
    like planning, execution, synthesis, or reflection.
    """

    def __init__(
        self,
        configuration: BasePromptStrategyConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.logger = logger

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
    speak: str = Field(description="Summary of thoughts, to say to user")
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

    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategy_map[strategy_type]
