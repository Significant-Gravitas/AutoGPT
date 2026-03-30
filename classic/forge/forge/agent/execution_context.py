"""Execution context for sub-agent support.

This module provides the infrastructure for strategies to spawn and coordinate
sub-agents. The ExecutionContext is passed down the agent hierarchy and provides
access to shared resources while enforcing resource budgets.

Based on research from:
- Google ADK Multi-Agent Patterns
- Anthropic Multi-Agent Research System
- LATS (Language Agent Tree Search)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from forge.config.ai_directives import AIDirectives
    from forge.config.ai_profile import AIProfile
    from forge.file_storage.base import FileStorage
    from forge.llm.providers import MultiProvider


class SubAgentStatus(str, Enum):
    """Status of a sub-agent."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceBudget:
    """Resource limits for sub-agent execution.

    Based on design decisions from research:
    - Permissive defaults for flexibility
    - Inherited deny rules, explicit allow rules
    """

    max_depth: int = 5
    """Maximum nesting depth for sub-agents."""

    max_sub_agents: int = 25
    """Maximum number of sub-agents that can be spawned."""

    max_cycles_per_agent: int = 50
    """Maximum execution cycles per agent."""

    max_tokens_total: int = 0
    """Maximum total tokens (0 = unlimited)."""

    inherited_deny_rules: list[str] = field(default_factory=list)
    """Permission deny rules inherited from parent (always enforced)."""

    explicit_allow_rules: list[str] = field(default_factory=list)
    """Permission allow rules explicitly granted to this context."""

    def create_child_budget(self) -> "ResourceBudget":
        """Create a budget for a child agent with reduced limits."""
        return ResourceBudget(
            max_depth=self.max_depth - 1,
            max_sub_agents=self.max_sub_agents,
            max_cycles_per_agent=self.max_cycles_per_agent,
            max_tokens_total=self.max_tokens_total,
            inherited_deny_rules=self.inherited_deny_rules.copy(),
            explicit_allow_rules=[],  # Child must get explicit permissions
        )


@dataclass
class SubAgentHandle:
    """Handle to a spawned sub-agent.

    Provides methods to interact with the sub-agent without exposing
    the full agent internals. This maintains proper encapsulation.
    """

    agent_id: str
    """Unique identifier for this sub-agent."""

    task: str
    """The task assigned to this sub-agent."""

    status: SubAgentStatus = SubAgentStatus.PENDING
    """Current status of the sub-agent."""

    result: Optional[Any] = None
    """Result from the sub-agent (if completed)."""

    error: Optional[str] = None
    """Error message (if failed)."""

    summary: str = ""
    """Brief summary of what the sub-agent accomplished."""

    # Internal fields (not part of public API)
    _agent: Optional[Any] = field(default=None, repr=False)
    _task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)

    def is_running(self) -> bool:
        """Check if the sub-agent is currently running."""
        return self.status == SubAgentStatus.RUNNING

    def is_done(self) -> bool:
        """Check if the sub-agent has finished (success or failure)."""
        return self.status in (
            SubAgentStatus.COMPLETED,
            SubAgentStatus.FAILED,
            SubAgentStatus.CANCELLED,
        )


class AgentFactory(Protocol):
    """Protocol for agent factory implementations.

    This allows strategies to spawn sub-agents without knowing the
    concrete agent implementation details.
    """

    def create_agent(
        self,
        agent_id: str,
        task: str,
        context: "ExecutionContext",
        ai_profile: Optional["AIProfile"] = None,
        directives: Optional["AIDirectives"] = None,
        strategy: Optional[str] = None,
    ) -> Any:
        """Create a new agent instance.

        Args:
            agent_id: Unique identifier for the agent.
            task: The task the agent should accomplish.
            context: Execution context with shared resources.
            ai_profile: Optional AI profile override.
            directives: Optional directives override.
            strategy: Optional strategy name override.

        Returns:
            A new agent instance.
        """
        ...


@dataclass
class ExecutionContext:
    """Context passed down the agent hierarchy.

    The ExecutionContext provides sub-agents with access to shared resources
    (LLM provider, file storage, agent factory) while enforcing resource
    budgets and isolation.

    Key design decisions (based on research):
    1. File storage: Sub-agents can READ parent workspace but only WRITE
       to their own subdirectory (.sub_agents/{agent_id}/).
    2. Permissions: Inherit deny rules, explicit allow rules per sub-agent.
    3. Context isolation: Sub-agents get minimal context (just task description).
    4. History visibility: Parent sees sub-agent results only, not full history.
    """

    llm_provider: "MultiProvider"
    """Shared LLM provider for all agents in the hierarchy."""

    file_storage: "FileStorage"
    """File storage (may have write restrictions for sub-agents)."""

    agent_factory: Optional[AgentFactory] = None
    """Factory for creating sub-agents."""

    parent_agent_id: Optional[str] = None
    """ID of the parent agent (None for root agent)."""

    depth: int = 0
    """Current depth in the agent hierarchy (0 = root)."""

    budget: ResourceBudget = field(default_factory=ResourceBudget)
    """Resource budget for this context."""

    sub_agents: dict[str, SubAgentHandle] = field(default_factory=dict)
    """Active sub-agents spawned by the owning agent."""

    _cancelled: bool = field(default=False, repr=False)
    """Whether this context has been cancelled."""

    _app_config: Optional[Any] = field(default=None, repr=False)
    """Application config (for agent creation)."""

    @property
    def is_root(self) -> bool:
        """Check if this is a root (top-level) agent context."""
        return self.parent_agent_id is None

    @property
    def cancelled(self) -> bool:
        """Check if this context has been cancelled."""
        return self._cancelled

    def can_spawn_sub_agent(self) -> bool:
        """Check if spawning a sub-agent is allowed.

        Returns:
            True if spawning is allowed, False otherwise.
        """
        if self._cancelled:
            return False
        if self.budget.max_depth <= 0:
            return False
        if len(self.sub_agents) >= self.budget.max_sub_agents:
            return False
        if self.agent_factory is None:
            return False
        return True

    def create_child_context(self, child_agent_id: str) -> "ExecutionContext":
        """Create a context for a child agent with appropriate restrictions.

        The child context has:
        - Same LLM provider (shared)
        - Write-restricted file storage (writes to .sub_agents/{child_agent_id}/)
        - Reduced budget (depth - 1)
        - Same agent factory

        Args:
            child_agent_id: ID of the child agent.

        Returns:
            A new ExecutionContext for the child.
        """
        # Create write-restricted file storage for child
        child_storage = self._create_child_storage(child_agent_id)

        return ExecutionContext(
            llm_provider=self.llm_provider,
            file_storage=child_storage,
            agent_factory=self.agent_factory,
            parent_agent_id=child_agent_id,
            depth=self.depth + 1,
            budget=self.budget.create_child_budget(),
            _app_config=self._app_config,
        )

    def _create_child_storage(self, child_agent_id: str) -> "FileStorage":
        """Create a write-restricted file storage for a child agent.

        The child can:
        - READ from the entire parent workspace
        - WRITE only to .sub_agents/{child_agent_id}/

        Args:
            child_agent_id: ID of the child agent.

        Returns:
            A FileStorage with write restrictions.
        """
        # Use clone_with_subroot for the write directory
        # The child's "root" for writing is the sub_agents directory
        sub_agent_path = f".sub_agents/{child_agent_id}"

        # For now, we create a subroot storage for the child
        # This restricts ALL access to the subroot (both read and write)
        # A more sophisticated implementation would allow read from parent
        # but that requires extending FileStorage
        return self.file_storage.clone_with_subroot(sub_agent_path)

    def register_sub_agent(self, handle: SubAgentHandle) -> None:
        """Register a sub-agent handle.

        Args:
            handle: The sub-agent handle to register.
        """
        self.sub_agents[handle.agent_id] = handle

    def get_sub_agent(self, agent_id: str) -> Optional[SubAgentHandle]:
        """Get a sub-agent handle by ID.

        Args:
            agent_id: The sub-agent ID.

        Returns:
            The sub-agent handle, or None if not found.
        """
        return self.sub_agents.get(agent_id)

    def cancel(self) -> None:
        """Cancel this context and all sub-agents.

        This sets the cancelled flag and attempts to cancel any running
        sub-agent tasks.
        """
        self._cancelled = True
        for handle in self.sub_agents.values():
            if handle._task and not handle._task.done():
                handle._task.cancel()
            handle.status = SubAgentStatus.CANCELLED

    async def wait_for_sub_agents(
        self,
        timeout: Optional[float] = None,
    ) -> dict[str, SubAgentHandle]:
        """Wait for all running sub-agents to complete.

        Args:
            timeout: Maximum time to wait (in seconds).

        Returns:
            Dictionary of all sub-agent handles.
        """
        tasks = [
            handle._task
            for handle in self.sub_agents.values()
            if handle._task and not handle._task.done()
        ]

        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

        return self.sub_agents


def generate_sub_agent_id(parent_id: Optional[str] = None) -> str:
    """Generate a unique ID for a sub-agent.

    Args:
        parent_id: Optional parent agent ID for hierarchical naming.

    Returns:
        A unique sub-agent ID.
    """
    short_uuid = str(uuid.uuid4())[:8]
    if parent_id:
        return f"{parent_id}-sub-{short_uuid}"
    return f"sub-{short_uuid}"
