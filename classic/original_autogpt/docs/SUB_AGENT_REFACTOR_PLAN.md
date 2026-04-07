# Sub-Agent Spawning Architecture Refactor

## Overview

This document outlines a comprehensive refactor to enable prompt strategies to spawn and coordinate sub-agents. This enables advanced patterns like:

- **LATS** (Language Agent Tree Search) - parallel exploration branches
- **Multi-agent debate** - consensus through agent interaction
- **Hierarchical decomposition** - delegate subtasks to specialists
- **Agent-as-tool** - use agents like functions

## Current Architecture (Before)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Loop                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ while running:                                           │    │
│  │   prompt = strategy.build_prompt(messages, task, ...)   │    │
│  │   response = llm.call(prompt)                           │    │
│  │   proposal = strategy.parse_response(response)          │    │
│  │   result = agent.execute(proposal)  ← tools only        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Strategy has NO access to:
- Agent factory
- LLM provider
- File storage
- Execution context
- Other agents
```

## Proposed Architecture (After)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Execution Context                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Agent Factory│  │ LLM Provider │  │ File Storage │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                  │                  │
│         └─────────────────┼──────────────────┘                  │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Parent Agent                          │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              Prompt Strategy                     │    │    │
│  │  │  - Has access to ExecutionContext               │    │    │
│  │  │  - Can spawn sub-agents via context             │    │    │
│  │  │  - Can await sub-agent results                  │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                         │                                │    │
│  │           ┌─────────────┼─────────────┐                 │    │
│  │           ▼             ▼             ▼                 │    │
│  │    ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │    │ SubAgent1 │ │ SubAgent2 │ │ SubAgent3 │           │    │
│  │    │ (searcher)│ │ (analyzer)│ │ (coder)   │           │    │
│  │    └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Infrastructure

### 1.1 ExecutionContext Model

**File: `forge/agent/execution_context.py`** (NEW)

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from forge.agent.base import BaseAgent
    from forge.file_storage.base import FileStorage
    from forge.llm.providers import MultiProvider


class ResourceBudget(BaseModel):
    """Resource limits for an agent and its children."""

    max_tokens: Optional[int] = None
    max_cycles: Optional[int] = None
    max_sub_agents: int = 10
    max_depth: int = 3  # Nesting depth limit
    deadline: Optional[datetime] = None

    def remaining_time(self) -> Optional[timedelta]:
        if self.deadline:
            return self.deadline - datetime.now()
        return None

    def create_child_budget(self, fraction: float = 0.5) -> "ResourceBudget":
        """Create a budget for a child agent."""
        return ResourceBudget(
            max_tokens=int(self.max_tokens * fraction) if self.max_tokens else None,
            max_cycles=int(self.max_cycles * fraction) if self.max_cycles else None,
            max_sub_agents=max(1, self.max_sub_agents // 2),
            max_depth=self.max_depth - 1,
            deadline=self.deadline,
        )


class SubAgentHandle(BaseModel):
    """Reference to a spawned sub-agent."""

    agent_id: str
    task: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None

    # Internal (excluded from serialization)
    _agent: Optional["BaseAgent"] = None
    _task: Optional[asyncio.Task] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True


@dataclass
class ExecutionContext:
    """Context passed down the agent hierarchy."""

    # Core dependencies
    llm_provider: "MultiProvider"
    file_storage: "FileStorage"

    # Agent factory function
    agent_factory: "AgentFactory"

    # Hierarchy tracking
    parent_agent_id: Optional[str] = None
    depth: int = 0

    # Resource management
    budget: ResourceBudget = field(default_factory=ResourceBudget)

    # Active sub-agents
    sub_agents: dict[str, SubAgentHandle] = field(default_factory=dict)

    # Cancellation
    cancelled: bool = False

    def can_spawn_sub_agent(self) -> bool:
        """Check if spawning another sub-agent is allowed."""
        if self.cancelled:
            return False
        if self.budget.max_depth <= 0:
            return False
        if len(self.sub_agents) >= self.budget.max_sub_agents:
            return False
        if self.budget.deadline and datetime.now() >= self.budget.deadline:
            return False
        return True

    def create_child_context(self, child_agent_id: str) -> "ExecutionContext":
        """Create a context for a child agent."""
        return ExecutionContext(
            llm_provider=self.llm_provider,
            file_storage=self.file_storage,
            agent_factory=self.agent_factory,
            parent_agent_id=child_agent_id,
            depth=self.depth + 1,
            budget=self.budget.create_child_budget(),
        )

    async def cancel_all_sub_agents(self):
        """Cancel all running sub-agents."""
        self.cancelled = True
        for handle in self.sub_agents.values():
            if handle._task and not handle._task.done():
                handle._task.cancel()
                handle.status = "cancelled"
```

### 1.2 AgentFactory Protocol

**File: `forge/agent/factory.py`** (NEW)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile

if TYPE_CHECKING:
    from forge.agent.base import BaseAgent
    from forge.agent.execution_context import ExecutionContext


class AgentFactory(Protocol):
    """Protocol for creating agents."""

    def create_agent(
        self,
        agent_id: str,
        task: str,
        context: ExecutionContext,
        ai_profile: Optional[AIProfile] = None,
        directives: Optional[AIDirectives] = None,
        strategy: Optional[str] = None,
    ) -> "BaseAgent":
        """Create a new agent instance."""
        ...


class DefaultAgentFactory:
    """Default implementation of AgentFactory."""

    def __init__(self, app_config: "AppConfig"):
        self.app_config = app_config

    def create_agent(
        self,
        agent_id: str,
        task: str,
        context: ExecutionContext,
        ai_profile: Optional[AIProfile] = None,
        directives: Optional[AIDirectives] = None,
        strategy: Optional[str] = None,
    ) -> "BaseAgent":
        from autogpt.agent_factory.configurators import create_agent_state
        from autogpt.agents.agent import Agent

        # Use provided or default profile/directives
        ai_profile = ai_profile or AIProfile(ai_name=f"SubAgent-{agent_id[:8]}")
        directives = directives or AIDirectives()

        # Create state
        state = create_agent_state(
            agent_id=agent_id,
            task=task,
            ai_profile=ai_profile,
            directives=directives,
            app_config=self.app_config,
        )

        # Override strategy if specified
        config = self.app_config.model_copy()
        if strategy:
            config.prompt_strategy = strategy

        return Agent(
            settings=state,
            llm_provider=context.llm_provider,
            file_storage=context.file_storage,
            app_config=config,
            execution_context=context,  # NEW: pass context
        )
```

---

## Phase 2: Strategy Interface Updates

### 2.1 Update BasePromptStrategyConfiguration

**File: `original_autogpt/autogpt/agents/prompt_strategies/base.py`**

```python
class BasePromptStrategyConfiguration(SystemConfiguration):
    # ... existing fields ...

    # Sub-agent configuration
    enable_sub_agents: bool = UserConfigurable(default=False)
    max_sub_agents: int = UserConfigurable(default=5)
    sub_agent_timeout_seconds: int = UserConfigurable(default=300)
```

### 2.2 Update BaseMultiStepPromptStrategy

**File: `original_autogpt/autogpt/agents/prompt_strategies/base.py`**

```python
from forge.agent.execution_context import ExecutionContext, SubAgentHandle

class BaseMultiStepPromptStrategy(PromptStrategy, ABC):
    """Base class for multi-step strategies with sub-agent support."""

    def __init__(
        self,
        configuration: BasePromptStrategyConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.logger = logger
        self._execution_context: Optional[ExecutionContext] = None

    def set_execution_context(self, context: ExecutionContext) -> None:
        """Inject the execution context. Called by Agent after creation."""
        self._execution_context = context

    @property
    def execution_context(self) -> Optional[ExecutionContext]:
        return self._execution_context

    def can_spawn_sub_agent(self) -> bool:
        """Check if this strategy can spawn sub-agents."""
        if not self.config.enable_sub_agents:
            return False
        if not self._execution_context:
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

        Args:
            task: The task for the sub-agent
            ai_profile: Optional custom profile
            directives: Optional custom directives
            strategy: Optional prompt strategy override

        Returns:
            Handle to the spawned sub-agent
        """
        if not self.can_spawn_sub_agent():
            raise RuntimeError("Cannot spawn sub-agent: disabled or limit reached")

        ctx = self._execution_context
        agent_id = f"sub-{uuid4().hex[:8]}"

        # Create child context
        child_ctx = ctx.create_child_context(agent_id)

        # Create the sub-agent
        sub_agent = ctx.agent_factory.create_agent(
            agent_id=agent_id,
            task=task,
            context=child_ctx,
            ai_profile=ai_profile,
            directives=directives,
            strategy=strategy,
        )

        # Create handle
        handle = SubAgentHandle(
            agent_id=agent_id,
            task=task,
            status="pending",
        )
        handle._agent = sub_agent

        # Track in context
        ctx.sub_agents[agent_id] = handle

        return handle

    async def run_sub_agent(
        self,
        handle: SubAgentHandle,
        max_cycles: Optional[int] = None,
    ) -> Any:
        """Run a sub-agent until completion.

        Args:
            handle: The sub-agent handle from spawn_sub_agent
            max_cycles: Maximum execution cycles

        Returns:
            The final result from the sub-agent
        """
        if handle._agent is None:
            raise RuntimeError("Sub-agent not initialized")

        agent = handle._agent
        handle.status = "running"
        cycles = 0
        max_cycles = max_cycles or self.config.max_sub_agent_cycles

        try:
            while cycles < max_cycles:
                # Check for cancellation
                if self._execution_context and self._execution_context.cancelled:
                    handle.status = "cancelled"
                    return None

                # Propose and execute
                proposal = await agent.propose_action()

                # Check for finish command
                if proposal.use_tool.name == "finish":
                    handle.status = "completed"
                    handle.result = proposal.use_tool.arguments.get("reason", "")
                    return handle.result

                # Execute the action
                result = await agent.execute(proposal)
                cycles += 1

            # Max cycles reached
            handle.status = "completed"
            handle.result = "Max cycles reached"
            return handle.result

        except Exception as e:
            handle.status = "failed"
            handle.error = str(e)
            raise

    async def spawn_and_run(
        self,
        task: str,
        ai_profile: Optional[AIProfile] = None,
        directives: Optional[AIDirectives] = None,
        strategy: Optional[str] = None,
        max_cycles: Optional[int] = None,
    ) -> Any:
        """Convenience: spawn and immediately run a sub-agent."""
        handle = await self.spawn_sub_agent(task, ai_profile, directives, strategy)
        return await self.run_sub_agent(handle, max_cycles)

    async def run_parallel(
        self,
        tasks: list[str],
        strategy: Optional[str] = None,
        max_cycles: Optional[int] = None,
    ) -> list[Any]:
        """Run multiple sub-agents in parallel.

        Args:
            tasks: List of tasks to run
            strategy: Prompt strategy for all sub-agents
            max_cycles: Max cycles per sub-agent

        Returns:
            List of results in same order as tasks
        """
        handles = []
        for task in tasks:
            handle = await self.spawn_sub_agent(task, strategy=strategy)
            handles.append(handle)

        # Run all in parallel
        coros = [self.run_sub_agent(h, max_cycles) for h in handles]
        results = await asyncio.gather(*coros, return_exceptions=True)

        return results
```

---

## Phase 3: Agent Integration

### 3.1 Update Agent Class

**File: `original_autogpt/autogpt/agents/agent.py`**

```python
class Agent(BaseAgent[AnyActionProposal], Configurable[AgentSettings]):

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        app_config: AppConfig,
        permission_manager: Optional[CommandPermissionManager] = None,
        execution_context: Optional[ExecutionContext] = None,  # NEW
    ):
        super().__init__(settings, permission_manager=permission_manager)

        self.llm_provider = llm_provider
        self.app_config = app_config

        # Create or use provided execution context
        if execution_context:
            self.execution_context = execution_context
        else:
            # Root agent - create new context
            from forge.agent.factory import DefaultAgentFactory
            self.execution_context = ExecutionContext(
                llm_provider=llm_provider,
                file_storage=file_storage,
                agent_factory=DefaultAgentFactory(app_config),
            )

        # Create strategy and inject context
        self.prompt_strategy = self._create_prompt_strategy(app_config)
        if hasattr(self.prompt_strategy, 'set_execution_context'):
            self.prompt_strategy.set_execution_context(self.execution_context)

        # ... rest of __init__ ...
```

### 3.2 Update Agent Factory

**File: `original_autogpt/autogpt/agent_factory/configurators.py`**

```python
def create_agent(
    agent_id: str,
    task: str,
    app_config: AppConfig,
    file_storage: FileStorage,
    llm_provider: MultiProvider,
    ai_profile: Optional[AIProfile] = None,
    directives: Optional[AIDirectives] = None,
    permission_manager: Optional[CommandPermissionManager] = None,
    execution_context: Optional[ExecutionContext] = None,  # NEW
) -> Agent:
    # ... existing code ...

    return Agent(
        settings=agent_state,
        llm_provider=llm_provider,
        file_storage=file_storage,
        app_config=app_config,
        permission_manager=permission_manager,
        execution_context=execution_context,  # NEW
    )
```

---

## Phase 4: Example Strategy - LATS

### 4.1 LATS Strategy Implementation

**File: `original_autogpt/autogpt/agents/prompt_strategies/lats.py`** (NEW)

```python
"""Language Agent Tree Search (LATS) Strategy.

Implements LATS from the paper "Language Agent Tree Search Unifies Reasoning,
Acting, and Planning in Language Models" (arxiv.org/abs/2310.04406).

LATS uses Monte Carlo Tree Search (MCTS) with LLM-based:
- Node expansion (generate candidate actions)
- Evaluation (score candidates)
- Simulation (run sub-agents to explore branches)
- Backpropagation (update scores based on outcomes)
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from enum import Enum
from logging import Logger
from typing import Any, Optional

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.llm.prompting import ChatPrompt
from forge.llm.providers.schema import AssistantChatMessage, ChatMessage
from forge.models.action import ActionProposal
from forge.models.config import UserConfigurable
from pydantic import Field

from .base import (
    BaseMultiStepPromptStrategy,
    BasePromptStrategyConfiguration,
)


class LATSPhase(str, Enum):
    SELECT = "select"
    EXPAND = "expand"
    SIMULATE = "simulate"
    BACKPROPAGATE = "backpropagate"


@dataclass
class LATSNode:
    """A node in the LATS search tree."""

    state: str  # Description of current state
    action: Optional[str] = None  # Action that led here
    parent: Optional["LATSNode"] = None
    children: list["LATSNode"] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    value: float = 0.0

    # Simulation results
    simulated: bool = False
    simulation_result: Optional[str] = None

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound for tree policy."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.value / self.visits

        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return (self.value / self.visits) + exploration

    def best_child(self) -> Optional["LATSNode"]:
        """Select best child by UCB1."""
        if not self.children:
            return None
        return max(self.children, key=lambda n: n.ucb1)


class LATSPromptConfiguration(BasePromptStrategyConfiguration):
    """Configuration for LATS strategy."""

    # MCTS parameters
    num_simulations: int = UserConfigurable(default=5)
    exploration_constant: float = UserConfigurable(default=1.414)
    max_depth: int = UserConfigurable(default=10)
    branching_factor: int = UserConfigurable(default=3)

    # Sub-agent configuration
    enable_sub_agents: bool = True  # Required for LATS
    simulation_strategy: str = UserConfigurable(default="one_shot")
    simulation_max_cycles: int = UserConfigurable(default=10)

    # Prompts
    DEFAULT_EXPAND_INSTRUCTION: str = (
        "Given the current state, generate {branching_factor} distinct "
        "candidate actions. Each should be a different approach.\n\n"
        "Current state:\n{state}\n\n"
        "Previous actions:\n{action_history}\n\n"
        "Generate candidates as a JSON array."
    )

    DEFAULT_EVALUATE_INSTRUCTION: str = (
        "Evaluate how promising this action is for solving the task.\n\n"
        "Task: {task}\n"
        "Current state: {state}\n"
        "Proposed action: {action}\n\n"
        "Score from 0-10 and explain briefly."
    )

    expand_instruction: str = UserConfigurable(default=DEFAULT_EXPAND_INSTRUCTION)
    evaluate_instruction: str = UserConfigurable(default=DEFAULT_EVALUATE_INSTRUCTION)


class LATSActionProposal(ActionProposal):
    """Action proposal for LATS."""

    thoughts: dict[str, Any]
    search_tree_summary: str = ""


class LATSPromptStrategy(BaseMultiStepPromptStrategy):
    """LATS: Language Agent Tree Search."""

    default_configuration = LATSPromptConfiguration()

    def __init__(
        self,
        configuration: LATSPromptConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: LATSPromptConfiguration = configuration
        self.root: Optional[LATSNode] = None
        self.current_phase = LATSPhase.SELECT
        self.simulations_completed = 0

    async def run_mcts(self, task: str, initial_state: str) -> LATSNode:
        """Run MCTS to find best action."""

        # Initialize root
        self.root = LATSNode(state=initial_state)

        for _ in range(self.config.num_simulations):
            # SELECT: traverse to promising leaf
            node = self._select(self.root)

            # EXPAND: generate candidate children
            if not node.children and node.visits > 0:
                await self._expand(node, task)

            # SIMULATE: run sub-agent to evaluate
            if node.children:
                # Select a child to simulate
                child = node.children[0]  # Or random
                if not child.simulated:
                    value = await self._simulate(child, task)
                    child.simulated = True
                    child.value = value

            # BACKPROPAGATE: update ancestor values
            self._backpropagate(node)

            self.simulations_completed += 1

        return self.root

    def _select(self, node: LATSNode) -> LATSNode:
        """Select most promising leaf node."""
        while node.children:
            best = node.best_child()
            if best is None:
                break
            node = best
        return node

    async def _expand(self, node: LATSNode, task: str) -> None:
        """Expand node by generating candidate actions."""
        # This would call the LLM to generate candidates
        # For now, placeholder
        pass

    async def _simulate(self, node: LATSNode, task: str) -> float:
        """Simulate by running a sub-agent.

        This is where sub-agent spawning happens!
        """
        if not self.can_spawn_sub_agent():
            self.logger.warning("Cannot spawn sub-agent for simulation")
            return 0.5  # Neutral score

        # Build simulation task
        simulation_task = (
            f"Task: {task}\n\n"
            f"Current state: {node.state}\n"
            f"Action to take: {node.action}\n\n"
            f"Execute this action and report the outcome."
        )

        try:
            result = await self.spawn_and_run(
                task=simulation_task,
                strategy=self.config.simulation_strategy,
                max_cycles=self.config.simulation_max_cycles,
            )

            node.simulation_result = str(result)

            # Evaluate outcome (would call LLM)
            # For now, simple heuristic
            if "success" in str(result).lower():
                return 1.0
            elif "error" in str(result).lower():
                return 0.0
            else:
                return 0.5

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return 0.0

    def _backpropagate(self, node: LATSNode) -> None:
        """Propagate simulation value up the tree."""
        while node is not None:
            node.visits += 1
            if node.simulated:
                # Update running average
                pass
            node = node.parent

    # Required interface methods

    @property
    def llm_classification(self):
        from forge.llm.prompting import LanguageModelClassification
        return LanguageModelClassification.SMART_MODEL

    def build_prompt(self, **kwargs) -> ChatPrompt:
        # Build prompt based on current MCTS phase
        pass

    def parse_response_content(self, response: AssistantChatMessage) -> LATSActionProposal:
        # Parse response and update tree
        pass
```

---

## Phase 5: Additional Infrastructure

### 5.1 Sub-Agent Communication Protocol

**File: `forge/agent/sub_agent_protocol.py`** (NEW)

```python
"""Protocol for sub-agent communication."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    CANCEL = "cancel"


class SubAgentMessage(BaseModel):
    """Message passed between parent and sub-agent."""

    type: MessageType
    sender_id: str
    recipient_id: str
    content: Any
    correlation_id: Optional[str] = None


class SubAgentRequest(SubAgentMessage):
    """Request from parent to sub-agent."""

    type: MessageType = MessageType.REQUEST
    task: str
    context: dict[str, Any] = {}


class SubAgentResponse(SubAgentMessage):
    """Response from sub-agent to parent."""

    type: MessageType = MessageType.RESPONSE
    success: bool
    result: Any
    error: Optional[str] = None
```

### 5.2 Resource Tracking Component

**File: `forge/components/resource_tracker.py`** (NEW)

```python
"""Component for tracking resource usage across agent hierarchy."""

from forge.agent.components import AgentComponent
from forge.agent.protocols import AfterExecute
from forge.models.action import ActionResult


class ResourceTrackerComponent(AgentComponent, AfterExecute):
    """Tracks resource usage for budget enforcement."""

    def __init__(self, execution_context):
        self.context = execution_context
        self.tokens_used = 0
        self.cycles_completed = 0

    def after_execute(self, result: ActionResult) -> None:
        self.cycles_completed += 1

        # Check budget
        budget = self.context.budget
        if budget.max_cycles and self.cycles_completed >= budget.max_cycles:
            raise BudgetExceededError("Cycle budget exceeded")
```

---

## Implementation Order

### Week 1: Core Infrastructure
1. Create `ExecutionContext` model
2. Create `AgentFactory` protocol
3. Update `BaseMultiStepPromptStrategy` with sub-agent methods
4. Update `Agent.__init__` to accept context

### Week 2: Integration
5. Update agent factory functions
6. Add sub-agent communication protocol
7. Create resource tracking component
8. Add tests for sub-agent spawning

### Week 3: Example Strategy
9. Implement LATS strategy skeleton
10. Implement MCTS core logic
11. Integrate sub-agent simulation
12. End-to-end testing

### Week 4: Polish
13. Error handling and cleanup
14. Documentation
15. Performance optimization
16. Additional example strategies

---

## Migration Notes

### Backward Compatibility

- Existing strategies continue to work unchanged
- `enable_sub_agents=False` by default
- ExecutionContext is optional (created automatically for root agents)

### Breaking Changes

- None for existing code
- New strategies using sub-agents require updated Agent class

### Testing Strategy

1. Unit tests for ExecutionContext
2. Integration tests for sub-agent lifecycle
3. LATS strategy tests with mocked sub-agents
4. End-to-end tests with real LLM calls

---

## Open Questions

1. **Shared State**: Should sub-agents share file storage? Separate workspaces?
2. **Permissions**: Inherit from parent or independent permission checks?
3. **History**: Should parent see sub-agent action history?
4. **Cancellation**: How to handle partial results when cancelled?
5. **Debugging**: How to trace execution across agent hierarchy?
