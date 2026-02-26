"""LATS (Language Agent Tree Search) prompt strategy.

This strategy implements the LATS algorithm from the paper:
"Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models"

LATS uses sub-agents to explore different reasoning paths with Monte Carlo Tree Search,
combining the benefits of tree search with LLM-based evaluation.

Key features:
- Sub-agents explore different action paths in parallel
- Monte Carlo Tree Search for intelligent exploration
- Value function learned from sub-agent outcomes
- Reflection on failed paths to improve future exploration
"""

from __future__ import annotations

import json
import re
from enum import Enum
from logging import Logger
from typing import Any, Optional

from pydantic import BaseModel, Field

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.json.parsing import extract_dict_from_json
from forge.llm.prompting import ChatPrompt, LanguageModelClassification
from forge.llm.providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.action import ActionProposal
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.models.utils import ModelWithSummary
from forge.utils.exceptions import InvalidAgentResponseError

from .base import BaseMultiStepPromptStrategy, BasePromptStrategyConfiguration


class LATSPhase(str, Enum):
    """Phases of the LATS algorithm."""

    SELECTION = "selection"  # Select node to expand using UCT
    EXPANSION = "expansion"  # Generate candidate actions via sub-agents
    EVALUATION = "evaluation"  # Evaluate candidates
    BACKPROPAGATION = "backpropagation"  # Update value estimates
    EXECUTION = "execution"  # Execute best action


class LATSNode(BaseModel):
    """A node in the LATS search tree."""

    state_description: str = Field(description="Description of the state at this node")
    action_taken: Optional[str] = Field(
        default=None, description="Action that led to this state"
    )
    value: float = Field(default=0.0, description="Estimated value (Q-value)")
    visits: int = Field(default=0, description="Number of times this node was visited")
    children: list["LATSNode"] = Field(default_factory=list)
    parent: Optional["LATSNode"] = Field(default=None, exclude=True)
    depth: int = Field(default=0)
    is_terminal: bool = Field(default=False)
    reward: float = Field(default=0.0, description="Reward received at this node")
    reflection: str = Field(default="", description="Reflection on failures")

    model_config = {"arbitrary_types_allowed": True}

    def uct_score(self, exploration_weight: float = 1.41) -> float:
        """Calculate UCT (Upper Confidence Bound for Trees) score."""
        if self.visits == 0:
            return float("inf")  # Encourage exploration of unvisited nodes

        if self.parent is None or self.parent.visits == 0:
            return self.value

        import math

        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


class LATSThoughts(ModelWithSummary):
    """Thoughts for LATS strategy."""

    observations: str = Field(description="Current observations from the state")
    reasoning: str = Field(description="Reasoning about which path to take")
    candidate_actions: list[str] = Field(
        default_factory=list, description="Candidate actions being considered"
    )
    selected_action: str = Field(description="The action selected for execution")
    confidence: float = Field(
        default=0.5, description="Confidence in the selected action (0-1)"
    )

    def summary(self) -> str:
        return self.selected_action


class LATSActionProposal(ActionProposal):
    """Action proposal for LATS strategy."""

    thoughts: LATSThoughts  # type: ignore


class LATSPromptConfiguration(BasePromptStrategyConfiguration):
    """Configuration for LATS strategy."""

    # MCTS parameters
    num_candidates: int = UserConfigurable(default=3)
    """Number of candidate actions to generate per expansion."""

    max_depth: int = UserConfigurable(default=5)
    """Maximum depth of the search tree."""

    exploration_weight: float = UserConfigurable(default=1.41)
    """UCT exploration weight (sqrt(2) is theoretically optimal)."""

    num_simulations: int = UserConfigurable(default=3)
    """Number of MCTS simulations per decision."""

    # Sub-agent configuration (inherited, but with LATS-specific defaults)
    enable_sub_agents: bool = UserConfigurable(default=True)
    max_sub_agents: int = UserConfigurable(default=10)
    sub_agent_timeout_seconds: int = UserConfigurable(default=120)
    sub_agent_max_cycles: int = UserConfigurable(default=10)

    DEFAULT_EXPANSION_INSTRUCTION: str = (
        "You are exploring possible actions for a task. "
        "Generate {num_candidates} distinct candidate actions "
        "that could make progress.\n\n"
        "## Action Prioritization\n"
        "1. Running tests (pytest) - highest value\n"
        "2. Writing/modifying code - direct progress\n"
        "3. Reading files - only when needed\n"
        "Avoid: excessive todo management, clipboard operations\n\n"
        "Current state: {state}\n"
        "Task: {task}\n\n"
        "For each candidate, provide:\n"
        "1. The action name and arguments\n"
        "2. Expected outcome\n"
        "3. Potential risks\n\n"
        "Format as JSON array of objects with "
        "'action', 'expected_outcome', 'risks' keys."
    )

    DEFAULT_EVALUATION_INSTRUCTION: str = (
        "Evaluate the following action outcome.\n\n"
        "Action: {action}\n"
        "Result: {result}\n"
        "Task goal: {task}\n\n"
        "## Scoring Criteria\n"
        "- 1.0: All tests pass\n"
        "- 0.7-0.9: Most tests pass\n"
        "- 0.4-0.6: Partial implementation\n"
        "- 0.0-0.3: No progress or regression\n"
        "Key: Fewer test failures = higher score\n\n"
        "Provide a score from 0.0 to 1.0 indicating progress toward the goal.\n"
        "Also provide a brief reflection on what worked or didn't work.\n\n"
        "Format: {{'score': 0.X, 'reflection': '...'}}"
    )

    expansion_instruction: str = UserConfigurable(default=DEFAULT_EXPANSION_INSTRUCTION)
    evaluation_instruction: str = UserConfigurable(
        default=DEFAULT_EVALUATION_INSTRUCTION
    )


class LATSPromptStrategy(BaseMultiStepPromptStrategy):
    """LATS (Language Agent Tree Search) prompt strategy.

    Uses sub-agents to explore different action paths with MCTS,
    combining tree search with LLM-based value estimation.
    """

    default_configuration = LATSPromptConfiguration()

    def __init__(
        self,
        configuration: LATSPromptConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: LATSPromptConfiguration = configuration
        self.response_schema = JSONSchema.from_dict(
            LATSActionProposal.model_json_schema()
        )

        # LATS state
        self.root: Optional[LATSNode] = None
        self.current_node: Optional[LATSNode] = None
        self.phase = LATSPhase.SELECTION
        self.simulation_count = 0
        self.candidate_actions: list[dict[str, Any]] = []

    @property
    def llm_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.SMART_MODEL

    def build_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
        **extras,
    ) -> ChatPrompt:
        """Build prompt based on current LATS phase."""
        # Initialize root node if needed
        if self.root is None:
            self.root = LATSNode(
                state_description=f"Initial state for task: {task}",
                depth=0,
            )
            self.current_node = self.root

        system_prompt = self._build_system_prompt(
            ai_profile, ai_directives, commands, include_os_info
        )

        # Add LATS-specific context
        lats_context = self._build_lats_context(task)

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(lats_context),
                ChatMessage.user(self._get_phase_instruction()),
            ],
            prefill_response='{\n    "thoughts":',
            functions=commands,
        )

    def _build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> str:
        """Build the system prompt."""
        intro = self.generate_intro_prompt(ai_profile)
        body = self.build_body(ai_directives, commands)

        lats_intro = (
            "\n\n## LATS Strategy\n"
            "You are using Language Agent Tree Search (LATS) to explore actions.\n"
            "This involves:\n"
            "1. Generating candidate actions\n"
            "2. Evaluating their potential\n"
            "3. Selecting the most promising path\n"
            "4. Learning from outcomes to improve future decisions\n"
        )

        coding_guidance = (
            "\n\n## Coding Task Strategy\n"
            "For coding tasks:\n"
            "1. Read tests/specs first to understand requirements\n"
            "2. Write complete initial implementation quickly\n"
            "3. Run tests frequently (pytest after each change)\n"
            "4. Fix failures systematically, one by one\n"
            "5. Minimize overhead - code directly, avoid excessive planning\n"
            "\n## Shell Command Tips\n"
            "- For pytest: just use `pytest -v` or `python -m pytest`\n"
            "- Env vars: use `env VAR=val cmd` instead of `VAR=val cmd`\n"
            "- Prefer execute_python_code for importing and running modules\n"
        )

        response_format = self._build_response_format()

        parts = intro + [body, lats_intro, coding_guidance, response_format]
        if include_os_info:
            parts.extend(self.generate_os_info())

        return "\n\n".join(parts)

    def _build_lats_context(self, task: str) -> str:
        """Build context about current LATS state."""
        if self.current_node is None:
            return ""

        context_parts = [
            "## Current Search State",
            f"Phase: {self.phase.value}",
            f"Tree depth: {self.current_node.depth}",
            f"Simulations completed: "
            f"{self.simulation_count}/{self.config.num_simulations}",
        ]

        if self.current_node.reflection:
            context_parts.append(f"Previous reflection: {self.current_node.reflection}")

        if self.candidate_actions:
            context_parts.append(
                f"Candidate actions under consideration: {len(self.candidate_actions)}"
            )

        return "\n".join(context_parts)

    def _get_phase_instruction(self) -> str:
        """Get instruction for current phase."""
        if self.phase == LATSPhase.SELECTION:
            return (
                "Select the next action to execute. Consider the search tree "
                "state and choose the most promising action based on UCT scores "
                "and your reasoning. "
                "For coding tasks, prioritize: "
                "1) Running tests (pytest) to see current state, "
                "2) Writing/fixing code to make tests pass, "
                "3) Reading files only when needed. "
                "Avoid todo commands and clipboard - focus on coding."
            )
        elif self.phase == LATSPhase.EXPANSION:
            return (
                f"Generate {self.config.num_candidates} candidate actions. "
                "Each should be a distinct approach to making progress on the task. "
                "For coding: prefer write_file or execute_python_code to implement, "
                "then pytest to test. Minimize planning overhead."
            )
        elif self.phase == LATSPhase.EVALUATION:
            return (
                "Evaluate the outcome of the last action. "
                "Score progress from 0.0 to 1.0 and reflect on what worked. "
                "For coding: count passing/failing tests. More passing = higher score. "
                "Score 1.0 for all tests passing, scale down based on failures."
            )
        else:
            return "Execute the selected action."

    def _build_response_format(self) -> str:
        """Build response format instruction."""
        response_schema = self.response_schema.model_copy(deep=True)
        if response_schema.properties and "use_tool" in response_schema.properties:
            del response_schema.properties["use_tool"]

        return (
            "## Response Format\n"
            "Respond with a JSON object containing your thoughts and invoke a tool.\n"
            f"{response_schema.to_typescript_object_interface('LATSResponse')}"
        )

    async def expand_with_sub_agents(self, task: str, state: str) -> list[dict]:
        """Use sub-agents to generate candidate actions."""
        if not self.can_spawn_sub_agent():
            self.logger.warning("Cannot spawn sub-agents for LATS expansion")
            return []

        expansion_tasks = []
        for i in range(self.config.num_candidates):
            sub_task = (
                f"You are candidate explorer #{i + 1}. "
                f"Task: {task}\n"
                f"Current state: {state}\n"
                f"Propose ONE specific action to make progress. "
                f"Focus on a unique approach different from other explorers."
            )
            expansion_tasks.append(sub_task)

        # Run sub-agents in parallel
        try:
            results = await self.run_parallel(
                expansion_tasks,
                strategy="one_shot",
                max_cycles=self.config.sub_agent_max_cycles,
            )

            candidates = []
            for i, result in enumerate(results):
                if result:
                    candidates.append(
                        {
                            "index": i,
                            "suggestion": str(result)[:500],
                            "source": f"sub-agent-{i}",
                        }
                    )

            self.candidate_actions = candidates
            return candidates

        except Exception as e:
            self.logger.error(f"LATS expansion failed: {e}")
            return []

    async def evaluate_with_sub_agent(
        self, action: str, result: str, task: str
    ) -> tuple[float, str]:
        """Use a sub-agent to evaluate an action outcome."""
        if not self.can_spawn_sub_agent():
            return 0.5, "Unable to evaluate (no sub-agent available)"

        eval_task = self.config.evaluation_instruction.format(
            action=action,
            result=result,
            task=task,
        )

        try:
            eval_result = await self.spawn_and_run(
                eval_task,
                strategy="one_shot",
                max_cycles=5,
            )

            if eval_result:
                # Try to parse score and reflection
                try:
                    parsed = json.loads(str(eval_result))
                    score = float(parsed.get("score", 0.5))
                    reflection = parsed.get("reflection", "")
                    return score, reflection
                except (json.JSONDecodeError, ValueError):
                    # Extract score from text
                    score_match = re.search(r"(\d+\.?\d*)", str(eval_result))
                    score = float(score_match.group(1)) if score_match else 0.5
                    return min(score, 1.0), str(eval_result)[:200]

            return 0.5, "Evaluation completed without result"

        except Exception as e:
            self.logger.error(f"LATS evaluation failed: {e}")
            return 0.5, f"Evaluation error: {e}"

    def select_node(self) -> LATSNode:
        """Select node to expand using UCT."""
        if self.root is None:
            raise RuntimeError("LATS tree not initialized")

        node = self.root
        while node.children and not node.is_terminal:
            # Select child with highest UCT score
            node = max(
                node.children,
                key=lambda n: n.uct_score(self.config.exploration_weight),
            )

        return node

    def backpropagate(self, node: LATSNode, reward: float) -> None:
        """Backpropagate reward through the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> LATSActionProposal:
        """Parse the LLM response into a LATS action proposal."""
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        self.logger.debug(f"LLM response content:\n{response.content[:500]}")

        assistant_reply_dict = extract_dict_from_json(response.content)

        if not response.tool_calls:
            raise InvalidAgentResponseError("Assistant did not use a tool")

        assistant_reply_dict["use_tool"] = response.tool_calls[0].function

        parsed_response = LATSActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        # Update LATS state based on response
        self._update_state_from_response(parsed_response)

        return parsed_response

    def _update_state_from_response(self, response: LATSActionProposal) -> None:
        """Update LATS state after receiving a response."""
        if self.current_node is None:
            return

        # Create child node for the action taken
        child = LATSNode(
            state_description=f"After: {response.use_tool.name}",
            action_taken=response.use_tool.name,
            parent=self.current_node,
            depth=self.current_node.depth + 1,
        )
        self.current_node.children.append(child)
        self.current_node = child

        # Advance phase
        self.simulation_count += 1
        if self.simulation_count >= self.config.num_simulations:
            self.phase = LATSPhase.EXECUTION
        else:
            self.phase = LATSPhase.SELECTION

    def record_execution_result(
        self, variable_name: str, result: str, error: Optional[str] = None
    ) -> None:
        """Record execution result for backpropagation."""
        if self.current_node is None:
            return

        # Simple reward based on success/failure
        if error:
            reward = 0.0
            self.current_node.reflection = f"Action failed: {error}"
        else:
            reward = 0.5  # Base reward for successful execution
            if "success" in result.lower() or "completed" in result.lower():
                reward = 1.0

        self.current_node.reward = reward
        self.backpropagate(self.current_node, reward)
