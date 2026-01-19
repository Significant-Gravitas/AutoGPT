"""Tree of Thoughts (ToT) Prompt Strategy.

This strategy implements the Tree of Thoughts pattern from research including:
- Tree of Thoughts: Deliberate Problem Solving (arxiv.org/abs/2305.10601)
- ToTRL: Unlock Tree-of-Thought via Puzzles (arxiv.org/abs/2505.12717)
- Tree of Uncertain Thoughts (arxiv.org/abs/2309.07694)

Key benefits:
- Frames problem solving as search over a tree of partial solutions
- Enables backtracking when a reasoning path fails
- Self-evaluates choices to decide next action
- Better for complex reasoning and exploration tasks

Pattern:
1. GENERATE: Multiple candidate thoughts at each step
2. EVALUATE: Score each thought (1-10)
3. SEARCH: BFS/DFS through the thought tree
4. BACKTRACK: If dead end, try alternative paths
"""

from __future__ import annotations

import json
import re
from enum import Enum
from logging import Logger
from typing import Any, Literal, Optional

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.json.parsing import extract_dict_from_json
from forge.llm.prompting import ChatPrompt, LanguageModelClassification
from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.action import ActionProposal
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.models.utils import ModelWithSummary
from forge.utils.exceptions import InvalidAgentResponseError
from pydantic import Field

from .base import BaseMultiStepPromptStrategy, BasePromptStrategyConfiguration, Thought


class ToTPhase(str, Enum):
    """Current phase of Tree of Thoughts execution."""

    GENERATING = "generating"
    EVALUATING = "evaluating"
    SELECTING = "selecting"


class ToTThoughts(ModelWithSummary):
    """Thoughts model for Tree of Thoughts strategy."""

    current_path: list[str] = Field(
        default_factory=list,
        description="The reasoning path taken to reach this point",
    )
    alternatives_considered: int = Field(
        default=0,
        description="Number of alternative paths explored",
    )
    reasoning: str = Field(description="Current reasoning at this node")
    evaluation: str = Field(description="Self-evaluation of this reasoning path")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this path (0-1)",
    )
    speak: str = Field(description="Summary to say to user")

    def summary(self) -> str:
        return self.reasoning


class ThoughtCandidate(ModelWithSummary):
    """A candidate thought for evaluation."""

    thought: str = Field(description="The thought/reasoning step")
    leads_to_action: bool = Field(
        default=False, description="Whether this thought leads directly to an action"
    )
    action_name: Optional[str] = Field(
        default=None, description="Action name if leads_to_action is True"
    )
    action_arguments: Optional[dict[str, Any]] = Field(
        default=None, description="Action arguments if leads_to_action is True"
    )

    def summary(self) -> str:
        return self.thought


class ThoughtEvaluation(ModelWithSummary):
    """Evaluation of a thought candidate."""

    thought_index: int = Field(description="Index of the thought being evaluated")
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    reasoning: str = Field(description="Reasoning for the score")
    is_promising: bool = Field(description="Whether this path is worth exploring")

    def summary(self) -> str:
        return f"Score {self.score}: {self.reasoning}"


class ThoughtTree:
    """A tree of thoughts for exploration."""

    def __init__(self, root_content: str = ""):
        self.root = Thought(content=root_content, depth=0)
        self.current_node: Thought = self.root
        self.explored_paths: list[list[str]] = []
        self.best_path: list[Thought] = []
        self.best_score: float = 0.0

    def add_candidates(self, candidates: list[ThoughtCandidate]) -> list[Thought]:
        """Add candidate thoughts as children of current node."""
        new_nodes = []
        for candidate in candidates:
            thought = Thought(
                content=candidate.thought,
                depth=self.current_node.depth + 1,
                is_terminal=candidate.leads_to_action,
            )
            if candidate.leads_to_action and candidate.action_name:
                thought.action = AssistantFunctionCall(
                    name=candidate.action_name,
                    arguments=candidate.action_arguments or {},
                )
            self.current_node.add_child(thought)
            new_nodes.append(thought)
        return new_nodes

    def evaluate_candidates(self, evaluations: list[ThoughtEvaluation]) -> None:
        """Apply evaluations to the current node's children."""
        for eval in evaluations:
            if eval.thought_index < len(self.current_node.children):
                self.current_node.children[eval.thought_index].score = eval.score

    def select_best_child(self) -> Optional[Thought]:
        """Select and move to the best-scoring child."""
        best = self.current_node.best_child()
        if best:
            self.current_node = best
            return best
        return None

    def backtrack(self) -> bool:
        """Backtrack to find unexplored paths. Returns False if exhausted."""
        # Simple implementation: reset to root and try next best unexplored
        # In a full implementation, would maintain parent pointers
        self.explored_paths.append(self.get_current_path_contents())

        # For now, just check if we've explored too many paths
        return len(self.explored_paths) < 10  # Max paths to explore

    def get_current_path(self) -> list[Thought]:
        """Get the path from root to current node."""
        # Simple implementation - in full version would use parent pointers
        path = []
        node = self.current_node
        while node:
            path.append(node)
            # Would need parent pointer here
            break  # Simplified
        return list(reversed(path))

    def get_current_path_contents(self) -> list[str]:
        """Get the content of thoughts in current path."""
        return [t.content for t in self.get_current_path()]

    def get_best_terminal_action(self) -> Optional[AssistantFunctionCall]:
        """Get the action from the best terminal node found."""
        best_terminal: Optional[Thought] = None
        best_score = -1.0

        def find_best_terminal(node: Thought) -> None:
            nonlocal best_terminal, best_score
            if node.is_terminal and node.score > best_score:
                best_terminal = node
                best_score = node.score
            for child in node.children:
                find_best_terminal(child)

        find_best_terminal(self.root)

        if best_terminal and best_terminal.action:
            return best_terminal.action
        return None

    def summary(self) -> str:
        """Get a summary of the tree exploration."""
        total_nodes = self._count_nodes(self.root)
        terminal_nodes = self._count_terminal_nodes(self.root)
        return (
            f"Tree: {total_nodes} nodes explored, "
            f"{terminal_nodes} terminal, "
            f"{len(self.explored_paths)} paths tried"
        )

    def _count_nodes(self, node: Thought) -> int:
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _count_terminal_nodes(self, node: Thought) -> int:
        count = 1 if node.is_terminal else 0
        for child in node.children:
            count += self._count_terminal_nodes(child)
        return count


class ToTActionProposal(ActionProposal):
    """Action proposal for Tree of Thoughts strategy.

    Note: thought_path, alternatives_explored, and phase are stored in strategy state.
    """

    thoughts: ToTThoughts


class ToTPromptConfiguration(BasePromptStrategyConfiguration):
    """Configuration for Tree of Thoughts strategy."""

    DEFAULT_GENERATE_INSTRUCTION: str = (
        "Generate {branching_factor} different approaches to solve the current "
        "sub-problem. Each approach should be distinct and explore a different "
        "angle.\n\n"
        "Current context:\n{context}\n\n"
        "For each approach, specify:\n"
        "1. The reasoning/thought\n"
        "2. Whether it leads directly to an action (or needs more reasoning)\n"
        "3. If it leads to an action, specify the command and arguments\n\n"
        "Format as a JSON array of candidates."
    )

    DEFAULT_EVALUATE_INSTRUCTION: str = (
        "Evaluate each of the following candidate thoughts for solving the problem.\n\n"
        "Task: {task}\n"
        "Current path: {current_path}\n"
        "Candidates:\n{candidates}\n\n"
        "For each candidate, provide:\n"
        "1. A score from 0-10 (10 being most promising)\n"
        "2. Brief reasoning for the score\n"
        "3. Whether it's worth exploring further\n\n"
        "Format as a JSON array of evaluations."
    )

    DEFAULT_SELECT_INSTRUCTION: str = (
        "Based on the evaluations, select the best path forward.\n\n"
        "Provide your thoughts in the required JSON format, then invoke "
        "the command if the selected path leads to an action."
    )

    generate_instruction: str = UserConfigurable(default=DEFAULT_GENERATE_INSTRUCTION)
    evaluate_instruction: str = UserConfigurable(default=DEFAULT_EVALUATE_INSTRUCTION)
    select_instruction: str = UserConfigurable(default=DEFAULT_SELECT_INSTRUCTION)
    search_algorithm: Literal["bfs", "dfs"] = UserConfigurable(default="bfs")
    branching_factor: int = UserConfigurable(default=3)
    max_depth: int = UserConfigurable(default=5)
    min_score_threshold: float = UserConfigurable(default=5.0)


class TreeOfThoughtsPromptStrategy(BaseMultiStepPromptStrategy):
    """Tree of Thoughts prompt strategy implementation.

    Implements deliberate problem solving through tree search.
    """

    default_configuration: ToTPromptConfiguration = ToTPromptConfiguration()

    def __init__(
        self,
        configuration: ToTPromptConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: ToTPromptConfiguration = configuration
        self.tree: Optional[ThoughtTree] = None
        self.current_phase: ToTPhase = ToTPhase.GENERATING
        self.pending_candidates: list[ThoughtCandidate] = []
        self.iteration_count: int = 0
        self._response_schema = JSONSchema.from_dict(
            ToTActionProposal.model_json_schema()
        )

    @property
    def llm_classification(self) -> LanguageModelClassification:
        # ToT requires the smarter model for reasoning quality
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
        """Build prompt based on current phase."""
        if self.current_phase == ToTPhase.GENERATING:
            return self._build_generate_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )
        elif self.current_phase == ToTPhase.EVALUATING:
            return self._build_evaluate_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )
        else:  # SELECTING
            return self._build_select_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )

    def _build_generate_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the generation phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        # Build context from current tree path
        context = ""
        if self.tree:
            path = self.tree.get_current_path_contents()
            if path:
                context = "Reasoning path so far:\n" + "\n→ ".join(path)
        if not context:
            context = f"Starting fresh. Task: {task}"

        generate_instruction = self.config.generate_instruction.format(
            branching_factor=self.config.branching_factor,
            context=context,
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(generate_instruction),
            ],
            prefill_response=response_prefill if self.config.use_prefill else "",
            functions=commands,
        )

    def _build_evaluate_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the evaluation phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        # Format candidates for evaluation
        candidates_text = "\n".join(
            f"{i + 1}. {c.thought}"
            + (f" → Action: {c.action_name}" if c.leads_to_action else "")
            for i, c in enumerate(self.pending_candidates)
        )

        current_path = ""
        if self.tree:
            path = self.tree.get_current_path_contents()
            if path:
                current_path = " → ".join(path)

        evaluate_instruction = self.config.evaluate_instruction.format(
            task=task,
            current_path=current_path or "Starting point",
            candidates=candidates_text,
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(evaluate_instruction),
            ],
            prefill_response='[{"thought_index":',
            functions=commands,
        )

    def _build_select_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the selection phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        tree_summary = self.tree.summary() if self.tree else "No tree yet"

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(
                    f"{self.config.select_instruction}\n\n"
                    f"Exploration summary: {tree_summary}"
                ),
            ],
            prefill_response=response_prefill if self.config.use_prefill else "",
            functions=commands,
        )

    def _build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> tuple[str, str]:
        """Build the system prompt."""
        response_fmt_instruction, response_prefill = self._response_format_instruction()

        phase_desc = {
            ToTPhase.GENERATING: (
                f"GENERATING: Create {self.config.branching_factor} different "
                "approaches to explore."
            ),
            ToTPhase.EVALUATING: "EVALUATING: Score each candidate thought.",
            ToTPhase.SELECTING: "SELECTING: Choose the best path and take action.",
        }.get(self.current_phase, "")

        system_prompt_parts = (
            self.generate_intro_prompt(ai_profile)
            + (self.generate_os_info() if include_os_info else [])
            + [self.build_body(ai_directives, commands)]
            + [
                "## Tree of Thoughts Mode\n"
                "You solve problems by exploring multiple reasoning paths.\n"
                "Generate diverse thoughts, evaluate them, and select the best path.\n"
                f"Current phase: {phase_desc}"
            ]
            + ["## RESPONSE FORMAT\n" + response_fmt_instruction]
        )

        return (
            "\n\n".join(filter(None, system_prompt_parts)).strip("\n"),
            response_prefill,
        )

    def _response_format_instruction(self) -> tuple[str, str]:
        """Generate response format instruction."""
        schema = self._response_schema.model_copy(deep=True)

        assert schema.properties
        if "use_tool" in schema.properties:
            del schema.properties["use_tool"]

        response_format = re.sub(
            r"\n\s+",
            "\n",
            schema.to_typescript_object_interface("AssistantResponse"),
        )
        response_prefill = f'{{\n    "{list(schema.properties.keys())[0]}":'

        return (
            (
                f"YOU MUST ALWAYS RESPOND WITH A JSON OBJECT OF THE FOLLOWING TYPE:\n"
                f"{response_format}"
                "\n\nYOU MUST ALSO INVOKE A TOOL!"
            ),
            response_prefill,
        )

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> ToTActionProposal:
        """Parse the LLM response."""
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        self.logger.debug(
            "LLM response content:"
            + (
                f"\n{response.content}"
                if "\n" in response.content
                else f" '{response.content}'"
            )
        )

        # Handle different phases
        if self.current_phase == ToTPhase.GENERATING:
            self._process_generation(response.content)
            self.current_phase = ToTPhase.EVALUATING
        elif self.current_phase == ToTPhase.EVALUATING:
            self._process_evaluation(response.content)
            self.current_phase = ToTPhase.SELECTING

        # Parse the final response
        assistant_reply_dict = extract_dict_from_json(response.content)
        self.logger.debug(
            "Parsing object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )

        if not response.tool_calls:
            # If no tool call but we found a good path, use that action
            if self.tree:
                best_action = self.tree.get_best_terminal_action()
                if best_action:
                    assistant_reply_dict["use_tool"] = best_action.model_dump()
                else:
                    raise InvalidAgentResponseError("Assistant did not use a tool")
            else:
                raise InvalidAgentResponseError("Assistant did not use a tool")
        else:
            assistant_reply_dict["use_tool"] = response.tool_calls[0].function

        # Note: thought_path, alternatives_explored, and phase are stored in
        # strategy state, not in the proposal

        parsed_response = ToTActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        # Move to next iteration
        self.iteration_count += 1
        self.current_phase = ToTPhase.GENERATING

        return parsed_response

    def _process_generation(self, content: str) -> None:
        """Process the generation phase response."""
        # Initialize tree if needed
        if not self.tree:
            self.tree = ThoughtTree(root_content="Initial problem analysis")

        # Try to extract candidates from response
        # Look for JSON array of candidates
        try:
            # Try to find array in response
            array_match = re.search(r"\[.*\]", content, re.DOTALL)
            if array_match:
                candidates_data = json.loads(array_match.group())
                self.pending_candidates = [
                    ThoughtCandidate(
                        thought=c.get("thought", c.get("reasoning", "")),
                        leads_to_action=c.get("leads_to_action", False),
                        action_name=c.get("action_name"),
                        action_arguments=c.get("action_arguments"),
                    )
                    for c in candidates_data
                    if isinstance(c, dict)
                ]
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract thoughts from numbered list
            pattern = re.compile(r"(\d+)\.\s+(.+?)(?=\n\d+\.|\n*$)", re.DOTALL)
            matches = pattern.findall(content)
            self.pending_candidates = [
                ThoughtCandidate(thought=match[1].strip())
                for match in matches[: self.config.branching_factor]
            ]

        if self.pending_candidates:
            self.tree.add_candidates(self.pending_candidates)

    def _process_evaluation(self, content: str) -> None:
        """Process the evaluation phase response."""
        if not self.tree:
            return

        evaluations: list[ThoughtEvaluation] = []

        try:
            # Try to find array in response
            array_match = re.search(r"\[.*\]", content, re.DOTALL)
            if array_match:
                eval_data = json.loads(array_match.group())
                evaluations = [
                    ThoughtEvaluation(
                        thought_index=e.get("thought_index", i),
                        score=float(e.get("score", 0)),
                        reasoning=e.get("reasoning", ""),
                        is_promising=e.get("is_promising", True),
                    )
                    for i, e in enumerate(eval_data)
                    if isinstance(e, dict)
                ]
        except (json.JSONDecodeError, ValueError):
            # Fallback: assign equal scores
            evaluations = [
                ThoughtEvaluation(
                    thought_index=i,
                    score=5.0,
                    reasoning="Default score",
                    is_promising=True,
                )
                for i in range(len(self.pending_candidates))
            ]

        self.tree.evaluate_candidates(evaluations)

        # Select best child based on search algorithm
        if self.config.search_algorithm == "dfs":
            # DFS: always go to highest scoring child
            self.tree.select_best_child()
        else:
            # BFS: would maintain a queue of nodes to explore
            # Simplified: just select best for now
            self.tree.select_best_child()

    def should_continue_search(self) -> bool:
        """Check if we should continue searching the tree."""
        if not self.tree:
            return True

        # Stop if we've hit max depth
        if self.tree.current_node.depth >= self.config.max_depth:
            return False

        # Stop if we've found a terminal node with good score
        if self.tree.current_node.is_terminal:
            if self.tree.current_node.score >= self.config.min_score_threshold:
                return False

        # Stop if we've explored too many iterations
        if self.iteration_count >= self.config.max_depth * self.config.branching_factor:
            return False

        return True

    def reset(self) -> None:
        """Reset the strategy for a new task."""
        self.tree = None
        self.current_phase = ToTPhase.GENERATING
        self.pending_candidates = []
        self.iteration_count = 0
