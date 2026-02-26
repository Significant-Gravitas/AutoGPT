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
from collections import Counter
from enum import Enum
from logging import Logger
from typing import Any, Literal, Optional

from pydantic import Field

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


class CategoricalEvaluation(ModelWithSummary):
    """Categorical evaluation of a thought (from ToT paper).

    The paper uses three categories:
    - 'sure': Definitely correct, can be solved
    - 'maybe': Might be correct, needs more exploration
    - 'impossible': Definitely wrong or a dead end
    """

    thought_index: int = Field(description="Index of the thought being evaluated")
    evaluation: Literal["sure", "maybe", "impossible"] = Field(
        description="Categorical evaluation"
    )
    reasoning: str = Field(description="Reasoning for the evaluation")

    def summary(self) -> str:
        return f"{self.evaluation}: {self.reasoning}"


class ThoughtEvaluation(ModelWithSummary):
    """Evaluation of a thought candidate."""

    thought_index: int = Field(description="Index of the thought being evaluated")
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    reasoning: str = Field(description="Reasoning for the score")
    is_promising: bool = Field(description="Whether this path is worth exploring")

    # Categorical evaluation support (from ToT paper)
    categorical: Optional[Literal["sure", "maybe", "impossible"]] = Field(
        default=None, description="Categorical evaluation if using categorical mode"
    )

    def summary(self) -> str:
        if self.categorical:
            return f"{self.categorical} (score {self.score}): {self.reasoning}"
        return f"Score {self.score}: {self.reasoning}"

    @classmethod
    def from_categorical(cls, cat: CategoricalEvaluation) -> "ThoughtEvaluation":
        """Create ThoughtEvaluation from a CategoricalEvaluation."""
        score_map = {"sure": 10.0, "maybe": 5.0, "impossible": 0.0}
        return cls(
            thought_index=cat.thought_index,
            score=score_map[cat.evaluation],
            reasoning=cat.reasoning,
            is_promising=cat.evaluation != "impossible",
            categorical=cat.evaluation,
        )


class ThoughtTree:
    """A tree of thoughts for exploration.

    Uses parent pointers in Thought nodes for proper backtracking (per ToT paper).
    """

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
            # add_child sets the parent pointer automatically
            self.current_node.add_child(thought)
            new_nodes.append(thought)
        return new_nodes

    def evaluate_candidates(self, evaluations: list[ThoughtEvaluation]) -> None:
        """Apply evaluations to the current node's children."""
        for eval_result in evaluations:
            if eval_result.thought_index < len(self.current_node.children):
                child = self.current_node.children[eval_result.thought_index]
                child.score = eval_result.score
                # Store categorical evaluation if present
                if eval_result.categorical:
                    child.categorical_evaluation = eval_result.categorical

    def select_best_child(self) -> Optional[Thought]:
        """Select and move to the best-scoring child."""
        best = self.current_node.best_child()
        if best:
            self.current_node = best
            return best
        return None

    def backtrack(self) -> bool:
        """Backtrack to find unexplored paths using parent pointers.

        Returns False if no more paths to explore (tree exhausted).
        """
        # Record current path as explored
        self.explored_paths.append(self.get_current_path_contents())

        # Walk up parent pointers looking for unexplored siblings
        node = self.current_node
        while node.parent is not None:
            parent = node.parent
            # Look for unexplored siblings
            for sibling in parent.children:
                if sibling is not node and not self._is_fully_explored(sibling):
                    self.current_node = sibling
                    return True
            # Move up to grandparent
            node = parent

        # No unexplored paths found
        return False

    def _is_fully_explored(self, node: Thought) -> bool:
        """Check if a node and all its descendants are fully explored."""
        # Terminal nodes are explored
        if node.is_terminal:
            return True

        # Nodes with categorical evaluation of 'impossible' are explored
        if node.categorical_evaluation == "impossible":
            return True

        # Nodes without children haven't been expanded yet
        if not node.children:
            return False

        # Node is explored if all children are explored
        return all(self._is_fully_explored(child) for child in node.children)

    def get_current_path(self) -> list[Thought]:
        """Get the path from root to current node using parent pointers."""
        return self.current_node.get_path_to_root()

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

    thoughts: ToTThoughts  # type: ignore[assignment]


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

    # Categorical evaluation instruction (from ToT paper)
    DEFAULT_EVALUATE_CATEGORICAL_INSTRUCTION: str = (
        "Evaluate each candidate thought's likelihood of leading to a correct "
        "solution.\n\n"
        "Task: {task}\n"
        "Current path: {current_path}\n"
        "Candidates:\n{candidates}\n\n"
        "For each candidate, classify as:\n"
        "- 'sure': Definitely correct, will solve the problem\n"
        "- 'maybe': Might be correct, worth exploring further\n"
        "- 'impossible': Definitely wrong or a dead end\n\n"
        "Format: [{{'thought_index': N, 'evaluation': 'sure|maybe|impossible', "
        "'reasoning': '...'}}]"
    )

    DEFAULT_SELECT_INSTRUCTION: str = (
        "Based on the evaluations, select the best path forward.\n\n"
        "Provide your thoughts in the required JSON format, then invoke "
        "the command if the selected path leads to an action."
    )

    generate_instruction: str = UserConfigurable(default=DEFAULT_GENERATE_INSTRUCTION)
    evaluate_instruction: str = UserConfigurable(default=DEFAULT_EVALUATE_INSTRUCTION)
    evaluate_categorical_instruction: str = UserConfigurable(
        default=DEFAULT_EVALUATE_CATEGORICAL_INSTRUCTION
    )
    select_instruction: str = UserConfigurable(default=DEFAULT_SELECT_INSTRUCTION)
    search_algorithm: Literal["bfs", "dfs"] = UserConfigurable(default="bfs")
    branching_factor: int = UserConfigurable(default=3)
    max_depth: int = UserConfigurable(default=5)
    min_score_threshold: float = UserConfigurable(default=5.0)

    # Evaluation mode: numeric (0-10 scores) or categorical (sure/maybe/impossible)
    evaluation_mode: Literal["numeric", "categorical"] = UserConfigurable(
        default="numeric"
    )

    # Number of evaluation samples for aggregation (paper uses 3)
    evaluation_samples: int = UserConfigurable(default=1)


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
        system_prompt, _ = self._build_system_prompt(
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

        # Use categorical or numeric evaluation instruction
        if self.config.evaluation_mode == "categorical":
            evaluate_instruction = self.config.evaluate_categorical_instruction.format(
                task=task,
                current_path=current_path or "Starting point",
                candidates=candidates_text,
            )
            prefill = '[{"thought_index": 0, "evaluation": "'
        else:
            evaluate_instruction = self.config.evaluate_instruction.format(
                task=task,
                current_path=current_path or "Starting point",
                candidates=candidates_text,
            )
            prefill = '[{"thought_index":'

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(evaluate_instruction),
            ],
            prefill_response=prefill,
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

                if self.config.evaluation_mode == "categorical":
                    # Parse categorical evaluations
                    for i, e in enumerate(eval_data):
                        if not isinstance(e, dict):
                            continue
                        cat_eval = CategoricalEvaluation(
                            thought_index=e.get("thought_index", i),
                            evaluation=e.get("evaluation", "maybe"),
                            reasoning=e.get("reasoning", ""),
                        )
                        evaluations.append(ThoughtEvaluation.from_categorical(cat_eval))
                else:
                    # Parse numeric evaluations
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
            # Fallback: assign default scores/categories
            if self.config.evaluation_mode == "categorical":
                evaluations = [
                    ThoughtEvaluation(
                        thought_index=i,
                        score=5.0,
                        reasoning="Default evaluation",
                        is_promising=True,
                        categorical="maybe",
                    )
                    for i in range(len(self.pending_candidates))
                ]
            else:
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

    def _aggregate_evaluations(
        self,
        all_samples: list[list[ThoughtEvaluation]],
    ) -> list[ThoughtEvaluation]:
        """Aggregate multiple evaluation samples (from ToT paper).

        For categorical mode, uses majority voting.
        For numeric mode, uses average scores.

        Args:
            all_samples: List of evaluation lists from multiple sampling runs

        Returns:
            Aggregated evaluations
        """
        if not all_samples:
            return []

        if len(all_samples) == 1:
            return all_samples[0]

        num_thoughts = len(all_samples[0])
        aggregated: list[ThoughtEvaluation] = []

        for i in range(num_thoughts):
            if self.config.evaluation_mode == "categorical":
                # Majority voting for categorical
                votes: Counter[str] = Counter()
                for sample in all_samples:
                    cat = sample[i].categorical if i < len(sample) else None
                    if cat is not None:
                        votes[cat] += 1

                if votes:
                    winner = votes.most_common(1)[0][0]
                    vote_count = votes[winner]
                else:
                    winner = "maybe"
                    vote_count = 0

                score_map = {"sure": 10.0, "maybe": 5.0, "impossible": 0.0}
                num_samples = len(all_samples)
                reasoning = f"Majority: {winner} ({vote_count}/{num_samples})"
                aggregated.append(
                    ThoughtEvaluation(
                        thought_index=i,
                        score=score_map.get(winner, 5.0),
                        reasoning=reasoning,
                        is_promising=winner != "impossible",
                        categorical=winner,  # type: ignore[arg-type]
                    )
                )
            else:
                # Average scores for numeric
                scores = [sample[i].score for sample in all_samples if i < len(sample)]
                avg_score = sum(scores) / len(scores) if scores else 5.0

                aggregated.append(
                    ThoughtEvaluation(
                        thought_index=i,
                        score=avg_score,
                        reasoning=f"Average of {len(scores)} samples",
                        is_promising=avg_score >= self.config.min_score_threshold,
                    )
                )

        return aggregated

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
