"""Reflexion Prompt Strategy.

This strategy implements the Reflexion pattern from research including:
- Reflexion: Verbal Reinforcement Learning (arxiv.org/abs/2303.11366)
- Self-Refine: Iterative Self-Feedback (arxiv.org/abs/2303.17651)
- Self-Reflection in LLM Agents (arxiv.org/abs/2405.06682)

Key benefits:
- 91% pass@1 on HumanEval (vs GPT-4's 80%)
- No training required - same LLM generates, critiques, refines
- Agents store reflections in episodic memory for better future decisions
- Supports 8 types of self-reflection that improve problem-solving

Pattern:
1. GENERATE: Propose action
2. EXECUTE: Run action
3. REFLECT: Critique result, extract lessons
4. RETRY: Use reflection to improve next attempt
"""

from __future__ import annotations

import json
import re
from datetime import datetime
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
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.action import ActionProposal
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.models.utils import ModelWithSummary
from forge.utils.exceptions import InvalidAgentResponseError

from .base import (
    BaseMultiStepPromptStrategy,
    BasePromptStrategyConfiguration,
    Reflection,
    ReflexionMemory,
)


class ReflexionPhase(str, Enum):
    """Current phase of Reflexion execution."""

    PROPOSING = "proposing"
    REFLECTING = "reflecting"


class EvaluatorType(str, Enum):
    """Type of evaluator for determining action success (from Reflexion paper)."""

    LLM = "llm"  # Use LLM to evaluate result
    HEURISTIC = "heuristic"  # Use pattern-based heuristics


class EvaluationResult(ModelWithSummary):
    """Result from the Evaluator component."""

    success: bool = Field(description="Whether the action was successful")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Score from 0-1 if available"
    )
    feedback: str = Field(default="", description="Feedback about the result")

    def summary(self) -> str:
        status = "Success" if self.success else "Failure"
        return f"{status}: {self.feedback}"


class ReflexionThoughts(ModelWithSummary):
    """Thoughts model for Reflexion strategy.

    Extends standard thoughts with explicit reflection on past attempts.
    """

    observations: str = Field(
        description="Relevant observations from context and last action"
    )
    reasoning: str = Field(description="Reasoning about the current approach")
    self_reflection: str = Field(
        description="Explicit reflection on what you've learned from past attempts"
    )
    lessons_applied: list[str] = Field(
        default_factory=list,
        description="Specific lessons from past reflections being applied now",
    )
    self_criticism: str = Field(
        description="Constructive self-criticism of current approach"
    )
    plan: list[str] = Field(description="Short list of planned steps")

    def summary(self) -> str:
        return self.reasoning


class ReflectionOutput(ModelWithSummary):
    """Output from the reflection phase."""

    action_summary: str = Field(description="Summary of what action was taken")
    result_analysis: str = Field(description="Analysis of the result")
    what_worked: str = Field(description="What aspects worked well")
    what_failed: str = Field(description="What aspects didn't work")
    root_cause: str = Field(
        default="", description="Root cause analysis if something failed"
    )
    lesson_learned: str = Field(
        description="Key lesson to remember for future attempts"
    )
    should_retry: bool = Field(
        default=False, description="Whether the action should be retried differently"
    )
    alternative_approach: str = Field(
        default="", description="Alternative approach to try if retrying"
    )

    def summary(self) -> str:
        return self.lesson_learned


class ReflexionActionProposal(ActionProposal):
    """Action proposal for Reflexion strategy.

    Note: phase and reflection_context are stored in strategy state.
    """

    thoughts: ReflexionThoughts  # type: ignore[assignment]


class ReflexionPromptConfiguration(BasePromptStrategyConfiguration):
    """Configuration for Reflexion strategy."""

    DEFAULT_PROPOSE_INSTRUCTION: str = (
        "Based on the task and your past experiences, propose the next action.\n\n"
        "Consider any relevant lessons from previous attempts shown below.\n"
        "Apply what you've learned to avoid repeating mistakes.\n\n"
        "{reflections_section}"
        "Provide your thoughts in the required JSON format, including which "
        "lessons you're applying, then invoke the appropriate command."
    )

    DEFAULT_REFLECT_INSTRUCTION: str = (
        "Reflect on the action you just took and its result.\n\n"
        "Action: {action_name}({action_args})\n"
        "Result: {result}\n\n"
        "Analyze what happened:\n"
        "1. What worked well?\n"
        "2. What didn't work?\n"
        "3. If it failed, what's the root cause?\n"
        "4. What's the key lesson for future attempts?\n"
        "5. Should you retry with a different approach?\n\n"
        "Provide your reflection in the JSON format below."
    )

    # Verbal reflection instruction (from Reflexion paper - free-form reflections)
    DEFAULT_VERBAL_REFLECT_INSTRUCTION: str = (
        "Reflect on your last action in natural language.\n\n"
        "Action: {action_name}({action_args})\n"
        "Result: {result}\n"
        "Evaluator Feedback: {evaluator_feedback}\n\n"
        "Write a brief reflection covering:\n"
        "- What happened and whether the goal was achieved\n"
        "- What went wrong (if anything)\n"
        "- What specific changes to make in the next attempt\n\n"
        "Be concise and actionable. Focus on lessons learned."
    )

    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Based on your reflection on past attempts, determine the best next action."
    )

    propose_instruction: str = UserConfigurable(default=DEFAULT_PROPOSE_INSTRUCTION)
    reflect_instruction: str = UserConfigurable(default=DEFAULT_REFLECT_INSTRUCTION)
    verbal_reflect_instruction: str = UserConfigurable(
        default=DEFAULT_VERBAL_REFLECT_INSTRUCTION
    )
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    max_reflections_in_prompt: int = UserConfigurable(default=5)
    always_reflect: bool = UserConfigurable(default=True)
    reflect_on_success: bool = UserConfigurable(default=True)
    max_retry_attempts: int = UserConfigurable(default=3)

    # Evaluator configuration (from Reflexion paper)
    evaluator_type: EvaluatorType = UserConfigurable(default=EvaluatorType.HEURISTIC)

    # Reflection format: structured (JSON fields), verbal (free-form), or auto
    reflection_format: Literal["structured", "verbal", "auto"] = UserConfigurable(
        default="structured"
    )


class ReflexionPromptStrategy(BaseMultiStepPromptStrategy):
    """Reflexion prompt strategy implementation.

    Implements verbal reinforcement learning through self-reflection.
    """

    default_configuration: ReflexionPromptConfiguration = ReflexionPromptConfiguration()

    def __init__(
        self,
        configuration: ReflexionPromptConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: ReflexionPromptConfiguration = configuration
        self.memory = ReflexionMemory(max_reflections=20)
        self.current_phase: ReflexionPhase = ReflexionPhase.PROPOSING
        self.last_action: Optional[dict[str, Any]] = None
        self.last_result: Optional[str] = None
        self.last_evaluation: Optional[EvaluationResult] = None
        self.retry_count: int = 0
        self._response_schema = JSONSchema.from_dict(
            ReflexionActionProposal.model_json_schema()
        )
        self._reflection_schema = JSONSchema.from_dict(
            ReflectionOutput.model_json_schema()
        )

    @property
    def llm_classification(self) -> LanguageModelClassification:
        # Reflection benefits from the smarter model
        if self.current_phase == ReflexionPhase.REFLECTING:
            return LanguageModelClassification.SMART_MODEL
        return LanguageModelClassification.FAST_MODEL

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
        """Build prompt based on current phase.

        Also processes any execution results from the message history to
        trigger reflection when appropriate. This allows Reflexion to work
        without requiring changes to agent.py.
        """
        # Extract and process the latest execution result from messages
        # This enables Reflexion to learn from past actions without
        # requiring the agent to explicitly call record_result()
        self._process_latest_result_from_messages(messages)

        if self.current_phase == ReflexionPhase.REFLECTING:
            return self._build_reflection_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )
        else:
            return self._build_propose_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )

    def _build_propose_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the proposal phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        # Get relevant reflections from memory
        reflections = self.memory.get_relevant_reflections(
            limit=self.config.max_reflections_in_prompt
        )

        reflections_section = ""
        if reflections:
            reflections_text = "\n".join(f"- {r.to_prompt_text()}" for r in reflections)
            reflections_section = (
                f"## Lessons from Past Attempts\n{reflections_text}\n\n"
            )

        propose_instruction = self.config.propose_instruction.format(
            reflections_section=reflections_section
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(propose_instruction),
            ],
            prefill_response=response_prefill if self.config.use_prefill else "",
            functions=commands,
        )

    def _build_reflection_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the reflection phase prompt."""
        # Check if we should use verbal reflection format
        reflection_format = self._get_reflection_format()
        if reflection_format == "verbal":
            return self._build_verbal_reflection_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )

        # Structured reflection (original behavior)
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
            for_reflection=True,
        )

        action_name = (
            self.last_action.get("name", "unknown") if self.last_action else "unknown"
        )
        action_args = (
            json.dumps(self.last_action.get("arguments", {}))
            if self.last_action
            else "{}"
        )
        result = self.last_result or "No result"

        reflect_instruction = self.config.reflect_instruction.format(
            action_name=action_name,
            action_args=action_args,
            result=result,
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Original Task: """{task}"""'),
                *messages,
                ChatMessage.user(reflect_instruction),
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
        for_reflection: bool = False,
    ) -> tuple[str, str]:
        """Build the system prompt."""
        if for_reflection:
            response_fmt_instruction = self._reflection_format_instruction()
            response_prefill = '{\n    "action_summary":'
        else:
            (
                response_fmt_instruction,
                response_prefill,
            ) = self._response_format_instruction()

        phase_desc = (
            "You are reflecting on your last action to extract lessons for improvement."
            if for_reflection
            else "You are proposing an action, informed by lessons from past attempts."
        )

        system_prompt_parts = (
            self.generate_intro_prompt(ai_profile)
            + (self.generate_os_info() if include_os_info else [])
            + [self.build_body(ai_directives, commands)]
            + [
                "## Reflexion Mode\n"
                "You learn from your mistakes through explicit self-reflection.\n"
                f"{phase_desc}"
            ]
            + ["## RESPONSE FORMAT\n" + response_fmt_instruction]
        )

        return (
            "\n\n".join(filter(None, system_prompt_parts)).strip("\n"),
            response_prefill,
        )

    def _response_format_instruction(self) -> tuple[str, str]:
        """Generate response format instruction for proposal phase."""
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

    def _reflection_format_instruction(self) -> str:
        """Generate response format instruction for reflection phase."""
        schema = self._reflection_schema.model_copy(deep=True)

        response_format = re.sub(
            r"\n\s+",
            "\n",
            schema.to_typescript_object_interface("ReflectionOutput"),
        )

        return (
            f"Provide your reflection as a JSON object of this type:\n"
            f"{response_format}"
            "\n\nAfter reflecting, invoke a command to continue with the task."
        )

    def _evaluate_heuristic(self, result: str) -> EvaluationResult:
        """Simple heuristic evaluation based on error patterns.

        This is the default evaluator that looks for common error indicators
        in the result string. For more sophisticated evaluation, use
        evaluator_type=EvaluatorType.LLM.
        """
        error_patterns = [
            "error",
            "failed",
            "exception",
            "traceback",
            "invalid",
            "not found",
            "permission denied",
            "timeout",
            "refused",
            "cannot",
            "unable to",
        ]

        result_lower = result.lower()
        has_error = any(pattern in result_lower for pattern in error_patterns)

        # Check for success patterns that might override error detection
        success_patterns = [
            "success",
            "completed",
            "done",
            "finished",
            "created",
            "saved",
        ]
        has_success = any(pattern in result_lower for pattern in success_patterns)

        # If both error and success patterns, look at which appears first
        if has_error and has_success:
            # Find first occurrence of each
            first_error_idx = min(
                (result_lower.find(p) for p in error_patterns if p in result_lower),
                default=len(result_lower),
            )
            first_success_idx = min(
                (result_lower.find(p) for p in success_patterns if p in result_lower),
                default=len(result_lower),
            )
            has_error = first_error_idx < first_success_idx

        if has_error:
            return EvaluationResult(
                success=False,
                score=0.2,
                feedback="Detected error patterns in output",
            )
        else:
            return EvaluationResult(
                success=True,
                score=0.8,
                feedback="Execution completed without detected errors",
            )

    def _get_reflection_format(self) -> Literal["structured", "verbal"]:
        """Determine the reflection format to use."""
        if self.config.reflection_format == "auto":
            # Auto mode: use verbal if there was an evaluation, structured otherwise
            return "verbal" if self.last_evaluation is not None else "structured"
        elif self.config.reflection_format == "verbal":
            return "verbal"
        else:
            return "structured"

    def _build_verbal_reflection_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build prompt for free-form verbal reflection (from Reflexion paper)."""
        system_prompt, _ = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
            for_reflection=True,
        )

        action_name = (
            self.last_action.get("name", "unknown") if self.last_action else "unknown"
        )
        action_args = (
            json.dumps(self.last_action.get("arguments", {}))
            if self.last_action
            else "{}"
        )
        result = self.last_result or "No result"
        evaluator_feedback = (
            self.last_evaluation.feedback
            if self.last_evaluation
            else "No evaluation available"
        )

        verbal_instruction = self.config.verbal_reflect_instruction.format(
            action_name=action_name,
            action_args=action_args,
            result=result,
            evaluator_feedback=evaluator_feedback,
        )

        # For verbal reflection, we want free-form text, not JSON
        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Original Task: """{task}"""'),
                *messages,
                ChatMessage.user(verbal_instruction),
            ],
            prefill_response="Reflection: ",
            functions=commands,
        )

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> ReflexionActionProposal:
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

        assistant_reply_dict = extract_dict_from_json(response.content)
        self.logger.debug(
            "Parsing object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )

        if not response.tool_calls:
            raise InvalidAgentResponseError("Assistant did not use a tool")

        assistant_reply_dict["use_tool"] = response.tool_calls[0].function

        # If we're in reflection phase, process the reflection
        if self.current_phase == ReflexionPhase.REFLECTING:
            # Check if this is a verbal reflection
            reflection_format = self._get_reflection_format()
            if reflection_format == "verbal":
                # Extract verbal reflection text from response
                # It starts after "Reflection: " prefix
                verbal_text = response.content
                if verbal_text.startswith("Reflection:"):
                    verbal_text = verbal_text[len("Reflection:") :].strip()
                self._process_reflection(assistant_reply_dict, verbal_text=verbal_text)
            else:
                self._process_reflection(assistant_reply_dict)
            # After reflection, move back to proposing
            self.current_phase = ReflexionPhase.PROPOSING

        # Phase and reflection_context are stored in strategy state, not in the proposal

        # Ensure thoughts has all required fields for ReflexionThoughts model
        thoughts = assistant_reply_dict.get("thoughts", {})
        if not isinstance(thoughts, dict):
            thoughts = {}
        # Set defaults for all required fields
        if "observations" not in thoughts:
            thoughts["observations"] = thoughts.get("text", "")
        if "reasoning" not in thoughts:
            thoughts["reasoning"] = ""
        if "self_reflection" not in thoughts:
            thoughts["self_reflection"] = thoughts.get("reasoning", "")
        if "self_criticism" not in thoughts:
            thoughts["self_criticism"] = thoughts.get("criticism", "")
        if "plan" not in thoughts:
            thoughts["plan"] = thoughts.get("plan", [])
            if isinstance(thoughts["plan"], str):
                thoughts["plan"] = [thoughts["plan"]] if thoughts["plan"] else []
        if "lessons_applied" not in thoughts:
            thoughts["lessons_applied"] = []
        assistant_reply_dict["thoughts"] = thoughts

        parsed_response = ReflexionActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        # Record the action for later reflection (when in proposing phase)
        # This ensures we track what action was taken so we can reflect on it
        # after seeing the result in the next build_prompt() call
        if self.current_phase == ReflexionPhase.PROPOSING and parsed_response.use_tool:
            self.record_action(
                action_name=parsed_response.use_tool.name,
                action_arguments=parsed_response.use_tool.arguments,
            )
            self.logger.debug(
                f"Reflexion: Recorded action {parsed_response.use_tool.name} "
                "for reflection"
            )

        return parsed_response

    def _process_reflection(
        self, response_dict: dict[str, Any], verbal_text: Optional[str] = None
    ) -> None:
        """Process a reflection response and store in memory.

        Args:
            response_dict: Parsed JSON response (for structured reflections)
            verbal_text: Raw verbal reflection text (for verbal format)
        """
        reflection_format = self._get_reflection_format()

        if reflection_format == "verbal" and verbal_text:
            # Verbal reflection format (from Reflexion paper)
            # Extract evaluation score if available
            evaluation_score = (
                self.last_evaluation.score if self.last_evaluation else None
            )
            success = self.last_evaluation.success if self.last_evaluation else True

            reflection = Reflection(
                action_name=(
                    self.last_action.get("name", "unknown")
                    if self.last_action
                    else "unknown"
                ),
                action_arguments=(
                    self.last_action.get("arguments", {}) if self.last_action else {}
                ),
                result_summary=self.last_result or "",
                verbal_reflection=verbal_text,
                reflection_format="verbal",
                evaluation_score=evaluation_score,
                success=success,
                timestamp=datetime.now(),
            )

            self.memory.add_reflection(reflection)
            self.logger.debug(
                f"Stored verbal reflection: {reflection.to_prompt_text()}"
            )

            # Handle retry logic based on evaluation
            if not success and self.retry_count < self.config.max_retry_attempts:
                self.retry_count += 1
                self.logger.info(
                    f"Evaluation suggests retry (attempt {self.retry_count})"
                )
            else:
                self.retry_count = 0
            return

        # Structured reflection format (original behavior)
        action_summary = response_dict.get("action_summary", "")
        result_analysis = response_dict.get("result_analysis", "")
        what_failed = response_dict.get("what_failed", "")
        lesson_learned = response_dict.get("lesson_learned", "")
        # root_cause is extracted but not used in the current implementation
        # It could be added to the Reflection model in the future
        _ = response_dict.get("root_cause", "")
        should_retry = response_dict.get("should_retry", False)

        # Determine success
        success = not what_failed or what_failed.lower() in (
            "",
            "nothing",
            "none",
            "n/a",
        )

        # Include evaluation score if available
        evaluation_score = self.last_evaluation.score if self.last_evaluation else None

        # Create and store reflection
        reflection = Reflection(
            action_name=(
                self.last_action.get("name", "unknown")
                if self.last_action
                else "unknown"
            ),
            action_arguments=(
                self.last_action.get("arguments", {}) if self.last_action else {}
            ),
            result_summary=result_analysis or action_summary,
            what_went_wrong=what_failed if not success else "",
            what_to_do_differently=lesson_learned,
            success=success,
            evaluation_score=evaluation_score,
            timestamp=datetime.now(),
        )

        self.memory.add_reflection(reflection)
        self.logger.debug(f"Stored reflection: {reflection.to_prompt_text()}")

        # Handle retry logic
        if should_retry and self.retry_count < self.config.max_retry_attempts:
            self.retry_count += 1
            self.logger.info(f"Reflection suggests retry (attempt {self.retry_count})")
        else:
            self.retry_count = 0

    def record_action(self, action_name: str, action_arguments: dict[str, Any]) -> None:
        """Record the action being taken for later reflection."""
        self.last_action = {
            "name": action_name,
            "arguments": action_arguments,
        }

    def record_result(self, result: str, success: Optional[bool] = None) -> None:
        """Record the result of an action and trigger reflection if needed.

        Args:
            result: The result string from executing the action
            success: Override for success determination. If None, uses evaluator.
        """
        self.last_result = result

        # Run evaluator if success is not explicitly provided
        if success is None:
            if self.config.evaluator_type == EvaluatorType.HEURISTIC:
                self.last_evaluation = self._evaluate_heuristic(result)
                success = self.last_evaluation.success
            else:
                # For LLM evaluator, would need to make an LLM call
                # For now, fall back to heuristic
                self.last_evaluation = self._evaluate_heuristic(result)
                success = self.last_evaluation.success
        else:
            # Create evaluation result from explicit success
            self.last_evaluation = EvaluationResult(
                success=success,
                score=0.9 if success else 0.1,
                feedback="Explicit success/failure provided",
            )

        if self.config.always_reflect or not success:
            if success and not self.config.reflect_on_success:
                # Skip reflection for successful actions if configured
                return
            self.current_phase = ReflexionPhase.REFLECTING

    def _process_latest_result_from_messages(self, messages: list[ChatMessage]) -> None:
        """Extract and process the latest execution result from message history.

        The ActionHistoryComponent includes tool results in the messages.
        We look for the most recent tool result to process for reflection.
        This is called at the start of build_prompt() to ensure results are
        processed before building the next prompt.

        This enables Reflexion to work without requiring changes to agent.py -
        the strategy self-extracts results from the standard message flow.
        """
        if not self.last_action:
            # No action recorded yet, nothing to process
            return

        # Look for tool result messages (from ActionHistoryComponent)
        for msg in reversed(messages):
            content = None
            is_error = False

            # Check for ToolResultMessage (has tool_call_id attribute)
            if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
                content = msg.content
                is_error = getattr(msg, "is_error", False)
            # Also check for user messages that contain result format
            elif hasattr(msg, "role") and getattr(msg, "role", None) == "user":
                msg_content = getattr(msg, "content", str(msg))
                if isinstance(msg_content, str):
                    if (
                        " returned:" in msg_content
                        or " raised an error:" in msg_content
                    ):
                        content = msg_content
                        is_error = " raised an error:" in msg_content

            if content is not None:
                # Only process if we haven't already processed this result
                if self.last_result is None or self.last_result != content:
                    self.logger.debug(
                        f"Reflexion: Extracted result from messages (error={is_error})"
                    )
                    self.record_result(content, success=not is_error)
                return

    def get_reflections_summary(self) -> str:
        """Get a summary of all stored reflections."""
        if not self.memory.reflections:
            return "No reflections stored yet."

        lines = ["Stored Reflections:"]
        for i, r in enumerate(self.memory.reflections, 1):
            lines.append(f"{i}. {r.to_prompt_text()}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the strategy for a new task."""
        # Keep memory across tasks - that's the point of Reflexion!
        self.current_phase = ReflexionPhase.PROPOSING
        self.last_action = None
        self.last_result = None
        self.last_evaluation = None
        self.retry_count = 0

    def clear_memory(self) -> None:
        """Clear the reflection memory (use sparingly)."""
        self.memory = ReflexionMemory(max_reflections=20)
