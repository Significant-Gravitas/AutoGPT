"""ReWOO (Reasoning Without Observation) Prompt Strategy.

This strategy implements the ReWOO pattern from the paper:
"ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models"
(https://arxiv.org/abs/2305.18323)

Key benefits:
- 5x token efficiency compared to ReAct
- Generates complete plan upfront, then executes all tools
- Robust under tool-failure scenarios
- Enables parallel tool execution

Pattern:
1. PLAN: Generate full reasoning plan with placeholder variables (#E1, #E2, etc.)
2. EXECUTE: Run all tools (potentially in parallel)
3. SYNTHESIZE: Combine tool results with plan to generate final response
"""

from __future__ import annotations

import json
import re
from enum import Enum
from logging import Logger
from typing import Any, Optional

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

from .base import (
    BaseMultiStepPromptStrategy,
    BasePromptStrategyConfiguration,
    PlannedStep,
)


class ReWOOPhase(str, Enum):
    """Current phase of the ReWOO execution."""

    PLANNING = "planning"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"


class ReWOOThoughts(ModelWithSummary):
    """Unified thoughts model for ReWOO strategy.

    Works for both planning and synthesis phases with optional fields.
    """

    observations: str = Field(
        description="Relevant observations from context or results"
    )
    reasoning: str = Field(
        default="", description="Reasoning about the task or analysis of results"
    )
    plan: list[str] = Field(
        default_factory=list, description="Planned steps or conclusions"
    )
    speak: str = Field(description="Summary to say to user")

    def summary(self) -> str:
        return self.reasoning if self.reasoning else self.speak


class ReWOOPlan(ModelWithSummary):
    """A complete ReWOO plan with multiple steps."""

    steps: list[PlannedStep] = Field(default_factory=list)
    current_step_index: int = Field(default=0)
    execution_results: dict[str, str] = Field(default_factory=dict)

    def summary(self) -> str:
        return f"Plan with {len(self.steps)} steps, {self.current_step_index} completed"

    def get_next_step(self) -> Optional[PlannedStep]:
        """Get the next step to execute."""
        if self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]

    def get_executable_steps(self) -> list[PlannedStep]:
        """Get all steps that can be executed (dependencies satisfied)."""
        executable = []
        for step in self.steps:
            if step.status != "pending":
                continue
            # Check if all dependencies are satisfied
            deps_satisfied = all(
                dep in self.execution_results for dep in step.depends_on
            )
            if deps_satisfied:
                executable.append(step)
        return executable

    def mark_step_complete(self, variable_name: str, result: str) -> None:
        """Mark a step as complete with its result."""
        self.execution_results[variable_name] = result
        for step in self.steps:
            if step.variable_name == variable_name:
                step.status = "completed"
                step.result = result
                break
        self.current_step_index += 1

    def substitute_variables(self, text: str) -> str:
        """Substitute variable placeholders with actual results."""
        for var_name, result in self.execution_results.items():
            text = text.replace(var_name, str(result))
        return text

    def all_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(step.status == "completed" for step in self.steps)


class ReWOOActionProposal(ActionProposal):
    """Action proposal for ReWOO strategy.

    Can represent either a single action from the plan or the full plan.
    Note: plan, phase, is_synthesis are stored in strategy state, not in the proposal.
    """

    thoughts: ReWOOThoughts  # type: ignore[assignment]


class ReWOOPromptConfiguration(BasePromptStrategyConfiguration):
    """Configuration for ReWOO prompt strategy."""

    DEFAULT_PLANNER_INSTRUCTION: str = (
        "Create a complete plan to accomplish the task. For each step:\n"
        "1. Write your reasoning (Plan:)\n"
        "2. Specify the tool to use and its arguments\n"
        "3. Assign a variable name (#E1, #E2, etc.) to store the result\n"
        "4. Later steps can reference earlier results using variable names\n\n"
        "Format each step as:\n"
        "Plan: [Your reasoning for this step]\n"
        '#E[n] = tool_name(arg1="value1", arg2=#E[m])\n\n'
        "After all steps, provide the response in the required JSON format."
    )

    DEFAULT_SYNTHESIZER_INSTRUCTION: str = (
        "You have executed the following plan and received these results.\n"
        "Analyze the results and provide a final response to the original task.\n\n"
        "Plan and Results:\n{plan_with_results}\n\n"
        "Original Task: {task}\n\n"
        "Provide your synthesis in the required JSON format, then use the "
        "appropriate command to complete the task or report findings."
    )

    planner_instruction: str = UserConfigurable(default=DEFAULT_PLANNER_INSTRUCTION)
    synthesizer_instruction: str = UserConfigurable(
        default=DEFAULT_SYNTHESIZER_INSTRUCTION
    )
    max_plan_steps: int = UserConfigurable(default=10)
    allow_parallel_execution: bool = UserConfigurable(default=True)


class ReWOOPromptStrategy(BaseMultiStepPromptStrategy):
    """ReWOO prompt strategy implementation.

    Implements the Reasoning Without Observation pattern for efficient
    multi-step task execution.
    """

    default_configuration: ReWOOPromptConfiguration = ReWOOPromptConfiguration()

    def __init__(
        self,
        configuration: ReWOOPromptConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: ReWOOPromptConfiguration = configuration
        self.current_plan: Optional[ReWOOPlan] = None
        self.current_phase: ReWOOPhase = ReWOOPhase.PLANNING
        self._response_schema = JSONSchema.from_dict(
            ReWOOActionProposal.model_json_schema()
        )

    @property
    def llm_classification(self) -> LanguageModelClassification:
        # ReWOO benefits from a smarter model for planning
        if self.current_phase == ReWOOPhase.PLANNING:
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
        """Build prompt based on current phase."""
        if self.current_phase == ReWOOPhase.SYNTHESIZING:
            return self._build_synthesis_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )
        else:
            return self._build_planning_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )

    def _build_planning_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the planning phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
            is_synthesis=False,
        )

        planning_msg = ChatMessage.user(
            f"{self.config.planner_instruction}\n\n" f'Your task:\n"""{task}"""'
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                *messages,
                planning_msg,
            ],
            prefill_response=response_prefill if self.config.use_prefill else "",
            functions=commands,
        )

    def _build_synthesis_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the synthesis phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
            is_synthesis=True,
        )

        # Format plan with results
        plan_with_results = self._format_plan_with_results()

        synthesis_instruction = self.config.synthesizer_instruction.format(
            plan_with_results=plan_with_results,
            task=task,
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                *messages,
                ChatMessage.user(synthesis_instruction),
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
        is_synthesis: bool,
    ) -> tuple[str, str]:
        """Build the system prompt."""
        response_fmt_instruction, response_prefill = self._response_format_instruction()

        system_prompt_parts = (
            self.generate_intro_prompt(ai_profile)
            + (self.generate_os_info() if include_os_info else [])
            + [self.build_body(ai_directives, commands)]
            + [
                "## Your Task\n"
                "The user will specify a task for you to execute. "
                + (
                    "Create a complete multi-step plan before taking any action."
                    if not is_synthesis
                    else "Synthesize the results of your plan into a final response."
                )
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

    def _format_plan_with_results(self) -> str:
        """Format the plan with execution results for synthesis."""
        if not self.current_plan:
            return "No plan executed."

        lines = []
        for i, step in enumerate(self.current_plan.steps):
            lines.append(f"Step {i + 1}: {step.thought}")
            lines.append(f"  Tool: {step.tool_name}({json.dumps(step.tool_arguments)})")
            lines.append(f"  Variable: {step.variable_name}")
            if step.result:
                result_preview = (
                    step.result[:500] + "..." if len(step.result) > 500 else step.result
                )
                lines.append(f"  Result: {result_preview}")
            lines.append("")

        return "\n".join(lines)

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> ReWOOActionProposal:
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

        # Parse plan from response if in planning phase
        if self.current_phase == ReWOOPhase.PLANNING:
            plan = self._extract_plan_from_response(response.content)
            if plan and plan.steps:
                self.current_plan = plan

        # Ensure thoughts dict has required fields
        thoughts_dict = assistant_reply_dict.get("thoughts", {})
        if not isinstance(thoughts_dict, dict):
            thoughts_dict = {"observations": "", "reasoning": "", "speak": ""}
        if "observations" not in thoughts_dict:
            thoughts_dict["observations"] = ""
        if "reasoning" not in thoughts_dict:
            thoughts_dict["reasoning"] = ""
        if "speak" not in thoughts_dict:
            thoughts_dict["speak"] = ""
        assistant_reply_dict["thoughts"] = thoughts_dict

        parsed_response = ReWOOActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        return parsed_response

    def _extract_plan_from_response(self, content: str) -> Optional[ReWOOPlan]:
        """Extract a structured plan from the LLM response.

        Looks for patterns like:
        Plan: [reasoning]
        #E1 = tool_name(arg1="value1", arg2=#E0)
        """
        plan = ReWOOPlan()

        # Pattern to match plan steps
        step_pattern = re.compile(
            r"Plan:\s*(.+?)\n\s*#(E\d+)\s*=\s*(\w+)\s*\(([^)]*)\)",
            re.MULTILINE | re.DOTALL,
        )

        matches = step_pattern.findall(content)

        for match in matches[: self.config.max_plan_steps]:
            thought, var_num, tool_name, args_str = match
            variable_name = f"#{var_num}"

            # Parse arguments
            tool_arguments = self._parse_tool_arguments(args_str)

            # Find dependencies (references to other variables)
            depends_on = re.findall(r"#E\d+", args_str)

            step = PlannedStep(
                thought=thought.strip(),
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                variable_name=variable_name,
                depends_on=depends_on,
            )
            plan.steps.append(step)

        return plan if plan.steps else None

    def _parse_tool_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse tool arguments from a string like 'arg1="value1", arg2=123'."""
        arguments: dict[str, Any] = {}

        # Simple pattern matching for key=value pairs
        # This handles strings, numbers, and variable references
        arg_pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(#E\d+)|(\d+\.?\d*))')

        for match in arg_pattern.finditer(args_str):
            key = match.group(1)
            if match.group(2) is not None:  # String value
                arguments[key] = match.group(2)
            elif match.group(3) is not None:  # Variable reference
                arguments[key] = match.group(3)
            elif match.group(4) is not None:  # Number
                num_str = match.group(4)
                arguments[key] = float(num_str) if "." in num_str else int(num_str)

        return arguments

    def get_next_action(self) -> Optional[AssistantFunctionCall]:
        """Get the next action to execute from the current plan."""
        if not self.current_plan:
            return None

        next_step = self.current_plan.get_next_step()
        if not next_step:
            return None

        # Substitute any variable references in arguments
        substituted_args = {}
        for key, value in next_step.tool_arguments.items():
            if isinstance(value, str) and value.startswith("#E"):
                substituted_args[key] = self.current_plan.execution_results.get(
                    value, value
                )
            else:
                substituted_args[key] = value

        return AssistantFunctionCall(
            name=next_step.tool_name,
            arguments=substituted_args,
        )

    def record_execution_result(self, variable_name: str, result: str) -> None:
        """Record the result of a step execution."""
        if self.current_plan:
            self.current_plan.mark_step_complete(variable_name, result)

            # Check if we should move to synthesis phase
            if self.current_plan.all_complete():
                self.current_phase = ReWOOPhase.SYNTHESIZING

    def reset(self) -> None:
        """Reset the strategy state for a new task."""
        self.current_plan = None
        self.current_phase = ReWOOPhase.PLANNING

    def is_plan_complete(self) -> bool:
        """Check if the current plan is fully executed."""
        if not self.current_plan:
            return False
        return self.current_plan.all_complete()
