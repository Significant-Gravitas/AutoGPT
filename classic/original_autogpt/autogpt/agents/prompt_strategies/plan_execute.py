"""Plan-and-Execute Prompt Strategy.

This strategy implements the Plan-and-Execute pattern from research including:
- Plan-and-Act: Long-horizon Task Planning (arxiv.org/html/2503.09572v3)
- Plan-and-Solve Prompting (arxiv.org/abs/2305.04091)
- Routine: Enterprise-Grade Planning (arxiv.org/html/2507.14447)

Key benefits:
- Separates planning from execution for better predictability
- Supports replanning on failure
- Better for long-horizon tasks
- 96.3% accuracy reported in Routine paper

Pattern:
1. PLAN: Generate high-level plan with steps
2. EXECUTE: For each step, generate specific action and execute
3. REPLAN: If step fails, regenerate plan from current state
"""

from __future__ import annotations

import json
import re
from enum import Enum
from logging import Logger
from typing import Optional, Union

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
    PlannedStep,
)


class PlanExecutePhase(str, Enum):
    """Current phase of Plan-and-Execute."""

    VARIABLE_EXTRACTION = "variable_extraction"  # PS+ phase
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"


# PS+ (Plan-and-Solve Plus) Data Models
# From the paper: "Plan-and-Solve Prompting" (arxiv.org/abs/2305.04091)


class ExtractedVariable(ModelWithSummary):
    """A variable extracted from the problem statement (PS+ feature)."""

    name: str = Field(description="Variable name or label")
    value: Optional[Union[float, int, str]] = Field(
        default=None, description="Extracted value if available"
    )
    unit: str = Field(default="", description="Unit of measurement")
    description: str = Field(
        default="", description="Description of what this represents"
    )

    def summary(self) -> str:
        if self.value is not None:
            return f"{self.name} = {self.value} {self.unit}".strip()
        return f"{self.name}: {self.description}"


class CalculationStep(ModelWithSummary):
    """A calculation step with verification (PS+ feature)."""

    expression: str = Field(description="The mathematical expression")
    result: str = Field(description="The calculated result")
    verification_method: str = Field(
        default="", description="Method used to verify the result"
    )
    is_valid: bool = Field(default=True, description="Whether verification passed")

    def summary(self) -> str:
        status = "✓" if self.is_valid else "✗"
        return f"{status} {self.expression} = {self.result}"


class PSPlusContext(ModelWithSummary):
    """Context for PS+ (Plan-and-Solve Plus) features.

    Stores extracted variables and calculation steps for improved
    mathematical and multi-step reasoning tasks.
    """

    extracted_variables: list[ExtractedVariable] = Field(default_factory=list)
    calculation_steps: list[CalculationStep] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.extracted_variables)} vars, {len(self.calculation_steps)} calcs"
        )

    def format_for_prompt(self) -> str:
        """Format the context for inclusion in prompts."""
        lines = []

        if self.extracted_variables:
            lines.append("## Extracted Variables")
            for var in self.extracted_variables:
                if var.value is not None:
                    lines.append(
                        f"- {var.name} = {var.value} {var.unit}: {var.description}"
                    )
                else:
                    lines.append(f"- {var.name}: {var.description}")
            lines.append("")

        if self.calculation_steps:
            lines.append("## Calculations Performed")
            for calc in self.calculation_steps:
                status = "valid" if calc.is_valid else "INVALID"
                lines.append(f"- {calc.expression} = {calc.result} [{status}]")
                if calc.verification_method:
                    lines.append(f"  Verified by: {calc.verification_method}")
            lines.append("")

        return "\n".join(lines) if lines else ""

    def add_variable(
        self,
        name: str,
        value: Optional[Union[float, int, str]] = None,
        unit: str = "",
        description: str = "",
    ) -> None:
        """Add an extracted variable."""
        self.extracted_variables.append(
            ExtractedVariable(
                name=name, value=value, unit=unit, description=description
            )
        )

    def add_calculation(
        self,
        expression: str,
        result: str,
        verification_method: str = "",
        is_valid: bool = True,
    ) -> None:
        """Add a calculation step."""
        self.calculation_steps.append(
            CalculationStep(
                expression=expression,
                result=result,
                verification_method=verification_method,
                is_valid=is_valid,
            )
        )


class PlanExecuteThoughts(ModelWithSummary):
    """Thoughts for Plan-and-Execute strategy."""

    observations: str = Field(
        description="Relevant observations from context or last action"
    )
    reasoning: str = Field(description="Reasoning about the current situation")
    current_step_analysis: str = Field(
        default="", description="Analysis of the current step being executed"
    )
    plan_status: str = Field(default="", description="Status of the overall plan")
    self_criticism: str = Field(description="Constructive self-criticism")

    def summary(self) -> str:
        return self.reasoning


class ExecutionPlan(ModelWithSummary):
    """A high-level plan with steps to execute."""

    goal: str = Field(description="The goal this plan is trying to achieve")
    steps: list[PlannedStep] = Field(default_factory=list)
    current_step_index: int = Field(default=0)
    completed_steps: list[str] = Field(
        default_factory=list, description="Summaries of completed steps"
    )
    failed_attempts: int = Field(default=0)
    max_retries: int = Field(default=3)

    def summary(self) -> str:
        return f"Plan: {self.goal} ({self.current_step_index}/{len(self.steps)} steps)"

    def get_current_step(self) -> Optional[PlannedStep]:
        """Get the current step to execute."""
        if self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]

    def advance_step(self, result_summary: str) -> None:
        """Mark current step complete and advance to next."""
        current = self.get_current_step()
        if current:
            current.status = "completed"
            current.result = result_summary
            step_num = self.current_step_index + 1
            self.completed_steps.append(
                f"Step {step_num}: {current.thought} -> {result_summary}"
            )
        self.current_step_index += 1
        self.failed_attempts = 0  # Reset on success

    def mark_step_failed(self, error: str) -> bool:
        """Mark current step as failed. Returns True if should replan."""
        current = self.get_current_step()
        if current:
            current.status = "failed"
            current.result = f"Failed: {error}"
        self.failed_attempts += 1
        return self.failed_attempts >= self.max_retries

    def is_complete(self) -> bool:
        """Check if plan is fully executed."""
        return self.current_step_index >= len(self.steps)

    def get_progress_summary(self) -> str:
        """Get a summary of progress so far."""
        lines = [f"Goal: {self.goal}", "", "Progress:"]
        for i, summary in enumerate(self.completed_steps):
            lines.append(f"  ✓ {summary}")
        current = self.get_current_step()
        if current:
            lines.append(f"  → Current: {current.thought}")
        remaining = len(self.steps) - self.current_step_index - 1
        if remaining > 0:
            lines.append(f"  ... {remaining} more steps remaining")
        return "\n".join(lines)


class PlanExecuteActionProposal(ActionProposal):
    """Action proposal for Plan-and-Execute strategy.

    Note: plan and phase are stored in strategy state, not in the proposal.
    """

    thoughts: PlanExecuteThoughts  # type: ignore[assignment]


class PlanExecutePromptConfiguration(BasePromptStrategyConfiguration):
    """Configuration for Plan-and-Execute strategy."""

    DEFAULT_PLANNER_INSTRUCTION: str = (
        "You need to create a step-by-step plan to accomplish the given task.\n\n"
        "For each step:\n"
        "1. Describe what needs to be done\n"
        "2. Specify which command/tool will be used\n"
        "3. Note any dependencies on previous steps\n\n"
        "Format your plan as a numbered list:\n"
        "1. [Description] - Command: [command_name]\n"
        "2. [Description] - Command: [command_name]\n"
        "...\n\n"
        "After the plan, provide your response in the required JSON format "
        "and invoke the command for the FIRST step only."
    )

    DEFAULT_EXECUTOR_INSTRUCTION: str = (
        "You are executing step {step_num} of your plan.\n\n"
        "Plan Progress:\n{progress}\n\n"
        "Current Step: {current_step}\n\n"
        "Execute this step by invoking the appropriate command. "
        "Provide your reasoning in the required JSON format."
    )

    DEFAULT_REPLANNER_INSTRUCTION: str = (
        "The previous step failed. You need to adjust your plan.\n\n"
        "Original Goal: {goal}\n"
        "Progress So Far:\n{progress}\n"
        "Failed Step: {failed_step}\n"
        "Error: {error}\n\n"
        "Create a new plan to achieve the goal from the current state. "
        "Consider what went wrong and how to avoid similar issues."
    )

    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Based on your plan and current progress, execute the next step."
    )

    # PS+ (Plan-and-Solve Plus) Variable Extraction Instruction
    DEFAULT_VARIABLE_EXTRACTION_INSTRUCTION: str = (
        "Before solving this problem, first extract all relevant variables "
        "and quantities mentioned in the problem statement.\n\n"
        "For each variable, identify:\n"
        "- Name/label (e.g., 'initial_speed', 'total_cost', 'num_items')\n"
        "- Value (if given explicitly)\n"
        "- Unit of measurement (if applicable)\n"
        "- Description of what it represents\n\n"
        "Format as:\n"
        "Variables:\n"
        "- [name]: [value] [unit] - [description]\n"
        "- [name]: [value] [unit] - [description]\n"
        "...\n\n"
        "This will help structure the solution approach."
    )

    # PS+ Calculation Verification Instruction
    DEFAULT_CALCULATION_VERIFICATION_INSTRUCTION: str = (
        "After each calculation step, verify the result:\n\n"
        "1. Show the calculation: [expression] = [result]\n"
        "2. Verify using an alternative method or sanity check\n"
        "3. Confirm: Valid [yes/no] - [reason]\n\n"
        "This ensures accuracy in mathematical reasoning."
    )

    planner_instruction: str = UserConfigurable(default=DEFAULT_PLANNER_INSTRUCTION)
    executor_instruction: str = UserConfigurable(default=DEFAULT_EXECUTOR_INSTRUCTION)
    replanner_instruction: str = UserConfigurable(default=DEFAULT_REPLANNER_INSTRUCTION)
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    variable_extraction_instruction: str = UserConfigurable(
        default=DEFAULT_VARIABLE_EXTRACTION_INSTRUCTION
    )
    calculation_verification_instruction: str = UserConfigurable(
        default=DEFAULT_CALCULATION_VERIFICATION_INSTRUCTION
    )
    enable_replanning: bool = UserConfigurable(default=True)
    max_replan_attempts: int = UserConfigurable(default=3)

    # PS+ feature flags
    enable_variable_extraction: bool = UserConfigurable(default=False)
    enable_calculation_verification: bool = UserConfigurable(default=False)


class PlanExecutePromptStrategy(BaseMultiStepPromptStrategy):
    """Plan-and-Execute prompt strategy implementation."""

    default_configuration: PlanExecutePromptConfiguration = (
        PlanExecutePromptConfiguration()
    )

    def __init__(
        self,
        configuration: PlanExecutePromptConfiguration,
        logger: Logger,
    ):
        super().__init__(configuration, logger)
        self.config: PlanExecutePromptConfiguration = configuration
        self.current_plan: Optional[ExecutionPlan] = None

        # Start with variable extraction if enabled, otherwise planning
        self.current_phase: PlanExecutePhase = (
            PlanExecutePhase.VARIABLE_EXTRACTION
            if configuration.enable_variable_extraction
            else PlanExecutePhase.PLANNING
        )

        self.replan_count: int = 0
        self._pending_step_advance: bool = False
        self._response_schema = JSONSchema.from_dict(
            PlanExecuteActionProposal.model_json_schema()
        )

        # PS+ context for variable extraction and calculation verification
        self.ps_plus_context: Optional[PSPlusContext] = None

    @property
    def llm_classification(self) -> LanguageModelClassification:
        # Use smart model for planning and variable extraction, fast for execution
        if self.current_phase in (
            PlanExecutePhase.VARIABLE_EXTRACTION,
            PlanExecutePhase.PLANNING,
            PlanExecutePhase.REPLANNING,
        ):
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
        if self.current_phase == PlanExecutePhase.VARIABLE_EXTRACTION:
            return self._build_variable_extraction_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )
        elif self.current_phase == PlanExecutePhase.REPLANNING:
            return self._build_replan_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
                error=extras.get("last_error", "Unknown error"),
            )
        elif self.current_phase == PlanExecutePhase.EXECUTING and self.current_plan:
            return self._build_execute_prompt(
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
        )

        planning_msg = ChatMessage.user(
            f"{self.config.planner_instruction}\n\n" f'Task:\n"""{task}"""'
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

    def _build_variable_extraction_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the variable extraction phase prompt (PS+ feature)."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        extraction_msg = ChatMessage.user(
            f"{self.config.variable_extraction_instruction}\n\n" f'Task:\n"""{task}"""'
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                *messages,
                extraction_msg,
            ],
            prefill_response=response_prefill if self.config.use_prefill else "",
            functions=commands,
        )

    def _build_execute_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
    ) -> ChatPrompt:
        """Build the execution phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        current_step = (
            self.current_plan.get_current_step() if self.current_plan else None
        )
        step_description = current_step.thought if current_step else "No current step"

        executor_instruction = self.config.executor_instruction.format(
            step_num=(
                self.current_plan.current_step_index + 1 if self.current_plan else 1
            ),
            progress=(
                self.current_plan.get_progress_summary() if self.current_plan else ""
            ),
            current_step=step_description,
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'Task: """{task}"""'),
                *messages,
                ChatMessage.user(executor_instruction),
            ],
            prefill_response=response_prefill if self.config.use_prefill else "",
            functions=commands,
        )

    def _build_replan_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: list[CompletionModelFunction],
        include_os_info: bool,
        error: str,
    ) -> ChatPrompt:
        """Build the replanning phase prompt."""
        system_prompt, response_prefill = self._build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        current_step = (
            self.current_plan.get_current_step() if self.current_plan else None
        )

        replan_instruction = self.config.replanner_instruction.format(
            goal=self.current_plan.goal if self.current_plan else task,
            progress=(
                self.current_plan.get_progress_summary()
                if self.current_plan
                else "No progress yet"
            ),
            failed_step=current_step.thought if current_step else "Unknown step",
            error=error,
        )

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                *messages,
                ChatMessage.user(replan_instruction),
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

        phase_instruction = {
            PlanExecutePhase.VARIABLE_EXTRACTION: (
                "You are in VARIABLE EXTRACTION mode (PS+). Extract all relevant "
                "variables and quantities from the problem before solving."
            ),
            PlanExecutePhase.PLANNING: (
                "You are in PLANNING mode. Create a comprehensive plan "
                "before taking any action."
            ),
            PlanExecutePhase.EXECUTING: (
                "You are in EXECUTION mode. Execute the current step of your plan."
            ),
            PlanExecutePhase.REPLANNING: (
                "You are in REPLANNING mode. A step failed and you need to adjust "
                "your approach."
            ),
        }.get(self.current_phase, "")

        # Include PS+ context if available
        ps_plus_section = ""
        if self.ps_plus_context:
            formatted_context = self.ps_plus_context.format_for_prompt()
            if formatted_context:
                ps_plus_section = f"\n## Problem Context (PS+)\n{formatted_context}"

        system_prompt_parts = (
            self.generate_intro_prompt(ai_profile)
            + (self.generate_os_info() if include_os_info else [])
            + [self.build_body(ai_directives, commands)]
            + ([ps_plus_section] if ps_plus_section else [])
            + [f"## Current Mode\n{phase_instruction}"]
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
    ) -> PlanExecuteActionProposal:
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

        # Handle variable extraction phase (PS+)
        if self.current_phase == PlanExecutePhase.VARIABLE_EXTRACTION:
            self._process_variable_extraction(response.content)
            # After extraction, move to planning phase
            self.current_phase = PlanExecutePhase.PLANNING
        # Extract plan from response if in planning phase
        elif self.current_phase == PlanExecutePhase.PLANNING:
            plan = self._extract_plan_from_response(response.content)
            if plan:
                self.current_plan = plan
                self.current_phase = PlanExecutePhase.EXECUTING
                # Mark that we need to advance after this action executes
                self._pending_step_advance = False
        elif self.current_phase == PlanExecutePhase.REPLANNING:
            plan = self._extract_plan_from_response(response.content)
            if plan:
                # Preserve completed steps from old plan
                if self.current_plan:
                    plan.completed_steps = self.current_plan.completed_steps
                self.current_plan = plan
                self.current_phase = PlanExecutePhase.EXECUTING
                self.replan_count += 1
                self._pending_step_advance = False
        elif self.current_phase == PlanExecutePhase.EXECUTING and self.current_plan:
            # If we have a pending advance from the previous action, do it now
            if getattr(self, "_pending_step_advance", False):
                current_step = self.current_plan.get_current_step()
                if current_step:
                    self.current_plan.advance_step("Executed")
            # Mark that the current action needs to be advanced after execution
            self._pending_step_advance = True

        # Plan and phase are stored in strategy state, not in the proposal
        parsed_response = PlanExecuteActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        return parsed_response

    def _extract_plan_from_response(self, content: str) -> Optional[ExecutionPlan]:
        """Extract a plan from the LLM response.

        Looks for numbered lists like:
        1. Description - Command: command_name
        2. Description - Command: command_name
        """
        plan = ExecutionPlan(goal="")

        # Try to extract goal from content
        goal_match = re.search(
            r"(?:Goal|Task|Objective):\s*(.+?)(?:\n|$)", content, re.IGNORECASE
        )
        if goal_match:
            plan.goal = goal_match.group(1).strip()

        # Pattern for numbered steps
        # Matches: "1. Description - Command: tool_name"
        step_regex = (
            r"(\d+)\.\s+(.+?)"
            r"(?:\s*[-–—]\s*(?:Command|Tool|Action):\s*(\w+))?"
            r"(?=\n\d+\.|\n*$)"
        )
        step_pattern = re.compile(step_regex, re.MULTILINE | re.DOTALL)

        matches = step_pattern.findall(content)

        for match in matches:
            _num, description, tool_name = match
            step = PlannedStep(
                thought=description.strip(),
                tool_name=tool_name.strip() if tool_name else "",
                tool_arguments={},
            )
            plan.steps.append(step)

        # If no structured plan found, try simpler numbered list
        if not plan.steps:
            simple_pattern = re.compile(r"^\s*(\d+)\.\s+(.+)$", re.MULTILINE)
            for match in simple_pattern.finditer(content):
                step = PlannedStep(
                    thought=match.group(2).strip(),
                    tool_name="",
                    tool_arguments={},
                )
                plan.steps.append(step)

        return plan if plan.steps else None

    def record_step_success(self, result_summary: str) -> None:
        """Record successful step execution."""
        if self.current_plan:
            self.current_plan.advance_step(result_summary)

    def record_step_failure(self, error: str) -> bool:
        """Record step failure. Returns True if should trigger replan."""
        if not self.current_plan:
            return False

        should_replan = self.current_plan.mark_step_failed(error)

        if should_replan and self.config.enable_replanning:
            if self.replan_count < self.config.max_replan_attempts:
                self.current_phase = PlanExecutePhase.REPLANNING
                return True

        return False

    def is_plan_complete(self) -> bool:
        """Check if the current plan is complete."""
        if not self.current_plan:
            return False
        return self.current_plan.is_complete()

    def reset(self) -> None:
        """Reset the strategy for a new task."""
        self.current_plan = None
        # Start with variable extraction if enabled, otherwise planning
        self.current_phase = (
            PlanExecutePhase.VARIABLE_EXTRACTION
            if self.config.enable_variable_extraction
            else PlanExecutePhase.PLANNING
        )
        self.replan_count = 0
        self._pending_step_advance = False
        self.ps_plus_context = None

    def _process_variable_extraction(self, content: str) -> None:
        """Process the variable extraction phase response (PS+ feature).

        Extracts variables from the LLM response and stores them in ps_plus_context.
        """
        self.ps_plus_context = PSPlusContext()

        # Pattern to match variables in format: "- name: value unit - description"
        # or "- name = value unit : description"
        var_pattern = re.compile(
            r"-\s*(\w+)\s*[:=]\s*"
            r"(?:(\d+(?:\.\d+)?)\s*(\w*)\s*[-:]?\s*)?"
            r"(.+?)(?=\n-|\n\n|$)",
            re.MULTILINE | re.IGNORECASE,
        )

        matches = var_pattern.findall(content)
        for match in matches:
            name, value_str, unit, description = match
            value: Optional[Union[float, int, str]] = None
            if value_str:
                try:
                    if "." in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str

            self.ps_plus_context.add_variable(
                name=name.strip(),
                value=value,
                unit=unit.strip() if unit else "",
                description=description.strip(),
            )

        self.logger.debug(
            f"Extracted {len(self.ps_plus_context.extracted_variables)} variables"
        )

    def get_ps_plus_context(self) -> Optional[PSPlusContext]:
        """Get the PS+ context for use in prompts."""
        return self.ps_plus_context
