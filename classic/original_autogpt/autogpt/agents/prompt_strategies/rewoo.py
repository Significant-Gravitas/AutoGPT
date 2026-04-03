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

from .base import (
    BaseMultiStepPromptStrategy,
    BasePromptStrategyConfiguration,
    PlannedStep,
    WorkerExecution,
)


class ReWOOPhase(str, Enum):
    """Current phase of the ReWOO execution."""

    PLANNING = "planning"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"


class UseCachedActionException(Exception):
    """Raised during EXECUTING phase to signal that cached action should be used.

    ReWOO pre-plans all actions during PLANNING phase, so EXECUTING phase
    should retrieve actions from the cached plan rather than making LLM calls.
    This exception allows the agent to skip the LLM call and use the pre-planned action.
    """

    def __init__(self, action_proposal: "ReWOOActionProposal"):
        self.action_proposal = action_proposal
        super().__init__("Use cached action from ReWOO plan")


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

    def summary(self) -> str:
        return self.reasoning if self.reasoning else self.observations


class ReWOOPlan(ModelWithSummary):
    """A complete ReWOO plan with multiple steps."""

    steps: list[PlannedStep] = Field(default_factory=list)
    current_step_index: int = Field(default=0)
    execution_results: dict[str, str] = Field(default_factory=dict)

    # Worker execution tracking (per ReWOO paper)
    worker_executions: list[WorkerExecution] = Field(default_factory=list)

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

    def record_worker_execution(
        self,
        step: PlannedStep,
        substituted_args: dict[str, Any],
        output: str,
        error: Optional[str] = None,
    ) -> None:
        """Record a Worker module execution (per ReWOO paper).

        Args:
            step: The planned step that was executed
            substituted_args: Arguments after variable substitution
            output: Raw output from tool execution
            error: Error message if execution failed
        """
        execution = WorkerExecution(
            step=step,
            input_substituted=substituted_args,
            raw_output=output,
            error=error,
        )
        self.worker_executions.append(execution)

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
        "Create a complete plan to FULLY ACCOMPLISH the task. Your plan must include "
        "all steps needed to produce the final deliverable - not just exploration.\n\n"
        "IMPORTANT:\n"
        "- Do NOT end with exploration, research, or todo/planning steps\n"
        "- Your plan must result in the actual task being COMPLETE\n"
        "- Include steps that create/modify files, write code, or produce output\n"
        "- The final steps should verify the task is done, not plan future work\n\n"
        "For each step:\n"
        "1. Write your reasoning (Plan:)\n"
        "2. Specify the tool to use and its arguments\n"
        "3. Assign a variable name (#E1, #E2, etc.) to store the result\n"
        "4. Later steps can reference earlier results using variable names\n\n"
        "Format each step EXACTLY as:\n"
        "Plan: [Your reasoning for this step]\n"
        '#E1 = tool_name(arg1="value1", arg2="value2")\n\n'
        "Example plan:\n"
        "Plan: First, I need to list the files to understand the structure.\n"
        '#E1 = list_folder(folder=".")\n'
        "Plan: Next, I will read the main file to understand its contents.\n"
        '#E2 = read_file(filename="main.py")\n'
        "Plan: Finally, I will write the solution to a new file.\n"
        '#E3 = write_to_file(filename="solution.txt", contents="The answer is 42")\n\n'
        "Now create your plan following this EXACT format."
    )

    # Paper-style planner instruction (uses bracket syntax like the original paper)
    DEFAULT_PAPER_PLANNER_INSTRUCTION: str = (
        "For the following task, make plans that can FULLY SOLVE the problem "
        "step by step. "
        "For each plan, indicate which external tool together with tool input to "
        "retrieve evidence. You can store the evidence into a variable #E[n] that "
        "can be called by later tools.\n\n"
        "IMPORTANT: Your plan must COMPLETE the task, not just explore or prepare. "
        "Do not end with research or planning steps - include all actions needed "
        "to produce the final deliverable.\n\n"
        "Tools can be one of the following:\n"
        "{available_tools}\n\n"
        "Format:\n"
        "Plan: [first action to take based on input question]\n"
        "#E1 = ToolName[tool input]\n"
        "Plan: [next action to take, based on result of #E1]\n"
        "#E2 = ToolName[tool input, possibly referencing #E1]\n"
        "...\n\n"
        "Begin! Describe your plans with rich details."
    )

    DEFAULT_SYNTHESIZER_INSTRUCTION: str = (
        "You have executed the following plan and received these results.\n"
        "Analyze the results and determine if the ORIGINAL TASK has been "
        "accomplished.\n\n"
        "Plan and Results:\n{plan_with_results}\n\n"
        "Original Task: {task}\n\n"
        "IMPORTANT: The task is only complete if you have PRODUCED THE DELIVERABLE "
        "(created/modified files, written code, generated output, etc). If you only "
        "explored or planned, the task is NOT complete.\n\n"
        "If the task is truly complete, call `finish` with your final answer. "
        "If the task is NOT complete (only explored/planned), you must call other "
        "commands to actually complete the work."
    )

    # Paper-style synthesizer instruction (Solver module from paper)
    DEFAULT_PAPER_SYNTHESIZER_INSTRUCTION: str = (
        "Solve the following task or problem. To assist you, we provide some "
        "plans and corresponding evidences that might be helpful. Notice that "
        "some of these information may contain noise, so you should use them "
        "with caution.\n\n"
        "Task: {task}\n\n"
        "Plans and Evidences:\n{plan_with_results}\n\n"
        "IMPORTANT: The task is only complete if you have PRODUCED THE DELIVERABLE. "
        "If you only explored or planned, call commands to complete the actual work. "
        "If the task is truly complete, call `finish` with your final answer."
    )

    planner_instruction: str = UserConfigurable(default=DEFAULT_PLANNER_INSTRUCTION)
    paper_planner_instruction: str = UserConfigurable(
        default=DEFAULT_PAPER_PLANNER_INSTRUCTION
    )
    synthesizer_instruction: str = UserConfigurable(
        default=DEFAULT_SYNTHESIZER_INSTRUCTION
    )
    paper_synthesizer_instruction: str = UserConfigurable(
        default=DEFAULT_PAPER_SYNTHESIZER_INSTRUCTION
    )
    max_plan_steps: int = UserConfigurable(default=10)
    allow_parallel_execution: bool = UserConfigurable(default=True)

    # Use paper-style bracket format (#E1 = Tool[arg]) vs function format
    use_paper_format: bool = UserConfigurable(default=False)


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
        # Worker execution tracking
        self._pending_step: Optional[PlannedStep] = None
        self._pending_substituted_args: Optional[dict[str, Any]] = None

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
        """Build prompt based on current phase.

        Raises:
            UseCachedActionException: During EXECUTING phase, signals that
                the cached action from the plan should be used instead of
                making an LLM call. This is the core ReWOO optimization.
        """
        self.logger.info(
            f"ReWOO build_prompt: current_phase={self.current_phase.value}"
        )

        # EXECUTING phase: use pre-planned actions without LLM calls
        if self.current_phase == ReWOOPhase.EXECUTING:
            cached_action = self._get_cached_action_proposal()
            if cached_action:
                # current_plan is guaranteed to be set in EXECUTING phase
                assert self.current_plan is not None
                self.logger.debug(
                    f"ReWOO EXECUTING: Using cached action "
                    f"(step {self.current_plan.current_step_index + 1} "
                    f"of {len(self.current_plan.steps)})"
                )
                raise UseCachedActionException(cached_action)
            # No more steps - transition to SYNTHESIZING
            self.logger.info("ReWOO: All steps executed, transitioning to SYNTHESIZING")
            self.current_phase = ReWOOPhase.SYNTHESIZING

        if self.current_phase == ReWOOPhase.SYNTHESIZING:
            return self._build_synthesis_prompt(
                messages=messages,
                task=task,
                ai_profile=ai_profile,
                ai_directives=ai_directives,
                commands=commands,
                include_os_info=include_os_info,
            )
        else:  # PLANNING phase
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
        # During planning, we want plan text format, not tool calls
        is_planning = not is_synthesis and self.current_phase == ReWOOPhase.PLANNING
        response_fmt_instruction, response_prefill = self._response_format_instruction(
            is_planning=is_planning
        )

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

    def _response_format_instruction(
        self, is_planning: bool = False
    ) -> tuple[str, str]:
        """Generate response format instruction.

        Args:
            is_planning: If True, we're in PLANNING phase and want plan text,
                not tool calls.
        """
        if is_planning:
            # During planning, we want the plan in text format, not tool calls
            return (
                "Output your plan following the EXACT format specified in the "
                "instructions. Each step must have:\n"
                '- "Plan:" followed by your reasoning\n'
                '- "#E[n] = tool_name(arg1=\\"value1\\", ...)" on the next line\n\n'
                "Do NOT call any tools directly - just write out the plan steps.",
                "",  # No prefill for planning
            )

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

        # Parse plan from response FIRST if in planning phase.
        # During PLANNING, we expect plan text format, not JSON
        if self.current_phase == ReWOOPhase.PLANNING:
            self.logger.info("ReWOO: Attempting to extract plan from PLANNING response")
            plan = self._extract_plan_from_response(response.content)
            if plan and plan.steps:
                self.current_plan = plan
                # Transition to EXECUTING phase now that we have a plan
                self.current_phase = ReWOOPhase.EXECUTING
                self.logger.info(
                    f"ReWOO: Extracted plan with {len(plan.steps)} steps, "
                    f"transitioning to EXECUTING phase"
                )

                # Use the first step of the plan as the action to execute
                first_step = plan.steps[0]
                first_action = AssistantFunctionCall(
                    name=first_step.tool_name,
                    arguments=first_step.tool_arguments,
                )

                # Build a complete proposal from the plan
                thoughts = ReWOOThoughts(
                    observations="Created ReWOO execution plan",
                    reasoning=first_step.thought,
                    plan=[f"{s.variable_name}: {s.thought}" for s in plan.steps],
                )

                # Create synthetic raw message
                from forge.llm.providers.schema import AssistantToolCall

                raw_message = AssistantChatMessage(
                    content=response.content,
                    tool_calls=[
                        AssistantToolCall(
                            id="rewoo_plan_step_0",
                            type="function",
                            function=first_action,
                        )
                    ],
                )

                return ReWOOActionProposal(
                    thoughts=thoughts,
                    use_tool=first_action,
                    raw_message=raw_message,
                )
            else:
                self.logger.warning(
                    "ReWOO: Failed to extract plan from response, staying in PLANNING. "
                    f"Plan: {plan}, Steps: {plan.steps if plan else 'N/A'}"
                )
                # Fall through to standard JSON parsing if plan extraction fails

        # For non-planning phases or if plan extraction failed, parse as JSON
        assistant_reply_dict = extract_dict_from_json(response.content)
        self.logger.debug(
            "Parsing object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )

        if not response.tool_calls:
            raise InvalidAgentResponseError("Assistant did not use a tool")

        assistant_reply_dict["use_tool"] = response.tool_calls[0].function

        # Ensure thoughts dict has required fields
        thoughts_dict = assistant_reply_dict.get("thoughts", {})
        if not isinstance(thoughts_dict, dict):
            thoughts_dict = {"observations": "", "reasoning": ""}
        if "observations" not in thoughts_dict:
            thoughts_dict["observations"] = ""
        if "reasoning" not in thoughts_dict:
            thoughts_dict["reasoning"] = ""
        assistant_reply_dict["thoughts"] = thoughts_dict

        parsed_response = ReWOOActionProposal.model_validate(assistant_reply_dict)
        parsed_response.raw_message = response.model_copy()

        return parsed_response

    def _extract_plan_from_response(self, content: str) -> Optional[ReWOOPlan]:
        """Extract a structured plan from the LLM response.

        Supports two formats:
        1. Paper-style bracket format: #E1 = Tool[argument]
        2. Function-style parenthesis format: #E1 = tool(arg1="value1")
        """
        self.logger.debug(f"ReWOO: Extracting plan from content:\n{content[:1000]}...")
        plan = ReWOOPlan()

        # Pattern for paper-style bracket format: #E1 = Tool[argument]
        bracket_pattern = re.compile(
            r"Plan:\s*(.+?)\n\s*#(E\d+)\s*=\s*(\w+)\s*\[([^\]]*)\]",
            re.MULTILINE | re.DOTALL,
        )

        # Pattern for function-style: #E1 = tool(arg="value")
        paren_pattern = re.compile(
            r"Plan:\s*(.+?)\n\s*#(E\d+)\s*=\s*(\w+)\s*\(([^)]*)\)",
            re.MULTILINE | re.DOTALL,
        )

        # Try bracket format first (paper-style)
        matches = bracket_pattern.findall(content)
        is_bracket_format = bool(matches)
        self.logger.debug(f"ReWOO: Bracket pattern matches: {len(matches)}")

        if not matches:
            # Fall back to parenthesis format
            matches = paren_pattern.findall(content)
            is_bracket_format = False
            self.logger.debug(f"ReWOO: Paren pattern matches: {len(matches)}")

        if not matches:
            self.logger.warning(
                "ReWOO: No plan pattern matched. Expected format:\n"
                'Plan: [reasoning]\n#E1 = tool_name(arg="value")'
            )

        for match in matches[: self.config.max_plan_steps]:
            thought, var_num, tool_name, args_str = match
            variable_name = f"#{var_num}"

            # Parse arguments based on format
            if is_bracket_format:
                tool_arguments = self._parse_bracket_arguments(args_str)
            else:
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

    def _parse_bracket_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse paper-style bracket arguments: Tool[query, #E1].

        The bracket format from the paper uses simpler argument syntax:
        - Tool[search query]
        - Tool[#E1]
        - Tool[query, #E1]
        """
        args_str = args_str.strip()
        arguments: dict[str, Any] = {}

        # Empty arguments
        if not args_str:
            return arguments

        # Single variable reference
        if args_str.startswith("#E") and "," not in args_str:
            return {"input": args_str}

        # Check for key=value pairs first
        if "=" in args_str:
            # Mixed format: key=value pairs
            parts = [p.strip() for p in args_str.split(",")]
            for i, part in enumerate(parts):
                if "=" in part:
                    key, value = part.split("=", 1)
                    value = value.strip().strip("\"'")
                    arguments[key.strip()] = value
                else:
                    # Positional argument
                    arg_key = "query" if i == 0 else f"arg{i}"
                    arguments[arg_key] = part.strip().strip("\"'")
            return arguments

        # Simple format: comma-separated values
        if "," in args_str:
            parts = [p.strip() for p in args_str.split(",")]
            arguments["query"] = parts[0].strip("\"'")
            for i, part in enumerate(parts[1:], 1):
                arguments[f"arg{i}"] = part.strip().strip("\"'")
        else:
            # Single argument - treat as query
            arguments["query"] = args_str.strip("\"'")

        return arguments

    def _parse_tool_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse tool arguments from a string like 'arg1="value1", arg2=123'.

        Supports:
        - String values: arg="value"
        - Numbers: arg=123 or arg=1.5
        - Variable references: arg=#E1
        - JSON arrays: arg=[{...}]
        - JSON objects: arg={...}
        - Booleans: arg=true/false
        """
        arguments: dict[str, Any] = {}

        # Try to parse as JSON-like structure
        # Handle both JSON format ("key": value) and Python format (key=value)
        try:
            json_str = args_str.strip()

            # Convert Python-style key=value to JSON-style "key": value
            # Match: word= at start or after comma, followed by any value
            json_str = re.sub(r"(?:^|,\s*)(\w+)\s*=\s*", r'"\1": ', json_str)

            # Wrap in braces
            json_str = "{" + json_str + "}"

            # Fix common Python-isms: single quotes -> double, True/False -> true/false
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r"\bTrue\b", "true", json_str)
            json_str = re.sub(r"\bFalse\b", "false", json_str)
            json_str = re.sub(r"\bNone\b", "null", json_str)

            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back to regex-based parsing for simpler cases
        # Pattern for key=value where value can be:
        # - quoted string: "..."
        # - variable reference: #E1
        # - number: 123 or 1.5
        # - boolean: true/false
        arg_pattern = re.compile(
            r'(\w+)\s*=\s*(?:"([^"]*)"|(#E\d+)|(\d+\.?\d*)|(true|false))',
            re.IGNORECASE,
        )

        for match in arg_pattern.finditer(args_str):
            key = match.group(1)
            if match.group(2) is not None:  # String value
                arguments[key] = match.group(2)
            elif match.group(3) is not None:  # Variable reference
                arguments[key] = match.group(3)
            elif match.group(4) is not None:  # Number
                num_str = match.group(4)
                arguments[key] = float(num_str) if "." in num_str else int(num_str)
            elif match.group(5) is not None:  # Boolean
                arguments[key] = match.group(5).lower() == "true"

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

        # Store the substituted args for worker execution tracking
        self._pending_substituted_args = substituted_args
        self._pending_step = next_step

        return AssistantFunctionCall(
            name=next_step.tool_name,
            arguments=substituted_args,
        )

    def _get_cached_action_proposal(self) -> Optional["ReWOOActionProposal"]:
        """Get the next action from the cached plan as a full ActionProposal.

        This wraps get_next_action() to return a complete ReWOOActionProposal
        that can be used directly by the agent without making an LLM call.

        Returns:
            ReWOOActionProposal if there's a next step in the plan, None if complete.
        """
        from forge.llm.providers.schema import AssistantToolCall

        if not self.current_plan:
            return None

        next_action = self.get_next_action()  # Returns AssistantFunctionCall
        if not next_action:
            return None

        # Get the current step for context
        current_step = self.current_plan.get_next_step()

        # Build plan summary for thoughts
        remaining_steps = self.current_plan.steps[
            self.current_plan.current_step_index :
        ]
        plan_summary = [
            f"Step {i + 1}: {s.thought}" for i, s in enumerate(remaining_steps)
        ]

        # Create thoughts with plan context
        thoughts = ReWOOThoughts(
            observations=(
                f"Executing step {self.current_plan.current_step_index + 1} "
                f"of {len(self.current_plan.steps)} from pre-computed ReWOO plan"
            ),
            reasoning=(
                current_step.thought if current_step else "Executing planned step"
            ),
            plan=plan_summary[:5],  # Limit to 5 steps for brevity
        )

        # Create a synthetic raw message for action history compatibility
        raw_message = AssistantChatMessage(
            content=f"[ReWOO EXECUTING] Planned action: {next_action.name}",
            tool_calls=[
                AssistantToolCall(
                    id=f"rewoo_step_{self.current_plan.current_step_index}",
                    type="function",
                    function=next_action,
                )
            ],
        )

        return ReWOOActionProposal(
            thoughts=thoughts,
            use_tool=next_action,
            raw_message=raw_message,
        )

    def record_execution_result(
        self, variable_name: str, result: str, error: Optional[str] = None
    ) -> None:
        """Record the result of a step execution (Worker module output).

        Args:
            variable_name: The variable name (#E1, etc.) for this step
            result: The raw output from tool execution
            error: Optional error message if execution failed
        """
        if self.current_plan:
            # Record worker execution if we have the pending step info
            pending_step = self._pending_step
            pending_args = self._pending_substituted_args
            if pending_step is not None and pending_args is not None:
                self.current_plan.record_worker_execution(
                    step=pending_step,
                    substituted_args=pending_args,
                    output=result,
                    error=error,
                )
                self._pending_step = None
                self._pending_substituted_args = None

            self.current_plan.mark_step_complete(variable_name, result)

            # Check if we should move to synthesis phase
            if self.current_plan.all_complete():
                self.current_phase = ReWOOPhase.SYNTHESIZING

    def reset(self) -> None:
        """Reset the strategy state for a new task."""
        self.current_plan = None
        self.current_phase = ReWOOPhase.PLANNING
        self._pending_step = None
        self._pending_substituted_args = None

    def is_plan_complete(self) -> bool:
        """Check if the current plan is fully executed."""
        if not self.current_plan:
            return False
        return self.current_plan.all_complete()
