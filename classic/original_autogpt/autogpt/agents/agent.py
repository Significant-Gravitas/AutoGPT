from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import sentry_sdk
from pydantic import Field

from forge.agent.base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
from forge.agent.execution_context import ExecutionContext
from forge.agent.protocols import (
    AfterExecute,
    AfterParse,
    CommandProvider,
    DirectiveProvider,
    MessageProvider,
)
from forge.command.command import Command
from forge.components.action_history import (
    ActionHistoryComponent,
    EpisodicActionHistory,
)
from forge.components.action_history.action_history import ActionHistoryConfiguration
from forge.components.archive_handler import ArchiveHandlerComponent
from forge.components.clipboard import ClipboardComponent
from forge.components.code_executor.code_executor import (
    CodeExecutorComponent,
    CodeExecutorConfiguration,
)
from forge.components.context.context import AgentContext, ContextComponent
from forge.components.data_processor import DataProcessorComponent
from forge.components.file_manager import FileManagerComponent
from forge.components.git_operations import GitOperationsComponent
from forge.components.http_client import HTTPClientComponent
from forge.components.image_gen import ImageGeneratorComponent
from forge.components.math_utils import MathUtilsComponent
from forge.components.platform_blocks import PlatformBlocksComponent
from forge.components.skills import SkillComponent, SkillConfiguration
from forge.components.system import SystemComponent
from forge.components.text_utils import TextUtilsComponent
from forge.components.todo import TodoComponent
from forge.components.user_interaction import UserInteractionComponent
from forge.components.watchdog import WatchdogComponent
from forge.components.web import WebPlaywrightComponent, WebSearchComponent
from forge.file_storage.base import FileStorage
from forge.llm.prompting.schema import ChatPrompt
from forge.llm.prompting.utils import dump_prompt
from forge.llm.providers import (
    AssistantFunctionCall,
    ChatMessage,
    ChatModelResponse,
    MultiProvider,
)
from forge.llm.providers.utils import function_specs_from_commands
from forge.models.action import (
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionProposal,
    ActionResult,
    ActionSuccessResult,
)
from forge.models.config import Configurable
from forge.permissions import CommandPermissionManager
from forge.utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    UnknownCommandError,
)

from .prompt_strategies.lats import LATSActionProposal
from .prompt_strategies.multi_agent_debate import DebateActionProposal
from .prompt_strategies.one_shot import (
    OneShotAgentActionProposal,
    OneShotAgentPromptStrategy,
)
from .prompt_strategies.plan_execute import (
    PlanExecuteActionProposal,
    PlanExecutePromptStrategy,
)
from .prompt_strategies.reflexion import (
    ReflexionActionProposal,
    ReflexionPromptStrategy,
)
from .prompt_strategies.rewoo import ReWOOActionProposal, ReWOOPromptStrategy
from .prompt_strategies.tree_of_thoughts import (
    ToTActionProposal,
    TreeOfThoughtsPromptStrategy,
)

# Union of all action proposal types from different prompt strategies
AnyActionProposal = (
    OneShotAgentActionProposal
    | PlanExecuteActionProposal
    | ReWOOActionProposal
    | ReflexionActionProposal
    | ToTActionProposal
    | LATSActionProposal
    | DebateActionProposal
)

if TYPE_CHECKING:
    from autogpt.app.config import AppConfig

logger = logging.getLogger(__name__)


class AgentConfiguration(BaseAgentConfiguration):
    pass


class AgentSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(  # type: ignore
        default_factory=AgentConfiguration
    )

    history: EpisodicActionHistory[AnyActionProposal] = Field(
        default_factory=EpisodicActionHistory[AnyActionProposal]
    )
    """(STATE) The action history of the agent."""

    context: AgentContext = Field(default_factory=AgentContext)


class Agent(BaseAgent[AnyActionProposal], Configurable[AgentSettings]):
    default_settings: ClassVar[AgentSettings] = AgentSettings(
        name="Agent",
        description=__doc__ if __doc__ else "",
    )

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        app_config: AppConfig,
        permission_manager: Optional[CommandPermissionManager] = None,
        execution_context: Optional[ExecutionContext] = None,
    ):
        super().__init__(settings, permission_manager=permission_manager)

        self.llm_provider = llm_provider
        self.app_config = app_config

        # Create or use provided execution context
        if execution_context:
            self.execution_context = execution_context
        else:
            # Root agent - create new context
            self.execution_context = self._create_root_execution_context(
                llm_provider, file_storage, app_config
            )

        # Create prompt strategy and inject execution context
        self.prompt_strategy = self._create_prompt_strategy(app_config)
        # Multi-step strategies have set_execution_context; one_shot doesn't
        set_ctx = getattr(self.prompt_strategy, "set_execution_context", None)
        if set_ctx is not None:
            set_ctx(self.execution_context)

        self.commands: list[Command] = []

        # Components
        self.system = SystemComponent()
        self.history = (
            ActionHistoryComponent(
                settings.history,
                lambda x: self.llm_provider.count_tokens(x, self.llm.name),
                llm_provider,
                ActionHistoryConfiguration(
                    llm_name=app_config.fast_llm, max_tokens=self.send_token_limit
                ),
            )
            .run_after(WatchdogComponent)
            .run_after(SystemComponent)
        )
        if not app_config.noninteractive_mode:
            self.user_interaction = UserInteractionComponent()
        # CLI mode: file storage rooted at workspace (not .autogpt)
        # Agents work directly in cwd; in server mode, they're sandboxed
        cli_mode = file_storage.root == app_config.workspace
        self.file_manager = FileManagerComponent(
            file_storage,
            settings,
            workspace_root=app_config.workspace if cli_mode else None,
        )
        self.code_executor = CodeExecutorComponent(
            self.file_manager.workspace,
            CodeExecutorConfiguration(
                docker_container_name=f"{settings.agent_id}_sandbox"
            ),
        )
        self.git_ops = GitOperationsComponent()
        self.image_gen = ImageGeneratorComponent(self.file_manager.workspace)
        self.web_search = WebSearchComponent()
        self.web_browser = WebPlaywrightComponent(
            llm_provider,
            app_config.app_data_dir,
        )
        self.context = ContextComponent(self.file_manager.workspace, settings.context)
        self.todo = TodoComponent(
            llm_provider=llm_provider,
            smart_llm=str(app_config.smart_llm),
        )
        self.archive_handler = ArchiveHandlerComponent(self.file_manager.workspace)
        self.clipboard = ClipboardComponent()
        self.data_processor = DataProcessorComponent()
        self.http_client = HTTPClientComponent()
        self.math_utils = MathUtilsComponent()
        self.text_utils = TextUtilsComponent()
        self.watchdog = WatchdogComponent(settings.config, settings.history).run_after(
            ContextComponent
        )
        # Platform blocks (enabled only if PLATFORM_API_KEY is set)
        self.platform_blocks = PlatformBlocksComponent()

        # Skills (SKILL.md support)
        self.skills = SkillComponent(
            SkillConfiguration(
                skill_directories=[
                    app_config.workspace / ".autogpt/skills",
                    Path.home() / ".autogpt/skills",
                ]
            )
        )

        self.event_history = settings.history

    def _create_root_execution_context(
        self,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        app_config: AppConfig,
    ) -> ExecutionContext:
        """Create execution context for a root (top-level) agent.

        Root agents create their own execution context with:
        - Full access to shared resources
        - Default resource budget
        - An agent factory for spawning sub-agents

        Args:
            llm_provider: The LLM provider instance.
            file_storage: The file storage instance.
            app_config: The application configuration.

        Returns:
            A new ExecutionContext for this root agent.
        """
        from autogpt.agent_factory.default_factory import DefaultAgentFactory

        factory = DefaultAgentFactory(app_config)

        return ExecutionContext(
            llm_provider=llm_provider,
            file_storage=file_storage,
            agent_factory=factory,
            parent_agent_id=None,  # Root agent has no parent
            depth=0,
            _app_config=app_config,
        )

    async def propose_action(self) -> AnyActionProposal:
        """Proposes the next action to execute, based on the task and current state.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        self.reset_trace()

        # Get directives
        resources = await self.run_pipeline(DirectiveProvider.get_resources)
        constraints = await self.run_pipeline(DirectiveProvider.get_constraints)
        best_practices = await self.run_pipeline(DirectiveProvider.get_best_practices)

        directives = self.state.directives.model_copy(deep=True)
        directives.resources += resources
        directives.constraints += constraints
        directives.best_practices += best_practices

        # Get commands
        self.commands = await self.run_pipeline(CommandProvider.get_commands)
        self._remove_disabled_commands()

        # Prepare messages (lazy compression) - skip if strategy will use cached actions
        # ReWOO EXECUTING phase doesn't need messages, so skip compression
        skip_message_prep = False
        current_phase = getattr(self.prompt_strategy, "current_phase", None)
        if current_phase is not None:
            from .prompt_strategies.rewoo import ReWOOPhase

            skip_message_prep = current_phase == ReWOOPhase.EXECUTING

        if not skip_message_prep and hasattr(self, "history"):
            await self.history.prepare_messages()

        # Get messages
        messages = await self.run_pipeline(MessageProvider.get_messages)

        include_os_info = (
            self.code_executor.config.execute_local_commands
            if hasattr(self, "code_executor")
            else False
        )

        # Try to build prompt - some strategies (like ReWOO in EXECUTING phase)
        # may raise UseCachedActionException to skip LLM calls
        try:
            prompt: ChatPrompt = self.prompt_strategy.build_prompt(
                messages=messages,
                task=self.state.task,
                ai_profile=self.state.ai_profile,
                ai_directives=directives,
                commands=function_specs_from_commands(self.commands),
                include_os_info=include_os_info,
            )
        except Exception as e:
            # Check if this is a UseCachedActionException from ReWOO
            # We use string comparison to avoid import cycles
            if type(e).__name__ == "UseCachedActionException":
                # ReWOO EXECUTING phase - use pre-planned action, skip LLM call
                logger.debug("Using cached action from ReWOO plan (no LLM call)")
                output = e.action_proposal  # type: ignore
                # Register the action with history (same as complete_and_parse does)
                # This is required so execute() can later register the result
                await self.run_pipeline(AfterParse.after_parse, output)
                self.config.cycle_count += 1
                return output
            # Re-raise other exceptions
            raise

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        output = await self.complete_and_parse(prompt)
        self.config.cycle_count += 1

        return output

    async def complete_and_parse(
        self, prompt: ChatPrompt, exception: Optional[Exception] = None
    ) -> AnyActionProposal:
        if exception:
            prompt.messages.append(ChatMessage.user(f"Error: {exception}"))

        # Build thinking/reasoning kwargs from app config
        thinking_kwargs: dict[str, Any] = {}
        if hasattr(self, "app_config") and self.app_config:
            if self.app_config.thinking_budget_tokens:
                thinking_kwargs["thinking_budget_tokens"] = (
                    self.app_config.thinking_budget_tokens
                )
            if self.app_config.reasoning_effort:
                thinking_kwargs["reasoning_effort"] = self.app_config.reasoning_effort

        response: ChatModelResponse[AnyActionProposal] = (
            await self.llm_provider.create_chat_completion(
                prompt.messages,
                model_name=self.llm.name,
                completion_parser=self.prompt_strategy.parse_response_content,
                functions=prompt.functions,
                prefill_response=prompt.prefill_response,
                **thinking_kwargs,
            )
        )
        result = response.parsed_result

        await self.run_pipeline(AfterParse.after_parse, result)

        return result

    async def execute(
        self,
        proposal: ActionProposal,
        user_feedback: str = "",
    ) -> ActionResult:
        # Get all tools to execute (supports parallel execution)
        tools = proposal.get_tools()

        # Get commands
        self.commands = await self.run_pipeline(CommandProvider.get_commands)
        self._remove_disabled_commands()

        # Check permissions for all tools before execution
        feedback_to_append = None
        if self.permission_manager:
            for tool in tools:
                perm_result = self.permission_manager.check_command(
                    tool.name, tool.arguments
                )
                if not perm_result.allowed:
                    # Permission denied - pass feedback to agent if provided
                    if perm_result.feedback:
                        return await self.do_not_execute(proposal, perm_result.feedback)
                    return ActionErrorResult(
                        reason=f"Permission denied for command '{tool.name}'",
                    )
                # Permission granted - save feedback if any
                if perm_result.feedback:
                    feedback_to_append = perm_result.feedback

        # Execute tool(s)
        if len(tools) == 1:
            # Single tool - original behavior
            tool = tools[0]
            try:
                return_value = await self._execute_tool(tool)
                result = ActionSuccessResult(outputs=return_value)
            except AgentTerminated:
                raise
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)
                logger.warning(f"{tool} raised an error: {e}")
                sentry_sdk.capture_exception(e)
        else:
            # Multiple tools - execute in parallel
            logger.info(f"Executing {len(tools)} tools in parallel")
            result = await self._execute_tools_parallel(tools)

        result_tlength = self.llm_provider.count_tokens(str(result), self.llm.name)
        if result_tlength > self.send_token_limit // 3:
            result = ActionErrorResult(
                reason="Command(s) returned too much output. "
                "Do not execute these commands again with the same arguments."
            )

        # Notify ReWOO strategy of execution result for variable tracking
        # This allows ReWOO to record results and substitute variables in later steps
        record_result = getattr(self.prompt_strategy, "record_execution_result", None)
        plan = getattr(self.prompt_strategy, "current_plan", None)
        if record_result is not None and plan is not None:
            if plan.current_step_index < len(plan.steps):
                step = plan.steps[plan.current_step_index]
                error_msg = None
                if isinstance(result, ActionErrorResult):
                    error_msg = getattr(result, "reason", None) or str(result)
                result_str = str(getattr(result, "outputs", result))
                record_result(
                    step.variable_name,
                    result_str,
                    error=error_msg,
                )
                logger.debug(
                    f"ReWOO: Recorded result for {step.variable_name}, "
                    f"step {plan.current_step_index + 1}/{len(plan.steps)}"
                )

        await self.run_pipeline(AfterExecute.after_execute, result)

        # If user provided feedback along with approval, append it to history
        # so the agent sees it in the next iteration
        if feedback_to_append:
            self.event_history.append_user_feedback(feedback_to_append)

        logger.debug("\n".join(self.trace))

        return result

    async def do_not_execute(
        self, denied_proposal: ActionProposal, user_feedback: str
    ) -> ActionResult:
        result = ActionInterruptedByHuman(feedback=user_feedback)

        await self.run_pipeline(AfterExecute.after_execute, result)

        # Store feedback so it also appears as a prominent user message
        # in the next prompt (in addition to the tool result)
        if user_feedback:
            self.event_history.append_user_feedback(user_feedback)

        logger.debug("\n".join(self.trace))

        return result

    async def _execute_tool(self, tool_call: AssistantFunctionCall) -> Any:
        """Execute the command and return the result

        Args:
            tool_call (AssistantFunctionCall): The tool call to execute

        Returns:
            str: The execution result
        """
        # Execute a native command with the same name or alias, if it exists
        command = self._get_command(tool_call.name)
        try:
            result = command(**tool_call.arguments)
            if inspect.isawaitable(result):
                return await result
            return result
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    async def _execute_tools_parallel(
        self, tools: list[AssistantFunctionCall]
    ) -> ActionResult:
        """Execute multiple tools in parallel and combine results.

        Args:
            tools: List of tool calls to execute in parallel

        Returns:
            Combined ActionResult with all outputs or errors
        """

        async def execute_single(tool: AssistantFunctionCall) -> tuple[str, Any, str]:
            """Execute a single tool and return (name, result, error)."""
            try:
                result = await self._execute_tool(tool)
                return (tool.name, result, "")
            except AgentTerminated:
                raise
            except AgentException as e:
                logger.warning(f"{tool} raised an error: {e}")
                sentry_sdk.capture_exception(e)
                return (tool.name, None, str(e))

        # Execute all tools in parallel
        results = await asyncio.gather(
            *[execute_single(tool) for tool in tools],
            return_exceptions=True,
        )

        # Process results
        outputs: dict[str, Any] = {}
        errors: list[str] = []

        for i, res in enumerate(results):
            tool = tools[i]
            if isinstance(res, BaseException):
                # Unexpected exception from gather
                errors.append(f"{tool.name}: {res}")
                logger.warning(f"{tool} raised unexpected error: {res}")
                sentry_sdk.capture_exception(res)
            elif isinstance(res, tuple):
                name, output, error = res
                if error:
                    errors.append(f"{name}: {error}")
                else:
                    outputs[name] = output

        # Return combined result
        if errors and not outputs:
            # All failed
            return ActionErrorResult(reason="; ".join(errors))
        elif errors:
            # Partial success - include errors in output
            outputs["_errors"] = errors
            return ActionSuccessResult(outputs=outputs)
        else:
            # All succeeded
            return ActionSuccessResult(outputs=outputs)

    def _get_command(self, command_name: str) -> Command:
        for command in reversed(self.commands):
            if command_name in command.names:
                return command

        raise UnknownCommandError(
            f"Cannot execute command '{command_name}': unknown command."
        )

    def _remove_disabled_commands(self) -> None:
        self.commands = [
            command
            for command in self.commands
            if not any(
                name in self.app_config.disabled_commands for name in command.names
            )
        ]

    def find_obscured_commands(self) -> list[Command]:
        seen_names = set()
        obscured_commands = []
        for command in reversed(self.commands):
            # If all of the command's names have been seen, it's obscured
            if seen_names.issuperset(command.names):
                obscured_commands.append(command)
            else:
                seen_names.update(command.names)
        return list(reversed(obscured_commands))

    def _create_prompt_strategy(self, app_config: AppConfig):
        """Create the appropriate prompt strategy based on configuration.

        Args:
            app_config: The application configuration containing the
                prompt_strategy setting.

        Returns:
            An instance of the selected prompt strategy.
        """
        strategy_name = app_config.prompt_strategy
        use_prefill = self.llm.provider_name != "anthropic"

        if strategy_name == "rewoo":
            config = ReWOOPromptStrategy.default_configuration.model_copy(deep=True)
            config.use_prefill = use_prefill
            return ReWOOPromptStrategy(config, logger)

        elif strategy_name == "plan_execute":
            pe_config = PlanExecutePromptStrategy.default_configuration.model_copy(
                deep=True
            )
            pe_config.use_prefill = use_prefill
            return PlanExecutePromptStrategy(pe_config, logger)

        elif strategy_name == "reflexion":
            ref_config = ReflexionPromptStrategy.default_configuration.model_copy(
                deep=True
            )
            ref_config.use_prefill = use_prefill
            return ReflexionPromptStrategy(ref_config, logger)

        elif strategy_name == "tree_of_thoughts":
            tot_config = TreeOfThoughtsPromptStrategy.default_configuration.model_copy(
                deep=True
            )
            tot_config.use_prefill = use_prefill
            return TreeOfThoughtsPromptStrategy(tot_config, logger)

        elif strategy_name == "lats":
            from .prompt_strategies.lats import LATSPromptStrategy

            lats_config = LATSPromptStrategy.default_configuration.model_copy(deep=True)
            lats_config.use_prefill = use_prefill
            return LATSPromptStrategy(lats_config, logger)

        elif strategy_name == "multi_agent_debate":
            from .prompt_strategies.multi_agent_debate import MultiAgentDebateStrategy

            debate_config = MultiAgentDebateStrategy.default_configuration.model_copy(
                deep=True
            )
            debate_config.use_prefill = use_prefill
            return MultiAgentDebateStrategy(debate_config, logger)

        else:  # Default to one_shot
            os_config = OneShotAgentPromptStrategy.default_configuration.model_copy(
                deep=True
            )
            os_config.use_prefill = use_prefill
            return OneShotAgentPromptStrategy(os_config, logger)
