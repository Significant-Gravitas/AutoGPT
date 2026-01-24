import inspect
import logging
from typing import Any, Optional
from uuid import uuid4

from forge.agent.base import BaseAgent, BaseAgentSettings
from forge.agent.protocols import (
    AfterExecute,
    AfterParse,
    CommandProvider,
    DirectiveProvider,
    MessageProvider,
)
from forge.agent_protocol.agent import ProtocolAgent
from forge.agent_protocol.database.db import AgentDB
from forge.agent_protocol.models.task import (
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
)
from forge.command.command import Command
from forge.components.system.system import SystemComponent
from forge.config.ai_profile import AIProfile
from forge.file_storage.base import FileStorage
from forge.llm.prompting.schema import ChatPrompt
from forge.llm.prompting.utils import dump_prompt
from forge.llm.providers import (
    AssistantFunctionCall,
    ChatMessage,
    ChatModelResponse,
    MultiProvider,
)
from forge.llm.providers.schema import AssistantChatMessage
from forge.llm.providers.utils import function_specs_from_commands
from forge.models.action import (
    ActionErrorResult,
    ActionProposal,
    ActionResult,
    ActionSuccessResult,
)
from forge.utils.exceptions import AgentException, AgentTerminated

# Import from the correct path relative to the project root
from original_autogpt.autogpt.agents.prompt_strategies.one_shot import (
    OneShotAgentActionProposal,
    OneShotAgentPromptStrategy,
)

logger = logging.getLogger(__name__)


class ForgeAgent(ProtocolAgent, BaseAgent[OneShotAgentActionProposal]):
    """
    The goal of the Forge is to take care of the boilerplate code,
    so you can focus on agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possibilities.

    ForgeAgent provides component support; https://docs.agpt.co/classic/forge/components/introduction/
    Using Components is a new way of building agents that is more flexible and easier to extend.
    """  # noqa: E501

    def __init__(self, database: AgentDB, workspace: FileStorage, llm_provider: MultiProvider):
        """
        The database is used to store tasks, steps and artifact metadata.
        The workspace is used to store artifacts (files).
        The llm_provider is used to interact with language models.
        """

        # An example agent information; you can modify this to suit your needs
        state = BaseAgentSettings(
            name="Forge Agent",
            description="The Forge Agent is a generic agent that can solve tasks.",
            agent_id=str(uuid4()),
            ai_profile=AIProfile(
                ai_name="ForgeAgent", ai_role="Generic Agent", ai_goals=["Solve tasks"]
            ),
            task="Solve tasks",
        )

        # ProtocolAgent adds the Agent Protocol (API) functionality
        ProtocolAgent.__init__(self, database, workspace)
        # BaseAgent provides the component handling functionality
        BaseAgent.__init__(self, state)

        # LLM Provider and Prompt Strategy
        self.llm_provider = llm_provider
        prompt_config = OneShotAgentPromptStrategy.default_configuration.model_copy(
            deep=True
        )
        prompt_config.use_functions_api = (
            state.config.use_functions_api
            # Anthropic currently doesn't support tools + prefilling :( 
            and self.llm.provider_name != "anthropic"
        )
        self.prompt_strategy = OneShotAgentPromptStrategy(prompt_config, logger)

        # AGENT COMPONENTS
        # Components provide additional functionality to the agent
        # There are NO components added by default in the BaseAgent
        # You can create your own components or add existing ones
        # Built-in components:
        #   https://docs.agpt.co/classic/forge/components/built-in-components/

        # System component provides "finish" command and adds some prompt information
        self.system = SystemComponent()

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge,
        works by creating a task and then executing steps for that task.
        This method is called when the agent is asked to create a task.

        We are hooking into function to add a custom log message.
        Though you can do anything you want here.
        """
        task = await super().create_task(task_request)
        logger.info(
            f"📦 Task created with ID: {task.task_id} and "
            f"input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        Preffered method to add agent logic is to add custom components:
        https://docs.agpt.co/classic/forge/components/creating-components/

        Outdated tutorial on how to add custom logic:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """  # noqa: E501

        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )

        proposal = await self.propose_action()

        output = await self.execute(proposal)

        if isinstance(output, ActionSuccessResult):
            step.output = str(output.outputs)
        elif isinstance(output, ActionErrorResult):
            step.output = output.reason

        return step

    async def complete_and_parse(
        self, prompt: ChatPrompt, exception: Optional[Exception] = None
    ) -> OneShotAgentActionProposal:
        if exception:
            prompt.messages.append(ChatMessage.system(f"Error: {exception}"))

        response: ChatModelResponse[
            OneShotAgentActionProposal
        ] = await self.llm_provider.create_chat_completion(
            prompt.messages,
            model_name=self.llm.name,
            completion_parser=self.prompt_strategy.parse_response_content,
            functions=prompt.functions,
            prefill_response=prompt.prefill_response,
        )
        result = response.parsed_result

        await self.run_pipeline(AfterParse.after_parse, result)

        return result

    async def propose_action(self) -> OneShotAgentActionProposal:
        self.reset_trace()

        # Get directives
        directives = self.state.directives.model_copy(deep=True)
        directives.resources += await self.run_pipeline(DirectiveProvider.get_resources)
        directives.constraints += await self.run_pipeline(
            DirectiveProvider.get_constraints
        )
        directives.best_practices += await self.run_pipeline(
            DirectiveProvider.get_best_practices
        )

        # Get commands
        self.commands = await self.run_pipeline(CommandProvider.get_commands)

        # Get messages
        messages = await self.run_pipeline(MessageProvider.get_messages)

        # Build prompt using the prompt strategy
        prompt: ChatPrompt = self.prompt_strategy.build_prompt(
            messages=messages,
            task=self.state.task,
            ai_profile=self.state.ai_profile,
            ai_directives=directives,
            commands=function_specs_from_commands(self.commands),
            include_os_info=False,
        )

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")

        # Call the LLM and parse result
        proposal = await self.complete_and_parse(prompt)

        self.config.cycle_count += 1

        return proposal

    async def execute(self, proposal: OneShotAgentActionProposal, user_feedback: str = "") -> ActionResult:
        tool = proposal.use_tool

        # Get commands
        self.commands = await self.run_pipeline(CommandProvider.get_commands)

        # Execute the command
        try:
            command: Optional[Command] = None
            for c in reversed(self.commands):
                if tool.name in c.names:
                    command = c

            if command is None:
                raise AgentException(f"Command {tool.name} not found")

            command_result = command(**tool.arguments)
            if inspect.isawaitable(command_result):
                command_result = await command_result

            result = ActionSuccessResult(outputs=command_result)
        except AgentTerminated:
            result = ActionSuccessResult(outputs="Agent terminated or finished")
        except AgentException as e:
            result = ActionErrorResult.from_exception(e)
            logger.warning(f"{tool} raised an error: {e}")

        await self.run_pipeline(AfterExecute.after_execute, result)

        logger.debug("\n".join(self.trace))

        return result

    async def do_not_execute(
        self, denied_proposal: OneShotAgentActionProposal, user_feedback: str
    ) -> ActionResult:
        result = ActionErrorResult(reason="Action denied")

        await self.run_pipeline(AfterExecute.after_execute, result)

        logger.debug("\n".join(self.trace))

        return result
