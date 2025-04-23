import inspect
import logging
from typing import Any, Optional
from uuid import uuid4

from forge.agent.base import BaseAgent, BaseAgentSettings
from forge.agent.protocols import (
    AfterExecute,
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
from forge.llm.providers.schema import AssistantChatMessage, AssistantFunctionCall
from forge.llm.providers.utils import function_specs_from_commands
from forge.models.action import (
    ActionErrorResult,
    ActionProposal,
    ActionResult,
    ActionSuccessResult,
)
from forge.utils.exceptions import AgentException, AgentTerminated

logger = logging.getLogger(__name__)


class ForgeAgent(ProtocolAgent, BaseAgent):
    """
    The goal of the Forge is to take care of the boilerplate code,
    so you can focus on agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possibilities.

    ForgeAgent provides component support; https://docs.agpt.co/classic/forge/components/introduction/
    Using Components is a new way of building agents that is more flexible and easier to extend.
    Components replace some agent's logic and plugins with a more modular and composable system.
    """  # noqa: E501

    def __init__(self, database: AgentDB, workspace: FileStorage):
        """
        The database is used to store tasks, steps and artifact metadata.
        The workspace is used to store artifacts (files).
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

    async def propose_action(self) -> ActionProposal:
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

        prompt: ChatPrompt = ChatPrompt(
            messages=messages, functions=function_specs_from_commands(self.commands)
        )

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")

        # Call the LLM and parse result
        # THIS NEEDS TO BE REPLACED WITH YOUR LLM CALL/LOGIC
        # Have a look at classic/original_autogpt/agents/agent.py
        # for an example (complete_and_parse)
        proposal = ActionProposal(
            thoughts="I cannot solve the task!",
            use_tool=AssistantFunctionCall(
                name="finish", arguments={"reason": "Unimplemented logic"}
            ),
            raw_message=AssistantChatMessage(
                content="finish(reason='Unimplemented logic')"
            ),
        )

        self.config.cycle_count += 1

        return proposal

    async def execute(self, proposal: Any, user_feedback: str = "") -> ActionResult:
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
        self, denied_proposal: Any, user_feedback: str
    ) -> ActionResult:
        result = ActionErrorResult(reason="Action denied")

        await self.run_pipeline(AfterExecute.after_execute, result)

        logger.debug("\n".join(self.trace))

        return result
