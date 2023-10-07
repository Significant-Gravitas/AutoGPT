from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable, List, Dict, Optional
from typing_extensions import TypedDict

from autogpt.core.agents.simple.lib.models.action import (
    ActionHistory,
    ActionResult,
    ActionInterruptedByHuman,
    ActionSuccessResult,
    ActionErrorResult,
)
from autogpt.core.agents.simple.lib.models.context_items import ContextItem
from autogpt.core.tools import ToolOutput
from autogpt.core.agents.simple.lib.models.plan import Plan
from autogpt.core.agents.simple.lib.models.tasks import Task, TaskStatusList

from autogpt.core.tools import ToolResult
from autogpt.core.agents.base import BaseLoop, BaseLoopHook, UserFeedback
from autogpt.core.runner.client_lib.parser import (
    parse_ability_result,
    parse_agent_plan,
    parse_next_ability,
)
from autogpt.core.agents.base.exceptions import (
    AgentException,
    ToolExecutionError,
    InvalidAgentResponseError,
    UnknownToolError,
)


if TYPE_CHECKING:
    from autogpt.core.agents.simple import PlannerAgent, PlannerAgentSettings
    #from autogpt.core.prompting.schema import ChatModelResponse
    from autogpt.core.resource.model_providers import ChatMessage, ChatModelResponse


# NOTE : This is an example of customization that allow to share part of a project in Github while keeping part not released
aaas  =  {}
try:
    from autogpt.core.agents.usercontext import (
        UserContextAgent,
        UserContextAgentSettings,
    )
    aaas["usercontext"] = True
except ImportError:
    aaas["usercontext"] = False
try:
    from autogpt.core.agents.whichway import (
        RoutingAgent,
        RoutingAgentSettings,
    )
    aaas["whichway"] = True
except ImportError:
    aaas["whichway"] = False


aaas["usercontext"] = False
aaas["whichway"] = False

class PlannerLoop(BaseLoop):
    _agent : PlannerAgent
    class LoophooksDict(BaseLoop.LoophooksDict):
        after_plan: Dict[BaseLoopHook]
        after_determine_next_ability: Dict[BaseLoopHook]

    def __init__(self) -> None:
        super().__init__()
        # AgentMixin.__init__()

        self._active = False
        self.remaining_cycles = 1

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

    async def run(
        self,
        agent: PlannerAgent,
        hooks: LoophooksDict,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        if isinstance(user_input_handler, Callable) and user_input_handler is not None:
            self._user_input_handler = user_input_handler
        elif self._user_input_handler is None:
            raise TypeError(
                "`user_input_handler` must be a callable or set previously."
            )

        if (
            isinstance(user_message_handler, Callable)
            and user_message_handler is not None
        ):
            self._user_message_handler = user_message_handler
        elif self._user_message_handler is None:
            raise TypeError(
                "`user_message_handler` must be a callable or set previously."
            )

        ##############################################################
        ### Step 1 : BEGIN WITH A HOOK
        ##############################################################
        await self.handle_hooks(
            hook_key="begin_run",
            hooks=hooks,
            agent=agent,
            user_input_handler=self._user_input_handler,
            user_message_handler=self._user_message_handler,
        )
        ##############################################################
        ### Step 2 : USER CONTEXT AGENT : IF USER CONTEXT AGENT EXIST
        ##############################################################
        if not aaas["usercontext"]:
            self._agent.agent_goals = [self._agent.agent_goal_sentence]
        else:
            agent_goal_sentence, agent_goals = await self.run_user_context_agent()
            self._agent.agent_goal_sentence = agent_goal_sentence
            self._agent.agent_goals = agent_goals

        ##############################################################
        ### Step 3 : USER CONTEXT AGENT : IF USER CONTEXT AGENT EXIST
        ##############################################################
        routing_feedbacks = ''
        if aaas["whichway"]:
            routing_feedbacks = await self.run_whichway_agent()

        ##############################################################
        ### Step 3 : Saving agent with its new goals
        ##############################################################
        await self.save_agent()

        ##############################################################
        ### Step 4 : Start with an plan !
        ##############################################################
        plan = await self.build_initial_plan(routing_feedbacks = routing_feedbacks)
        self._agent.plan = Plan()
        for task in plan['task_list'] :
            self._agent.plan.tasks.append(Task(data = task))

        parse_agent_plan(plan)
        self._agent._logger.info(parse_agent_plan)

        ##############################################################
        # NOTE : Important KPI to log during crashes
        ##############################################################
        # Count the number of cycle, usefull to bench for stability before crash or hallunation
        self._loop_count = 0

        # _is_running is important because it avoid having two concurent loop in the same agent (cf : Agent.run())
        while self._is_running:

            # if _active is false, then the loop is paused
            # FIXME replace _active by self.remaining_cycles > 0:
            if self._active:
                self._loop_count += 1

                ##############################################################
                ### Step 5 : select_tool()
                ##############################################################
                command_name, command_args, assistant_reply_dict = await self.select_tool()

                ##############################################################
                ### Step 6 : execute_tool() #
                ##############################################################
                result = await self.execute_tool(command_name, command_args)

            self.save_agent()

    async def run_user_context_agent(self):
        """
        Configures the user context agent based on the current agent settings and executes the user context agent.
        Returns the updated agent goals.
        """
        
        # USER CONTEXT AGENT : Configure the agent to our context
        usercontextagent_configuration = {
            "user_id": self._agent.user_id,
            "parent_agent_id": self._agent.agent_id,
            "agent_name": "UCC (User Context Checker)",
            "agent_goals": self._agent.agent_goals,
            "agent_goal_sentence": self._agent.agent_goal_sentence,
            "memory": self._agent._memory._settings.dict(),
            "workspace": self._agent._workspace._settings.dict(),
            "openai_provider": self._agent._openai_provider._settings.dict()
            # _type_ = 'autogpt.core.agents.usercontext.main.UserContextAgent',
            # agent_class = 'UserContextAgent'
        }

        # USER CONTEXT AGENT : Create Agent Settings
        usercontext_settings: UserContextAgentSettings = (
            UserContextAgent.compile_settings(
                logger=self._agent._logger,
                user_configuration=usercontextagent_configuration,
            )
        )
        
        # USER CONTEXT AGENT : Save UserContextAgent Settings in DB (for POW / POC)
        new_user_context_agent = UserContextAgent.create_agent(
            agent_settings=usercontext_settings, logger=self._agent._logger
        )

        # USER CONTEXT AGENT : Get UserContextAgent from DB (for POW / POC)
        usercontext_settings.agent_id = new_user_context_agent.agent_id
        user_context_agent = UserContextAgent.get_agent_from_settings(
            agent_settings=usercontext_settings,
            logger=self._agent._logger,
        )

        user_context_return: dict = await user_context_agent.run(
            user_input_handler=self._user_input_handler,
            user_message_handler=self._user_message_handler,
        )
        
        return user_context_return["agent_goal_sentence"], user_context_return["agent_goals"]

    async def run_whichway_agent(): 
        pass

    async def start(
        self,
        agent: PlannerAgent = None,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        await super().start(
            agent=agent,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )

        if not self.remaining_cycles:
            self.remaining_cycles = 1

    async def build_initial_plan(self, routing_feedbacks = '') -> dict:
        # plan =  self.execute_strategy(
        self.tool_registry().list_tools_descriptions()
        plan = await self.execute_strategy(
            strategy_name="make_initial_plan",
            agent_name=self._agent.agent_name,
            agent_role=self._agent.agent_role,
            agent_goals=self._agent.agent_goals,
            agent_goal_sentence=self._agent.agent_goal_sentence,
            routing_feedbacks = routing_feedbacks,
            tools=self.tool_registry().list_tools_descriptions(),
        )

        # TODO: Should probably do a step to evaluate the quality of the generated tasks,
        #  and ensure that they have actionable ready and acceptance criteria

        self.plan = Plan(
            tasks=[Task.parse_obj(task) for task in plan.parsed_result["task_list"]]
        )
        self.plan.tasks.sort(key=lambda t: t.priority, reverse=True)
        self.plan[-1].context.status = TaskStatusList.READY
        return plan.parsed_result

    async def determine_next_ability(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._agent._configuration.cycle_count += 1
        task = self._task_queue.pop()
        self._agent._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        next_ability = await self._choose_next_ability(
            task,
            self.tool_registry().dump_tools(),
        )
        self._current_task = task
        self._next_ability = next_ability.content
        return self._current_task, self._next_ability

    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability = self.tool_registry().get_tool(self._next_ability["next_ability"])
            ability_response = await ability(**self._next_ability["ability_arguments"])
            await self._update_tasks_and_memory(ability_response)
            if self._current_task.context.status == TaskStatusList.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                self._task_queue.append(self._current_task)
            self._current_task = None
            self._next_ability = None

            return ability_response.dict()
        else:
            raise NotImplementedError

    # async def _evaluate_task_and_add_context(self, task: Task) -> Task:
    #     """Evaluate the task and add context to it."""
    #     if task.context.status == TaskStatusList.IN_PROGRESS:
    #         # Nothing to do here
    #         return task
    #     else:
    #         self._agent._logger.debug(
    #             f"Evaluating task {task} and adding relevant context."
    #         )
    #         # TODO: Look up relevant memories (need working memory system)
    #         # TODO: Evaluate whether there is enough information to start the task (language model call).
    #         task.context.enough_info = True
    #         task.context.status = TaskStatusList.IN_PROGRESS
    #         return task

    # async def _choose_next_ability(self, task: Task, ability_schema: list[dict]):
    #     """Choose the next ability to use for the task."""
    #     self._agent._logger.debug(f"Choosing next ability for task {task}.")
    #     if task.context.cycle_count > self._agent._configuration.max_task_cycle_count:
    #         # Don't hit the LLM, just set the next action as "breakdown_task" with an appropriate reason
    #         raise NotImplementedError
    #     elif not task.context.enough_info:
    #         # Don't ask the LLM, just set the next action as "breakdown_task" with an appropriate reason
    #         raise NotImplementedError
    #     else:
    #         next_ability = await self._agent._planning.determine_next_ability(
    #             task, ability_schema
    #         )
    #         return next_ability

    # async def _update_tasks_and_memory(self, ability_result: ToolResult):
    #     self._current_task.context.cycle_count += 1
    #     self._current_task.context.prior_actions.append(ability_result)
    #     # TODO: Summarize new knowledge
    #     # TODO: store knowledge and summaries in memory and in relevant tasks
    #     # TODO: evaluate whether the task is complete

    def __repr__(self):
        return "SimpleLoop()"

    from typing import Literal, Any

    ToolName = str
    ToolArgs = dict[str, str]
    AgentThoughts = dict[str, Any]
    ThoughtProcessOutput = tuple[ToolName, ToolArgs, AgentThoughts]
    from autogpt.core.resource.model_providers.chat_schema import (
        ChatMessage,
        ChatPrompt,
    )

    async def select_tool(
        self,
    ) :
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        raw_response: ChatModelResponse = await self.execute_strategy(
            strategy_name= "select_tool",
            agent=self._agent,
            tools=self.get_tools(),
        )

        command_name = raw_response.parsed_result[0]
        command_args = raw_response.parsed_result[1]
        assistant_reply_dict = raw_response.parsed_result[2]

        self._agent._logger.info(
            (
                f"command_name : {command_name} \n\n"
                + f"command_args : {str(command_args)}\n\n"
                + f"assistant_reply_dict : {str(assistant_reply_dict)}\n\n"
            )
        )
        return command_name, command_args, assistant_reply_dict 

    async def execute_tool(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        if command_name == "humand_feedback":
            result = ActionInterruptedByHuman(user_input)

            # self.log_cycle_handler.log_cycle(
            #     self._agent.agent_name,
            #     self._agent.created_at,
            #     self.cycle_count,
            #     user_input,
            #     USER_INPUT_FILE_NAME,
            # )

        else:
            # for plugin in self.config.plugins:
            #     if not plugin.can_handle_pre_command():
            #         continue
            #     command_name, arguments = plugin.pre_command(command_name, command_args)

            # NOTE : Test tools individually
            command_name =  "web_search" 
            command_args : {'query': 'instructions for building a Pizza oven'}
            try:
                return_value = await execute_command(
                    command_name=command_name,
                    arguments=command_args,
                    agent=self._agent,
                )

                # Intercept ContextItem if one is returned by the command
                if type(return_value) == tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    context_item = return_value[1]
                    return_value = return_value[0]
                    self._agent._logger.debug(
                        f"Tool {command_name} returned a ContextItem: {context_item}"
                    )
                    self.context.add(context_item)

                result = ActionSuccessResult(outputs=return_value)
            except AgentException as e:
                result = ActionErrorResult(reason=e.message, error=e)

            # for plugin in self.config.plugins:
            #     if not plugin.can_handle_post_command():
            #         continue
            #     if result.status == "success":
            #         result.outputs = plugin.post_command(command_name, result.outputs)
            #     elif result.status == "error":
            #         result.reason = plugin.post_command(command_name, result.reason)

        # Check if there's a result from the command append it to the message
        if result.status == "success":
            self._agent.message_history.add(
                "system",
                f"Tool {command_name} returned: {result.outputs}",
                "action_result",
            )
        elif result.status == "error":
            message = f"Tool {command_name} failed: {result.reason}"

            # Append hint to the error message if the exception has a hint
            if (
                result.error
                and isinstance(result.error, AgentException)
                and result.error.hint
            ):
                message = message.rstrip(".") + f". {result.error.hint}"

            self._agent.message_history.add("system", message, "action_result")

        # Update action history
        self._agent.event_history.register_result(result)

        if result.status == "success":
            self._agent._logger.info(result)
        elif result.status == "error":
            self._agent._logger.warn(
                f"Tool {command_name} returned an error: {result.error or result.reason}"
            )

        return result


def execute_command(
    command_name: str,
    arguments: dict[str, str],
    agent: PlannerAgent,
) -> ToolOutput:
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command
        agent (Agent): The agent that is executing the command

    Returns:
        str: The result of the command
    """
    # Execute a native command with the same name or alias, if it exists
    if command := agent._tool_registry.get_tool(tool_name = command_name):
        try:
            return command(**arguments, agent=agent)
        except AgentException:
            raise
        except Exception as e:
            raise ToolExecutionError(str(e))

    raise UnknownToolError(
        f"Cannot execute command '{command_name}': unknown command."
    )
