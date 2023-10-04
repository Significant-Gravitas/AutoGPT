from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable, List, Dict, Optional
from typing_extensions import TypedDict

from autogpt.core.planning.models.action import (
    ActionHistory,
    ActionResult,
    ActionInterruptedByHuman,
    ActionSuccessResult,
    ActionErrorResult,
)
from autogpt.core.planning.models.context_items import ContextItem
from autogpt.core.tools import ToolOutput
from autogpt.core.planning.models.plan import Plan
from autogpt.core.planning.models.tasks import Task, TaskStatusList

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
    from autogpt.core.prompting.schema import ChatModelResponse
    from autogpt.core.resource.model_providers import ChatMessage


# NOTE : This is an example of customization that allow to share part of a project in Github while keeping part not released
try:
    from autogpt.core.agents.usercontext import (
        UserContextAgent,
        UserContextAgentSettings,
    )

    usercontext = True
except ImportError:
    usercontext = False


usercontext = False

# from autogpt.logs.log_cycle import (
#     CURRENT_CONTEXT_FILE_NAME,
#     FULL_MESSAGE_HISTORY_FILE_NAME,
#     NEXT_ACTION_FILE_NAME,
#     USER_INPUT_FILE_NAME,
#     LogCycleHandler,
# )


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
        if not usercontext:
            self._agent.agent_goals = [self._agent.agent_goal_sentence]
        else:
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
            # usercontext_settings = UserContextAgent.build_agent_configuration(configuration=agent_settings)

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

            self._agent.agent_goal_sentence = user_context_return["agent_goal_sentence"]
            self._agent.agent_goals = user_context_return["agent_goals"]

        ##############################################################
        ### Step 3 : Saving agent with its new goals
        ##############################################################
        self.save_agent()

        ##############################################################
        ### Step 4 : Start with an plan !
        ##############################################################
        plan = await self.build_initial_plan()
        self._agent.plan = Plan()
        for task in plan['task_list'] :
            self._agent.plan.tasks.append(Task(data = task))


        parse_agent_plan(plan)
        self._agent._logger.info(parse_agent_plan)

        consecutive_failures = 0
               
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
                # logger.debug(f"Cycle budget: {cycle_budget}; remaining: {self.remaining_cycles}")
                self._loop_count += 1

                try:
                    ###
                    ### Step 4 a : think()
                    ###
                    response: ChatModelResponse = await self.think()

                    command_name = response.parsed_result["name"]
                    del response.parsed_result["name"]
                    command_args = response.parsed_result
                    assistant_reply_dict = response.content
                except InvalidAgentResponseError as e:
                    self._agent._logger.warn(f"The agent's thoughts could not be parsed: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        self._agent._logger.error(
                            f"The agent failed to output valid thoughts {consecutive_failures} "
                            "times in a row. Terminating..."
                        )

                # Print the assistant's thoughts and the next command to the user.
                self._agent._logger.info(
                    (
                        f"command_name : {command_name} \n\n"
                        + f"command_args : {str(command_args)}\n\n"
                        + f"assistant_reply_dict : {str(assistant_reply_dict)}\n\n"
                    )
                )

                user_input = ""
                # Get user input if there is no more automated cycles
                if self.remaining_cycles == 1:
                    user_input = await self._user_input_handler(
                        f"Enter y to authorise command, "
                        f"y -N' to run N continuous commands, "
                        f"q to exit program, or enter feedback for "
                        + self._agent.agent_name
                        + "..."
                    )

                    if user_input.lower().strip() == "y":
                        pass
                    else:
                        command_name = "ask_user"

                ###################
                # Execute Tool #
                ###################
                # Decrement the cycle counter first to reduce the likelihood of a SIGINT
                # happening during command execution, setting the cycles remaining to 1,
                # and then having the decrement set it to 0, exiting the application.
                if command_name != "ask_user":
                    self.remaining_cycles -= 1

                if not command_name:
                    continue

                result = await self.execute(command_name, command_args, user_input)

                if result.status == "success":
                    self._agent._logger.info(result)
                elif result.status == "error":
                    self._agent._logger.warn(
                        f"Tool {command_name} returned an error: {result.error or result.reason}"
                    )

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

    async def build_initial_plan(self) -> dict:
        # plan =  self.execute_strategy(
        self.tool_registry().list_tools_descriptions()
        plan = await self.execute_strategy(
            strategy_name="make_initial_plan",
            agent_name=self._agent.agent_name,
            agent_role=self._agent.agent_role,
            agent_goals=self._agent.agent_goals,
            agent_goal_sentence=self._agent.agent_goal_sentence,
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

    async def think(
        self,
    ) :
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        raw_response: ChatModelResponse = await self.execute_strategy(
            strategy_name="think",
            agent=self._agent,
            tools=self.get_tools(),
        )
        return raw_response

    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        if command_name == "ask_user":
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
