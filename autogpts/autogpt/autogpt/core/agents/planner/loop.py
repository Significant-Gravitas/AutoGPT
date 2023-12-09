from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Optional

from pydantic import Field

from autogpts.AFAAS.app.lib.action import (ActionErrorResult, ActionResult,
                                           ActionSuccessResult)
from autogpts.AFAAS.app.lib.context_items import ContextItem
from autogpts.AFAAS.app.lib.task import TaskStatusList
from autogpts.AFAAS.app.lib.task.plan import Plan
from autogpts.AFAAS.app.lib.task.task import Task
from autogpts.autogpt.autogpt.core.agents.base.exceptions import (
    AgentException, ToolExecutionError, UnknownToolError)
from autogpts.autogpt.autogpt.core.tools import ToolOutput

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.agents.planner import PlannerAgent

    # from autogpts.autogpt.autogpt.core.prompting.schema import ChatModelResponse
    from autogpts.autogpt.autogpt.core.resource.model_providers import (
        ChatModelResponse,
    )

from autogpts.autogpt.autogpt.core.agents.base import BaseLoop, BaseLoopHook

from AFAAS.core.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)

# aaas = {}
# try:
#     pass

#     Task.command: Optional[str] = Field(default="afaas_routing")
#     aaas["routing"] = True
# except:
#     aaas["routing"] = False

# try:
#     pass

#     aaas["usercontext"] = True
# except:
#     aaas["usercontext"] = False

# # FIXME: Deactivated for as long as we don't have the UI to support it
# aaas["usercontext"] = False
# # aaas['routing'] = False


class PlannerLoop(BaseLoop):
    _agent: PlannerAgent

    class LoophooksDict(BaseLoop.LoophooksDict):
        after_plan: Dict[BaseLoopHook]
        after_determine_next_ability: Dict[BaseLoopHook]

    def __init__(self) -> None:
        super().__init__()
        # AgentMixin.__init__()

        self._active = False
        self.remaining_cycles = 1

    def set_current_task(self, task=Task):
        self._current_task: Task = task

    """
    def add_initial_tasks(self):
        ###
        ### Step 1 : add routing to the tasks
        ###
        if aaas["routing"]:
            initial_task = Task(
                # task_parent = self.plan() ,
                task_parent_id=None,
                task_predecessor_id=None,
                responsible_agent_id=None,
                # task_goal="Define an agent approach to tackle a tasks",
                task_goal= self._agent.agent_goal_sentence,
                command="afaas_routing",
                arguments = {'note_to_agent_length' : 400},
                acceptance_criteria=[
                    "A plan has been made to achieve the specific task"
                ],
                state=TaskStatusList.READY,
            )
        else:
            initial_task = Task(
                # task_parent = self.plan() ,
                task_parent_id=None,
                task_predecessor_id=None,
                responsible_agent_id=None,
                # task_goal="Make a plan to tacke a tasks",
                task_goal= self._agent.agent_goal_sentence,
                command="afaas_make_initial_plan",
                arguments={},
                acceptance_criteria=[
                    "Contextual information related to the task has been provided"
                ],
                state=TaskStatusList.READY,
            )

        self._current_task = initial_task  # .task_id
        initial_task_list = [initial_task]

        ###
        ### Step 2 : Prepend usercontext
        ###
        if aaas["usercontext"]:
            refine_user_context_task = Task(
                # task_parent = self.plan() ,
                task_parent_id=None,
                task_predecessor_id=None,
                responsible_agent_id=None,
                name="afaas_refine_user_context",
                task_goal="Refine a user requirements for better exploitation by Agents",
                command="afaas_refine_user_context",
                arguments={},
                state=TaskStatusList.READY,
            )
            initial_task_list = [refine_user_context_task] + initial_task_list
            self._current_task = refine_user_context_task  # .task_id
        else:
            self._agent.agent_goals = [self._agent.agent_goal_sentence]

        self._current_task_routing_description = ""
        self._current_task_routing_feedbacks = ""
        self.plan().add_tasks(tasks=initial_task_list, agent=self._agent)
    """

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

        if self.plan() is None and False:
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
            ### Step 3 : Define the approach
            ##############################################################
            routing_feedbacks = ""
            description = ""
            if aaas["routing"]:
                description, routing_feedbacks = await self.run_routing_agent()

            ##############################################################
            ### Step 4 : Saving agent with its new goals
            ##############################################################
            await self.save_agent()

            ##############################################################
            ### Step 5 : Make an plan !
            ##############################################################
            llm_response = await self.build_initial_plan(
                description=description, routing_feedbacks=routing_feedbacks
            )
            # self._agent.plan = Plan()
            # for task in plan['task_list'] :
            #     self._agent.plan.tasks.append(Task(data = task))

            # Debugging :)
            self._agent._logger.info(Plan.info_parse_task(self._agent.plan))

            ###
            ### Assign task
            ###

        ##############################################################
        # NOTE : Important KPI to log during crashes
        ##############################################################
        # Count the number of cycle, usefull to bench for stability before crash or hallunation
        self._loop_count = 0

        # _is_running is important because it avoid having two concurent loop in the same agent (cf : Agent.run())
        current_task =  self._current_task 
        while self._is_running:
            # if _active is false, then the loop is paused
            # FIXME replace _active by self.remaining_cycles > 0:
            if self._active:
                self._loop_count += 1

                ##############################################################
                ### Step 5 : select_tool()
                ##############################################################
                if current_task.command is not None:
                    command_name = current_task.command
                    command_args = current_task.arguments
                    assistant_reply_dict = current_task.long_decription
                else:
                    LOG.error("No command to execute")
                    (
                        command_name,
                        command_args,
                        assistant_reply_dict,
                    ) = await self.select_tool()



                ##############################################################
                ### Step 6 : execute_tool() #
                ##############################################################
                result = await self.execute_tool(
                    command_name=command_name,
                    command_args=command_args,
                    current_task=current_task
                    # user_input = assistant_reply_dict
                )
            
            self._current_task = self.plan().get_next_task(task = current_task)

            self.save_plan()

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

    # async def build_initial_plan(self, description="", routing_feedbacks="") -> dict:
    #     # plan =  self.execute_strategy(
    #     self.tool_registry().list_tools_descriptions()
    #     plan = await self.execute_strategy(
    #         strategy_name="make_initial_plan",
    #         agent_name=self._agent.agent_name,
    #         agent_role=self._agent.agent_role,
    #         agent_goals=self._agent.agent_goals,
    #         agent_goal_sentence=self._agent.agent_goal_sentence,
    #         description=description,
    #         routing_feedbacks=routing_feedbacks,
    #         tools=self.tool_registry().list_tools_descriptions(),
    #     )

    #     # TODO: Should probably do a step to evaluate the quality of the generated tasks,
    #     #  and ensure that they have actionable ready and acceptance criteria

    #     self._agent.plan = Plan(
    #         tasks=[Task.parse_obj(task) for task in plan.parsed_result["task_list"]]
    #     )
    #     self._agent.plan.tasks.sort(key=lambda t: t.priority, reverse=True)
    #     self._agent.current_task = self._agent.plan[-1]
    #     self._agent.current_task.context.status = TaskStatusList.READY
    #     return plan

    def __repr__(self):
        return "SimpleLoop()"

    from typing import Any, Literal

    ToolName = str
    ToolArgs = dict[str, str]
    AgentThoughts = dict[str, Any]
    ThoughtProcessOutput = tuple[ToolName, ToolArgs, AgentThoughts]
    from autogpts.autogpt.autogpt.core.resource.model_providers.chat_schema import (
        ChatMessage, ChatPrompt)

    async def select_tool(
        self,
    ):
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        raw_response: ChatModelResponse = await self._execute_strategy(
            strategy_name="select_tool",
            agent=self._agent,
            tools=self.get_tool_list(),
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
        current_task: Task,
        command_args: dict[str, str] = {},
    ) -> ActionResult:
        result: ActionResult

        # NOTE : Test tools individually
        # command_name = "web_search"
        # command_args: {"query": "instructions for building a Pizza oven"}

        try:
            return_value = await execute_command(
                command_name=command_name,
                arguments=command_args,
                task=current_task,
                agent=self._agent,
            )

            # # Intercept ContextItem if one is returned by the command
            # if type(return_value) == tuple and isinstance(return_value[1], ContextItem):
            #     context_item = return_value[1]
            #     return_value = return_value[0]
            #     self._agent._logger.debug(
            #         f"Tool {command_name} returned a ContextItem: {context_item}"
            #     )
            #     self.context.add(context_item)

            # result = ActionSuccessResult(outputs=return_value)
        except AgentException as e:
            #FIXME : Implement retry mechanism if a fail
            return_value = ActionErrorResult(reason=e.message, error=e)

        current_task.state = TaskStatusList.DONE
        #self.plan().set_task_status(task= current_task, status= TaskStatusList.DONE)

        return return_value

        ###
        ### TODO : Low priority : Save results of tool execution & manage errors
        ###

        # # Check if there's a result from the command append it to the message
        # if result.status == "success":
        #     ###
        #     ### NOT IMPLEMENTED : Save the result of all tool execution
        #     ###
        #     self._agent.tool_result_history = []
        #     def add_result_history(command_name, result) :
        #         # TODO : Implement tool result history
        #         self._agent.tool_result_history = self._agent.tool_result_history_table.list()
        #         self._agent.tool_result_history_table = self.get_table("tool_result_history")
        #         self._agent.tool_result_history_table.add(
        #             "system",
        #             f"Tool {command_name} returned: {result.outputs}",
        #             "action_result",
        #         )
        #     self._agent.tool_result_history.append(command_name, result)

        # elif result.status == "error":
        #     message = f"Tool {command_name} failed: {result.reason}"

        #     # Append hint to the error message if the exception has a hint
        #     if (
        #         result.error
        #         and isinstance(result.error, AgentException)
        #         and result.error.hint
        #     ):
        #         message = message.rstrip(".") + f". {result.error.hint}"

        #     self._agent.message_history.add("system", message, "action_result")

        if result.status == "success":
            self._agent._logger.info(result)
        elif result.status == "error":
            self._agent._logger.warn(
                f"Tool {command_name} returned an error: {result.error or result.reason}"
            )

        self.plan().set_task_status(task=current_task, status=TaskStatusList.DONE.value)

        return result


def execute_command(
    command_name: str, arguments: dict[str, str], agent: PlannerAgent, task: Task
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
    agent._logger.info(f"Executing command : {command_name}")
    agent._logger.info(f"with arguments : {arguments}")
    if tool := agent._tool_registry.get_tool(tool_name=command_name):
        try:
            result = tool(**arguments, task=task, agent=agent)
            tool.success_check_callback(task=task, tool_output=result)
            return result
        except AgentException:
            raise
        except Exception as e:
            raise ToolExecutionError(str(e))

    raise UnknownToolError(f"Cannot execute command '{command_name}': unknown command.")
