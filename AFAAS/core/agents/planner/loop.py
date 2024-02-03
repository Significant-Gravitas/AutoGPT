from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Optional

from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.interfaces.tools.tool_output import ToolOutput
from AFAAS.lib.sdk.errors import AgentException, ToolExecutionError, UnknownToolError
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import Task, TaskStatusList

if TYPE_CHECKING:
    from AFAAS.core.agents.planner.main import PlannerAgent
    from AFAAS.interfaces.adapters import (
        AbstractChatModelResponse,
    )

from AFAAS.interfaces.agent.loop import BaseLoop, BaseLoopHook
from AFAAS.lib.sdk.logger import NOTICE, TRACE, AFAASLogger

LOG = AFAASLogger(name=__name__)


class PlannerLoop(BaseLoop):
    _agent: PlannerAgent

    class LoophooksDict(BaseLoop.LoophooksDict):
        after_plan: Dict[BaseLoopHook]
        after_determine_next_ability: Dict[BaseLoopHook]

    def __init__(self) -> None:
        super().__init__()

        self._active = False
        self.remaining_cycles = 1

    def set_current_task(self, task=Task):
        self._current_task: Task = task

    async def run(
        self,
        agent: PlannerAgent,
        hooks: LoophooksDict,
        user_input_handler: Optional[Callable[[str], Awaitable[str]]] = None,
        user_message_handler: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:

        # current_task = self._current_task
        # # NOTE : Test tools individually
        # command_name = "web_search"
        # command_args= {"query": "instructions for building a Pizza oven"}
        # return_value = await execute_command(
        #     command_name=command_name,
        #     arguments=command_args,
        #     task=current_task,
        #     agent=self._agent,
        # )
        # print(return_value)

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
        # NOTE : Reminicence opf a previous plugg in system to remove once Pipeline implemented
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
            # Note to remove once user_context tool rewritten
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

        ##############################################################
        # NOTE : Important KPI to log during crashes
        ##############################################################
        # Count the number of cycle, usefull to bench for stability before crash or hallunation
        self._loop_count = 0

        # _is_running is important because it avoid having two concurent loop in the same agent (cf : Agent.run())
        current_task = self._current_task
        while self._is_running:
            # if _active is false, then the loop is paused
            # FIXME replace _active by self.remaining_cycles > 0:
            if self._active:
                self._loop_count += 1

                # ##############################################################
                # ### Step 5 : select_tool()
                # ##############################################################
                # if current_task.command is not None:
                #     command_name = current_task.command
                #     command_args = current_task.arguments
                #     assistant_reply_dict = current_task.long_description
                # else:
                #     raise Exception("Honney pot ! In order to ensure we can remove this section of code securely, we raise an exception.")
                #     LOG.error("No command to execute")
                #     (
                #         command_name,
                #         command_args,
                #         assistant_reply_dict,
                #     ) = await self.select_tool()

                ##############################################################
                ### Step 6 : Prepare RAG #
                ##############################################################
                # NOTE : Anayse swap between step 5 and 6
                await current_task.task_preprossessing()

                ##############################################################
                ### Step 7 : execute_tool() #
                ##############################################################
                # current_task.command = command_name
                # current_task.arguments = command_args
                try:
                    tool, result = await current_task.task_execute()
                    # await tool.success_check_callback(self=tool, task=self, tool_output=result)
                except AgentException as e:
                    # FIXME : Implement retry mechanism if a fail
                    result = AgentException(reason=e.message, error=e)
                LOG.debug(f"result : {str(result)}")

                successfull_closure : bool = await current_task.task_postprocessing(tool =tool,
                                                                                tool_output = result)

                LOG.debug(f"successfull_closure : {successfull_closure}")
                LOG.debug(await self.plan().debug_dump_str(depth=2))

                self._current_task = await self.plan().get_next_task(task=current_task)
                await self.save_plan()

                if len(self.plan().get_all_tasks_ids()) == len(
                    self.plan().get_all_done_tasks_ids()
                ):
                    self._is_running = False
                    LOG.info("All tasks are done 😄")
                elif self._current_task is None:
                    self._is_running = False
                    raise AgentException(  "The agent can't find the next task to execute 😱 ! This is an anomaly and we would be working on it." )
                else:
                    self.prepare_next_iteration()

    async def prepare_next_iteration(self):
        LOG.trace(f"Next task : {self._current_task.debug_formated_str()}")
        LOG.info("Task history : (Max. 10 tasks)")
        plan_history: list[Task] = await self.plan().get_last_achieved_tasks(count=10)
        for i, task in enumerate(plan_history):
            LOG.info(
                f"{i+1}.Task : {task.debug_formated_str()} : {getattr(task, 'task_text_output', '')}"
            )

        if LOG.isEnabledFor(TRACE):
            input("Press Enter to continue...")

        LOG.info(f"Task Path : {await self._current_task.get_formated_task_path()}")
        task_path: list[Task] = await self._current_task.get_task_path()
        for i, task in enumerate(task_path):
            LOG.trace(
                f"{i+1}.Task : {task.debug_formated_str()} : {getattr(task, 'task_text_output', '')}"
            )

        if LOG.isEnabledFor(TRACE):
            input("Press Enter to continue...")

        LOG.trace(f"Task Sibblings :")
        task_sibblings: list[Task] = await self._current_task.get_siblings()
        for i, task in enumerate(task_sibblings):
            LOG.trace(
                f"{i+1}.Task : {task.debug_formated_str()} : {getattr(task, 'task_text_output', '')}"
            )

        if LOG.isEnabledFor(TRACE):
            input("Press Enter to continue...")

        LOG.trace(f"Task Predecessor :")
        task_predecessors: list[Task] = (
            await self._current_task.task_predecessors.get_all_tasks_from_stack()
        )
        for i, task in enumerate(task_predecessors):
            LOG.trace(
                f"{i+1}.Task : {task.debug_formated_str()} : {getattr(task, 'task_text_output', '')}"
            )

        if LOG.isEnabledFor(NOTICE):
            input("Press Enter to continue...")

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

    #     # TODO: Should probably do a step to evaluate the quality of the generated tasks,
    #     #  and ensure that they have actionable ready and acceptance criteria

    def __repr__(self):
        return "SimpleLoop()"

    from typing import Any, Literal

    ToolName = str
    ToolArgs = dict[str, str]
    AgentThoughts = dict[str, Any]
    ThoughtProcessOutput = tuple[ToolName, ToolArgs, AgentThoughts]
    from AFAAS.interfaces.adapters.chatmodel import ChatMessage, ChatPrompt

    async def select_tool(
        self,
    ):
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        raw_response: AbstractChatModelResponse = await self._execute_strategy(
            strategy_name="select_tool",
            agent=self._agent,
            tools=self.get_tool_list(),
        )

        command_name = raw_response.parsed_result[0]
        command_args = raw_response.parsed_result[1]
        assistant_reply_dict = raw_response.parsed_result[2]

        LOG.info(
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
    ) -> Any:
        result: any

        current_task.command = command_name
        current_task.arguments = command_args

        try:
            return_value = await current_task.task_execute()

        except AgentException as e:
            # FIXME : Implement retry mechanism if a fail
            return_value = AgentException(reason=e.message, error=e)

        return return_value
