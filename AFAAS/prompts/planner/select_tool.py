from __future__ import annotations

import enum
import uuid

from typing import TYPE_CHECKING, Callable, Optional

from AFAAS.interfaces.agent.agent_directives import BaseAgentDirectives

if TYPE_CHECKING:
    from AFAAS.core.agents.planner import PlannerAgent


# prompting
from AFAAS.lib.action_history import Episode

from AFAAS.interfaces.task import AbstractTask
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)

from AFAAS.interfaces.prompts.strategy_planning import (
    PlanningPromptStrategiesConfiguration,
    AbstractPlanningPromptStrategy,
)
from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)


class SelectToolFunctionNames(str, enum.Enum):
    SELECT_TOOL: str = "select_tool"


###
### CONFIGURATION
####
class SelectToolStrategyConfiguration(PlanningPromptStrategiesConfiguration):
    temperature: float = 0.5
    default_tool_choice: SelectToolFunctionNames = "ask_user"


###
### STRATEGY
####
class SelectToolStrategy(AbstractPlanningPromptStrategy):
    default_configuration: SelectToolStrategyConfiguration = (
        SelectToolStrategyConfiguration()
    )
    STRATEGY_NAME = "select_tool"

    def __init__(
        self,
        default_tool_choice: SelectToolFunctionNames,
        note_to_agent_length: int,
        temperature: float,  # if coding 0.05,
        count=0,
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
    ):
        self._count = count
        self._config = self.default_configuration
        self.note_to_agent_length = note_to_agent_length
        self.default_tool_choice = default_tool_choice

    def build_message(self, *_, **kwargs) -> ChatPrompt:
        return self.build_prompt(*_, **kwargs)

    def build_prompt(
        self,
        task: AbstractTask,
        agent: "PlannerAgent",
        # instruction: str,
        **kwargs,
    ) -> ChatPrompt:
        ###
        ### To Facilitate merge with AutoGPT changes
        ###
        event_history = False
        include_os_info = True
        agent_directives = BaseAgentDirectives.from_file(agent=agent)
        del kwargs["tools"]
        self._tools = agent._tool_registry.dump_tools()
        tools = self._tools
        system_prompt = self._construct_system_prompt(
            agent=agent,
            agent_directives=agent_directives,
            tools=tools,
            include_os_info=include_os_info,
            **kwargs,
        )

        progress = (
            self.compile_progress(
                event_history,
            )
            if event_history
            else ""
        )
        response_format_instr = self.response_format_instruction()
        extra_messages: list[ChatMessage] = []
        extra_messages.append(ChatMessage.system(response_format_instr))
        extra_messages = [msg.content for msg in extra_messages]

        context = {
            "system_prompt": system_prompt,
            "progress": progress,
            "response_format_instr": response_format_instr,
            "extra_messages": extra_messages,
            "final_instruction_msg": self._config.choose_action_instruction,
        }

        self._function = agent._tool_registry.dump_tools()

        messages = []
        messages.append(
            ChatMessage.system(
                self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=context,
                )
            )
        )
        messages.append(
            ChatMessage.system(response_format_instr=self.response_format_instruction())
        )

        # prompt = ChatPrompt(
        #     messages=messages,
        #     tools=self._function,
        #     tool_choice="auto",
        #     default_tool_choice="ask_user",
        # )
        return self.build_chat_prompt(messages=messages)

    #
    # response_format_instruction
    #
    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()

    #
    # _generate_intro_prompt
    #
    def _generate_intro_prompt(self, agent: "PlannerAgent", **kargs) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return super()._generate_intro_prompt(agent, **kargs)

    #
    # _generate_os_info
    #
    def _generate_os_info(self, **kwargs) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """

        return super()._generate_os_info(**kwargs)

    #
    #     def _generate_budget_constraint
    #
    def _generate_budget_constraint(self, api_budget: float, **kargs) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        return super()._generate_budget_constraint(api_budget, **kargs)

    #
    # _generate_goals_info
    #
    def _generate_goals_info(self, goals: list[str], **kargs) -> list[str]:
        """Generates the goals information part of the prompt.

        Returns:
            str: The goals information part of the prompt.
        """
        return super()._generate_goals_info(goals, **kargs)

    #
    # _generate_tools_list
    #
    def _generate_tools_list(
        self, tools: list[CompletionModelFunction], **kargs
    ) -> str:
        """Lists the tools available to the agent.

        Params:
            agent: The agent for which the tools are being listed.

        Returns:
            str: A string containing a numbered list of tools.
        """
        return super()._generate_tools_list(tools, **kargs)

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content)

    def save(self):
        pass

    #############
    # Utilities #
    #############
    def extract_command(
        assistant_reply_json: dict,
        assistant_reply: AssistantChatMessageDict,
        use_openai_functions_api: bool,
    ) -> tuple[str, dict[str, str]]:
        super().extract_command(
            assistant_reply_json=assistant_reply_json,
            assistant_reply=assistant_reply,
            use_openai_functions_api=use_openai_functions_api,
        )

    # # NOTE : based on planning_agent.py
    # def construct_base_prompt(
    #     self, agent: "PlannerAgent", **kwargs
    # ) -> list[ChatMessage]:

    #     # Add the current plan to the prompt, if any
    #     if agent.plan:
    #         plan_section = [
    #             "## Plan",
    #             "To complete your task, you have composed the following plan:",
    #         ]
    #         plan_section += [f"{i}. {s}" for i, s in enumerate(agent.plan, 1)]

    #         # Add the actions so far to the prompt
    #         if agent.event_history:
    #             plan_section += [
    #                 "\n### Progress",
    #                 "So far, you have executed the following actions based on the plan:",
    #             ]
    #             for i, cycle in enumerate(agent.event_history, 1):
    #                 if not (cycle.action and cycle.result):
    #                     LOG.warn(f"Incomplete action in history: {cycle}")
    #                     continue

    #                 plan_section.append(
    #                     f"{i}. You executed the command `{cycle.action.format_call()}`, "
    #                     f"which gave the result `{cycle.result}`."
    #                 )

    #         self._prepend_messages.append(ChatMessage.system("\n".join(plan_section)))

    #     messages = super().construct_base_prompt(
    #         agent=agent,  **kwargs
    #     )

    # return messages

    def compile_progress(
        self,
        episode_history: list[Episode],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        start: int = len(episode_history)

        for i, c in reversed(list(enumerate(episode_history))):
            step = f"### Step {i+1}: Executed `{c.action.format_call()}`\n"
            step += f'- **Reasoning:** "{c.action.reasoning}"\n'
            step += (
                f"- **Status:** `{c.result.status if c.result else 'did_not_finish'}`\n"
            )
            if c.result:
                if c.result.status == "success":
                    result = str(c.result)
                    result = "\n" + indent(result) if "\n" in result else result
                    step += f"- **Output:** {result}"
                elif c.result.status == "error":
                    step += f"- **Reason:** {c.result.reason}\n"
                    if c.result.error:
                        step += f"- **Error:** {c.result.error}\n"
                elif c.result.status == "interrupted_by_human":
                    step += f"- **Feedback:** {c.result.feedback}\n"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)
            start = i

        # TODO: summarize remaining

        part = slice(0, start)

        return "\n\n".join(steps)
