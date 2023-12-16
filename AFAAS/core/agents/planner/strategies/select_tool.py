from __future__ import annotations

import enum

from typing import TYPE_CHECKING, Callable, Optional

from AFAAS.interfaces.agent.agent_directives import \
    BaseAgentDirectives

if TYPE_CHECKING:
    from AFAAS.core.agents.planner import PlannerAgent


# prompting
from AFAAS.core.lib.action_history import Episode
from AFAAS.interfaces.prompts.strategy import (
    DefaultParsedResponse,  PromptStrategyLanguageModelClassification)
from AFAAS.interfaces.prompts.strategy_planning import (
    PlanningPromptStrategiesConfiguration, PlanningPromptStrategy)
from AFAAS.interfaces.prompts.utils import indent
from AFAAS.core.resource.model_providers import (
    AssistantChatMessageDict, ChatMessage, ChatPrompt, CompletionModelFunction)


class ThinkStrategyFunctionNames(str, enum.Enum):
    THINK: str = "select_tool"


###
### CONFIGURATION
####
class SelectToolStrategyConfiguration(PlanningPromptStrategiesConfiguration):
    model_classification:  PromptStrategyLanguageModelClassification = (
         PromptStrategyLanguageModelClassification.FAST_MODEL_16K
    )
    temperature: float = 0.5


###
### STRATEGY
####
class SelectToolStrategy(PlanningPromptStrategy):
    default_configuration: SelectToolStrategyConfiguration = (
        SelectToolStrategyConfiguration()
    )
    STRATEGY_NAME = "select_tool"

    def __init__(
        self,
        model_classification:  PromptStrategyLanguageModelClassification,
        **kwargs,
    ):
        super().__init__(model_classification=model_classification, **kwargs
        )

    @property
    def model_classification(self) ->  PromptStrategyLanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        agent: "PlannerAgent",
        # instruction: str,
        **kwargs,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary as needed
        3. `cycle_instruction`

        Params:
            cycle_instruction: The final instruction for a select_tooling cycle
        """

        model_name = kwargs["model_name"]
        self._tools = agent._tool_registry.dump_tools()

        ###
        ### To Facilitate merge with AutoGPT changes
        ###
        event_history = False
        include_os_info = True
        del kwargs["tools"]
        tools = self._tools
        agent_directives = BaseAgentDirectives.from_file(agent=agent)
        extra_messages: list[ChatMessage] = []

        system_prompt = self._construct_system_prompt(
            agent=agent,
            agent_directives=agent_directives,
            tools=tools,
            include_os_info=include_os_info,
            **kwargs,
        )
        # system_prompt_tlength = count_message_tokens(ChatMessage.system(system_prompt))

        response_format_instr = self.response_format_instruction(
            agent=agent,
            model_name=model_name,
        )
        extra_messages.append(ChatMessage.system(response_format_instr))

        final_instruction_msg = ChatMessage.user(self._config.choose_action_instruction)
        # final_instruction_tlength = count_message_tokens(final_instruction_msg)

        if event_history:
            progress = self.compile_progress(
                event_history,
                # count_tokens=count_tokens,
                # max_tokens=(
                #     max_prompt_tokens
                #     - system_prompt_tlength
                #     - final_instruction_tlength
                #     - count_message_tokens(extra_messages)
                # ),
            )
            extra_messages.insert(
                0,
                ChatMessage.system(f"## Progress\n\n{progress}"),
            )

        messages = [
            ChatMessage.system(system_prompt),
            *extra_messages,
            final_instruction_msg,
        ]
        # messages: list[ChatMessage] = agent._loop.on_before_select_tool(
        #     messages=messages,
        # )

        # tools = get_openai_command_specs(
        #         agent._tool_registry.list_available_tools(self)
        #     ) ===== agent._tool_registry.dump_tools()
        self._function = agent._tool_registry.dump_tools()
        prompt = ChatPrompt(
            messages=messages,
            tools=self._function,
            tool_choice="auto",
            default_tool_choice="ask_user",
        )

        return prompt

    #
    # response_format_instruction
    #
    def response_format_instruction(self, model_name: str) -> str:
        model_provider = self._agent._chat_model_provider
        return super().response_format_instruction(
            language_model_provider=model_provider, model_name=model_name
        )

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
