from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from AFAAS.core.agents.planner.main import PlannerAgent


from AFAAS.interfaces.adapters.chatmodel import AIMessage , HumanMessage, SystemMessage , ChatMessage
# prompting
from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatPrompt,
)
from AFAAS.interfaces.prompts.strategy import DefaultParsedResponse
from AFAAS.interfaces.prompts.strategy_planning import (
    AbstractPlanningPromptStrategy,
    PlanningPromptStrategiesConfiguration,
)
from AFAAS.interfaces.prompts.utils.utils import (
    indent,
    json_loads,
    to_dotted_list,
    to_md_quotation,
    to_numbered_list,
    to_string_list,
)
from AFAAS.interfaces.task.task import AbstractTask


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

    async def build_message(self, *_, **kwargs) -> ChatPrompt:
        return self.build_prompt(*_, **kwargs)

    async def build_prompt(
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
        del kwargs["tools"]
        self._tools = agent._tool_registry.dump_tools()

        progress = ""  # TODO:""
        response_format_instr = self.response_format_instruction()
        extra_messages: list[ChatMessage] = []
        extra_messages.append(SystemMessage(response_format_instr))
        extra_messages = [msg.content for msg in extra_messages]

        context = {
            "progress": progress,
            "response_format_instr": response_format_instr,
            "extra_messages": extra_messages,
            "final_instruction_msg": self._config.choose_action_instruction,
            "tools": self._tools,
        }

        self._function = agent._tool_registry.dump_tools()

        messages = []
        messages.append(
            SystemMessage(
                await self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=context,
                )
            )
        )
        messages.append(
            SystemMessage(response_format_instr=self.response_format_instruction())
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

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content=response_content)

    def save(self):
        pass
