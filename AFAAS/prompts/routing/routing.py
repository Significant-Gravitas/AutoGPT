from __future__ import annotations


from AFAAS.interfaces.adapters.chatmodel import AIMessage , HumanMessage, SystemMessage , ChatMessage

import enum
import os
import uuid
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


class RoutingStrategyFunctionNames(str, enum.Enum):
    EVALUATE_AND_SELECT: str = "afaas_evaluate_and_select"
    MAKE_PLAN: str = "afaas_select_planning"


class RoutingStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: RoutingStrategyFunctionNames = (
        RoutingStrategyFunctionNames.EVALUATE_AND_SELECT
    )
    note_to_agent_length: int = 250
    temperature: float = 0.4


class RoutingStrategy(AbstractPromptStrategy):
    default_configuration = RoutingStrategyConfiguration()
    STRATEGY_NAME = "routing_assess_context"

    ###
    ### PROMPTS
    ###
    def __init__(
        self,
        default_tool_choice: RoutingStrategyFunctionNames,
        temperature: float,  # if coding 0.05
        count=0,
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
        note_to_agent_length: int = 250,
    ):
        """
        Initialize the RoutingStrategy.

        Parameters:
        -----------
        logger: Logger
            The logger object.
        model_classification:  PromptStrategyLanguageModelClassification
            Classification of the language model.
        default_tool_choice: RoutingFunctionNames
            Default function call for the strategy.
        context_min_tokens: int
            Minimum number of tokens in the context.
        context_max_tokens: int
            Maximum number of tokens in the context.
        count: int, optional (default = 0)
            The count for the iterative process.
        user_last_goal: str, optional (default = "")
            Last goal provided by the user.
        exit_token: str, optional
            Token to indicate exit from the process.
        use_message: bool, optional (default = False)
            Flag to determine whether to use messages.
        """
        self._count = count
        self._config = self.default_configuration
        self.note_to_agent_length = note_to_agent_length

    def set_tools(self, **kwargs):
        if "note_to_agent_length" in kwargs:
            self.note_to_agent_length = kwargs["note_to_agent_length"]
        self.function_evaluate_and_select: CompletionModelFunction = (
            CompletionModelFunction(
                name=RoutingStrategyFunctionNames.EVALUATE_AND_SELECT.value,
                description="Determine the most appropriate approach to address the given challenge using the context provided.",
                parameters={
                    "strategy": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        enum=[
                            RoutingStrategyFunctionNames.EVALUATE_AND_SELECT.value,
                            RoutingStrategyFunctionNames.MAKE_PLAN.value,
                        ],
                        description=f"Define the next step. If the value is '{RoutingStrategyFunctionNames.EVALUATE_AND_SELECT.value}' the agent will go for an evaluation phase, if the value is '{RoutingStrategyFunctionNames.EVALUATE_AND_SELECT.value}' make a plan for the given chalenge",
                        required=True,
                    ),
                    "note_to_agent": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"This is a note / tips you can give to the agent, typicaly explains the dificulties the agent would face, the limitations he might want to condider. This note should be {str(self.note_to_agent_length * 0.8)} to {str(self.note_to_agent_length *  1.25)}  words long.",
                        required=True,
                    ),
                    "debug": JSONSchema(
                        type=JSONSchema.Type.INTEGER,
                        description=f" Value between 0 & 100 : Rate your confidence when choosing between '{RoutingStrategyFunctionNames.EVALUATE_AND_SELECT.value}' and '{RoutingStrategyFunctionNames.EVALUATE_AND_SELECT.value}'",
                        required=True,
                    ),
                },
            )
        )

        # function_make_plan = (
        #     RoutingStrategy.FUNCTION_MAKE_PLAN
        # )
        self._tools = [
            self.function_evaluate_and_select,
            # function_make_plan,
        ]

    async def build_message(self, task: AbstractTask, **kwargs) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())

        # Get the directory containing the currently executing script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        file_loader = FileSystemLoader(current_directory)
        env = Environment(loader=file_loader)
        template = env.get_template("10_routing.jinja")

        routing_param = {
            "step": "ROUTING",
            "task": task,
            "additional_context_description": str(task.task_context),
        }
        content = template.render(routing_param)
        messages = [SystemMessage(content)]
        strategy_tools = self.get_tools()

        prompt = ChatPrompt(
            messages=messages,
            tools=strategy_tools,
            tool_choice="auto",
            default_tool_choice=RoutingStrategyFunctionNames.EVALUATE_AND_SELECT,
            # TODO
            tokens_used=0,
        )

        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content)

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
