from __future__ import annotations

from AFAAS.interfaces.adapters.chatmodel import AIMessage , HumanMessage, SystemMessage , ChatMessage

import enum
import uuid

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


class EvaluateSelectStrategyFunctionNames(str, enum.Enum):
    EVALUATE: str = "afaas_evaluate_and_select"


class EvaluateSelectStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: EvaluateSelectStrategyFunctionNames = (
        EvaluateSelectStrategyFunctionNames.EVALUATE
    )
    note_to_agent_length: int = 800
    temperature: float = 0.75


class EvaluateSelectStrategy(AbstractPromptStrategy):
    default_configuration = EvaluateSelectStrategyConfiguration()
    STRATEGY_NAME = "routing_evaluate"

    ###
    ### PROMPTS
    ###
    def __init__(
        self,
        default_tool_choice: EvaluateSelectStrategyFunctionNames,
        note_to_agent_length: int,
        temperature: float,  # if coding 0.05,
        count=0,
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
    ):
        """
        Initialize the EvaluateSelectStrategy.

        Parameters:
        -----------
        logger: Logger
            The logger object.
        model_classification:  PromptStrategyLanguageModelClassification
            Classification of the language model.
        default_tool_choice: EvaluateSelectFunctionNames
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
        self.default_tool_choice = default_tool_choice

    def set_tools(self, **kwargs):
        if "note_to_agent_length" in kwargs:
            self.note_to_agent_length = "note_to_agent_length"

        self.function_evaluate_and_select: CompletionModelFunction = (
            CompletionModelFunction(
                name=EvaluateSelectStrategyFunctionNames.EVALUATE,
                description="Determine the most appropriate approach to address the given challenge using the context provided.",
                parameters={
                    "note_to_agent": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"This are actionable instructions & tips you can give the Autonomous Agent in charge of writing a plan to solve the given challenge thus helping him to overcome Large Language Models limitations. This note should be {str(self.note_to_agent_length * 0.8)} to {str(self.note_to_agent_length *  1.25)}  words long.",
                        required=True,
                    ),
                },
            )
        )

        self._tools = [
            self.function_evaluate_and_select,
        ]

    async def build_message(self, task: AbstractTask, **kwargs) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())
        self._task: AbstractTask = task
        evaluate_and_select_param = {
            "step": self.STRATEGY_NAME,
            "task": task,
            "additional_context_description": str(task.task_context),
        }

        messages = []
        messages.append(
            SystemMessage(
                await self._build_jinja_message(
                    task=task,
                    template_name=f"20_evaluate_and_select.jinja",
                    template_params=evaluate_and_select_param,
                )
            )
        )
        messages.append(SystemMessage(self.response_format_instruction()))

        return self.build_chat_prompt(messages=messages)

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
