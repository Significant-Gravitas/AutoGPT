from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.task import AbstractTask

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


class AFAAS_SMART_RAGStrategyFunctionNames(str, enum.Enum):
    MAKE_SMART_RAG: str = "afaas_smart_rag"


class AFAAS_SMART_RAGStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: AFAAS_SMART_RAGStrategyFunctionNames = (
        AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG
    )
    task_context_length: int = 300
    temperature: float = 0.4


class AFAAS_SMART_RAG_Strategy(AbstractPromptStrategy):
    default_configuration = AFAAS_SMART_RAGStrategyConfiguration()
    STRATEGY_NAME = "afaas_smart_rag"

    def __init__(
        self,
        default_tool_choice: AFAAS_SMART_RAGStrategyFunctionNames,
        temperature: float,  # if coding 0.05
        # top_p: Optional[float] ,
        # max_tokens : Optional[int] ,
        # frequency_penalty: Optional[float], # Avoid repeting oneselfif coding 0.3
        # presence_penalty : Optional[float], # Avoid certain subjects
        count=0,
        exit_token: str = str(uuid.uuid4()),
        task_context_length: int = 300,
    ):
        self._count = count
        self._config = self.default_configuration
        self.default_tool_choice = default_tool_choice
        self.task_context_length = task_context_length

    def set_tools(
        self,
        task: AbstractTask,
        task_history: list[AbstractTask],
        task_sibblings: list[AbstractTask],
        **kwargs,
    ):
        self.afaas_smart_rag: CompletionModelFunction = CompletionModelFunction(
            name=AFAAS_SMART_RAGStrategyFunctionNames.MAKE_SMART_RAG.value,
            description="Provide accurate information to perform a task",
            parameters={
                "uml_diagrams": JSONSchema(
                    type=JSONSchema.Type.ARRAY,
                    items=JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"A list of the task identified by their IF with UML diagrams relevant to the task",
                        required=True,
                        enum=[task.task_id for task in task_history + task_sibblings],
                    ),
                ),
                "resume": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description=f'Information related to past tasks that can be relevant to the execution of the task {task.task_goal}task but the "situational context" and the "task goal". This note should be {str(self.task_context_length * 0.8)} to {str(self.task_context_length *  1.25)}  words long.',
                    required=True,
                ),
                "long_description": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description=f"Description of the tasks (minimum 80 words long).",
                    required=True,
                ),
            },
        )

        self._tools = [
            self.afaas_smart_rag,
        ]

    def build_message(self, task: AbstractTask, **kwargs) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + task.debug_dump_str())
        self._task: AbstractTask = task
        smart_rag_param = {
            "task_goal": task.task_goal,
            "additional_context_description": str(task.task_context),
            "task_history": kwargs.get("task_history", None),
            "task_sibblings": kwargs.get("task_sibblings", None),
            "task_path": kwargs.get("task_path", None),
            "related_tasks": kwargs.get("related_tasks", None),
        }

        messages = []
        messages.append(
            ChatMessage.system(
                self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=smart_rag_param,
                )
            )
        )
        messages.append(ChatMessage.system(self.response_format_instruction()))

        return self.build_chat_prompt(messages=messages)

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content=response_content)
        # parsed_response : DefaultParsedResponse = self.default_parse_response_content(response_content )
        # parsed_response.command_name
        # self._task.task_context = response_content.get("task_context", None)
        # self._task.task_context = response_content.get("task_context", None)

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
