from __future__ import annotations

import enum
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


class BaseTaskRagStrategyFunctionNames(str, enum.Enum):
    MAKE_SMART_RAG: str = "afaas_smart_rag"


class BaseTaskRagStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: BaseTaskRagStrategyFunctionNames = (
        BaseTaskRagStrategyFunctionNames.MAKE_SMART_RAG
    )
    task_context_length: int
    temperature: float = 0.4


class BaseTaskRagStrategy(AbstractPromptStrategy):
    def __init__(
        self,
        default_tool_choice: BaseTaskRagStrategyFunctionNames,
        temperature: float,
        count=0,
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
            name=BaseTaskRagStrategyFunctionNames.MAKE_SMART_RAG.value,
            description="Add a new paragraph to the Agent Memo",
            parameters={
                "title": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description=f"Title of the paragraph",
                ),
                "paragraph": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description=f"New paragraph that should be {str(self.task_context_length * 0.8)} to {str(self.task_context_length *  1.25)} words long.",
                    required=True,
                ),
                "uml_diagrams": JSONSchema(
                    type=JSONSchema.Type.ARRAY,
                    items=JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"List of Task that contains UML diagrams that may help the Agent",
                        required=True,
                        enum=[task.task_id for task in task_history],
                    ),
                ),
            },
        )

        self._tools = [
            self.afaas_smart_rag,
        ]

    def build_message(
        self,
        task: AbstractTask,
        task_path: list[AbstractTask] = None,
        task_history: list[AbstractTask] = None,
        task_followup: list[AbstractTask] = None,
        task_sibblings: list[AbstractTask] = None,
        related_tasks: list[AbstractTask] = None,
        **kwargs,
    ) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + task.debug_dump_str())
        self._task: AbstractTask = task
        smart_rag_param = {
            "task_goal": task.task_goal,
            "additional_context_description": str(task.task_context),
            "task_history": task_history,
            "task_sibblings": task_sibblings,
            "task_path": task_path,
            "related_tasks": related_tasks,
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

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
