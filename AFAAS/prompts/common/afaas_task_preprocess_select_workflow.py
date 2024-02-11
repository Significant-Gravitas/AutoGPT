from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.task.task import AbstractTask

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.adapters.language_model import AbstractPromptConfiguration
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


class SelectWorkflowStrategyFunctionNames(str, enum.Enum):
    SELECT_WORKFLOW: str = "afaas_task_preprocess_select_workflow"


class SelectWorkflowStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: SelectWorkflowStrategyFunctionNames = (
        SelectWorkflowStrategyFunctionNames.SELECT_WORKFLOW
    )
    task_context_length: int = 150
    temperature: float = 0.4


class AfaasSelectWorkflowStrategy(AbstractPromptStrategy):
    STRATEGY_NAME = "afaas_task_preprocess_select_workflow"
    default_configuration = SelectWorkflowStrategyConfiguration()

    def __init__(
        self,
        default_tool_choice: SelectWorkflowStrategyFunctionNames,
        temperature: float,
        task_context_length: int,
        count=0,
    ):
        self._count = count
        self._config = self.default_configuration
        self.default_tool_choice = default_tool_choice
        self.task_context_length = task_context_length

    def set_tools(
        self,
        task: AbstractTask,
        **kwargs,
    ):
        self.afaas_task_post_rag_update_function: CompletionModelFunction = (
            CompletionModelFunction(
                name=SelectWorkflowStrategyFunctionNames.SELECT_WORKFLOW.value,
                description="Select a workflow",
                parameters={
                    "task_workflow": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"The workflow to be used for the task",
                        required=True,
                        enum=[
                            workflow.name
                            for workflow in self._agent.workflow_registry.workflows.values()
                        ],
                    ),
                    "justifications": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"Explain the reasons that led you to select this workflow specificaly and not select the others",
                    ),
                },
            )
        )

        self._tools = [
            self.afaas_task_post_rag_update_function,
        ]

    async def build_message(
        self,
        task: AbstractTask,
        task_path: list[AbstractTask] = None,
        task_history: list[AbstractTask] = None,
        task_followup: list[AbstractTask] = None,
        task_sibblings: list[AbstractTask] = None,
        related_tasks: list[AbstractTask] = None,
        **kwargs,
    ) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())
        self._task: AbstractTask = task
        smart_rag_param = {
            "task_history": task_history,
            "task_sibblings": task_sibblings,
            "task_path": task_path,
            "related_tasks": related_tasks,
            "workflows": self._agent.workflow_registry.workflows,
        }

        messages = []
        messages.append(
            ChatMessage.system(
                await self._build_jinja_message(
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
