from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from AFAAS.interfaces.task.task import AbstractTask

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

from AFAAS.interfaces.adapters.chatmodel import AIMessage , HumanMessage, SystemMessage , ChatMessage
LOG = AFAASLogger(name=__name__)


class UserProxyStrategyFunctionNames(str, enum.Enum):
    ANSWER_WITH_DOCUMENT = "answer_with_document"
    USER_INTERACTION = "user_interaction"
    ANSWER_WITHOUT_DOCUMENT = "answer_without_document"


class UserProxyStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: str = "user_interaction"
    task_context_length: int = 300
    temperature: float = 0.4


class UserProxyStrategy(AbstractPromptStrategy):
    default_configuration = UserProxyStrategyConfiguration()
    STRATEGY_NAME = "user_proxy"

    def __init__(
        self,
        default_tool_choice: UserProxyStrategyFunctionNames,
        temperature: float,
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
        **kwargs,
    ):
        self._tools: list[CompletionModelFunction] = []
        documents = kwargs.get("documents", [])
        user_input_description = ""
        if len(documents) > 0:
            self.answer_with_document: CompletionModelFunction = (
                CompletionModelFunction(
                    name=UserProxyStrategyFunctionNames.ANSWER_WITH_DOCUMENT.value,
                    description="Provide an answer based on documents.",
                    parameters={
                        "answer": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description=f"Provide answer with as many details and justifications as possible. 1. Rephase the problematic ; 2. List options & informations ; 3. Provive a response with justifications ; 4(Optional). List possible chalenges.",
                            required=True,
                        ),
                        "documents": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description=f"List of documents to use for the answer.",
                                required=True,
                                enum=[
                                    doc.metadata["document_id"]
                                    for doc in documents
                                    if "document_id" in doc.metadata.keys()
                                ],
                            ),
                        ),
                    },
                )
            )
            self._tools.append(self.answer_with_document)
            user_input_description = " OR when documents made available to you don't produce a clear information"

        self.answer_without_document: CompletionModelFunction = CompletionModelFunction(
            name=UserProxyStrategyFunctionNames.ANSWER_WITHOUT_DOCUMENT.value,
            description="Provide an answer without documents. ONLY USE FOR TRIVIAL QUESTIONS WITH LOW STAKES AND NO STRATEGIC CONSEQUENCES.",
            parameters={
                "answer": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description=f"Provide answer with as many details and justifications as possible. 1. Rephase the problematic ; 2. List options & informations ; 3. Provive a response with justifications ; 4(Optional). List possible chalenges.",
                    required=True,
                ),
            },
        )
        self._tools.append(self.answer_without_document)

        self.user_interaction = CompletionModelFunction(
            name=UserProxyStrategyFunctionNames.USER_INTERACTION.value,
            description=f"Forward the question to someone else. USE FOR HIGH STAKES QUESTIONS WITH STRATEGIC CONSEQUENCES{user_input_description}.",
            parameters={
                "query": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="The original query enriched with elements from documents made available to you.",
                    required=True,
                )
            },
        )
        self._tools.append(self.user_interaction)

    async def build_message(
        self,
        task: AbstractTask,
        agent: BaseAgent,
        query: str,
        documents: list[Document],
        **kwargs,
    ) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())
        self._task: AbstractTask = task
        query_llm_param = {
            "query": query,
            "documents": documents,
        }

        messages = []
        messages.append(
            SystemMessage(
                await self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=query_llm_param,
                )
            )
        )

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
        # FIXME: Alike code, set a low temparature for this prompt (0.2 ?)
        return super().get_prompt_config()
