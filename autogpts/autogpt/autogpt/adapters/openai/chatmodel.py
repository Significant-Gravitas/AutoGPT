import os
from typing import Any, Callable, Dict, ParamSpec, Tuple, TypeVar, Optional

import tiktoken
from openai.resources import AsyncCompletions

from autogpt.adapters.openai.configuration import (
    OPEN_AI_CHAT_MODELS,
    OPEN_AI_DEFAULT_CHAT_CONFIGS,
    OPEN_AI_MODELS,
    OpenAIModelName,
    OpenAIPromptConfiguration,
    OpenAISettings,
)

from langchain_core.messages  import AIMessage , ChatMessage
from autogpt.core.configuration import Configurable
from autogpt.interfaces.adapters.chatmodel.chatmodel import (
    AbstractChatModelProvider,
    AbstractChatModelResponse,
    AssistantChatMessage,
    CompletionModelFunction,
)
from autogpt.interfaces.adapters.language_model import  ModelTokenizer, LanguageModelResponse
import logging

LOG = logging.getLogger(__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")

# Example : LangChain Client
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback, OpenAICallbackHandler

# Example : OAClient : 
# from openai import AsyncOpenAI

class ChatOpenAIAdapter(Configurable[OpenAISettings], AbstractChatModelProvider): 

    # Example : LangChain Client
    callback : Optional[OpenAICallbackHandler] = None
    llm_api_client = ChatOpenAI()

    # Example : OAClient : 
    # llm_model = = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    llmmodel_default : str = "gpt-3.5-turbo"
    llmmodel_fine_tuned : str = "gpt-3.5-turbo"
    llmmodel_cheap : str = "gpt-3.5-turbo"
    llmmodel_code_expert_model : str = "gpt-3.5-turbo"
    llmmodel_long_context_model : str = "gpt-3.5-turbo"



    def __init__(
        self,
        settings: OpenAISettings = OpenAISettings(),
    ):
        super().__init__(settings)
        self._credentials = settings.credentials
        self._budget = settings.budget

    def get_token_limit(self, model_name: str) -> int:
        return OPEN_AI_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        return self._budget.remaining_budget

    @classmethod
    def get_tokenizer(cls, model_name: OpenAIModelName) -> ModelTokenizer:
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: OpenAIModelName) -> int:
        encoding = cls.get_tokenizer(model_name)
        return len(encoding.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
            encoding_model = "gpt-3.5-turbo"
        elif model_name.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
            encoding_model = "gpt-4"
        else:
            raise NotImplementedError(
                f"count_message_tokens() is not implemented for model {model_name}.\n"
                " See https://github.com/openai/openai-python/blob/main/chatml.md for"
                " information on how messages are converted to tokens."
            )
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            LOG.warning(
                f"Model {model_name} not found. Defaulting to cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def extract_response_details(
        self, response: AsyncCompletions, model_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if (isinstance(response, AsyncCompletions)) : 
            response_args = LanguageModelResponse(
                llm_model_info=OPEN_AI_CHAT_MODELS[model_name],
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            #response_message = response.choices[0].message.model_dump()
        elif (isinstance(response, AIMessage)) :
            # AGPT retro compatibility
            response_args = LanguageModelResponse(
                llm_model_info=OPEN_AI_CHAT_MODELS[model_name],
                prompt_tokens= self.callback.prompt_tokens,
                completion_tokens= self.callback.completion_tokens,
            )
            response.base_response = response_args 
        return response

    def should_retry_function_call(
        self, tools: list[CompletionModelFunction], response_message: Dict[str, Any]
    ) -> bool:
        if not tools :
            return False

        if not isinstance(response_message, AIMessage):
            # AGPT retro compatibility
            if "tool_calls" not in response_message:
                return True
        else:
            if "tool_calls" not in response_message.additional_kwargs:
                return True

        return False

    def formulate_final_response(
        self,
        response_message: Dict[str, Any],
        completion_parser: Callable[[AssistantChatMessage], _T],
        **kwargs
    ) -> AbstractChatModelResponse[_T]:

        response_info = response_message.base_response.model_dump()

        response_message_dict = response_message.dict()
        parsed_result = completion_parser(response_message)

        response = AbstractChatModelResponse(
            response=response_message_dict,
            parsed_result=parsed_result,
            **response_info,
        )
        self._budget.update_usage_and_cost(model_response=response)
        return response

    def __repr__(self):
        return "OpenAIProvider()"

    def get_default_config(self) -> OpenAIPromptConfiguration:
        LOG.warning(f"Using {__class__.__name__} default config, we recommend setting individual model configs")
        return OPEN_AI_DEFAULT_CHAT_CONFIGS.SMART_MODEL_32K

    def make_tools_arg(self, tools : list[CompletionModelFunction]) -> dict:
        return { "tools" : [self.make_tool(f) for f in tools] }

    def make_tool(self, f : CompletionModelFunction) -> dict:
        return  {"type": "function", "function": f.schema(schema_builder=self.tool_builder)}

    @staticmethod
    def tool_builder(func: CompletionModelFunction) -> dict[str, str | dict | list]:
            return {
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: param.to_dict() for name, param in func.parameters.items()
                    },
                    "required": [
                        name for name, param in func.parameters.items() if param.required
                    ],
                },
            }

    def make_tool_choice_arg(self , name : str) -> dict:
        return {
            "tool_choice" : {
            "type": "function",
            "function": {"name": name},
            }
        }

    def make_model_arg(self, model_name : str) -> dict:
        return { "model" : model_name }


    async def chat(
        self, messages: list[ChatMessage], *_, **llm_kwargs
    ) -> AsyncCompletions: 

        with get_openai_callback() as callback:
            self.callback : OpenAICallbackHandler = callback
            return await self.llm_api_client.ainvoke(input = messages , **llm_kwargs)

        # Example : OAClient : 
        # return await self.llm_api_client.chat.completions.create( messages=messages, **llm_kwargs )
