import os
from typing import Any, Callable, Dict, ParamSpec, Tuple, TypeVar

import tiktoken
from openai import AsyncOpenAI
from openai.resources import AsyncCompletions

from AFAAS.core.adapters.openai.common import (
    OPEN_AI_CHAT_MODELS,
    OPEN_AI_DEFAULT_CHAT_CONFIGS,
    OPEN_AI_MODELS,
    OpenAIChatMessage,
    OpenAIModelName,
    OpenAIPromptConfiguration,
    OpenAISettings,
)
aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

from AFAAS.configs.schema import Configurable
from AFAAS.interfaces.adapters.chatmodel import (
    _RetryHandler,
    AbstractChatModelProvider,
    AbstractChatModelResponse,
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
    ChatCompletionKwargs
)
from AFAAS.interfaces.adapters.language_model import  ModelTokenizer
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIChatParser = Callable[[str], dict]


class AFAASChatOpenAI(Configurable[OpenAISettings], AbstractChatModelProvider):

    def __init__(
        self,
        settings: OpenAISettings = OpenAISettings(),
    ):
        super().__init__(settings)
        self._credentials = settings.credentials
        self._budget = settings.budget
        self._chat = settings.chat

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
        messages: OpenAIChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, OpenAIChatMessage):
            messages = [messages]

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

    def _extract_response_details(
        self, response: AsyncCompletions, model_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        response_args = {
            "llm_model_info": OPEN_AI_CHAT_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response_message = response.choices[0].message.model_dump()
        return response_message, response_args

    def _should_retry_function_call(
        self, tools: list[CompletionModelFunction], response_message: Dict[str, Any]
    ) -> bool:
        if tools is not None and "tool_calls" not in response_message:
            return True
        return False

    def _formulate_final_response(
        self,
        response_message: Dict[str, Any],
        completion_parser: Callable[[AssistantChatMessageDict], _T],
        response_args: Dict[str, Any],
    ) -> AbstractChatModelResponse[_T]:
        response = AbstractChatModelResponse(
            response=response_message,
            parsed_result=completion_parser(response_message),
            **response_args,
        )
        self._budget.update_usage_and_cost(model_response=response)
        return response

    def __repr__(self):
        return "OpenAIProvider()"

    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        return True # Always True for OpenAI
        #return OPEN_AI_CHAT_MODELS[model_name].has_function_call_api

    def get_default_config(self) -> OpenAIPromptConfiguration:
        return OPEN_AI_DEFAULT_CHAT_CONFIGS.SMART_MODEL_32K


    def make_tool(self, f : CompletionModelFunction) -> dict:
        return  {"type": "function", "function": f.schema}

    def make_tool_choice_arg(self , name : str) -> dict:
        return {
            "tool_choice" : {
            "type": "function",
            "function": {"name": name},
            }
        }

    def make_model_arg(self, model_name : str) -> dict:
        return { "model" : model_name }

    def make_tools_arg(self, tools : list[CompletionModelFunction]) -> dict:
        return { "tools" : [self.make_tool(f) for f in tools] }

    async def chat(
        self, messages: list[ChatMessage], *_, **llm_kwargs
    ) -> AsyncCompletions:
        self.llm_model = aclient.chat
        return await aclient.chat.completions.create(
            messages=messages, **llm_kwargs
        )
