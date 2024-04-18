import logging
from typing import Any, Iterator, Optional, TypeVar

import requests
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from overrides import overrides

from autogpt.core.resource.model_providers.openai import (
    OpenAICredentials,
    OpenAIProvider,
    OpenAIModelName,
    _functions_compat_fix_kwargs
)
from autogpt.core.resource.model_providers.schema import (
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    CompletionModelFunction,
    ModelTokenizer,
)
from autogpt.core.utils.json_utils import json_loads

_T = TypeVar("_T")


class LlamafileTokenizer(ModelTokenizer):

    def __init__(self, credentials: OpenAICredentials):
        self._credentials = credentials

    @property
    def _tokenizer_base_url(self):
        # The OpenAI-chat-compatible base url should look something like
        # 'http://localhost:8080/v1' but the tokenizer endpoint is
        # 'http://localhost:8080/tokenize'. So here we just strip off the '/v1'.
        api_base = self._credentials.api_base.get_secret_value()
        return api_base.strip("/v1")

    def encode(self, text: str) -> list[int]:
        response = requests.post(
            url=f"{self._tokenizer_base_url}/tokenize",
            json={"content": text}
        )
        response.raise_for_status()
        return response.json()["tokens"]

    def decode(self, tokens: list[int]) -> str:
        response = requests.post(
            url=f"{self._tokenizer_base_url}/detokenize",
            json={"tokens": tokens}
        )
        response.raise_for_status()
        return response.json()["content"]


class LlamafileProvider(OpenAIProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def get_tokenizer(self, model_name: OpenAIModelName) -> ModelTokenizer:
        return LlamafileTokenizer(self._credentials)

    @overrides
    def count_tokens(self, text: str, model_name: OpenAIModelName) -> int:
        return len(self.get_tokenizer(model_name).encode(text))

    @overrides
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        if model_name == OpenAIModelName.LLAMAFILE_MISTRAL_7B_INSTRUCT:
            # For mistral-instruct, num added tokens depends on if the message
            # is a prompt/instruction or an assistant-generated message.
            # - prompt gets [INST], [/INST] added and the first instruction
            # begins with '<s>' ('beginning-of-sentence' token).
            # - assistant-generated messages get '</s>' added
            # see: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
            #
            prompt_added = 1  # one for '<s>' token
            assistant_num_added = 0
            ntokens = 0
            for message in messages:
                # if message.role == ChatMessage.Role.SYSTEM:
                #     raise ValueError(f"{model_name} does not support 'system' role")
                if (
                        message.role == ChatMessage.Role.USER or
                        message.role == ChatMessage.Role.SYSTEM  # note that 'system' messages will get converted to 'user' messages before being sent to the model
                ):
                    # 5 tokens for [INST], [/INST], which actually get
                    # tokenized into "[, INST, ]" and "[, /, INST, ]"
                    # by the mistral tokenizer
                    prompt_added += 5
                elif message.role == ChatMessage.Role.ASSISTANT:
                    assistant_num_added += 1  # for </s>
                else:
                    raise ValueError(f"{model_name} does not support role: {message.role}")

                ntokens += self.count_tokens(message.content, model_name)

            total_token_count = prompt_added + assistant_num_added + ntokens
            return total_token_count

        else:
            raise NotImplementedError(f"count_message_tokens not implemented for model {model_name}")

    @overrides
    def _get_chat_completion_args(
            self,
            model_prompt: list[ChatMessage],
            model_name: OpenAIModelName,
            functions: Optional[list[CompletionModelFunction]] = None,
            **kwargs,
    ) -> tuple[list[ChatCompletionMessageParam], dict[str, Any]]:
        """Prepare chat completion arguments and keyword arguments for API call.

        Args:
            model_prompt: List of ChatMessages.
            model_name: The model to use.
            functions: Optional list of functions available to the LLM.
            kwargs: Additional keyword arguments.

        Returns:
            list[ChatCompletionMessageParam]: Prompt messages for the OpenAI call
            dict[str, Any]: Any other kwargs for the OpenAI call
        """
        kwargs.update(self._credentials.get_model_access_kwargs(model_name))

        if functions:
            _functions_compat_fix_kwargs(functions, kwargs)

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        if "messages" in kwargs:
            model_prompt += kwargs["messages"]
            del kwargs["messages"]

        if model_name == OpenAIModelName.LLAMAFILE_MISTRAL_7B_INSTRUCT:
            model_prompt = self._adapt_chat_messages_for_mistral_instruct(model_prompt)

        openai_messages: list[ChatCompletionMessageParam] = [
            message.dict(
                include={"role", "content", "tool_calls", "name"},
                exclude_none=True,
            )
            for message in model_prompt
        ]

        return openai_messages, kwargs

    def _adapt_chat_messages_for_mistral_instruct(
            self,
            messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        """
        Munge the messages to be compatible with the mistral-7b-instruct chat
        template, which:
        - only supports 'user' and 'assistant' roles.
        - expects messages to alternate between user/assistant roles.

        See details here: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format

        """
        adapted_messages = []
        for m in messages:
            if m.role == ChatMessage.Role.SYSTEM:
                # convert to 'user' role
                adapted_messages.append(ChatMessage.user(m.content))
            else:
                adapted_messages.append(m)

        # if there are multiple adjacent user messages, glom them together
        # into a single user message
        glommed = []
        i = 0
        while i < len(adapted_messages):
            if len(glommed) == 0:
                glommed.append(adapted_messages[i])
            elif adapted_messages[i].role != glommed[-1].role:
                glommed.append(adapted_messages[i])
            else:
                glommed[-1].content += " " + adapted_messages[i].content
            i += 1

        return glommed

    @overrides
    async def _create_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: OpenAIModelName,
        *_,
        **kwargs,
    ) -> tuple[ChatCompletion, float, int, int]:
        if model == OpenAIModelName.LLAMAFILE_MISTRAL_7B_INSTRUCT:
            # validate that all messages have roles that are supported by
            # mistral-7b-instruct
            for m in messages:
                if m["role"] not in [
                    ChatMessage.Role.USER,
                    ChatMessage.Role.ASSISTANT
                ]:
                    raise ValueError(f"Role {m['role']} not supported by model {model}")

        return await super()._create_chat_completion(messages, model, **kwargs)

    @overrides
    def _parse_assistant_tool_calls(
        self, assistant_message: ChatCompletionMessage, compat_mode: bool = False
    ):
        if not assistant_message.content:
            raise ValueError("Assistant message content is empty")
        if not compat_mode:
            raise ValueError("compat_mode must be enabled for LlamafileProvider")

        tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []

        for tool_call in _tool_calls_compat_extract_calls(assistant_message.content):
            tool_calls.append(tool_call)

        # try:
        #     tool_calls = list(
        #         _tool_calls_compat_extract_calls(assistant_message.content)
        #     )
        # except Exception as e:
        #     parse_errors.append(e)

        return tool_calls, parse_errors


def _tool_calls_compat_extract_calls(response: str) -> Iterator[AssistantToolCall]:
    import re
    import uuid

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    response = response.strip()  # strip off any leading/trailing whitespace
    if response.startswith("```"):
        # attempt to remove any extraneous markdown artifacts like "```json"
        response = response.strip("```")
        if response.startswith("json"):
            response = response.strip("json")
        response = response.strip()  # any remaining whitespace

    if response[0] == "[":
        tool_calls: list[AssistantToolCallDict] = json_loads(response)
    else:
        block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find tool_calls block in response")
        tool_calls: list[AssistantToolCallDict] = json_loads(block.group(1))

    for t in tool_calls:
        t["id"] = str(uuid.uuid4())
        # t["function"]["arguments"] = str(t["function"]["arguments"])  # HACK

        yield AssistantToolCall.parse_obj(t)