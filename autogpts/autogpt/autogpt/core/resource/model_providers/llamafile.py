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

    def _adapt_chat_messages_for_mistral_instruct(
            self,
            messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        Munge the messages to be compatible with the mistral-7b-instruct chat
        template, which:
        - only supports 'user' and 'assistant' roles.
        - expects messages to alternate between user/assistant roles.

        See details here: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format

        """
        adapted_messages = []
        for message in messages:

            # convert 'system' role to 'user' role as mistral-7b-instruct does
            # not support 'system'
            if message["role"] == ChatMessage.Role.SYSTEM:
                message["role"] = ChatMessage.Role.USER

            if len(adapted_messages) == 0:
                adapted_messages.append(message)

            else:
                if message["role"] == adapted_messages[-1]["role"]:
                    # if the curr message has the same role as the previous one,
                    # concat the current message content to the prev message
                    adapted_messages[-1]["content"] += " " + message["content"]
                else:
                    adapted_messages.append(message)

        return adapted_messages

    @overrides
    async def _create_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: OpenAIModelName,
        *_,
        **kwargs,
    ) -> tuple[ChatCompletion, float, int, int]:
        if model == OpenAIModelName.LLAMAFILE_MISTRAL_7B_INSTRUCT:
            messages = self._adapt_chat_messages_for_mistral_instruct(messages)

        if "seed" not in kwargs:
            # TODO: temporarily hard-coded for reproducibility, instead the
            #  seed should be set from config
            kwargs["seed"] = 0

        return await super()._create_chat_completion(messages, model, **kwargs)

    @overrides
    def _parse_assistant_tool_calls(
        self, assistant_message: ChatCompletionMessage, compat_mode: bool = False
    ):
        tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []

        if compat_mode and assistant_message.content:
            try:
                tool_calls = list(
                    _tool_calls_compat_extract_calls(assistant_message.content)
                )
            except Exception as e:
                parse_errors.append(e)

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