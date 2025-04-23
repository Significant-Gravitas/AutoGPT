import enum
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import requests
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import SecretStr

from forge.json.parsing import json_loads
from forge.models.config import UserConfigurable

from .._openai_base import BaseOpenAIChatProvider
from ..schema import (
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    ChatModelInfo,
    CompletionModelFunction,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)


class LlamafileModelName(str, enum.Enum):
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct-v0.2"


LLAMAFILE_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=LlamafileModelName.MISTRAL_7B_INSTRUCT,
            provider_name=ModelProviderName.LLAMAFILE,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
    ]
}

LLAMAFILE_EMBEDDING_MODELS = {}


class LlamafileConfiguration(ModelProviderConfiguration):
    # TODO: implement 'seed' across forge.llm.providers
    seed: Optional[int] = None


class LlamafileCredentials(ModelProviderCredentials):
    api_key: Optional[SecretStr] = SecretStr("sk-no-key-required")
    api_base: SecretStr = UserConfigurable(  # type: ignore
        default=SecretStr("http://localhost:8080/v1"), from_env="LLAMAFILE_API_BASE"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class LlamafileSettings(ModelProviderSettings):
    configuration: LlamafileConfiguration  # type: ignore
    credentials: Optional[LlamafileCredentials] = None  # type: ignore


class LlamafileTokenizer(ModelTokenizer[int]):
    def __init__(self, credentials: LlamafileCredentials):
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
            url=f"{self._tokenizer_base_url}/tokenize", json={"content": text}
        )
        response.raise_for_status()
        return response.json()["tokens"]

    def decode(self, tokens: list[int]) -> str:
        response = requests.post(
            url=f"{self._tokenizer_base_url}/detokenize", json={"tokens": tokens}
        )
        response.raise_for_status()
        return response.json()["content"]


class LlamafileProvider(
    BaseOpenAIChatProvider[LlamafileModelName, LlamafileSettings],
    # TODO: add and test support for embedding models
    # BaseOpenAIEmbeddingProvider[LlamafileModelName, LlamafileSettings],
):
    EMBEDDING_MODELS = LLAMAFILE_EMBEDDING_MODELS
    CHAT_MODELS = LLAMAFILE_CHAT_MODELS
    MODELS = {**CHAT_MODELS, **EMBEDDING_MODELS}

    default_settings = LlamafileSettings(
        name="llamafile_provider",
        description=(
            "Provides chat completion and embedding services "
            "through a llamafile instance"
        ),
        configuration=LlamafileConfiguration(),
    )

    _settings: LlamafileSettings
    _credentials: LlamafileCredentials
    _configuration: LlamafileConfiguration

    async def get_available_models(self) -> Sequence[ChatModelInfo[LlamafileModelName]]:
        _models = (await self._client.models.list()).data
        # note: at the moment, llamafile only serves one model at a time (so this
        # list will only ever have one value). however, in the future, llamafile
        # may support multiple models, so leaving this method as-is for now.
        self._logger.debug(f"Retrieved llamafile models: {_models}")

        clean_model_ids = [clean_model_name(m.id) for m in _models]
        self._logger.debug(f"Cleaned llamafile model IDs: {clean_model_ids}")

        return [
            LLAMAFILE_CHAT_MODELS[id]
            for id in clean_model_ids
            if id in LLAMAFILE_CHAT_MODELS
        ]

    def get_tokenizer(self, model_name: LlamafileModelName) -> LlamafileTokenizer:
        return LlamafileTokenizer(self._credentials)

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: LlamafileModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        if model_name == LlamafileModelName.MISTRAL_7B_INSTRUCT:
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
                    message.role == ChatMessage.Role.USER
                    # note that 'system' messages will get converted
                    # to 'user' messages before being sent to the model
                    or message.role == ChatMessage.Role.SYSTEM
                ):
                    # 5 tokens for [INST], [/INST], which actually get
                    # tokenized into "[, INST, ]" and "[, /, INST, ]"
                    # by the mistral tokenizer
                    prompt_added += 5
                elif message.role == ChatMessage.Role.ASSISTANT:
                    assistant_num_added += 1  # for </s>
                else:
                    raise ValueError(
                        f"{model_name} does not support role: {message.role}"
                    )

                ntokens += self.count_tokens(message.content, model_name)

            total_token_count = prompt_added + assistant_num_added + ntokens
            return total_token_count

        else:
            raise NotImplementedError(
                f"count_message_tokens not implemented for model {model_name}"
            )

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: LlamafileModelName,
        functions: list[CompletionModelFunction] | None = None,
        max_output_tokens: int | None = None,
        **kwargs,
    ) -> tuple[
        list[ChatCompletionMessageParam], CompletionCreateParams, dict[str, Any]
    ]:
        messages, completion_kwargs, parse_kwargs = super()._get_chat_completion_args(
            prompt_messages, model, functions, max_output_tokens, **kwargs
        )

        if model == LlamafileModelName.MISTRAL_7B_INSTRUCT:
            messages = self._adapt_chat_messages_for_mistral_instruct(messages)

        if "seed" not in kwargs and self._configuration.seed is not None:
            completion_kwargs["seed"] = self._configuration.seed

        # Convert all messages with content blocks to simple text messages
        for message in messages:
            if isinstance(content := message.get("content"), list):
                message["content"] = "\n\n".join(
                    b["text"]
                    for b in content
                    if b["type"] == "text"
                    # FIXME: add support for images through image_data completion kwarg
                )

        return messages, completion_kwargs, parse_kwargs

    def _adapt_chat_messages_for_mistral_instruct(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        Munge the messages to be compatible with the mistral-7b-instruct chat
        template, which:
        - only supports 'user' and 'assistant' roles.
        - expects messages to alternate between user/assistant roles.

        See details here:
        https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format
        """
        adapted_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            # convert 'system' role to 'user' role as mistral-7b-instruct does
            # not support 'system'
            if message["role"] == ChatMessage.Role.SYSTEM:
                message["role"] = ChatMessage.Role.USER

            if (
                len(adapted_messages) == 0
                or message["role"] != (last_message := adapted_messages[-1])["role"]
            ):
                adapted_messages.append(message)
            else:
                if not message.get("content"):
                    continue

                # if the curr message has the same role as the previous one,
                # concat the current message content to the prev message
                if message["role"] == "user" and last_message["role"] == "user":
                    # user messages can contain other types of content blocks
                    if not isinstance(last_message["content"], list):
                        last_message["content"] = [
                            {"type": "text", "text": last_message["content"]}
                        ]

                    last_message["content"].extend(
                        message["content"]
                        if isinstance(message["content"], list)
                        else [{"type": "text", "text": message["content"]}]
                    )
                elif message["role"] != "user" and last_message["role"] != "user":
                    last_message["content"] = (
                        (last_message.get("content") or "")
                        + "\n\n"
                        + (message.get("content") or "")
                    ).strip()

        return adapted_messages

    def _parse_assistant_tool_calls(
        self,
        assistant_message: ChatCompletionMessage,
        compat_mode: bool = False,
        **kwargs,
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


def clean_model_name(model_file: str) -> str:
    """
    Clean up model names:
    1. Remove file extension
    2. Remove quantization info

    Examples:
    ```
    raw:   'mistral-7b-instruct-v0.2.Q5_K_M.gguf'
    clean: 'mistral-7b-instruct-v0.2'

    raw: '/Users/kate/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf'
    clean:                  'mistral-7b-instruct-v0.2'

    raw:   'llava-v1.5-7b-q4.gguf'
    clean: 'llava-v1.5-7b'
    ```
    """
    name_without_ext = Path(model_file).name.rsplit(".", 1)[0]
    name_without_Q = re.match(
        r"^[a-zA-Z0-9]+([.\-](?!([qQ]|B?F)\d{1,2})[a-zA-Z0-9]+)*",
        name_without_ext,
    )
    return name_without_Q.group() if name_without_Q else name_without_ext


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
