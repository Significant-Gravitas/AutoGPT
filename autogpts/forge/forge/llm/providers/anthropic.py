from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVar

import sentry_sdk
import tenacity
import tiktoken
from anthropic import APIConnectionError, APIStatusError
from pydantic import SecretStr

from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    ChatMessage,
    ChatModelInfo,
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
    ToolResultMessage,
)
from forge.models.config import Configurable, UserConfigurable

from .utils import validate_tool_calls

if TYPE_CHECKING:
    from anthropic.types.beta.tools import MessageCreateParams
    from anthropic.types.beta.tools import ToolsBetaMessage as Message
    from anthropic.types.beta.tools import ToolsBetaMessageParam as MessageParam

_T = TypeVar("_T")
_P = ParamSpec("_P")


class AnthropicModelName(str, enum.Enum):
    CLAUDE3_OPUS_v1 = "claude-3-opus-20240229"
    CLAUDE3_SONNET_v1 = "claude-3-sonnet-20240229"
    CLAUDE3_HAIKU_v1 = "claude-3-haiku-20240307"


ANTHROPIC_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_OPUS_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=15 / 1e6,
            completion_token_cost=75 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_SONNET_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=3 / 1e6,
            completion_token_cost=15 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=AnthropicModelName.CLAUDE3_HAIKU_v1,
            provider_name=ModelProviderName.ANTHROPIC,
            prompt_token_cost=0.25 / 1e6,
            completion_token_cost=1.25 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
    ]
}


class AnthropicConfiguration(ModelProviderConfiguration):
    fix_failed_parse_tries: int = UserConfigurable(3)


class AnthropicCredentials(ModelProviderCredentials):
    """Credentials for Anthropic."""

    api_key: SecretStr = UserConfigurable(from_env="ANTHROPIC_API_KEY")
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="ANTHROPIC_API_BASE_URL"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: (v.get_secret_value() if type(v) is SecretStr else v)
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class AnthropicSettings(ModelProviderSettings):
    configuration: AnthropicConfiguration
    credentials: Optional[AnthropicCredentials]
    budget: ModelProviderBudget


class AnthropicProvider(Configurable[AnthropicSettings], ChatModelProvider):
    default_settings = AnthropicSettings(
        name="anthropic_provider",
        description="Provides access to Anthropic's API.",
        configuration=AnthropicConfiguration(
            retries_per_request=7,
        ),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: AnthropicSettings
    _configuration: AnthropicConfiguration
    _credentials: AnthropicCredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[AnthropicSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not settings:
            settings = self.default_settings.copy(deep=True)
        if not settings.credentials:
            settings.credentials = AnthropicCredentials.from_env()

        super(AnthropicProvider, self).__init__(settings=settings, logger=logger)

        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(**self._credentials.get_api_access_kwargs())

    async def get_available_models(self) -> list[ChatModelInfo]:
        return list(ANTHROPIC_CHAT_MODELS.values())

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return ANTHROPIC_CHAT_MODELS[model_name].max_tokens

    @classmethod
    def get_tokenizer(cls, model_name: AnthropicModelName) -> ModelTokenizer:
        # HACK: No official tokenizer is available for Claude 3
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: AnthropicModelName) -> int:
        return 0  # HACK: No official tokenizer is available for Claude 3

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: AnthropicModelName,
    ) -> int:
        return 0  # HACK: No official tokenizer is available for Claude 3

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: AnthropicModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the Anthropic API."""
        anthropic_messages, completion_kwargs = self._get_chat_completion_args(
            prompt_messages=model_prompt,
            model=model_name,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        total_cost = 0.0
        attempts = 0
        while True:
            completion_kwargs["messages"] = anthropic_messages.copy()
            if prefill_response:
                completion_kwargs["messages"].append(
                    {"role": "assistant", "content": prefill_response}
                )

            (
                _assistant_msg,
                cost,
                t_input,
                t_output,
            ) = await self._create_chat_completion(completion_kwargs)
            total_cost += cost
            self._logger.debug(
                f"Completion usage: {t_input} input, {t_output} output "
                f"- ${round(cost, 5)}"
            )

            # Merge prefill into generated response
            if prefill_response:
                first_text_block = next(
                    b for b in _assistant_msg.content if b.type == "text"
                )
                first_text_block.text = prefill_response + first_text_block.text

            assistant_msg = AssistantChatMessage(
                content="\n\n".join(
                    b.text for b in _assistant_msg.content if b.type == "text"
                ),
                tool_calls=self._parse_assistant_tool_calls(_assistant_msg),
            )

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            attempts += 1
            tool_call_errors = []
            try:
                # Validate tool calls
                if assistant_msg.tool_calls and functions:
                    tool_call_errors = validate_tool_calls(
                        assistant_msg.tool_calls, functions
                    )
                    if tool_call_errors:
                        raise ValueError(
                            "Invalid tool use(s):\n"
                            + "\n".join(str(e) for e in tool_call_errors)
                        )

                parsed_result = completion_parser(assistant_msg)
                break
            except Exception as e:
                self._logger.debug(
                    f"Parsing failed on response: '''{_assistant_msg}'''"
                )
                self._logger.warning(f"Parsing attempt #{attempts} failed: {e}")
                sentry_sdk.capture_exception(
                    error=e,
                    extras={"assistant_msg": _assistant_msg, "i_attempt": attempts},
                )
                if attempts < self._configuration.fix_failed_parse_tries:
                    anthropic_messages.append(
                        _assistant_msg.dict(include={"role", "content"})
                    )
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": [
                                *(
                                    # tool_result is required if last assistant message
                                    # had tool_use block(s)
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tc.id,
                                        "is_error": True,
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Not executed because parsing "
                                                "of your last message failed"
                                                if not tool_call_errors
                                                else str(e)
                                                if (
                                                    e := next(
                                                        (
                                                            tce
                                                            for tce in tool_call_errors
                                                            if tce.name
                                                            == tc.function.name
                                                        ),
                                                        None,
                                                    )
                                                )
                                                else "Not executed because validation "
                                                "of tool input failed",
                                            }
                                        ],
                                    }
                                    for tc in assistant_msg.tool_calls or []
                                ),
                                {
                                    "type": "text",
                                    "text": (
                                        "ERROR PARSING YOUR RESPONSE:\n\n"
                                        f"{e.__class__.__name__}: {e}"
                                    ),
                                },
                            ],
                        }
                    )
                else:
                    raise

        if attempts > 1:
            self._logger.debug(
                f"Total cost for {attempts} attempts: ${round(total_cost, 5)}"
            )

        return ChatModelResponse(
            response=assistant_msg,
            parsed_result=parsed_result,
            model_info=ANTHROPIC_CHAT_MODELS[model_name],
            prompt_tokens_used=t_input,
            completion_tokens_used=t_output,
        )

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: AnthropicModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> tuple[list[MessageParam], MessageCreateParams]:
        """Prepare arguments for message completion API call.

        Args:
            prompt_messages: List of ChatMessages.
            model: The model to use.
            functions: Optional list of functions available to the LLM.
            kwargs: Additional keyword arguments.

        Returns:
            list[MessageParam]: Prompt messages for the Anthropic call
            dict[str, Any]: Any other kwargs for the Anthropic call
        """
        kwargs["model"] = model

        if functions:
            kwargs["tools"] = [
                {
                    "name": f.name,
                    "description": f.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            name: param.to_dict()
                            for name, param in f.parameters.items()
                        },
                        "required": [
                            name
                            for name, param in f.parameters.items()
                            if param.required
                        ],
                    },
                }
                for f in functions
            ]

        kwargs["max_tokens"] = max_output_tokens or 4096

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        system_messages = [
            m for m in prompt_messages if m.role == ChatMessage.Role.SYSTEM
        ]
        if (_n := len(system_messages)) > 1:
            self._logger.warning(
                f"Prompt has {_n} system messages; Anthropic supports only 1. "
                "They will be merged, and removed from the rest of the prompt."
            )
        kwargs["system"] = "\n\n".join(sm.content for sm in system_messages)

        messages: list[MessageParam] = []
        for message in prompt_messages:
            if message.role == ChatMessage.Role.SYSTEM:
                continue
            elif message.role == ChatMessage.Role.USER:
                # Merge subsequent user messages
                if messages and (prev_msg := messages[-1])["role"] == "user":
                    if isinstance(prev_msg["content"], str):
                        prev_msg["content"] += f"\n\n{message.content}"
                    else:
                        assert isinstance(prev_msg["content"], list)
                        prev_msg["content"].append(
                            {"type": "text", "text": message.content}
                        )
                else:
                    messages.append({"role": "user", "content": message.content})
                # TODO: add support for image blocks
            elif message.role == ChatMessage.Role.ASSISTANT:
                if isinstance(message, AssistantChatMessage) and message.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                *(
                                    [{"type": "text", "text": message.content}]
                                    if message.content
                                    else []
                                ),
                                *(
                                    {
                                        "type": "tool_use",
                                        "id": tc.id,
                                        "name": tc.function.name,
                                        "input": tc.function.arguments,
                                    }
                                    for tc in message.tool_calls
                                ),
                            ],
                        }
                    )
                elif message.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content,
                        }
                    )
            elif isinstance(message, ToolResultMessage):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": [{"type": "text", "text": message.content}],
                                "is_error": message.is_error,
                            }
                        ],
                    }
                )

        return messages, kwargs  # type: ignore

    async def _create_chat_completion(
        self, completion_kwargs: MessageCreateParams
    ) -> tuple[Message, float, int, int]:
        """
        Create a chat completion using the Anthropic API with retry handling.

        Params:
            completion_kwargs: Keyword arguments for an Anthropic Messages API call

        Returns:
            Message: The message completion object
            float: The cost ($) of this completion
            int: Number of input tokens used
            int: Number of output tokens used
        """

        @self._retry_api_request
        async def _create_chat_completion_with_retry(
            completion_kwargs: MessageCreateParams,
        ) -> Message:
            return await self._client.beta.tools.messages.create(
                **completion_kwargs  # type: ignore
            )

        response = await _create_chat_completion_with_retry(completion_kwargs)

        cost = self._budget.update_usage_and_cost(
            model_info=ANTHROPIC_CHAT_MODELS[completion_kwargs["model"]],
            input_tokens_used=response.usage.input_tokens,
            output_tokens_used=response.usage.output_tokens,
        )
        return response, cost, response.usage.input_tokens, response.usage.output_tokens

    def _parse_assistant_tool_calls(
        self, assistant_message: Message
    ) -> list[AssistantToolCall]:
        return [
            AssistantToolCall(
                id=c.id,
                type="function",
                function=AssistantFunctionCall(name=c.name, arguments=c.input),
            )
            for c in assistant_message.content
            if c.type == "tool_use"
        ]

    def _retry_api_request(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(APIConnectionError)
                | tenacity.retry_if_exception(
                    lambda e: isinstance(e, APIStatusError) and e.status_code >= 500
                )
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=tenacity.after_log(self._logger, logging.DEBUG),
        )(func)

    def __repr__(self):
        return "AnthropicProvider()"
