import logging
from typing import Any, Iterator, Optional, TypeVar

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from overrides import overrides

from autogpt.core.resource.model_providers.openai import (
    OpenAIProvider,
    OpenAIModelName,
    _functions_compat_fix_kwargs
)
from autogpt.core.resource.model_providers.schema import (
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_utils import json_loads

_T = TypeVar("_T")


class LlamafileProvider(OpenAIProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        openai_messages: list[ChatCompletionMessageParam] = [
            message.dict(
                include={"role", "content", "tool_calls", "name"},
                exclude_none=True,
            )
            for message in model_prompt
        ]

        return openai_messages, kwargs

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