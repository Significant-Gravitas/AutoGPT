import enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, ParamSpec, TypeVar, cast

import tenacity
import tiktoken
import yaml
from openai._exceptions import APIStatusError, RateLimitError
from openai.types import EmbeddingCreateParams
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import SecretStr

from forge.json.parsing import json_loads
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema

from ._openai_base import BaseOpenAIChatProvider, BaseOpenAIEmbeddingProvider
from .schema import (
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    ChatModelInfo,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]


class OpenAIModelName(str, enum.Enum):
    EMBEDDING_v2 = "text-embedding-ada-002"
    EMBEDDING_v3_S = "text-embedding-3-small"
    EMBEDDING_v3_L = "text-embedding-3-large"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_v3 = "gpt-3.5-turbo-1106"
    GPT3_v4 = "gpt-3.5-turbo-0125"
    GPT3_ROLLING = "gpt-3.5-turbo"
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k"
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_v3 = "gpt-4-1106-preview"
    GPT4_v3_VISION = "gpt-4-1106-vision-preview"
    GPT4_v4 = "gpt-4-0125-preview"
    GPT4_v5 = "gpt-4-turbo-2024-04-09"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4_O_v1 = "gpt-4o-2024-05-13"
    GPT4_O_ROLLING = "gpt-4o"
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k
    GPT4_O = GPT4_O_ROLLING


OPEN_AI_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v2,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0001 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v3_S,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.00002 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v3_L,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
    ]
}


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v1,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v2_16k,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v3,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v4,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0005 / 1000,
            completion_token_cost=0.0015 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_v1,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_v1_32k,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_TURBO,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_O,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=5 / 1_000_000,
            completion_token_cost=15 / 1_000_000,
            max_tokens=128_000,
            has_function_call_api=True,
        ),
    ]
}
# Copy entries for models with equivalent specs
chat_model_mapping = {
    OpenAIModelName.GPT3_v1: [OpenAIModelName.GPT3_v2],
    OpenAIModelName.GPT3_v2_16k: [OpenAIModelName.GPT3_16k],
    OpenAIModelName.GPT3_v4: [OpenAIModelName.GPT3_ROLLING],
    OpenAIModelName.GPT4_v1: [OpenAIModelName.GPT4_v2, OpenAIModelName.GPT4_ROLLING],
    OpenAIModelName.GPT4_v1_32k: [
        OpenAIModelName.GPT4_v2_32k,
        OpenAIModelName.GPT4_32k,
    ],
    OpenAIModelName.GPT4_TURBO: [
        OpenAIModelName.GPT4_v3,
        OpenAIModelName.GPT4_v3_VISION,
        OpenAIModelName.GPT4_VISION,
        OpenAIModelName.GPT4_v4,
        OpenAIModelName.GPT4_TURBO_PREVIEW,
        OpenAIModelName.GPT4_v5,
    ],
    OpenAIModelName.GPT4_O: [OpenAIModelName.GPT4_O_v1],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = OPEN_AI_CHAT_MODELS[base].model_copy(update={"name": copy})
        OPEN_AI_CHAT_MODELS[copy] = copy_info
        if copy.endswith(("-0301", "-0314")):
            copy_info.has_function_call_api = False


OPEN_AI_MODELS: Mapping[
    OpenAIModelName,
    ChatModelInfo[OpenAIModelName] | EmbeddingModelInfo[OpenAIModelName],
] = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAICredentials(ModelProviderCredentials):
    """Credentials for OpenAI."""

    api_key: SecretStr = UserConfigurable(from_env="OPENAI_API_KEY")  # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_API_BASE_URL"
    )
    organization: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_ORGANIZATION"
    )

    api_type: Optional[SecretStr] = UserConfigurable(
        default=None,
        from_env=lambda: cast(
            SecretStr | None,
            (
                "azure"
                if os.getenv("USE_AZURE") == "True"
                else os.getenv("OPENAI_API_TYPE")
            ),
        ),
    )
    api_version: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_API_VERSION"
    )
    azure_endpoint: Optional[SecretStr] = None
    azure_model_to_deploy_id_map: Optional[dict[str, str]] = None

    def get_api_access_kwargs(self) -> dict[str, str]:
        kwargs = {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
                "organization": self.organization,
                "api_version": self.api_version,
            }.items()
            if v is not None
        }
        if self.api_type == SecretStr("azure"):
            assert self.azure_endpoint, "Azure endpoint not configured"
            kwargs["azure_endpoint"] = self.azure_endpoint.get_secret_value()
        return kwargs

    def get_model_access_kwargs(self, model: str) -> dict[str, str]:
        kwargs = {"model": model}
        if self.api_type == SecretStr("azure") and model:
            azure_kwargs = self._get_azure_access_kwargs(model)
            kwargs.update(azure_kwargs)
        return kwargs

    def load_azure_config(self, config_file: Path) -> None:
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.SafeLoader) or {}

        try:
            assert config_params.get(
                "azure_model_map", {}
            ), "Azure model->deployment_id map is empty"
        except AssertionError as e:
            raise ValueError(*e.args)

        self.api_type = config_params.get("azure_api_type", "azure")
        self.api_version = config_params.get("azure_api_version", None)
        self.azure_endpoint = config_params.get("azure_endpoint")
        self.azure_model_to_deploy_id_map = config_params.get("azure_model_map")

    def _get_azure_access_kwargs(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""

        if not self.azure_model_to_deploy_id_map:
            raise ValueError("Azure model deployment map not configured")

        if model not in self.azure_model_to_deploy_id_map:
            raise ValueError(f"No Azure deployment ID configured for model '{model}'")
        deployment_id = self.azure_model_to_deploy_id_map[model]

        return {"model": deployment_id}


class OpenAISettings(ModelProviderSettings):
    credentials: Optional[OpenAICredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore


class OpenAIProvider(
    BaseOpenAIChatProvider[OpenAIModelName, OpenAISettings],
    BaseOpenAIEmbeddingProvider[OpenAIModelName, OpenAISettings],
):
    MODELS = OPEN_AI_MODELS
    CHAT_MODELS = OPEN_AI_CHAT_MODELS
    EMBEDDING_MODELS = OPEN_AI_EMBEDDING_MODELS

    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=ModelProviderConfiguration(),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: OpenAISettings
    _configuration: ModelProviderConfiguration
    _credentials: OpenAICredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[OpenAISettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(OpenAIProvider, self).__init__(settings=settings, logger=logger)

        if self._credentials.api_type == SecretStr("azure"):
            from openai import AsyncAzureOpenAI

            # API key and org (if configured) are passed, the rest of the required
            # credentials is loaded from the environment by the AzureOpenAI client.
            self._client = AsyncAzureOpenAI(
                **self._credentials.get_api_access_kwargs()  # type: ignore
            )
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                **self._credentials.get_api_access_kwargs()  # type: ignore
            )

    def get_tokenizer(self, model_name: OpenAIModelName) -> ModelTokenizer[int]:
        return tiktoken.encoding_for_model(model_name)

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        # TODO: check if this is still valid for gpt-4o
        elif model_name.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"count_message_tokens() is not implemented for model {model_name}.\n"
                "See https://github.com/openai/openai-python/blob/120d225b91a8453e15240a49fb1c6794d8119326/chatml.md "  # noqa
                "for information on how messages are converted to tokens."
            )
        tokenizer = self.get_tokenizer(model_name)

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.model_dump().items():
                num_tokens += len(tokenizer.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: OpenAIModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> tuple[
        list[ChatCompletionMessageParam], CompletionCreateParams, dict[str, Any]
    ]:
        """Prepare keyword arguments for an OpenAI chat completion call

        Args:
            prompt_messages: List of ChatMessages
            model: The model to use
            functions (optional): List of functions available to the LLM
            max_output_tokens (optional): Maximum number of tokens to generate

        Returns:
            list[ChatCompletionMessageParam]: Prompt messages for the OpenAI call
            CompletionCreateParams: Mapping of other kwargs for the OpenAI call
            Mapping[str, Any]: Any keyword arguments to pass on to the completion parser
        """
        tools_compat_mode = False
        if functions:
            if not OPEN_AI_CHAT_MODELS[model].has_function_call_api:
                # Provide compatibility with older models
                _functions_compat_fix_kwargs(functions, prompt_messages)
                tools_compat_mode = True
                functions = None

        openai_messages, kwargs, parse_kwargs = super()._get_chat_completion_args(
            prompt_messages=prompt_messages,
            model=model,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        kwargs.update(self._credentials.get_model_access_kwargs(model))  # type: ignore

        if tools_compat_mode:
            parse_kwargs["compat_mode"] = True

        return openai_messages, kwargs, parse_kwargs

    def _parse_assistant_tool_calls(
        self,
        assistant_message: ChatCompletionMessage,
        compat_mode: bool = False,
        **kwargs,
    ) -> tuple[list[AssistantToolCall], list[Exception]]:
        tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []

        if not compat_mode:
            return super()._parse_assistant_tool_calls(
                assistant_message=assistant_message, compat_mode=compat_mode, **kwargs
            )
        elif assistant_message.content:
            try:
                tool_calls = list(
                    _tool_calls_compat_extract_calls(assistant_message.content)
                )
            except Exception as e:
                parse_errors.append(e)

        return tool_calls, parse_errors

    def _get_embedding_kwargs(
        self, input: str | list[str], model: OpenAIModelName, **kwargs
    ) -> EmbeddingCreateParams:
        kwargs = super()._get_embedding_kwargs(input=input, model=model, **kwargs)
        kwargs.update(self._credentials.get_model_access_kwargs(model))  # type: ignore
        return kwargs

    _get_embedding_kwargs.__doc__ = (
        BaseOpenAIEmbeddingProvider._get_embedding_kwargs.__doc__
    )

    def _retry_api_request(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        _log_retry_debug_message = tenacity.after_log(self._logger, logging.DEBUG)

        def _log_on_fail(retry_state: tenacity.RetryCallState) -> None:
            _log_retry_debug_message(retry_state)

            if (
                retry_state.attempt_number == 0
                and retry_state.outcome
                and isinstance(retry_state.outcome.exception(), RateLimitError)
            ):
                self._logger.warning(
                    "Please double check that you have setup a PAID OpenAI API Account."
                    " You can read more here: "
                    "https://docs.agpt.co/setup/#getting-an-openai-api-key"
                )

        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(RateLimitError)
                | tenacity.retry_if_exception(
                    lambda e: isinstance(e, APIStatusError) and e.status_code == 502
                )
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=_log_on_fail,
        )(func)

    def __repr__(self):
        return "OpenAIProvider()"


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by OpenAI internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_tokens` to determine token usage of provided functions.

    Example:
    ```ts
    namespace functions {

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;

    } // namespace functions
    ```
    """

    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_openai_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )


def format_openai_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way OpenAI does it internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    Example:
    ```ts
    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;
    ```
    """

    def param_signature(name: str, spec: JSONSchema) -> str:
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {spec.typescript_type},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], count_tokens: Callable[[str], int]
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18  # noqa: E501
    """
    return count_tokens(
        "# Tools\n\n"
        "## functions\n\n"
        f"{format_function_specs_as_typescript_ns(functions)}"
    )


def _functions_compat_fix_kwargs(
    functions: list[CompletionModelFunction],
    prompt_messages: list[ChatMessage],
):
    function_definitions = format_function_specs_as_typescript_ns(functions)
    function_call_schema = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "name": JSONSchema(
                description="The name of the function to call",
                enum=[f.name for f in functions],
                required=True,
            ),
            "arguments": JSONSchema(
                description="The arguments for the function call",
                type=JSONSchema.Type.OBJECT,
                required=True,
            ),
        },
    )
    tool_calls_schema = JSONSchema(
        type=JSONSchema.Type.ARRAY,
        items=JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={
                "type": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    enum=["function"],
                ),
                "function": function_call_schema,
            },
        ),
    )
    prompt_messages.append(
        ChatMessage.system(
            "# tool usage instructions\n\n"
            "Specify a '```tool_calls' block in your response,"
            " with a valid JSON object that adheres to the following schema:\n\n"
            f"{tool_calls_schema.to_dict()}\n\n"
            "Specify any tools that you need to use through this JSON object.\n\n"
            "Put the tool_calls block at the end of your response"
            " and include its fences if it is not the only content.\n\n"
            "## functions\n\n"
            "For the function call itself, use one of the following"
            f" functions:\n\n{function_definitions}"
        ),
    )


def _tool_calls_compat_extract_calls(response: str) -> Iterator[AssistantToolCall]:
    import re
    import uuid

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    if response[0] == "[":
        tool_calls: list[AssistantToolCallDict] = json_loads(response)
    else:
        block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find tool_calls block in response")
        tool_calls: list[AssistantToolCallDict] = json_loads(block.group(1))

    for t in tool_calls:
        t["id"] = str(uuid.uuid4())
        yield AssistantToolCall.model_validate(t)
