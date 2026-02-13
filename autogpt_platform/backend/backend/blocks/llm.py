# This file contains a lot of prompt block strings that would trigger "line too long"
# flake8: noqa: E501
import logging
import re
import secrets
from abc import ABC
from enum import Enum
from json import JSONDecodeError
from typing import Any, Iterable, List, Literal, Optional

import anthropic
import ollama
import openai
from anthropic.types import ToolParam
from groq import AsyncGroq
from pydantic import BaseModel, GetCoreSchemaHandler, SecretStr
from pydantic_core import CoreSchema, core_schema

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data import llm_registry
from backend.data.llm_registry import ModelMetadata
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util import json
from backend.util.logging import TruncatedLogger
from backend.util.prompt import compress_context, estimate_token_count
from backend.util.text import TextFormatter

logger = TruncatedLogger(logging.getLogger(__name__), "[LLM-Block]")
fmt = TextFormatter(autoescape=False)

LLMProviderName = Literal[
    ProviderName.AIML_API,
    ProviderName.ANTHROPIC,
    ProviderName.GROQ,
    ProviderName.OLLAMA,
    ProviderName.OPENAI,
    ProviderName.OPEN_ROUTER,
    ProviderName.LLAMA_API,
    ProviderName.V0,
]
AICredentials = CredentialsMetaInput[LLMProviderName, Literal["api_key"]]

TEST_CREDENTIALS = APIKeyCredentials(
    id="769f6af7-820b-4d5d-9b7a-ab82bbc165f",
    provider="openai",
    api_key=SecretStr("mock-openai-api-key"),
    title="Mock OpenAI API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def AICredentialsField() -> AICredentials:
    """
    Returns a CredentialsField for LLM providers.
    The discriminator_mapping will be refreshed when the schema is generated
    if it's empty, ensuring the LLM registry is loaded.
    """
    # Get the mapping now - it may be empty initially, but will be refreshed
    # when the schema is generated via CredentialsMetaInput._add_json_schema_extra
    mapping = llm_registry.get_llm_discriminator_mapping()

    return CredentialsField(
        description="API key for the LLM provider.",
        discriminator="model",
        discriminator_mapping=mapping,  # May be empty initially, refreshed later
    )


def llm_model_schema_extra() -> dict[str, Any]:
    return {"options": llm_registry.get_llm_model_schema_options()}


class LlmModelMeta(type):
    """
    Metaclass for LlmModel that enables attribute-style access to dynamic models.

    This allows code like `LlmModel.GPT4O` to work by converting the attribute
    name to a slug format:
    - GPT4O -> gpt-4o
    - GPT4O_MINI -> gpt-4o-mini
    - CLAUDE_3_5_SONNET -> claude-3-5-sonnet
    """

    def __getattr__(cls, name: str):
        # Don't intercept private/dunder attributes
        if name.startswith("_"):
            raise AttributeError(f"type object 'LlmModel' has no attribute '{name}'")

        # Convert attribute name to slug format:
        # 1. Lowercase: GPT4O -> gpt4o
        # 2. Underscores to hyphens: GPT4O_MINI -> gpt4o-mini
        slug = name.lower().replace("_", "-")

        # Check for exact match in registry first (e.g., "o1" stays "o1")
        registry_slugs = llm_registry.get_dynamic_model_slugs()
        if slug in registry_slugs:
            return cls(slug)

        # If no exact match, try inserting hyphen between letter and digit
        # e.g., gpt4o -> gpt-4o
        transformed_slug = re.sub(r"([a-z])(\d)", r"\1-\2", slug)
        return cls(transformed_slug)

    def __iter__(cls):
        """Iterate over all models from the registry.

        Yields LlmModel instances for each model in the dynamic registry.
        Used by __get_pydantic_json_schema__ to build model metadata.
        """
        for model in llm_registry.iter_dynamic_models():
            yield cls(model.slug)


class LlmModel(str, metaclass=LlmModelMeta):
    """
    Dynamic LLM model type that accepts any model slug from the registry.

    This is a string subclass (not an Enum) that allows any model slug value.
    All models are managed via the LLM Registry in the database.

    Usage:
        model = LlmModel("gpt-4o")  # Direct construction
        model = LlmModel.GPT4O      # Attribute access (converted to "gpt-4o")
        model.value                  # Returns the slug string
        model.provider               # Returns the provider from registry
    """

    def __new__(cls, value: str):
        if isinstance(value, LlmModel):
            return value
        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Tell Pydantic how to validate LlmModel.

        Accepts strings and converts them to LlmModel instances.
        """
        return core_schema.no_info_after_validator_function(
            cls,  # The validator function (LlmModel constructor)
            core_schema.str_schema(),  # Accept string input
            serialization=core_schema.to_string_ser_schema(),  # Serialize as string
        )

    @property
    def value(self) -> str:
        """Return the model slug (for compatibility with enum-style access)."""
        return str(self)

    @classmethod
    def default(cls) -> "LlmModel":
        """
        Get the default model from the registry.

        Returns the recommended model if set, otherwise gpt-4o if available
        and enabled, otherwise the first enabled model from the registry.
        Falls back to "gpt-4o" if registry is empty (e.g., at module import time).
        """
        from backend.data.llm_registry import get_default_model_slug

        slug = get_default_model_slug()
        if slug is None:
            # Registry is empty (e.g., at module import time before DB connection).
            # Fall back to gpt-4o for backward compatibility.
            slug = "gpt-4o"
        return cls(slug)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        json_schema = handler(schema)
        llm_model_metadata = {}
        for model in cls:
            model_name = model.value
            # Skip disabled models - only show enabled models in the picker
            if not llm_registry.is_model_enabled(model_name):
                continue
            # Use registry directly with None check to gracefully handle
            # missing metadata during startup/import before registry is populated
            metadata = llm_registry.get_llm_model_metadata(model_name)
            if metadata is None:
                # Skip models without metadata (registry not yet populated)
                continue
            llm_model_metadata[model_name] = {
                "creator": metadata.creator_name,
                "creator_name": metadata.creator_name,
                "title": metadata.display_name,
                "provider": metadata.provider,
                "provider_name": metadata.provider_name,
                "name": model_name,
                "price_tier": metadata.price_tier,
            }
        json_schema["llm_model"] = True
        json_schema["llm_model_metadata"] = llm_model_metadata
        return json_schema

    @property
    def metadata(self) -> ModelMetadata:
        metadata = llm_registry.get_llm_model_metadata(self.value)
        if metadata:
            return metadata
        raise ValueError(
            f"Missing metadata for model: {self.value}. Model not found in LLM registry."
        )

    @property
    def provider(self) -> str:
        return self.metadata.provider

    @property
    def context_window(self) -> int:
        return self.metadata.context_window

    @property
    def max_output_tokens(self) -> int | None:
        return self.metadata.max_output_tokens


# Default model constant for backward compatibility
# Uses the dynamic registry to get the default model
DEFAULT_LLM_MODEL = LlmModel.default()


class ToolCall(BaseModel):
    name: str
    arguments: str


class ToolContentBlock(BaseModel):
    id: str
    type: str
    function: ToolCall


class LLMResponse(BaseModel):
    raw_response: Any
    prompt: List[Any]
    response: str
    tool_calls: Optional[List[ToolContentBlock]] | None
    prompt_tokens: int
    completion_tokens: int
    reasoning: Optional[str] = None


def convert_openai_tool_fmt_to_anthropic(
    openai_tools: list[dict] | None = None,
) -> Iterable[ToolParam] | anthropic.Omit:
    """
    Convert OpenAI tool format to Anthropic tool format.
    """
    if not openai_tools or len(openai_tools) == 0:
        return anthropic.omit

    anthropic_tools = []
    for tool in openai_tools:
        if "function" in tool:
            # Handle case where tool is already in OpenAI format with "type" and "function"
            function_data = tool["function"]
        else:
            # Handle case where tool is just the function definition
            function_data = tool

        anthropic_tool: anthropic.types.ToolParam = {
            "name": function_data["name"],
            "description": function_data.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": function_data.get("parameters", {}).get("properties", {}),
                "required": function_data.get("parameters", {}).get("required", []),
            },
        }
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


def extract_openai_reasoning(response) -> str | None:
    """Extract reasoning from OpenAI-compatible response if available."""
    """Note: This will likely not working since the reasoning is not present in another Response API"""
    reasoning = None
    choice = response.choices[0]
    if hasattr(choice, "reasoning") and getattr(choice, "reasoning", None):
        reasoning = str(getattr(choice, "reasoning"))
    elif hasattr(response, "reasoning") and getattr(response, "reasoning", None):
        reasoning = str(getattr(response, "reasoning"))
    elif hasattr(choice.message, "reasoning") and getattr(
        choice.message, "reasoning", None
    ):
        reasoning = str(getattr(choice.message, "reasoning"))
    return reasoning


def extract_openai_tool_calls(response) -> list[ToolContentBlock] | None:
    """Extract tool calls from OpenAI-compatible response."""
    if response.choices[0].message.tool_calls:
        return [
            ToolContentBlock(
                id=tool.id,
                type=tool.type,
                function=ToolCall(
                    name=tool.function.name,
                    arguments=tool.function.arguments,
                ),
            )
            for tool in response.choices[0].message.tool_calls
        ]
    return None


def get_parallel_tool_calls_param(
    llm_model: LlmModel, parallel_tool_calls: bool | None
) -> bool | openai.Omit:
    """Get the appropriate parallel_tool_calls parameter for OpenAI-compatible APIs."""
    # Check for o-series models (o1, o1-mini, o3-mini, etc.) which don't support
    # parallel tool calls. Use regex to avoid false positives like "openai/gpt-oss".
    is_o_series = re.match(r"^o\d", llm_model) is not None
    if is_o_series or parallel_tool_calls is None:
        return openai.NOT_GIVEN
    return parallel_tool_calls


async def llm_call(
    credentials: APIKeyCredentials,
    llm_model: LlmModel,
    prompt: list[dict],
    max_tokens: int | None,
    force_json_output: bool = False,
    tools: list[dict] | None = None,
    ollama_host: str = "localhost:11434",
    parallel_tool_calls=None,
    compress_prompt_to_fit: bool = True,
) -> LLMResponse:
    """
    Make a call to a language model.

    Args:
        credentials: The API key credentials to use.
        llm_model: The LLM model to use.
        prompt: The prompt to send to the LLM.
        force_json_output: Whether the response should be in JSON format.
        max_tokens: The maximum number of tokens to generate in the chat completion.
        tools: The tools to use in the chat completion.
        ollama_host: The host for ollama to use.

    Returns:
        LLMResponse object containing:
            - prompt: The prompt sent to the LLM.
            - response: The text response from the LLM.
            - tool_calls: Any tool calls the model made, if applicable.
            - prompt_tokens: The number of tokens used in the prompt.
            - completion_tokens: The number of tokens used in the completion.
    """
    # Get model metadata and check if enabled - with fallback support
    # The model we'll actually use (may differ if original is disabled)
    model_to_use = llm_model.value

    # Check if model is in registry and if it's enabled
    from backend.data.llm_registry import (
        get_fallback_model_for_disabled,
        get_model_info,
    )

    model_info = get_model_info(llm_model.value)

    if model_info and not model_info.is_enabled:
        # Model is disabled - try to find a fallback from the same provider
        fallback = get_fallback_model_for_disabled(llm_model.value)
        if fallback:
            logger.warning(
                f"Model '{llm_model.value}' is disabled. Using fallback model '{fallback.slug}' from the same provider ({fallback.metadata.provider})."
            )
            model_to_use = fallback.slug
            # Use fallback model's metadata
            provider = fallback.metadata.provider
            context_window = fallback.metadata.context_window
            model_max_output = fallback.metadata.max_output_tokens or int(2**15)
        else:
            # No fallback available - raise error
            raise ValueError(
                f"LLM model '{llm_model.value}' is disabled and no fallback model "
                f"from the same provider is available. Please enable the model or "
                f"select a different model in the block configuration."
            )
    else:
        # Model is enabled or not in registry (legacy/static model)
        try:
            provider = llm_model.metadata.provider
            context_window = llm_model.context_window
            model_max_output = llm_model.max_output_tokens or int(2**15)
        except ValueError:
            # Model not in cache - try refreshing the registry once if we have DB access
            logger.warning(f"Model {llm_model.value} not found in registry cache")

            # Try refreshing the registry if we have database access
            from backend.data.db import is_connected

            if is_connected():
                try:
                    logger.info(
                        f"Refreshing LLM registry and retrying lookup for {llm_model.value}"
                    )
                    await llm_registry.refresh_llm_registry()
                    # Try again after refresh
                    try:
                        provider = llm_model.metadata.provider
                        context_window = llm_model.context_window
                        model_max_output = llm_model.max_output_tokens or int(2**15)
                        logger.info(
                            f"Successfully loaded model {llm_model.value} metadata after registry refresh"
                        )
                    except ValueError:
                        # Still not found after refresh
                        raise ValueError(
                            f"LLM model '{llm_model.value}' not found in registry after refresh. "
                            "Please ensure the model is added and enabled in the LLM registry via the admin UI."
                        )
                except Exception as refresh_exc:
                    logger.error(f"Failed to refresh LLM registry: {refresh_exc}")
                    raise ValueError(
                        f"LLM model '{llm_model.value}' not found in registry and failed to refresh. "
                        "Please ensure the model is added to the LLM registry via the admin UI."
                    ) from refresh_exc
            else:
                # No DB access (e.g., in executor without direct DB connection)
                # The registry should have been loaded on startup
                raise ValueError(
                    f"LLM model '{llm_model.value}' not found in registry cache. "
                    "The registry may need to be refreshed. Please contact support or try again later."
                )

    # Create effective model for model-specific parameter resolution (e.g., o-series check)
    # This uses the resolved model_to_use which may differ from llm_model if fallback occurred
    effective_model = LlmModel(model_to_use)

    if compress_prompt_to_fit:
        result = await compress_context(
            messages=prompt,
            target_tokens=context_window // 2,
            client=None,  # Truncation-only, no LLM summarization
        )
        if result.error:
            logger.warning(
                f"Prompt compression did not meet target: {result.error}. "
                f"Proceeding with {result.token_count} tokens."
            )
        prompt = result.messages

    # Calculate available tokens based on context window and input length
    estimated_input_tokens = estimate_token_count(prompt)
    # model_max_output already set above
    user_max = max_tokens or model_max_output
    available_tokens = max(context_window - estimated_input_tokens, 0)
    max_tokens = max(min(available_tokens, model_max_output, user_max), 1)

    if provider == "openai":
        tools_param = tools if tools else openai.NOT_GIVEN
        oai_client = openai.AsyncOpenAI(api_key=credentials.api_key.get_secret_value())
        response_format = None

        parallel_tool_calls = get_parallel_tool_calls_param(
            effective_model, parallel_tool_calls
        )

        if force_json_output:
            response_format = {"type": "json_object"}

        response = await oai_client.chat.completions.create(
            model=model_to_use,
            messages=prompt,  # type: ignore
            response_format=response_format,  # type: ignore
            max_completion_tokens=max_tokens,
            tools=tools_param,  # type: ignore
            parallel_tool_calls=parallel_tool_calls,
        )

        tool_calls = extract_openai_tool_calls(response)
        reasoning = extract_openai_reasoning(response)

        return LLMResponse(
            raw_response=response.choices[0].message,
            prompt=prompt,
            response=response.choices[0].message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            reasoning=reasoning,
        )
    elif provider == "anthropic":

        an_tools = convert_openai_tool_fmt_to_anthropic(tools)

        system_messages = [p["content"] for p in prompt if p["role"] == "system"]
        sysprompt = " ".join(system_messages)

        messages = []
        last_role = None
        for p in prompt:
            if p["role"] in ["user", "assistant"]:
                if (
                    p["role"] == last_role
                    and isinstance(messages[-1]["content"], str)
                    and isinstance(p["content"], str)
                ):
                    # If the role is the same as the last one, combine the content
                    messages[-1]["content"] += p["content"]
                else:
                    messages.append({"role": p["role"], "content": p["content"]})
                    last_role = p["role"]

        client = anthropic.AsyncAnthropic(
            api_key=credentials.api_key.get_secret_value()
        )
        try:
            resp = await client.messages.create(
                model=model_to_use,
                system=sysprompt,
                messages=messages,
                max_tokens=max_tokens,
                tools=an_tools,
                timeout=600,
            )

            if not resp.content:
                raise ValueError("No content returned from Anthropic.")

            tool_calls = None
            for content_block in resp.content:
                # Antropic is different to openai, need to iterate through
                # the content blocks to find the tool calls
                if content_block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        ToolContentBlock(
                            id=content_block.id,
                            type=content_block.type,
                            function=ToolCall(
                                name=content_block.name,
                                arguments=json.dumps(content_block.input),
                            ),
                        )
                    )

            if not tool_calls and resp.stop_reason == "tool_use":
                logger.warning(
                    f"Tool use stop reason but no tool calls found in content. {resp}"
                )

            reasoning = None
            for content_block in resp.content:
                if hasattr(content_block, "type") and content_block.type == "thinking":
                    reasoning = content_block.thinking
                    break

            return LLMResponse(
                raw_response=resp,
                prompt=prompt,
                response=(
                    resp.content[0].name
                    if isinstance(resp.content[0], anthropic.types.ToolUseBlock)
                    else getattr(resp.content[0], "text", "")
                ),
                tool_calls=tool_calls,
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                reasoning=reasoning,
            )
        except anthropic.APIError as e:
            error_message = f"Anthropic API error: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)
    elif provider == "groq":
        if tools:
            raise ValueError("Groq does not support tools.")

        client = AsyncGroq(api_key=credentials.api_key.get_secret_value())
        response_format = {"type": "json_object"} if force_json_output else None
        response = await client.chat.completions.create(
            model=model_to_use,
            messages=prompt,  # type: ignore
            response_format=response_format,  # type: ignore
            max_tokens=max_tokens,
        )
        return LLMResponse(
            raw_response=response.choices[0].message,
            prompt=prompt,
            response=response.choices[0].message.content or "",
            tool_calls=None,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            reasoning=None,
        )
    elif provider == "ollama":
        if tools:
            raise ValueError("Ollama does not support tools.")

        client = ollama.AsyncClient(host=ollama_host)
        sys_messages = [p["content"] for p in prompt if p["role"] == "system"]
        usr_messages = [p["content"] for p in prompt if p["role"] != "system"]
        response = await client.generate(
            model=model_to_use,
            prompt=f"{sys_messages}\n\n{usr_messages}",
            stream=False,
            options={"num_ctx": max_tokens},
        )
        return LLMResponse(
            raw_response=response.get("response") or "",
            prompt=prompt,
            response=response.get("response") or "",
            tool_calls=None,
            prompt_tokens=response.get("prompt_eval_count") or 0,
            completion_tokens=response.get("eval_count") or 0,
            reasoning=None,
        )
    elif provider == "open_router":
        tools_param = tools if tools else openai.NOT_GIVEN
        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=credentials.api_key.get_secret_value(),
        )

        parallel_tool_calls_param = get_parallel_tool_calls_param(
            effective_model, parallel_tool_calls
        )

        response = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://agpt.co",
                "X-Title": "AutoGPT",
            },
            model=model_to_use,
            messages=prompt,  # type: ignore
            max_tokens=max_tokens,
            tools=tools_param,  # type: ignore
            parallel_tool_calls=parallel_tool_calls_param,
        )

        # If there's no response, raise an error
        if not response.choices:
            if response:
                raise ValueError(f"OpenRouter error: {response}")
            else:
                raise ValueError("No response from OpenRouter.")

        tool_calls = extract_openai_tool_calls(response)
        reasoning = extract_openai_reasoning(response)

        return LLMResponse(
            raw_response=response.choices[0].message,
            prompt=prompt,
            response=response.choices[0].message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            reasoning=reasoning,
        )
    elif provider == "llama_api":
        tools_param = tools if tools else openai.NOT_GIVEN
        client = openai.AsyncOpenAI(
            base_url="https://api.llama.com/compat/v1/",
            api_key=credentials.api_key.get_secret_value(),
        )

        parallel_tool_calls_param = get_parallel_tool_calls_param(
            effective_model, parallel_tool_calls
        )

        response = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://agpt.co",
                "X-Title": "AutoGPT",
            },
            model=model_to_use,
            messages=prompt,  # type: ignore
            max_tokens=max_tokens,
            tools=tools_param,  # type: ignore
            parallel_tool_calls=parallel_tool_calls_param,
        )

        # If there's no response, raise an error
        if not response.choices:
            if response:
                raise ValueError(f"Llama API error: {response}")
            else:
                raise ValueError("No response from Llama API.")

        tool_calls = extract_openai_tool_calls(response)
        reasoning = extract_openai_reasoning(response)

        return LLMResponse(
            raw_response=response.choices[0].message,
            prompt=prompt,
            response=response.choices[0].message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            reasoning=reasoning,
        )
    elif provider == "aiml_api":
        client = openai.AsyncOpenAI(
            base_url="https://api.aimlapi.com/v2",
            api_key=credentials.api_key.get_secret_value(),
            default_headers={
                "X-Project": "AutoGPT",
                "X-Title": "AutoGPT",
                "HTTP-Referer": "https://github.com/Significant-Gravitas/AutoGPT",
            },
        )

        completion = await client.chat.completions.create(
            model=model_to_use,
            messages=prompt,  # type: ignore
            max_tokens=max_tokens,
        )

        return LLMResponse(
            raw_response=completion.choices[0].message,
            prompt=prompt,
            response=completion.choices[0].message.content or "",
            tool_calls=None,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=(
                completion.usage.completion_tokens if completion.usage else 0
            ),
            reasoning=None,
        )
    elif provider == "v0":
        tools_param = tools if tools else openai.NOT_GIVEN
        client = openai.AsyncOpenAI(
            base_url="https://api.v0.dev/v1",
            api_key=credentials.api_key.get_secret_value(),
        )

        response_format = None
        if force_json_output:
            response_format = {"type": "json_object"}

        parallel_tool_calls_param = get_parallel_tool_calls_param(
            effective_model, parallel_tool_calls
        )

        response = await client.chat.completions.create(
            model=model_to_use,
            messages=prompt,  # type: ignore
            response_format=response_format,  # type: ignore
            max_tokens=max_tokens,
            tools=tools_param,  # type: ignore
            parallel_tool_calls=parallel_tool_calls_param,
        )

        tool_calls = extract_openai_tool_calls(response)
        reasoning = extract_openai_reasoning(response)

        return LLMResponse(
            raw_response=response.choices[0].message,
            prompt=prompt,
            response=response.choices[0].message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            reasoning=reasoning,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


class AIBlockBase(Block, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = []

    def merge_llm_stats(self, block: "AIBlockBase"):
        self.merge_stats(block.execution_stats)
        self.prompt = block.prompt


class AIStructuredResponseGeneratorBlock(AIBlockBase):
    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="The prompt to send to the language model.",
            placeholder="Enter your prompt here...",
        )
        expected_format: dict[str, str] = SchemaField(
            description="Expected format of the response. If provided, the response will be validated against this format. "
            "The keys should be the expected fields in the response, and the values should be the description of the field.",
        )
        list_result: bool = SchemaField(
            title="List Result",
            default=False,
            description="Whether the response should be a list of objects in the expected format.",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default_factory=LlmModel.default,
            description="The language model to use for answering the prompt.",
            advanced=False,
            json_schema_extra=llm_model_schema_extra(),
        )
        force_json_output: bool = SchemaField(
            title="Restrict LLM to pure JSON output",
            default=False,
            description=(
                "Whether to force the LLM to produce a JSON-only response. "
                "This can increase the block's reliability, "
                "but may also reduce the quality of the response "
                "because it prohibits the LLM from reasoning "
                "before providing its JSON response."
            ),
        )
        credentials: AICredentials = AICredentialsField()
        sys_prompt: str = SchemaField(
            title="System Prompt",
            default="",
            description="The system prompt to provide additional context to the model.",
        )
        conversation_history: list[dict] | None = SchemaField(
            default_factory=list,
            description="The conversation history to provide context for the prompt.",
        )
        retry: int = SchemaField(
            title="Retry Count",
            default=3,
            description="Number of times to retry the LLM call if the response does not match the expected format.",
        )
        prompt_values: dict[str, str] = SchemaField(
            advanced=False,
            default_factory=dict,
            description="Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}.",
        )
        max_tokens: int | None = SchemaField(
            advanced=True,
            default=None,
            description="The maximum number of tokens to generate in the chat completion.",
        )
        compress_prompt_to_fit: bool = SchemaField(
            advanced=True,
            default=True,
            description="Whether to compress the prompt to fit within the model's context window.",
        )
        ollama_host: str = SchemaField(
            advanced=True,
            default="localhost:11434",
            description="Ollama host for local  models",
        )

    class Output(BlockSchemaOutput):
        response: dict[str, Any] | list[dict[str, Any]] = SchemaField(
            description="The response object generated by the language model."
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")

    def __init__(self):
        super().__init__(
            id="ed55ac19-356e-4243-a6cb-bc599e9b716f",
            description="A block that generates structured JSON responses using a Large Language Model (LLM), with schema validation and format enforcement.",
            categories={BlockCategory.AI},
            input_schema=AIStructuredResponseGeneratorBlock.Input,
            output_schema=AIStructuredResponseGeneratorBlock.Output,
            test_input={
                "model": "gpt-4o",  # Using string value - enum accepts any model slug dynamically
                "credentials": TEST_CREDENTIALS_INPUT,
                "expected_format": {
                    "key1": "value1",
                    "key2": "value2",
                },
                "prompt": "User prompt",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("response", {"key1": "key1Value", "key2": "key2Value"}),
                ("prompt", list),
            ],
            test_mock={
                "llm_call": lambda *args, **kwargs: LLMResponse(
                    raw_response="",
                    prompt=[""],
                    response=(
                        '<json_output id="test123456">{\n'
                        '  "key1": "key1Value",\n'
                        '  "key2": "key2Value"\n'
                        "}</json_output>"
                    ),
                    tool_calls=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning=None,
                ),
                "get_collision_proof_output_tag_id": lambda *args: "test123456",
            },
        )

    async def llm_call(
        self,
        credentials: APIKeyCredentials,
        llm_model: LlmModel,
        prompt: list[dict],
        max_tokens: int | None,
        force_json_output: bool = False,
        compress_prompt_to_fit: bool = True,
        tools: list[dict] | None = None,
        ollama_host: str = "localhost:11434",
    ) -> LLMResponse:
        """
        Test mocks work only on class functions, this wraps the llm_call function
        so that it can be mocked withing the block testing framework.
        """
        self.prompt = prompt
        return await llm_call(
            credentials=credentials,
            llm_model=llm_model,
            prompt=prompt,
            max_tokens=max_tokens,
            force_json_output=force_json_output,
            tools=tools,
            ollama_host=ollama_host,
            compress_prompt_to_fit=compress_prompt_to_fit,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        logger.debug(f"Calling LLM with input data: {input_data}")
        prompt = [json.to_dict(p) for p in input_data.conversation_history or [] if p]

        values = input_data.prompt_values
        if values:
            input_data.prompt = fmt.format_string(input_data.prompt, values)
            input_data.sys_prompt = fmt.format_string(input_data.sys_prompt, values)

        if input_data.sys_prompt:
            prompt.append({"role": "system", "content": input_data.sys_prompt})

        # Use a one-time unique tag to prevent collisions with user/LLM content
        output_tag_id = self.get_collision_proof_output_tag_id()
        output_tag_start = f'<json_output id="{output_tag_id}">'
        if input_data.expected_format:
            sys_prompt = self.response_format_instructions(
                input_data.expected_format,
                list_mode=input_data.list_result,
                pure_json_mode=input_data.force_json_output,
                output_tag_start=output_tag_start,
            )
            prompt.append({"role": "system", "content": sys_prompt})

        if input_data.prompt:
            prompt.append({"role": "user", "content": input_data.prompt})

        def validate_response(parsed: object) -> str | None:
            try:
                if not isinstance(parsed, dict):
                    return f"Expected a dictionary, but got {type(parsed)}"
                miss_keys = set(input_data.expected_format.keys()) - set(parsed.keys())
                if miss_keys:
                    return f"Missing keys: {miss_keys}"
                return None
            except JSONDecodeError as e:
                return f"JSON decode error: {e}"

        error_feedback_message = ""
        llm_model = input_data.model

        for retry_count in range(input_data.retry):
            logger.debug(f"LLM request: {prompt}")
            try:
                llm_response = await self.llm_call(
                    credentials=credentials,
                    llm_model=llm_model,
                    prompt=prompt,
                    compress_prompt_to_fit=input_data.compress_prompt_to_fit,
                    force_json_output=(
                        input_data.force_json_output
                        and bool(input_data.expected_format)
                    ),
                    ollama_host=input_data.ollama_host,
                    max_tokens=input_data.max_tokens,
                )
                response_text = llm_response.response
                self.merge_stats(
                    NodeExecutionStats(
                        input_token_count=llm_response.prompt_tokens,
                        output_token_count=llm_response.completion_tokens,
                    )
                )
                logger.debug(f"LLM attempt-{retry_count} response: {response_text}")

                if input_data.expected_format:
                    try:
                        response_obj = self.get_json_from_response(
                            response_text,
                            pure_json_mode=input_data.force_json_output,
                            output_tag_start=output_tag_start,
                        )
                    except (ValueError, JSONDecodeError) as parse_error:
                        censored_response = re.sub(r"[A-Za-z0-9]", "*", response_text)
                        response_snippet = (
                            f"{censored_response[:50]}...{censored_response[-30:]}"
                        )
                        logger.warning(
                            f"Error getting JSON from LLM response: {parse_error}\n\n"
                            f"Response start+end: `{response_snippet}`"
                        )
                        prompt.append({"role": "assistant", "content": response_text})

                        error_feedback_message = self.invalid_response_feedback(
                            parse_error,
                            was_parseable=False,
                            list_mode=input_data.list_result,
                            pure_json_mode=input_data.force_json_output,
                            output_tag_start=output_tag_start,
                        )
                        prompt.append(
                            {"role": "user", "content": error_feedback_message}
                        )
                        continue

                    # Handle object response for `force_json_output`+`list_result`
                    if input_data.list_result and isinstance(response_obj, dict):
                        if "results" in response_obj and isinstance(
                            response_obj["results"], list
                        ):
                            response_obj = response_obj["results"]
                        else:
                            error_feedback_message = (
                                "Expected an array of objects in the 'results' key, "
                                f"but got: {response_obj}"
                            )
                            prompt.append(
                                {"role": "assistant", "content": response_text}
                            )
                            prompt.append(
                                {"role": "user", "content": error_feedback_message}
                            )
                            continue

                    validation_errors = "\n".join(
                        [
                            validation_error
                            for response_item in (
                                response_obj
                                if isinstance(response_obj, list)
                                else [response_obj]
                            )
                            if (validation_error := validate_response(response_item))
                        ]
                    )

                    if not validation_errors:
                        self.merge_stats(
                            NodeExecutionStats(
                                llm_call_count=retry_count + 1,
                                llm_retry_count=retry_count,
                            )
                        )
                        yield "response", response_obj
                        yield "prompt", self.prompt
                        return

                    prompt.append({"role": "assistant", "content": response_text})
                    error_feedback_message = self.invalid_response_feedback(
                        validation_errors,
                        was_parseable=True,
                        list_mode=input_data.list_result,
                        pure_json_mode=input_data.force_json_output,
                        output_tag_start=output_tag_start,
                    )
                    prompt.append({"role": "user", "content": error_feedback_message})
                else:
                    self.merge_stats(
                        NodeExecutionStats(
                            llm_call_count=retry_count + 1,
                            llm_retry_count=retry_count,
                        )
                    )
                    yield "response", {"response": response_text}
                    yield "prompt", self.prompt
                    return
            except Exception as e:
                logger.exception(f"Error calling LLM: {e}")
                if (
                    "maximum context length" in str(e).lower()
                    or "token limit" in str(e).lower()
                ):
                    if input_data.max_tokens is None:
                        input_data.max_tokens = llm_model.max_output_tokens or 4096
                    input_data.max_tokens = int(input_data.max_tokens * 0.85)
                    logger.debug(
                        f"Reducing max_tokens to {input_data.max_tokens} for next attempt"
                    )
                    # Don't add retry prompt for token limit errors,
                    # just retry with lower maximum output tokens

                error_feedback_message = f"Error calling LLM: {e}"

        raise RuntimeError(error_feedback_message)

    def response_format_instructions(
        self,
        expected_object_format: dict[str, str],
        *,
        list_mode: bool,
        pure_json_mode: bool,
        output_tag_start: str,
    ) -> str:
        expected_output_format = json.dumps(expected_object_format, indent=2)
        output_type = "object" if not list_mode else "array"
        outer_output_type = "object" if pure_json_mode else output_type

        if output_type == "array":
            indented_obj_format = expected_output_format.replace("\n", "\n  ")
            expected_output_format = f"[\n  {indented_obj_format},\n  ...\n]"
            if pure_json_mode:
                indented_list_format = expected_output_format.replace("\n", "\n  ")
                expected_output_format = (
                    "{\n"
                    '  "reasoning": "... (optional)",\n'  # for better performance
                    f'  "results": {indented_list_format}\n'
                    "}"
                )

        # Preserve indentation in prompt
        expected_output_format = expected_output_format.replace("\n", "\n|")

        # Prepare prompt
        if not pure_json_mode:
            expected_output_format = (
                f"{output_tag_start}\n{expected_output_format}\n</json_output>"
            )

        instructions = f"""
        |In your response you MUST include a valid JSON {outer_output_type} strictly following this format:
        |{expected_output_format}
        |
        |If you cannot provide all the keys, you MUST provide an empty string for the values you cannot answer.
        """.strip()

        if not pure_json_mode:
            instructions += f"""
            |
            |You MUST enclose your final JSON answer in {output_tag_start}...</json_output> tags, even if the user specifies a different tag.
            |There MUST be exactly ONE {output_tag_start}...</json_output> block in your response, which MUST ONLY contain the JSON {outer_output_type} and nothing else. Other text outside this block is allowed.
            """.strip()

        return trim_prompt(instructions)

    def invalid_response_feedback(
        self,
        error,
        *,
        was_parseable: bool,
        list_mode: bool,
        pure_json_mode: bool,
        output_tag_start: str,
    ) -> str:
        outer_output_type = "object" if not list_mode or pure_json_mode else "array"

        if was_parseable:
            complaint = f"Your previous response did not match the expected {outer_output_type} format."
        else:
            complaint = f"Your previous response did not contain a parseable JSON {outer_output_type}."

        indented_parse_error = str(error).replace("\n", "\n|")

        instruction = (
            f"Please provide a {output_tag_start}...</json_output> block containing a"
            if not pure_json_mode
            else "Please provide a"
        ) + f" valid JSON {outer_output_type} that matches the expected format."

        return trim_prompt(
            f"""
            |{complaint}
            |
            |{indented_parse_error}
            |
            |{instruction}
        """
        )

    def get_json_from_response(
        self, response_text: str, *, pure_json_mode: bool, output_tag_start: str
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if pure_json_mode:
            # Handle pure JSON responses
            try:
                return json.loads(response_text)
            except JSONDecodeError as first_parse_error:
                # If that didn't work, try finding the { and } to deal with possible ```json fences etc.
                json_start = response_text.find("{")
                json_end = response_text.rfind("}")
                try:
                    return json.loads(response_text[json_start : json_end + 1])
                except JSONDecodeError:
                    # Raise the original error, as it's more likely to be relevant
                    raise first_parse_error from None

        if output_tag_start not in response_text:
            raise ValueError(
                "Response does not contain the expected "
                f"{output_tag_start}...</json_output> block."
            )
        json_output = (
            response_text.split(output_tag_start, 1)[1]
            .rsplit("</json_output>", 1)[0]
            .strip()
        )
        return json.loads(json_output)

    def get_collision_proof_output_tag_id(self) -> str:
        return secrets.token_hex(8)


def trim_prompt(s: str) -> str:
    """Removes indentation up to and including `|` from a multi-line prompt."""
    lines = s.strip().split("\n")
    return "\n".join([line.strip().lstrip("|") for line in lines])


class AITextGeneratorBlock(AIBlockBase):
    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="The prompt to send to the language model. You can use any of the {keys} from Prompt Values to fill in the prompt with values from the prompt values dictionary by putting them in curly braces.",
            placeholder="Enter your prompt here...",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default_factory=LlmModel.default,
            description="The language model to use for answering the prompt.",
            advanced=False,
            json_schema_extra=llm_model_schema_extra(),
        )
        credentials: AICredentials = AICredentialsField()
        sys_prompt: str = SchemaField(
            title="System Prompt",
            default="",
            description="The system prompt to provide additional context to the model.",
        )
        retry: int = SchemaField(
            title="Retry Count",
            default=3,
            description="Number of times to retry the LLM call if the response does not match the expected format.",
        )
        prompt_values: dict[str, str] = SchemaField(
            advanced=False,
            default_factory=dict,
            description="Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}.",
        )
        ollama_host: str = SchemaField(
            advanced=True,
            default="localhost:11434",
            description="Ollama host for local  models",
        )
        max_tokens: int | None = SchemaField(
            advanced=True,
            default=None,
            description="The maximum number of tokens to generate in the chat completion.",
        )

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="The response generated by the language model."
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")

    def __init__(self):
        super().__init__(
            id="1f292d4a-41a4-4977-9684-7c8d560b9f91",
            description="A block that produces text responses using a Large Language Model (LLM) based on customizable prompts and system instructions.",
            categories={BlockCategory.AI},
            input_schema=AITextGeneratorBlock.Input,
            output_schema=AITextGeneratorBlock.Output,
            test_input={
                "prompt": "User prompt",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("response", "Response text"),
                ("prompt", list),
            ],
            test_mock={"llm_call": lambda *args, **kwargs: "Response text"},
        )

    async def llm_call(
        self,
        input_data: AIStructuredResponseGeneratorBlock.Input,
        credentials: APIKeyCredentials,
    ) -> dict:
        block = AIStructuredResponseGeneratorBlock()
        response = await block.run_once(input_data, "response", credentials=credentials)
        self.merge_llm_stats(block)
        return response["response"]

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        object_input_data = AIStructuredResponseGeneratorBlock.Input(
            **{
                attr: getattr(input_data, attr)
                for attr in AITextGeneratorBlock.Input.model_fields
            },
            expected_format={},
        )
        response = await self.llm_call(object_input_data, credentials)
        yield "response", response
        yield "prompt", self.prompt


class SummaryStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet points"
    NUMBERED_LIST = "numbered list"


class AITextSummarizerBlock(AIBlockBase):
    class Input(BlockSchemaInput):
        text: str = SchemaField(
            description="The text to summarize.",
            placeholder="Enter the text to summarize here...",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default_factory=LlmModel.default,
            description="The language model to use for summarizing the text.",
            json_schema_extra=llm_model_schema_extra(),
        )
        focus: str = SchemaField(
            title="Focus",
            default="general information",
            description="The topic to focus on in the summary",
        )
        style: SummaryStyle = SchemaField(
            title="Summary Style",
            default=SummaryStyle.CONCISE,
            description="The style of the summary to generate.",
        )
        credentials: AICredentials = AICredentialsField()
        # TODO: Make this dynamic
        max_tokens: int = SchemaField(
            title="Max Tokens",
            default=4096,
            description="The maximum number of tokens to generate in the chat completion.",
            ge=1,
        )
        chunk_overlap: int = SchemaField(
            title="Chunk Overlap",
            default=100,
            description="The number of overlapping tokens between chunks to maintain context.",
            ge=0,
        )
        ollama_host: str = SchemaField(
            advanced=True,
            default="localhost:11434",
            description="Ollama host for local  models",
        )

    class Output(BlockSchemaOutput):
        summary: str = SchemaField(description="The final summary of the text.")
        prompt: list = SchemaField(description="The prompt sent to the language model.")

    def __init__(self):
        super().__init__(
            id="a0a69be1-4528-491c-a85a-a4ab6873e3f0",
            description="A block that summarizes long texts using a Large Language Model (LLM), with configurable focus topics and summary styles.",
            categories={BlockCategory.AI, BlockCategory.TEXT},
            input_schema=AITextSummarizerBlock.Input,
            output_schema=AITextSummarizerBlock.Output,
            test_input={
                "text": "Lorem ipsum..." * 100,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("summary", "Final summary of a long text"),
                ("prompt", list),
            ],
            test_mock={
                "llm_call": lambda input_data, credentials: (
                    {"final_summary": "Final summary of a long text"}
                    if "final_summary" in input_data.expected_format
                    else {"summary": "Summary of a chunk of text"}
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        async for output_name, output_data in self._run(input_data, credentials):
            yield output_name, output_data

    async def _run(
        self, input_data: Input, credentials: APIKeyCredentials
    ) -> BlockOutput:
        chunks = self._split_text(
            input_data.text, input_data.max_tokens, input_data.chunk_overlap
        )
        summaries = []

        for chunk in chunks:
            chunk_summary = await self._summarize_chunk(chunk, input_data, credentials)
            summaries.append(chunk_summary)

        final_summary = await self._combine_summaries(
            summaries, input_data, credentials
        )
        yield "summary", final_summary
        yield "prompt", self.prompt

    @staticmethod
    def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
        # Security fix: Add validation to prevent DoS attacks
        # Limit text size to prevent memory exhaustion
        MAX_TEXT_LENGTH = 1_000_000  # 1MB character limit
        MAX_CHUNKS = 100  # Maximum number of chunks to prevent excessive memory use

        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        # Ensure chunk_size is at least 1 to prevent infinite loops
        chunk_size = max(1, max_tokens - overlap)

        # Ensure overlap is less than max_tokens to prevent invalid configurations
        if overlap >= max_tokens:
            overlap = max(0, max_tokens - 1)

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            if len(chunks) >= MAX_CHUNKS:
                break  # Limit the number of chunks to prevent memory exhaustion
            chunk = " ".join(words[i : i + max_tokens])
            chunks.append(chunk)

        return chunks

    async def llm_call(
        self,
        input_data: AIStructuredResponseGeneratorBlock.Input,
        credentials: APIKeyCredentials,
    ) -> dict:
        block = AIStructuredResponseGeneratorBlock()
        response = await block.run_once(input_data, "response", credentials=credentials)
        self.merge_llm_stats(block)
        return response

    async def _summarize_chunk(
        self, chunk: str, input_data: Input, credentials: APIKeyCredentials
    ) -> str:
        prompt = f"Summarize the following text in a {input_data.style} form. Focus your summary on the topic of `{input_data.focus}` if present, otherwise just provide a general summary:\n\n```{chunk}```"

        llm_response = await self.llm_call(
            AIStructuredResponseGeneratorBlock.Input(
                prompt=prompt,
                credentials=input_data.credentials,
                model=input_data.model,
                expected_format={"summary": "The summary of the given text."},
            ),
            credentials=credentials,
        )

        summary = llm_response["summary"]

        # Validate that the LLM returned a string and not a list or other type
        if not isinstance(summary, str):
            from backend.util.truncate import truncate

            truncated_summary = truncate(summary, 500)
            raise ValueError(
                f"LLM generation failed: Expected a string summary, but received {type(summary).__name__}. "
                f"The language model incorrectly formatted its response. "
                f"Received value: {json.dumps(truncated_summary)}"
            )

        return summary

    async def _combine_summaries(
        self, summaries: list[str], input_data: Input, credentials: APIKeyCredentials
    ) -> str:
        combined_text = "\n\n".join(summaries)

        if len(combined_text.split()) <= input_data.max_tokens:
            prompt = f"Provide a final summary of the following section summaries in a {input_data.style} form, focus your summary on the topic of `{input_data.focus}` if present:\n\n ```{combined_text}```\n\n Just respond with the final_summary in the format specified."

            llm_response = await self.llm_call(
                AIStructuredResponseGeneratorBlock.Input(
                    prompt=prompt,
                    credentials=input_data.credentials,
                    model=input_data.model,
                    expected_format={
                        "final_summary": "The final summary of all provided summaries."
                    },
                ),
                credentials=credentials,
            )

            final_summary = llm_response["final_summary"]

            # Validate that the LLM returned a string and not a list or other type
            if not isinstance(final_summary, str):
                from backend.util.truncate import truncate

                truncated_final_summary = truncate(final_summary, 500)
                raise ValueError(
                    f"LLM generation failed: Expected a string final summary, but received {type(final_summary).__name__}. "
                    f"The language model incorrectly formatted its response. "
                    f"Received value: {json.dumps(truncated_final_summary)}"
                )

            return final_summary
        else:
            # If combined summaries are still too long, recursively summarize
            block = AITextSummarizerBlock()
            return await block.run_once(
                AITextSummarizerBlock.Input(
                    text=combined_text,
                    credentials=input_data.credentials,
                    model=input_data.model,
                    max_tokens=input_data.max_tokens,
                    chunk_overlap=input_data.chunk_overlap,
                ),
                "summary",
                credentials=credentials,
            )


class AIConversationBlock(AIBlockBase):
    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="The prompt to send to the language model.",
            placeholder="Enter your prompt here...",
            default="",
            advanced=False,
        )
        messages: List[Any] = SchemaField(
            description="List of messages in the conversation.",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default_factory=LlmModel.default,
            description="The language model to use for the conversation.",
            json_schema_extra=llm_model_schema_extra(),
        )
        credentials: AICredentials = AICredentialsField()
        max_tokens: int | None = SchemaField(
            advanced=True,
            default=None,
            description="The maximum number of tokens to generate in the chat completion.",
        )
        ollama_host: str = SchemaField(
            advanced=True,
            default="localhost:11434",
            description="Ollama host for local  models",
        )

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="The model's response to the conversation."
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")

    def __init__(self):
        super().__init__(
            id="32a87eab-381e-4dd4-bdb8-4c47151be35a",
            description="A block that facilitates multi-turn conversations with a Large Language Model (LLM), maintaining context across message exchanges.",
            categories={BlockCategory.AI},
            input_schema=AIConversationBlock.Input,
            output_schema=AIConversationBlock.Output,
            test_input={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {
                        "role": "assistant",
                        "content": "The Los Angeles Dodgers won the World Series in 2020.",
                    },
                    {"role": "user", "content": "Where was it played?"},
                ],
                "model": "gpt-4o",  # Using string value - enum accepts any model slug dynamically
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "response",
                    "The 2020 World Series was played at Globe Life Field in Arlington, Texas.",
                ),
                ("prompt", list),
            ],
            test_mock={
                "llm_call": lambda *args, **kwargs: dict(
                    response="The 2020 World Series was played at Globe Life Field in Arlington, Texas."
                )
            },
        )

    async def llm_call(
        self,
        input_data: AIStructuredResponseGeneratorBlock.Input,
        credentials: APIKeyCredentials,
    ) -> dict:
        block = AIStructuredResponseGeneratorBlock()
        response = await block.run_once(input_data, "response", credentials=credentials)
        self.merge_llm_stats(block)
        return response

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        response = await self.llm_call(
            AIStructuredResponseGeneratorBlock.Input(
                prompt=input_data.prompt,
                credentials=input_data.credentials,
                model=input_data.model,
                conversation_history=input_data.messages,
                max_tokens=input_data.max_tokens,
                expected_format={},
                ollama_host=input_data.ollama_host,
            ),
            credentials=credentials,
        )
        yield "response", response["response"]
        yield "prompt", self.prompt


class AIListGeneratorBlock(AIBlockBase):
    class Input(BlockSchemaInput):
        focus: str | None = SchemaField(
            description="The focus of the list to generate.",
            placeholder="The top 5 most interesting news stories in the data.",
            default=None,
            advanced=False,
        )
        source_data: str | None = SchemaField(
            description="The data to generate the list from.",
            placeholder="News Today: Humans land on Mars: Today humans landed on mars. -- AI wins Nobel Prize: AI wins Nobel Prize for solving world hunger. -- New AI Model: A new AI model has been released.",
            default=None,
            advanced=False,
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default_factory=LlmModel.default,
            description="The language model to use for generating the list.",
            advanced=True,
            json_schema_extra=llm_model_schema_extra(),
        )
        credentials: AICredentials = AICredentialsField()
        max_retries: int = SchemaField(
            default=3,
            description="Maximum number of retries for generating a valid list.",
            ge=1,
            le=5,
        )
        force_json_output: bool = SchemaField(
            title="Restrict LLM to pure JSON output",
            default=False,
            description=(
                "Whether to force the LLM to produce a JSON-only response. "
                "This can increase the block's reliability, "
                "but may also reduce the quality of the response "
                "because it prohibits the LLM from reasoning "
                "before providing its JSON response."
            ),
        )
        max_tokens: int | None = SchemaField(
            advanced=True,
            default=None,
            description="The maximum number of tokens to generate in the chat completion.",
        )
        ollama_host: str = SchemaField(
            advanced=True,
            default="localhost:11434",
            description="Ollama host for local  models",
        )

    class Output(BlockSchemaOutput):
        generated_list: list[str] = SchemaField(description="The generated list.")
        list_item: str = SchemaField(
            description="Each individual item in the list.",
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")

    def __init__(self):
        super().__init__(
            id="9c0b0450-d199-458b-a731-072189dd6593",
            description="A block that creates lists of items based on prompts using a Large Language Model (LLM), with optional source data for context.",
            categories={BlockCategory.AI, BlockCategory.TEXT},
            input_schema=AIListGeneratorBlock.Input,
            output_schema=AIListGeneratorBlock.Output,
            test_input={
                "focus": "planets",
                "source_data": (
                    "Zylora Prime is a glowing jungle world with bioluminescent plants, "
                    "while Kharon-9 is a harsh desert planet with underground cities. "
                    "Vortexia's constant storms power floating cities, and Oceara is a water-covered world home to "
                    "intelligent marine life. On icy Draknos, ancient ruins lie buried beneath its frozen landscape, "
                    "drawing explorers to uncover its mysteries. Each planet showcases the limitless possibilities of "
                    "fictional worlds."
                ),
                "model": "gpt-4o",  # Using string value - enum accepts any model slug dynamically
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_retries": 3,
                "force_json_output": False,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "generated_list",
                    ["Zylora Prime", "Kharon-9", "Vortexia", "Oceara", "Draknos"],
                ),
                ("prompt", list),
                ("list_item", "Zylora Prime"),
                ("list_item", "Kharon-9"),
                ("list_item", "Vortexia"),
                ("list_item", "Oceara"),
                ("list_item", "Draknos"),
            ],
            test_mock={
                "llm_call": lambda input_data, credentials: {
                    "list": [
                        "Zylora Prime",
                        "Kharon-9",
                        "Vortexia",
                        "Oceara",
                        "Draknos",
                    ]
                },
            },
        )

    async def llm_call(
        self,
        input_data: AIStructuredResponseGeneratorBlock.Input,
        credentials: APIKeyCredentials,
    ) -> dict[str, Any]:
        llm_block = AIStructuredResponseGeneratorBlock()
        response = await llm_block.run_once(
            input_data, "response", credentials=credentials
        )
        self.merge_llm_stats(llm_block)
        return response

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        logger.debug(f"Starting AIListGeneratorBlock.run with input data: {input_data}")

        # Create a proper expected format for the structured response generator
        expected_format = {
            "list": "A JSON array containing the generated string values"
        }
        if input_data.force_json_output:
            # Add reasoning field for better performance
            expected_format = {
                "reasoning": "... (optional)",
                **expected_format,
            }

        # Build the prompt
        if input_data.focus:
            prompt = f"Generate a list with the following focus:\n<focus>\n\n{input_data.focus}</focus>"
        else:
            # If there's source data
            if input_data.source_data:
                prompt = "Extract the main focus of the source data to a list.\ni.e if the source data is a news website, the focus would be the news stories rather than the social links in the footer."
            else:
                # No focus or source data provided, generate a random list
                prompt = "Generate a random list."

        # If the source data is provided, add it to the prompt
        if input_data.source_data:
            prompt += f"\n\nUse the following source data to generate the list from:\n\n<source_data>\n\n{input_data.source_data}</source_data>\n\nDo not invent fictional data that is not present in the source data."
        # Else, tell the LLM to synthesize the data
        else:
            prompt += "\n\nInvent the data to generate the list from."

        # Use the structured response generator to handle all the complexity
        response_obj = await self.llm_call(
            AIStructuredResponseGeneratorBlock.Input(
                sys_prompt=self.SYSTEM_PROMPT,
                prompt=prompt,
                credentials=input_data.credentials,
                model=input_data.model,
                expected_format=expected_format,
                force_json_output=input_data.force_json_output,
                retry=input_data.max_retries,
                max_tokens=input_data.max_tokens,
                ollama_host=input_data.ollama_host,
            ),
            credentials=credentials,
        )
        logger.debug(f"Response object: {response_obj}")

        # Extract the list from the response object
        if isinstance(response_obj, dict) and "list" in response_obj:
            parsed_list = response_obj["list"]
        else:
            # Fallback - treat the whole response as the list
            parsed_list = response_obj

        # Validate that we got a list
        if not isinstance(parsed_list, list):
            raise ValueError(
                f"Expected a list, but got {type(parsed_list).__name__}: {parsed_list}"
            )

        logger.debug(f"Parsed list: {parsed_list}")

        # Yield the results
        yield "generated_list", parsed_list
        yield "prompt", self.prompt

        # Yield each item in the list
        for item in parsed_list:
            yield "list_item", item

    SYSTEM_PROMPT = trim_prompt(
        """
        |You are a JSON array generator. Your task is to generate a JSON array of string values based on the user's prompt.
        |
        |The 'list' field should contain a JSON array with the generated string values.
        |The array can contain ONLY strings.
        |
        |Valid JSON array formats include:
        |• ["string1", "string2", "string3"]
        |
        |Ensure you provide a proper JSON array with only string values in the 'list' field.
        """
    )
