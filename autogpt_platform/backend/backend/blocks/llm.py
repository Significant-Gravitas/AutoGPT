# This file contains a lot of prompt block strings that would trigger "line too long"
# flake8: noqa: E501
import asyncio
import logging
import math
import re
import secrets
from abc import ABC
from enum import Enum
from json import JSONDecodeError
from typing import Any, Iterable, List, Literal, Optional, cast

import anthropic
import openai
from anthropic.types import ToolParam
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from pydantic import BaseModel, SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.llm_registry.llm_models import DEFAULT_LLM_MODEL as DEFAULT_LLM_MODEL
from backend.data.llm_registry.llm_models import (
    LEGACY_MODEL_MAPPINGS as LEGACY_MODEL_MAPPINGS,
)
from backend.data.llm_registry.llm_models import MODEL_METADATA as MODEL_METADATA
from backend.data.llm_registry.llm_models import LLMModel as LLMModel
from backend.data.llm_registry.llm_models import LLMModelMeta as LLMModelMeta
from backend.data.llm_registry.llm_models import ModelMetadata as ModelMetadata
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util import json

# ``ToolCall`` and ``ToolContentBlock`` live in the shared
# ``util.llm.conversions`` module so every LLM caller (block layer,
# dream pass, copilot chat) references one canonical type. Re-export
# them here so existing ``from backend.blocks.llm import
# ToolContentBlock`` imports keep working.
from backend.util.llm.conversions import ToolCall, ToolContentBlock
from backend.util.logging import TruncatedLogger
from backend.util.prompt import compress_context, estimate_token_count
from backend.util.settings import Settings
from backend.util.text import TextFormatter

settings = Settings()
logger = TruncatedLogger(logging.getLogger(__name__), "[LLM-Block]")
fmt = TextFormatter(autoescape=False)

# HTTP status codes for user-caused errors that should not be reported to Sentry.
USER_ERROR_STATUS_CODES = (401, 403, 429)

# Hard cap on a single provider HTTP request. Healthy non-streaming Responses /
# Messages calls finish in seconds; anything past this is almost certainly a
# stalled socket (server keeping connection alive but starving response bytes,
# which the SDK's read-timeout doesn't reliably detect on its own). Lower than
# the SDK defaults (typically 600s) so retries-on-timeout don't compound into
# multi-hour worst cases when a block makes many sequential calls.
LLM_REQUEST_TIMEOUT_SECONDS = 120

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
    return CredentialsField(
        description="API key for the LLM provider.",
        discriminator="model",
        discriminator_mapping={
            model.value: model.metadata.provider for model in LLMModel
        },
    )


class LLMResponse(BaseModel):
    raw_response: Any
    prompt: List[Any]
    response: str
    tool_calls: Optional[List[ToolContentBlock]] | None
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning: Optional[str] = None
    provider_cost: float | None = None


def convert_openai_tool_fmt_to_anthropic(
    openai_tools: list[dict] | None = None,
) -> Iterable[ToolParam] | anthropic.NotGiven:
    """
    Convert OpenAI tool format to Anthropic tool format.
    """
    if not openai_tools or len(openai_tools) == 0:
        return anthropic.NOT_GIVEN

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


def extract_openrouter_cost(response: OpenAIChatCompletion) -> float | None:
    """Extract OpenRouter's per-request USD cost from a chat-completion response.

    OpenRouter populates a ``cost`` field on the standard ``usage`` object (a
    USD float) when the request body includes ``usage: {"include": True}``.
    The OpenAI SDK's typed ``CompletionUsage`` does not declare it, so we read
    it off ``model_extra`` (pydantic v2's typed extras container) — no
    ``getattr``. Mirrors backend/executor/simulator.py::_extract_cost_usd —
    keep the two aligned.
    """
    usage = response.usage
    if usage is None:
        return None
    extras = usage.model_extra or {}
    cost = extras.get("cost")
    if cost is None:
        return None
    try:
        cost_f = float(cost)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cost_f) or cost_f < 0:
        return None
    return cost_f


def extract_openai_reasoning(response) -> str | None:
    """Extract reasoning from OpenAI-compatible response if available."""
    """Note: This will likely not working since the reasoning is not present in another Response API"""
    if not response.choices:
        logger.warning("LLM response has empty choices in extract_openai_reasoning")
        return None
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
    if not response.choices:
        logger.warning("LLM response has empty choices in extract_openai_tool_calls")
        return None
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
    llm_model: LLMModel, parallel_tool_calls: bool | None
) -> bool | openai.Omit:
    """Get the appropriate parallel_tool_calls parameter for OpenAI-compatible APIs."""
    if llm_model.startswith("o") or parallel_tool_calls is None:
        return openai.omit
    return parallel_tool_calls


async def llm_call(
    credentials: APIKeyCredentials,
    llm_model: LLMModel,
    prompt: list[dict],
    max_tokens: int | None,
    force_json_output: bool = False,
    tools: list[dict] | None = None,
    ollama_host: str = "localhost:11434",
    parallel_tool_calls=None,
    compress_prompt_to_fit: bool = True,
) -> LLMResponse:
    """Public LLM-call entry point. Wraps the provider dispatch in a hard timeout
    so that no single request can park an executor thread indefinitely."""
    try:
        return await asyncio.wait_for(
            _llm_call(
                credentials=credentials,
                llm_model=llm_model,
                prompt=prompt,
                max_tokens=max_tokens,
                force_json_output=force_json_output,
                tools=tools,
                ollama_host=ollama_host,
                parallel_tool_calls=parallel_tool_calls,
                compress_prompt_to_fit=compress_prompt_to_fit,
            ),
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        raise TimeoutError(
            f"LLM request to {llm_model.metadata.provider}/{llm_model.value} "
            f"exceeded {LLM_REQUEST_TIMEOUT_SECONDS}s and was cancelled."
        ) from e


async def _llm_call(
    credentials: APIKeyCredentials,
    llm_model: LLMModel,
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
    provider = llm_model.metadata.provider
    context_window = llm_model.context_window

    if compress_prompt_to_fit:
        result = await compress_context(
            messages=prompt,
            target_tokens=llm_model.context_window // 2,
            client=None,  # Truncation-only, no LLM summarization
            reserve=0,  # Caller handles response token budget separately
        )
        if result.error:
            logger.warning(
                f"Prompt compression did not meet target: {result.error}. "
                f"Proceeding with {result.token_count} tokens."
            )
        prompt = result.messages

    # Sanitize unpaired surrogates in message content to prevent
    # UnicodeEncodeError when httpx encodes the JSON request body.
    for msg in prompt:
        content = msg.get("content")
        if isinstance(content, str):
            try:
                content.encode("utf-8")
            except UnicodeEncodeError:
                logger.warning("Sanitized unpaired surrogates in LLM prompt content")
                msg["content"] = content.encode("utf-8", errors="surrogatepass").decode(
                    "utf-8", errors="replace"
                )

    # Calculate available tokens based on context window and input length
    estimated_input_tokens = estimate_token_count(prompt)
    model_max_output = llm_model.max_output_tokens or int(2**15)
    user_max = max_tokens or model_max_output
    available_tokens = max(context_window - estimated_input_tokens, 0)
    max_tokens = max(min(available_tokens, model_max_output, user_max), 1)

    # ---- Delegate per-provider HTTP to the shared helper ----------------
    # All per-provider SDK quirks (Anthropic system-prompt wrapping,
    # OpenRouter cost extras, AI/ML branded headers, Ollama generate
    # API, etc.) live in ``backend/util/llm/providers.py`` so the dream
    # pass, copilot chat, and any future server-side caller route
    # through one implementation. This wrapper keeps the block-layer
    # framing on top: ``LLMModel``-aware token budget, retry-on-bad-
    # shape (caller), ``NodeExecutionStats`` (caller).
    from backend.util.llm.providers import ProviderResponse, call_provider

    provider_response = await call_provider(
        provider=cast(Any, provider),
        model=llm_model.value,
        api_key=credentials.api_key.get_secret_value(),
        messages=prompt,
        max_tokens=max_tokens,
        tools=tools,
        force_json_output=force_json_output,
        parallel_tool_calls=get_parallel_tool_calls_param(
            llm_model, parallel_tool_calls
        ),
        ollama_host=ollama_host,
        timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
    )
    # Block layer never opts into batch mode (that lives on the
    # orchestrator side for the dream pass and any future async
    # callers). The default ``execution_mode="sync"`` is hardcoded
    # above, so the helper always returns a ``ProviderResponse`` here.
    # The isinstance guard narrows the union for the type checker.
    if not isinstance(provider_response, ProviderResponse):
        raise RuntimeError(
            "block-layer _llm_call only supports execution_mode='sync' but "
            f"call_provider returned {type(provider_response).__name__}"
        )

    return LLMResponse(
        raw_response=provider_response.raw_response,
        prompt=prompt,
        response=provider_response.content,
        tool_calls=provider_response.tool_calls,
        prompt_tokens=provider_response.prompt_tokens,
        completion_tokens=provider_response.completion_tokens,
        cache_read_tokens=provider_response.cache_read_tokens,
        cache_creation_tokens=provider_response.cache_creation_tokens,
        reasoning=provider_response.reasoning,
        provider_cost=provider_response.cost_usd,
    )


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
        model: LLMModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use for answering the prompt.",
            advanced=False,
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
                "model": DEFAULT_LLM_MODEL,
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
        llm_model: LLMModel,
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
            input_data.prompt = await fmt.format_string(input_data.prompt, values)
            input_data.sys_prompt = await fmt.format_string(
                input_data.sys_prompt, values
            )

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
        total_provider_cost: float | None = None

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
                # Accumulate token counts and provider_cost for every attempt
                # (each call costs tokens and USD, regardless of validation outcome).
                token_stats = NodeExecutionStats(
                    input_token_count=llm_response.prompt_tokens,
                    output_token_count=llm_response.completion_tokens,
                    cache_read_token_count=llm_response.cache_read_tokens,
                    cache_creation_token_count=llm_response.cache_creation_tokens,
                )
                self.merge_stats(token_stats)
                if llm_response.provider_cost is not None:
                    total_provider_cost = (
                        total_provider_cost or 0.0
                    ) + llm_response.provider_cost
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
                                provider_cost=total_provider_cost,
                                provider_cost_type=(
                                    "cost_usd"
                                    if total_provider_cost is not None
                                    else None
                                ),
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
                            provider_cost=total_provider_cost,
                            provider_cost_type=(
                                "cost_usd" if total_provider_cost is not None else None
                            ),
                        )
                    )
                    yield "response", {"response": response_text}
                    yield "prompt", self.prompt
                    return
            except Exception as e:
                is_user_error = (
                    isinstance(e, (anthropic.APIStatusError, openai.APIStatusError))
                    and e.status_code in USER_ERROR_STATUS_CODES
                )
                if is_user_error:
                    logger.warning(f"Error calling LLM: {e}")
                    error_feedback_message = f"Error calling LLM: {e}"
                    break
                if isinstance(e, TimeoutError):
                    # A request that hung once will most likely hang again on
                    # retry — the underlying issue (server-side starvation,
                    # network partition, etc.) doesn't clear on a fresh socket.
                    # Skip retries to avoid the N×timeout wait cascade.
                    logger.warning(f"LLM call timed out, not retrying: {e}")
                    error_feedback_message = f"Error calling LLM: {e}"
                    break
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

        # All retries exhausted or user-error break: persist accumulated cost so
        # the executor can still charge/report the spend even on failure.
        if total_provider_cost is not None:
            self.merge_stats(
                NodeExecutionStats(
                    provider_cost=total_provider_cost,
                    provider_cost_type="cost_usd",
                )
            )
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
        model: LLMModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use for answering the prompt.",
            advanced=False,
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
        model: LLMModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use for summarizing the text.",
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
        model: LLMModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use for the conversation.",
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
                "model": DEFAULT_LLM_MODEL,
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
        has_messages = any(
            isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and bool(m["content"].strip())
            for m in (input_data.messages or [])
        )
        has_prompt = bool(input_data.prompt and input_data.prompt.strip())
        if not has_messages and not has_prompt:
            raise ValueError(
                "Cannot call LLM with no messages and no prompt. "
                "Provide at least one message or a non-empty prompt."
            )

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
        model: LLMModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model to use for generating the list.",
            advanced=True,
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
                "model": DEFAULT_LLM_MODEL,
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
