# This file contains a lot of prompt block strings that would trigger "line too long"
# flake8: noqa: E501
import ast
import logging
import re
import secrets
from abc import ABC
from enum import Enum, EnumMeta
from json import JSONDecodeError
from typing import Any, Iterable, List, Literal, NamedTuple, Optional

import anthropic
import ollama
import openai
from anthropic.types import ToolParam
from groq import AsyncGroq
from pydantic import BaseModel, SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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
from backend.util.prompt import compress_prompt, estimate_token_count
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
    return CredentialsField(
        description="API key for the LLM provider.",
        discriminator="model",
        discriminator_mapping={
            model.value: model.metadata.provider for model in LlmModel
        },
    )


class ModelMetadata(NamedTuple):
    provider: str
    context_window: int
    max_output_tokens: int | None


class LlmModelMeta(EnumMeta):
    pass


class LlmModel(str, Enum, metaclass=LlmModelMeta):
    # OpenAI models
    O3_MINI = "o3-mini"
    O3 = "o3-2025-04-16"
    O1 = "o1"
    O1_MINI = "o1-mini"
    # GPT-5 models
    GPT5 = "gpt-5-2025-08-07"
    GPT5_MINI = "gpt-5-mini-2025-08-07"
    GPT5_NANO = "gpt-5-nano-2025-08-07"
    GPT5_CHAT = "gpt-5-chat-latest"
    GPT41 = "gpt-4.1-2025-04-14"
    GPT41_MINI = "gpt-4.1-mini-2025-04-14"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    GPT4_TURBO = "gpt-4-turbo"
    GPT3_5_TURBO = "gpt-3.5-turbo"
    # Anthropic models
    CLAUDE_4_1_OPUS = "claude-opus-4-1-20250805"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_4_5_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    # AI/ML API models
    AIML_API_QWEN2_5_72B = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    AIML_API_LLAMA3_1_70B = "nvidia/llama-3.1-nemotron-70b-instruct"
    AIML_API_LLAMA3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    AIML_API_META_LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    AIML_API_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    # Groq models
    GEMMA2_9B = "gemma2-9b-it"
    LLAMA3_3_70B = "llama-3.3-70b-versatile"
    LLAMA3_1_8B = "llama-3.1-8b-instant"
    LLAMA3_70B = "llama3-70b-8192"
    LLAMA3_8B = "llama3-8b-8192"
    # Groq preview models
    DEEPSEEK_LLAMA_70B = "deepseek-r1-distill-llama-70b"
    # Ollama models
    OLLAMA_LLAMA3_3 = "llama3.3"
    OLLAMA_LLAMA3_2 = "llama3.2"
    OLLAMA_LLAMA3_8B = "llama3"
    OLLAMA_LLAMA3_405B = "llama3.1:405b"
    OLLAMA_DOLPHIN = "dolphin-mistral:latest"
    # OpenRouter models
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"
    OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
    GEMINI_FLASH_1_5 = "google/gemini-flash-1.5"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro-preview-03-25"
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GEMINI_2_0_FLASH = "google/gemini-2.0-flash-001"
    GEMINI_2_5_FLASH_LITE_PREVIEW = "google/gemini-2.5-flash-lite-preview-06-17"
    GEMINI_2_0_FLASH_LITE = "google/gemini-2.0-flash-lite-001"
    MISTRAL_NEMO = "mistralai/mistral-nemo"
    COHERE_COMMAND_R_08_2024 = "cohere/command-r-08-2024"
    COHERE_COMMAND_R_PLUS_08_2024 = "cohere/command-r-plus-08-2024"
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"  # Actually: DeepSeek V3
    DEEPSEEK_R1_0528 = "deepseek/deepseek-r1-0528"
    PERPLEXITY_SONAR = "perplexity/sonar"
    PERPLEXITY_SONAR_PRO = "perplexity/sonar-pro"
    PERPLEXITY_SONAR_DEEP_RESEARCH = "perplexity/sonar-deep-research"
    NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B = "nousresearch/hermes-3-llama-3.1-405b"
    NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B = "nousresearch/hermes-3-llama-3.1-70b"
    AMAZON_NOVA_LITE_V1 = "amazon/nova-lite-v1"
    AMAZON_NOVA_MICRO_V1 = "amazon/nova-micro-v1"
    AMAZON_NOVA_PRO_V1 = "amazon/nova-pro-v1"
    MICROSOFT_WIZARDLM_2_8X22B = "microsoft/wizardlm-2-8x22b"
    GRYPHE_MYTHOMAX_L2_13B = "gryphe/mythomax-l2-13b"
    META_LLAMA_4_SCOUT = "meta-llama/llama-4-scout"
    META_LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"
    GROK_4 = "x-ai/grok-4"
    KIMI_K2 = "moonshotai/kimi-k2"
    QWEN3_235B_A22B_THINKING = "qwen/qwen3-235b-a22b-thinking-2507"
    QWEN3_CODER = "qwen/qwen3-coder"
    # Llama API models
    LLAMA_API_LLAMA_4_SCOUT = "Llama-4-Scout-17B-16E-Instruct-FP8"
    LLAMA_API_LLAMA4_MAVERICK = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    LLAMA_API_LLAMA3_3_8B = "Llama-3.3-8B-Instruct"
    LLAMA_API_LLAMA3_3_70B = "Llama-3.3-70B-Instruct"
    # v0 by Vercel models
    V0_1_5_MD = "v0-1.5-md"
    V0_1_5_LG = "v0-1.5-lg"
    V0_1_0_MD = "v0-1.0-md"

    @property
    def metadata(self) -> ModelMetadata:
        return MODEL_METADATA[self]

    @property
    def provider(self) -> str:
        return self.metadata.provider

    @property
    def context_window(self) -> int:
        return self.metadata.context_window

    @property
    def max_output_tokens(self) -> int | None:
        return self.metadata.max_output_tokens


MODEL_METADATA = {
    # https://platform.openai.com/docs/models
    LlmModel.O3: ModelMetadata("openai", 200000, 100000),
    LlmModel.O3_MINI: ModelMetadata("openai", 200000, 100000),  # o3-mini-2025-01-31
    LlmModel.O1: ModelMetadata("openai", 200000, 100000),  # o1-2024-12-17
    LlmModel.O1_MINI: ModelMetadata("openai", 128000, 65536),  # o1-mini-2024-09-12
    # GPT-5 models
    LlmModel.GPT5: ModelMetadata("openai", 400000, 128000),
    LlmModel.GPT5_MINI: ModelMetadata("openai", 400000, 128000),
    LlmModel.GPT5_NANO: ModelMetadata("openai", 400000, 128000),
    LlmModel.GPT5_CHAT: ModelMetadata("openai", 400000, 16384),
    LlmModel.GPT41: ModelMetadata("openai", 1047576, 32768),
    LlmModel.GPT41_MINI: ModelMetadata("openai", 1047576, 32768),
    LlmModel.GPT4O_MINI: ModelMetadata(
        "openai", 128000, 16384
    ),  # gpt-4o-mini-2024-07-18
    LlmModel.GPT4O: ModelMetadata("openai", 128000, 16384),  # gpt-4o-2024-08-06
    LlmModel.GPT4_TURBO: ModelMetadata(
        "openai", 128000, 4096
    ),  # gpt-4-turbo-2024-04-09
    LlmModel.GPT3_5_TURBO: ModelMetadata("openai", 16385, 4096),  # gpt-3.5-turbo-0125
    # https://docs.anthropic.com/en/docs/about-claude/models
    LlmModel.CLAUDE_4_1_OPUS: ModelMetadata(
        "anthropic", 200000, 32000
    ),  # claude-opus-4-1-20250805
    LlmModel.CLAUDE_4_OPUS: ModelMetadata(
        "anthropic", 200000, 32000
    ),  # claude-4-opus-20250514
    LlmModel.CLAUDE_4_SONNET: ModelMetadata(
        "anthropic", 200000, 64000
    ),  # claude-4-sonnet-20250514
    LlmModel.CLAUDE_4_5_SONNET: ModelMetadata(
        "anthropic", 200000, 64000
    ),  # claude-sonnet-4-5-20250929
    LlmModel.CLAUDE_3_7_SONNET: ModelMetadata(
        "anthropic", 200000, 64000
    ),  # claude-3-7-sonnet-20250219
    LlmModel.CLAUDE_3_5_SONNET: ModelMetadata(
        "anthropic", 200000, 8192
    ),  # claude-3-5-sonnet-20241022
    LlmModel.CLAUDE_3_5_HAIKU: ModelMetadata(
        "anthropic", 200000, 8192
    ),  # claude-3-5-haiku-20241022
    LlmModel.CLAUDE_3_HAIKU: ModelMetadata(
        "anthropic", 200000, 4096
    ),  # claude-3-haiku-20240307
    # https://docs.aimlapi.com/api-overview/model-database/text-models
    LlmModel.AIML_API_QWEN2_5_72B: ModelMetadata("aiml_api", 32000, 8000),
    LlmModel.AIML_API_LLAMA3_1_70B: ModelMetadata("aiml_api", 128000, 40000),
    LlmModel.AIML_API_LLAMA3_3_70B: ModelMetadata("aiml_api", 128000, None),
    LlmModel.AIML_API_META_LLAMA_3_1_70B: ModelMetadata("aiml_api", 131000, 2000),
    LlmModel.AIML_API_LLAMA_3_2_3B: ModelMetadata("aiml_api", 128000, None),
    # https://console.groq.com/docs/models
    LlmModel.GEMMA2_9B: ModelMetadata("groq", 8192, None),
    LlmModel.LLAMA3_3_70B: ModelMetadata("groq", 128000, 32768),
    LlmModel.LLAMA3_1_8B: ModelMetadata("groq", 128000, 8192),
    LlmModel.LLAMA3_70B: ModelMetadata("groq", 8192, None),
    LlmModel.LLAMA3_8B: ModelMetadata("groq", 8192, None),
    LlmModel.DEEPSEEK_LLAMA_70B: ModelMetadata("groq", 128000, None),
    # https://ollama.com/library
    LlmModel.OLLAMA_LLAMA3_3: ModelMetadata("ollama", 8192, None),
    LlmModel.OLLAMA_LLAMA3_2: ModelMetadata("ollama", 8192, None),
    LlmModel.OLLAMA_LLAMA3_8B: ModelMetadata("ollama", 8192, None),
    LlmModel.OLLAMA_LLAMA3_405B: ModelMetadata("ollama", 8192, None),
    LlmModel.OLLAMA_DOLPHIN: ModelMetadata("ollama", 32768, None),
    # https://openrouter.ai/models
    LlmModel.GEMINI_FLASH_1_5: ModelMetadata("open_router", 1000000, 8192),
    LlmModel.GEMINI_2_5_PRO: ModelMetadata("open_router", 1050000, 8192),
    LlmModel.GEMINI_2_5_FLASH: ModelMetadata("open_router", 1048576, 65535),
    LlmModel.GEMINI_2_0_FLASH: ModelMetadata("open_router", 1048576, 8192),
    LlmModel.GEMINI_2_5_FLASH_LITE_PREVIEW: ModelMetadata(
        "open_router", 1048576, 65535
    ),
    LlmModel.GEMINI_2_0_FLASH_LITE: ModelMetadata("open_router", 1048576, 8192),
    LlmModel.MISTRAL_NEMO: ModelMetadata("open_router", 128000, 4096),
    LlmModel.COHERE_COMMAND_R_08_2024: ModelMetadata("open_router", 128000, 4096),
    LlmModel.COHERE_COMMAND_R_PLUS_08_2024: ModelMetadata("open_router", 128000, 4096),
    LlmModel.DEEPSEEK_CHAT: ModelMetadata("open_router", 64000, 2048),
    LlmModel.DEEPSEEK_R1_0528: ModelMetadata("open_router", 163840, 163840),
    LlmModel.PERPLEXITY_SONAR: ModelMetadata("open_router", 127000, 127000),
    LlmModel.PERPLEXITY_SONAR_PRO: ModelMetadata("open_router", 200000, 8000),
    LlmModel.PERPLEXITY_SONAR_DEEP_RESEARCH: ModelMetadata(
        "open_router",
        128000,
        128000,
    ),
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B: ModelMetadata(
        "open_router", 131000, 4096
    ),
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B: ModelMetadata(
        "open_router", 12288, 12288
    ),
    LlmModel.OPENAI_GPT_OSS_120B: ModelMetadata("open_router", 131072, 131072),
    LlmModel.OPENAI_GPT_OSS_20B: ModelMetadata("open_router", 131072, 32768),
    LlmModel.AMAZON_NOVA_LITE_V1: ModelMetadata("open_router", 300000, 5120),
    LlmModel.AMAZON_NOVA_MICRO_V1: ModelMetadata("open_router", 128000, 5120),
    LlmModel.AMAZON_NOVA_PRO_V1: ModelMetadata("open_router", 300000, 5120),
    LlmModel.MICROSOFT_WIZARDLM_2_8X22B: ModelMetadata("open_router", 65536, 4096),
    LlmModel.GRYPHE_MYTHOMAX_L2_13B: ModelMetadata("open_router", 4096, 4096),
    LlmModel.META_LLAMA_4_SCOUT: ModelMetadata("open_router", 131072, 131072),
    LlmModel.META_LLAMA_4_MAVERICK: ModelMetadata("open_router", 1048576, 1000000),
    LlmModel.GROK_4: ModelMetadata("open_router", 256000, 256000),
    LlmModel.KIMI_K2: ModelMetadata("open_router", 131000, 131000),
    LlmModel.QWEN3_235B_A22B_THINKING: ModelMetadata("open_router", 262144, 262144),
    LlmModel.QWEN3_CODER: ModelMetadata("open_router", 262144, 262144),
    # Llama API models
    LlmModel.LLAMA_API_LLAMA_4_SCOUT: ModelMetadata("llama_api", 128000, 4028),
    LlmModel.LLAMA_API_LLAMA4_MAVERICK: ModelMetadata("llama_api", 128000, 4028),
    LlmModel.LLAMA_API_LLAMA3_3_8B: ModelMetadata("llama_api", 128000, 4028),
    LlmModel.LLAMA_API_LLAMA3_3_70B: ModelMetadata("llama_api", 128000, 4028),
    # v0 by Vercel models
    LlmModel.V0_1_5_MD: ModelMetadata("v0", 128000, 64000),
    LlmModel.V0_1_5_LG: ModelMetadata("v0", 512000, 64000),
    LlmModel.V0_1_0_MD: ModelMetadata("v0", 128000, 64000),
}

for model in LlmModel:
    if model not in MODEL_METADATA:
        raise ValueError(f"Missing MODEL_METADATA metadata for model: {model}")


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
):
    """Get the appropriate parallel_tool_calls parameter for OpenAI-compatible APIs."""
    if llm_model.startswith("o") or parallel_tool_calls is None:
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
    provider = llm_model.metadata.provider
    context_window = llm_model.context_window

    if compress_prompt_to_fit:
        prompt = compress_prompt(
            messages=prompt,
            target_tokens=llm_model.context_window // 2,
            lossy_ok=True,
        )

    # Calculate available tokens based on context window and input length
    estimated_input_tokens = estimate_token_count(prompt)
    model_max_output = llm_model.max_output_tokens or int(2**15)
    user_max = max_tokens or model_max_output
    available_tokens = max(context_window - estimated_input_tokens, 0)
    max_tokens = max(min(available_tokens, model_max_output, user_max), 1)

    if provider == "openai":
        tools_param = tools if tools else openai.NOT_GIVEN
        oai_client = openai.AsyncOpenAI(api_key=credentials.api_key.get_secret_value())
        response_format = None

        parallel_tool_calls = get_parallel_tool_calls_param(
            llm_model, parallel_tool_calls
        )

        if force_json_output:
            response_format = {"type": "json_object"}

        response = await oai_client.chat.completions.create(
            model=llm_model.value,
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
                model=llm_model.value,
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
            model=llm_model.value,
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
            model=llm_model.value,
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
            llm_model, parallel_tool_calls
        )

        response = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://agpt.co",
                "X-Title": "AutoGPT",
            },
            model=llm_model.value,
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
            llm_model, parallel_tool_calls
        )

        response = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://agpt.co",
                "X-Title": "AutoGPT",
            },
            model=llm_model.value,
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
        client = openai.OpenAI(
            base_url="https://api.aimlapi.com/v2",
            api_key=credentials.api_key.get_secret_value(),
            default_headers={
                "X-Project": "AutoGPT",
                "X-Title": "AutoGPT",
                "HTTP-Referer": "https://github.com/Significant-Gravitas/AutoGPT",
            },
        )

        completion = client.chat.completions.create(
            model=llm_model.value,
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
            llm_model, parallel_tool_calls
        )

        response = await client.chat.completions.create(
            model=llm_model.value,
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
    class Input(BlockSchema):
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
            default=LlmModel.GPT4O,
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
        conversation_history: list[dict] = SchemaField(
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

    class Output(BlockSchema):
        response: dict[str, Any] | list[dict[str, Any]] = SchemaField(
            description="The response object generated by the language model."
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")
        error: str = SchemaField(description="Error message if the API call failed.")

    def __init__(self):
        super().__init__(
            id="ed55ac19-356e-4243-a6cb-bc599e9b716f",
            description="Call a Large Language Model (LLM) to generate formatted object based on the given prompt.",
            categories={BlockCategory.AI},
            input_schema=AIStructuredResponseGeneratorBlock.Input,
            output_schema=AIStructuredResponseGeneratorBlock.Output,
            test_input={
                "model": LlmModel.GPT4O,
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
        prompt = [json.to_dict(p) for p in input_data.conversation_history]

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
    class Input(BlockSchema):
        prompt: str = SchemaField(
            description="The prompt to send to the language model. You can use any of the {keys} from Prompt Values to fill in the prompt with values from the prompt values dictionary by putting them in curly braces.",
            placeholder="Enter your prompt here...",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=LlmModel.GPT4O,
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

    class Output(BlockSchema):
        response: str = SchemaField(
            description="The response generated by the language model."
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")
        error: str = SchemaField(description="Error message if the API call failed.")

    def __init__(self):
        super().__init__(
            id="1f292d4a-41a4-4977-9684-7c8d560b9f91",
            description="Call a Large Language Model (LLM) to generate a string based on the given prompt.",
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
    class Input(BlockSchema):
        text: str = SchemaField(
            description="The text to summarize.",
            placeholder="Enter the text to summarize here...",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=LlmModel.GPT4O,
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

    class Output(BlockSchema):
        summary: str = SchemaField(description="The final summary of the text.")
        prompt: list = SchemaField(description="The prompt sent to the language model.")
        error: str = SchemaField(description="Error message if the API call failed.")

    def __init__(self):
        super().__init__(
            id="a0a69be1-4528-491c-a85a-a4ab6873e3f0",
            description="Utilize a Large Language Model (LLM) to summarize a long text.",
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

        return llm_response["summary"]

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

            return llm_response["final_summary"]
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
    class Input(BlockSchema):
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
            default=LlmModel.GPT4O,
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

    class Output(BlockSchema):
        response: str = SchemaField(
            description="The model's response to the conversation."
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")
        error: str = SchemaField(description="Error message if the API call failed.")

    def __init__(self):
        super().__init__(
            id="32a87eab-381e-4dd4-bdb8-4c47151be35a",
            description="Advanced LLM call that takes a list of messages and sends them to the language model.",
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
                "model": LlmModel.GPT4O,
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
                "llm_call": lambda *args, **kwargs: "The 2020 World Series was played at Globe Life Field in Arlington, Texas."
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
        yield "response", response
        yield "prompt", self.prompt


class AIListGeneratorBlock(AIBlockBase):
    class Input(BlockSchema):
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
            default=LlmModel.GPT4O,
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

    class Output(BlockSchema):
        generated_list: List[str] = SchemaField(description="The generated list.")
        list_item: str = SchemaField(
            description="Each individual item in the list.",
        )
        prompt: list = SchemaField(description="The prompt sent to the language model.")
        error: str = SchemaField(
            description="Error message if the list generation failed."
        )

    def __init__(self):
        super().__init__(
            id="9c0b0450-d199-458b-a731-072189dd6593",
            description="Generate a Python list based on the given prompt using a Large Language Model (LLM).",
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
                "model": LlmModel.GPT4O,
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_retries": 3,
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
                    "response": "['Zylora Prime', 'Kharon-9', 'Vortexia', 'Oceara', 'Draknos']"
                },
            },
        )

    async def llm_call(
        self,
        input_data: AIStructuredResponseGeneratorBlock.Input,
        credentials: APIKeyCredentials,
    ) -> dict[str, str]:
        llm_block = AIStructuredResponseGeneratorBlock()
        response = await llm_block.run_once(
            input_data, "response", credentials=credentials
        )
        self.merge_llm_stats(llm_block)
        return response

    @staticmethod
    def string_to_list(string):
        """
        Converts a string representation of a list into an actual Python list object.
        """
        logger.debug(f"Converting string to list. Input string: {string}")
        try:
            # Use ast.literal_eval to safely evaluate the string
            python_list = ast.literal_eval(string)
            if isinstance(python_list, list):
                logger.debug(f"Successfully converted string to list: {python_list}")
                return python_list
            else:
                logger.error(f"The provided string '{string}' is not a valid list")
                raise ValueError(f"The provided string '{string}' is not a valid list.")
        except (SyntaxError, ValueError) as e:
            logger.error(f"Failed to convert string to list: {e}")
            raise ValueError("Invalid list format. Could not convert to list.")

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        logger.debug(f"Starting AIListGeneratorBlock.run with input data: {input_data}")

        # Check for API key
        api_key_check = credentials.api_key.get_secret_value()
        if not api_key_check:
            raise ValueError("No LLM API key provided.")

        # Prepare the system prompt
        sys_prompt = """You are a Python list generator. Your task is to generate a Python list based on the user's prompt. 
            |Respond ONLY with a valid python list. 
            |The list can contain strings, numbers, or nested lists as appropriate. 
            |Do not include any explanations or additional text.

            |Valid Example string formats:

            |Example 1:
            |```
            |['1', '2', '3', '4']
            |```

            |Example 2:
            |```
            |[['1', '2'], ['3', '4'], ['5', '6']]
            |```

            |Example 3:
            |```
            |['1', ['2', '3'], ['4', ['5', '6']]]
            |```

            |Example 4:
            |```
            |['a', 'b', 'c']
            |```

            |Example 5:
            |```
            |['1', '2.5', 'string', 'True', ['False', 'None']]
            |```

            |Do not include any explanations or additional text, just respond with the list in the format specified above.
            """
        # If a focus is provided, add it to the prompt
        if input_data.focus:
            prompt = f"Generate a list with the following focus:\n<focus>\n\n{input_data.focus}</focus>"
        else:
            # If there's source data
            if input_data.source_data:
                prompt = "Extract the main focus of the source data to a list.\ni.e if the source data is a news website, the focus would be the news stories rather than the social links in the footer."
            else:
                # No focus or source data provided, generat a random list
                prompt = "Generate a random list."

        # If the source data is provided, add it to the prompt
        if input_data.source_data:
            prompt += f"\n\nUse the following source data to generate the list from:\n\n<source_data>\n\n{input_data.source_data}</source_data>\n\nDo not invent fictional data that is not present in the source data."
        # Else, tell the LLM to synthesize the data
        else:
            prompt += "\n\nInvent the data to generate the list from."

        for attempt in range(input_data.max_retries):
            try:
                logger.debug("Calling LLM")
                llm_response = await self.llm_call(
                    AIStructuredResponseGeneratorBlock.Input(
                        sys_prompt=sys_prompt,
                        prompt=prompt,
                        credentials=input_data.credentials,
                        model=input_data.model,
                        expected_format={},  # Do not use structured response
                        ollama_host=input_data.ollama_host,
                    ),
                    credentials=credentials,
                )

                logger.debug(f"LLM response: {llm_response}")

                # Extract Response string
                response_string = llm_response["response"]
                logger.debug(f"Response string: {response_string}")

                # Convert the string to a Python list
                logger.debug("Converting string to Python list")
                parsed_list = self.string_to_list(response_string)
                logger.debug(f"Parsed list: {parsed_list}")

                # If we reach here, we have a valid Python list
                logger.debug("Successfully generated a valid Python list")
                yield "generated_list", parsed_list
                yield "prompt", self.prompt

                # Yield each item in the list
                for item in parsed_list:
                    yield "list_item", item
                return

            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == input_data.max_retries - 1:
                    logger.error(
                        f"Failed to generate a valid Python list after {input_data.max_retries} attempts"
                    )
                    raise RuntimeError(
                        f"Failed to generate a valid Python list after {input_data.max_retries} attempts. Last error: {str(e)}"
                    )
                else:
                    # Add a retry prompt
                    logger.debug("Preparing retry prompt")
                    prompt = f"""
                    The previous attempt failed due to `{e}`
                    Generate a valid Python list based on the original prompt.
                    Remember to respond ONLY with a valid Python list as per the format specified earlier.
                    Original prompt: 
                    ```{prompt}```
                    
                    Respond only with the list in the format specified with no commentary or apologies.
                    """
                    logger.debug(f"Retry prompt: {prompt}")

        logger.debug("AIListGeneratorBlock.run completed")
