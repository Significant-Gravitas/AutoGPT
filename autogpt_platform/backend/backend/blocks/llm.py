# This file contains a lot of prompt block strings that would trigger "line too long"
# flake8: noqa: E501
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

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
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
    display_name: str
    provider_name: str
    creator_name: str
    price_tier: Literal[1, 2, 3]


class LlmModelMeta(EnumMeta):
    pass


class LlmModel(str, Enum, metaclass=LlmModelMeta):
    # OpenAI models
    O3_MINI = "o3-mini"
    O3 = "o3-2025-04-16"
    O1 = "o1"
    O1_MINI = "o1-mini"
    # GPT-5 models
    GPT5_2 = "gpt-5.2-2025-12-11"
    GPT5_1 = "gpt-5.1-2025-11-13"
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
    CLAUDE_4_5_OPUS = "claude-opus-4-5-20251101"
    CLAUDE_4_5_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_4_5_HAIKU = "claude-haiku-4-5-20251001"
    CLAUDE_4_6_OPUS = "claude-opus-4-6"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    # AI/ML API models
    AIML_API_QWEN2_5_72B = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    AIML_API_LLAMA3_1_70B = "nvidia/llama-3.1-nemotron-70b-instruct"
    AIML_API_LLAMA3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    AIML_API_META_LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    AIML_API_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    # Groq models
    LLAMA3_3_70B = "llama-3.3-70b-versatile"
    LLAMA3_1_8B = "llama-3.1-8b-instant"
    # Ollama models
    OLLAMA_LLAMA3_3 = "llama3.3"
    OLLAMA_LLAMA3_2 = "llama3.2"
    OLLAMA_LLAMA3_8B = "llama3"
    OLLAMA_LLAMA3_405B = "llama3.1:405b"
    OLLAMA_DOLPHIN = "dolphin-mistral:latest"
    # OpenRouter models
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"
    OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro-preview-03-25"
    GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"
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
    GROK_4_FAST = "x-ai/grok-4-fast"
    GROK_4_1_FAST = "x-ai/grok-4.1-fast"
    GROK_CODE_FAST_1 = "x-ai/grok-code-fast-1"
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

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        json_schema = handler(schema)
        llm_model_metadata = {}
        for model in cls:
            model_name = model.value
            metadata = model.metadata
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
    LlmModel.O3: ModelMetadata("openai", 200000, 100000, "O3", "OpenAI", "OpenAI", 2),
    LlmModel.O3_MINI: ModelMetadata(
        "openai", 200000, 100000, "O3 Mini", "OpenAI", "OpenAI", 1
    ),  # o3-mini-2025-01-31
    LlmModel.O1: ModelMetadata(
        "openai", 200000, 100000, "O1", "OpenAI", "OpenAI", 3
    ),  # o1-2024-12-17
    LlmModel.O1_MINI: ModelMetadata(
        "openai", 128000, 65536, "O1 Mini", "OpenAI", "OpenAI", 2
    ),  # o1-mini-2024-09-12
    # GPT-5 models
    LlmModel.GPT5_2: ModelMetadata(
        "openai", 400000, 128000, "GPT-5.2", "OpenAI", "OpenAI", 3
    ),
    LlmModel.GPT5_1: ModelMetadata(
        "openai", 400000, 128000, "GPT-5.1", "OpenAI", "OpenAI", 2
    ),
    LlmModel.GPT5: ModelMetadata(
        "openai", 400000, 128000, "GPT-5", "OpenAI", "OpenAI", 1
    ),
    LlmModel.GPT5_MINI: ModelMetadata(
        "openai", 400000, 128000, "GPT-5 Mini", "OpenAI", "OpenAI", 1
    ),
    LlmModel.GPT5_NANO: ModelMetadata(
        "openai", 400000, 128000, "GPT-5 Nano", "OpenAI", "OpenAI", 1
    ),
    LlmModel.GPT5_CHAT: ModelMetadata(
        "openai", 400000, 16384, "GPT-5 Chat Latest", "OpenAI", "OpenAI", 2
    ),
    LlmModel.GPT41: ModelMetadata(
        "openai", 1047576, 32768, "GPT-4.1", "OpenAI", "OpenAI", 1
    ),
    LlmModel.GPT41_MINI: ModelMetadata(
        "openai", 1047576, 32768, "GPT-4.1 Mini", "OpenAI", "OpenAI", 1
    ),
    LlmModel.GPT4O_MINI: ModelMetadata(
        "openai", 128000, 16384, "GPT-4o Mini", "OpenAI", "OpenAI", 1
    ),  # gpt-4o-mini-2024-07-18
    LlmModel.GPT4O: ModelMetadata(
        "openai", 128000, 16384, "GPT-4o", "OpenAI", "OpenAI", 2
    ),  # gpt-4o-2024-08-06
    LlmModel.GPT4_TURBO: ModelMetadata(
        "openai", 128000, 4096, "GPT-4 Turbo", "OpenAI", "OpenAI", 3
    ),  # gpt-4-turbo-2024-04-09
    LlmModel.GPT3_5_TURBO: ModelMetadata(
        "openai", 16385, 4096, "GPT-3.5 Turbo", "OpenAI", "OpenAI", 1
    ),  # gpt-3.5-turbo-0125
    # https://docs.anthropic.com/en/docs/about-claude/models
    LlmModel.CLAUDE_4_1_OPUS: ModelMetadata(
        "anthropic", 200000, 32000, "Claude Opus 4.1", "Anthropic", "Anthropic", 3
    ),  # claude-opus-4-1-20250805
    LlmModel.CLAUDE_4_OPUS: ModelMetadata(
        "anthropic", 200000, 32000, "Claude Opus 4", "Anthropic", "Anthropic", 3
    ),  # claude-4-opus-20250514
    LlmModel.CLAUDE_4_SONNET: ModelMetadata(
        "anthropic", 200000, 64000, "Claude Sonnet 4", "Anthropic", "Anthropic", 2
    ),  # claude-4-sonnet-20250514
    LlmModel.CLAUDE_4_6_OPUS: ModelMetadata(
        "anthropic", 200000, 128000, "Claude Opus 4.6", "Anthropic", "Anthropic", 3
    ),  # claude-opus-4-6
    LlmModel.CLAUDE_4_5_OPUS: ModelMetadata(
        "anthropic", 200000, 64000, "Claude Opus 4.5", "Anthropic", "Anthropic", 3
    ),  # claude-opus-4-5-20251101
    LlmModel.CLAUDE_4_5_SONNET: ModelMetadata(
        "anthropic", 200000, 64000, "Claude Sonnet 4.5", "Anthropic", "Anthropic", 3
    ),  # claude-sonnet-4-5-20250929
    LlmModel.CLAUDE_4_5_HAIKU: ModelMetadata(
        "anthropic", 200000, 64000, "Claude Haiku 4.5", "Anthropic", "Anthropic", 2
    ),  # claude-haiku-4-5-20251001
    LlmModel.CLAUDE_3_HAIKU: ModelMetadata(
        "anthropic", 200000, 4096, "Claude 3 Haiku", "Anthropic", "Anthropic", 1
    ),  # claude-3-haiku-20240307
    # https://docs.aimlapi.com/api-overview/model-database/text-models
    LlmModel.AIML_API_QWEN2_5_72B: ModelMetadata(
        "aiml_api", 32000, 8000, "Qwen 2.5 72B Instruct Turbo", "AI/ML", "Qwen", 1
    ),
    LlmModel.AIML_API_LLAMA3_1_70B: ModelMetadata(
        "aiml_api",
        128000,
        40000,
        "Llama 3.1 Nemotron 70B Instruct",
        "AI/ML",
        "Nvidia",
        1,
    ),
    LlmModel.AIML_API_LLAMA3_3_70B: ModelMetadata(
        "aiml_api", 128000, None, "Llama 3.3 70B Instruct Turbo", "AI/ML", "Meta", 1
    ),
    LlmModel.AIML_API_META_LLAMA_3_1_70B: ModelMetadata(
        "aiml_api", 131000, 2000, "Llama 3.1 70B Instruct Turbo", "AI/ML", "Meta", 1
    ),
    LlmModel.AIML_API_LLAMA_3_2_3B: ModelMetadata(
        "aiml_api", 128000, None, "Llama 3.2 3B Instruct Turbo", "AI/ML", "Meta", 1
    ),
    # https://console.groq.com/docs/models
    LlmModel.LLAMA3_3_70B: ModelMetadata(
        "groq", 128000, 32768, "Llama 3.3 70B Versatile", "Groq", "Meta", 1
    ),
    LlmModel.LLAMA3_1_8B: ModelMetadata(
        "groq", 128000, 8192, "Llama 3.1 8B Instant", "Groq", "Meta", 1
    ),
    # https://ollama.com/library
    LlmModel.OLLAMA_LLAMA3_3: ModelMetadata(
        "ollama", 8192, None, "Llama 3.3", "Ollama", "Meta", 1
    ),
    LlmModel.OLLAMA_LLAMA3_2: ModelMetadata(
        "ollama", 8192, None, "Llama 3.2", "Ollama", "Meta", 1
    ),
    LlmModel.OLLAMA_LLAMA3_8B: ModelMetadata(
        "ollama", 8192, None, "Llama 3", "Ollama", "Meta", 1
    ),
    LlmModel.OLLAMA_LLAMA3_405B: ModelMetadata(
        "ollama", 8192, None, "Llama 3.1 405B", "Ollama", "Meta", 1
    ),
    LlmModel.OLLAMA_DOLPHIN: ModelMetadata(
        "ollama", 32768, None, "Dolphin Mistral Latest", "Ollama", "Mistral AI", 1
    ),
    # https://openrouter.ai/models
    LlmModel.GEMINI_2_5_PRO: ModelMetadata(
        "open_router",
        1050000,
        8192,
        "Gemini 2.5 Pro Preview 03.25",
        "OpenRouter",
        "Google",
        2,
    ),
    LlmModel.GEMINI_3_PRO_PREVIEW: ModelMetadata(
        "open_router", 1048576, 65535, "Gemini 3 Pro Preview", "OpenRouter", "Google", 2
    ),
    LlmModel.GEMINI_2_5_FLASH: ModelMetadata(
        "open_router", 1048576, 65535, "Gemini 2.5 Flash", "OpenRouter", "Google", 1
    ),
    LlmModel.GEMINI_2_0_FLASH: ModelMetadata(
        "open_router", 1048576, 8192, "Gemini 2.0 Flash 001", "OpenRouter", "Google", 1
    ),
    LlmModel.GEMINI_2_5_FLASH_LITE_PREVIEW: ModelMetadata(
        "open_router",
        1048576,
        65535,
        "Gemini 2.5 Flash Lite Preview 06.17",
        "OpenRouter",
        "Google",
        1,
    ),
    LlmModel.GEMINI_2_0_FLASH_LITE: ModelMetadata(
        "open_router",
        1048576,
        8192,
        "Gemini 2.0 Flash Lite 001",
        "OpenRouter",
        "Google",
        1,
    ),
    LlmModel.MISTRAL_NEMO: ModelMetadata(
        "open_router", 128000, 4096, "Mistral Nemo", "OpenRouter", "Mistral AI", 1
    ),
    LlmModel.COHERE_COMMAND_R_08_2024: ModelMetadata(
        "open_router", 128000, 4096, "Command R 08.2024", "OpenRouter", "Cohere", 1
    ),
    LlmModel.COHERE_COMMAND_R_PLUS_08_2024: ModelMetadata(
        "open_router", 128000, 4096, "Command R Plus 08.2024", "OpenRouter", "Cohere", 2
    ),
    LlmModel.DEEPSEEK_CHAT: ModelMetadata(
        "open_router", 64000, 2048, "DeepSeek Chat", "OpenRouter", "DeepSeek", 1
    ),
    LlmModel.DEEPSEEK_R1_0528: ModelMetadata(
        "open_router", 163840, 163840, "DeepSeek R1 0528", "OpenRouter", "DeepSeek", 1
    ),
    LlmModel.PERPLEXITY_SONAR: ModelMetadata(
        "open_router", 127000, 8000, "Sonar", "OpenRouter", "Perplexity", 1
    ),
    LlmModel.PERPLEXITY_SONAR_PRO: ModelMetadata(
        "open_router", 200000, 8000, "Sonar Pro", "OpenRouter", "Perplexity", 2
    ),
    LlmModel.PERPLEXITY_SONAR_DEEP_RESEARCH: ModelMetadata(
        "open_router",
        128000,
        16000,
        "Sonar Deep Research",
        "OpenRouter",
        "Perplexity",
        3,
    ),
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B: ModelMetadata(
        "open_router",
        131000,
        4096,
        "Hermes 3 Llama 3.1 405B",
        "OpenRouter",
        "Nous Research",
        1,
    ),
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B: ModelMetadata(
        "open_router",
        12288,
        12288,
        "Hermes 3 Llama 3.1 70B",
        "OpenRouter",
        "Nous Research",
        1,
    ),
    LlmModel.OPENAI_GPT_OSS_120B: ModelMetadata(
        "open_router", 131072, 131072, "GPT-OSS 120B", "OpenRouter", "OpenAI", 1
    ),
    LlmModel.OPENAI_GPT_OSS_20B: ModelMetadata(
        "open_router", 131072, 32768, "GPT-OSS 20B", "OpenRouter", "OpenAI", 1
    ),
    LlmModel.AMAZON_NOVA_LITE_V1: ModelMetadata(
        "open_router", 300000, 5120, "Nova Lite V1", "OpenRouter", "Amazon", 1
    ),
    LlmModel.AMAZON_NOVA_MICRO_V1: ModelMetadata(
        "open_router", 128000, 5120, "Nova Micro V1", "OpenRouter", "Amazon", 1
    ),
    LlmModel.AMAZON_NOVA_PRO_V1: ModelMetadata(
        "open_router", 300000, 5120, "Nova Pro V1", "OpenRouter", "Amazon", 1
    ),
    LlmModel.MICROSOFT_WIZARDLM_2_8X22B: ModelMetadata(
        "open_router", 65536, 4096, "WizardLM 2 8x22B", "OpenRouter", "Microsoft", 1
    ),
    LlmModel.GRYPHE_MYTHOMAX_L2_13B: ModelMetadata(
        "open_router", 4096, 4096, "MythoMax L2 13B", "OpenRouter", "Gryphe", 1
    ),
    LlmModel.META_LLAMA_4_SCOUT: ModelMetadata(
        "open_router", 131072, 131072, "Llama 4 Scout", "OpenRouter", "Meta", 1
    ),
    LlmModel.META_LLAMA_4_MAVERICK: ModelMetadata(
        "open_router", 1048576, 1000000, "Llama 4 Maverick", "OpenRouter", "Meta", 1
    ),
    LlmModel.GROK_4: ModelMetadata(
        "open_router", 256000, 256000, "Grok 4", "OpenRouter", "xAI", 3
    ),
    LlmModel.GROK_4_FAST: ModelMetadata(
        "open_router", 2000000, 30000, "Grok 4 Fast", "OpenRouter", "xAI", 1
    ),
    LlmModel.GROK_4_1_FAST: ModelMetadata(
        "open_router", 2000000, 30000, "Grok 4.1 Fast", "OpenRouter", "xAI", 1
    ),
    LlmModel.GROK_CODE_FAST_1: ModelMetadata(
        "open_router", 256000, 10000, "Grok Code Fast 1", "OpenRouter", "xAI", 1
    ),
    LlmModel.KIMI_K2: ModelMetadata(
        "open_router", 131000, 131000, "Kimi K2", "OpenRouter", "Moonshot AI", 1
    ),
    LlmModel.QWEN3_235B_A22B_THINKING: ModelMetadata(
        "open_router",
        262144,
        262144,
        "Qwen 3 235B A22B Thinking 2507",
        "OpenRouter",
        "Qwen",
        1,
    ),
    LlmModel.QWEN3_CODER: ModelMetadata(
        "open_router", 262144, 262144, "Qwen 3 Coder", "OpenRouter", "Qwen", 3
    ),
    # Llama API models
    LlmModel.LLAMA_API_LLAMA_4_SCOUT: ModelMetadata(
        "llama_api",
        128000,
        4028,
        "Llama 4 Scout 17B 16E Instruct FP8",
        "Llama API",
        "Meta",
        1,
    ),
    LlmModel.LLAMA_API_LLAMA4_MAVERICK: ModelMetadata(
        "llama_api",
        128000,
        4028,
        "Llama 4 Maverick 17B 128E Instruct FP8",
        "Llama API",
        "Meta",
        1,
    ),
    LlmModel.LLAMA_API_LLAMA3_3_8B: ModelMetadata(
        "llama_api", 128000, 4028, "Llama 3.3 8B Instruct", "Llama API", "Meta", 1
    ),
    LlmModel.LLAMA_API_LLAMA3_3_70B: ModelMetadata(
        "llama_api", 128000, 4028, "Llama 3.3 70B Instruct", "Llama API", "Meta", 1
    ),
    # v0 by Vercel models
    LlmModel.V0_1_5_MD: ModelMetadata("v0", 128000, 64000, "v0 1.5 MD", "V0", "V0", 1),
    LlmModel.V0_1_5_LG: ModelMetadata("v0", 512000, 64000, "v0 1.5 LG", "V0", "V0", 1),
    LlmModel.V0_1_0_MD: ModelMetadata("v0", 128000, 64000, "v0 1.0 MD", "V0", "V0", 1),
}

DEFAULT_LLM_MODEL = LlmModel.GPT5_2

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
    if llm_model.startswith("o") or parallel_tool_calls is None:
        return openai.omit
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
        model: LlmModel = SchemaField(
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
        model: LlmModel = SchemaField(
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
