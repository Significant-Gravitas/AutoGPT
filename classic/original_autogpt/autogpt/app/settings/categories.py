"""Settings category definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .introspection import SettingInfo


@dataclass
class Category:
    """A category of settings."""

    id: str
    name: str
    description: str
    env_vars: list[str] = field(default_factory=list)
    icon: str = ""

    def get_settings(
        self, all_settings: dict[str, "SettingInfo"]
    ) -> list["SettingInfo"]:
        """Get the settings in this category."""
        return [all_settings[env] for env in self.env_vars if env in all_settings]


# Define all categories with their env vars
CATEGORIES: list[Category] = [
    Category(
        id="api_keys",
        name="API Keys",
        description="LLM provider API keys",
        icon="🔑",
        env_vars=[
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
        ],
    ),
    Category(
        id="models",
        name="Models",
        description="LLM model configuration",
        icon="🤖",
        env_vars=[
            "SMART_LLM",
            "FAST_LLM",
            "TEMPERATURE",
            "EMBEDDING_MODEL",
            "PROMPT_STRATEGY",
            "THINKING_BUDGET_TOKENS",
            "REASONING_EFFORT",
        ],
    ),
    Category(
        id="search",
        name="Web Search",
        description="Search provider configuration",
        icon="🔍",
        env_vars=[
            "TAVILY_API_KEY",
            "SERPER_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_CUSTOM_SEARCH_ENGINE_ID",
        ],
    ),
    Category(
        id="storage",
        name="Storage",
        description="File storage configuration",
        icon="💾",
        env_vars=[
            "FILE_STORAGE_BACKEND",
            "STORAGE_BUCKET",
            "S3_ENDPOINT_URL",
            "RESTRICT_TO_WORKSPACE",
        ],
    ),
    Category(
        id="image",
        name="Image Gen",
        description="Image generation configuration",
        icon="🎨",
        env_vars=[
            "HUGGINGFACE_API_TOKEN",
            "SD_WEBUI_AUTH",
        ],
    ),
    Category(
        id="github",
        name="GitHub",
        description="GitHub integration",
        icon="🐙",
        env_vars=[
            "GITHUB_API_KEY",
            "GITHUB_USERNAME",
        ],
    ),
    Category(
        id="tts",
        name="Text-to-Speech",
        description="Text-to-speech configuration",
        icon="🔊",
        env_vars=[
            "TEXT_TO_SPEECH_PROVIDER",
            "ELEVENLABS_API_KEY",
            "ELEVENLABS_VOICE_ID",
            "STREAMELEMENTS_VOICE",
        ],
    ),
    Category(
        id="logging",
        name="Logging",
        description="Logging configuration",
        icon="📝",
        env_vars=[
            "LOG_LEVEL",
            "LOG_FORMAT",
            "LOG_FILE_FORMAT",
            "PLAIN_OUTPUT",
        ],
    ),
    Category(
        id="server",
        name="Server",
        description="Agent Protocol server settings",
        icon="🌐",
        env_vars=[
            "AP_SERVER_PORT",
            "AP_SERVER_DB_URL",
            "AP_SERVER_CORS_ALLOWED_ORIGINS",
        ],
    ),
    Category(
        id="app",
        name="Application",
        description="Application settings",
        icon="⚙️",
        env_vars=[
            "AUTHORISE_COMMAND_KEY",
            "EXIT_KEY",
            "NONINTERACTIVE_MODE",
            "DISABLED_COMMANDS",
            "TELEMETRY_OPT_IN",
            "COMPONENT_CONFIG_FILE",
        ],
    ),
    Category(
        id="openai",
        name="OpenAI",
        description="OpenAI-specific settings",
        icon="🟢",
        env_vars=[
            "OPENAI_API_BASE_URL",
            "OPENAI_ORGANIZATION",
            "OPENAI_API_TYPE",
            "OPENAI_API_VERSION",
            "AZURE_CONFIG_FILE",
        ],
    ),
    Category(
        id="anthropic",
        name="Anthropic",
        description="Anthropic-specific settings",
        icon="🟠",
        env_vars=[
            "ANTHROPIC_API_BASE_URL",
        ],
    ),
    Category(
        id="groq",
        name="Groq",
        description="Groq-specific settings",
        icon="🟣",
        env_vars=[
            "GROQ_API_BASE_URL",
        ],
    ),
    Category(
        id="platform",
        name="Platform",
        description="AutoGPT Platform integration",
        icon="🚀",
        env_vars=[
            "PLATFORM_API_KEY",
            "PLATFORM_BLOCKS_ENABLED",
            "PLATFORM_URL",
            "PLATFORM_TIMEOUT",
        ],
    ),
    Category(
        id="local_llm",
        name="Local LLM",
        description="Local LLM configuration (Llamafile, etc.)",
        icon="💻",
        env_vars=[
            "LLAMAFILE_API_BASE",
        ],
    ),
]


def get_category_by_id(category_id: str) -> Category | None:
    """Get a category by its ID."""
    for cat in CATEGORIES:
        if cat.id == category_id:
            return cat
    return None


def get_categories_for_display() -> list[Category]:
    """Get categories in display order, filtered to those with settings."""
    # Return categories that have at least one env var defined
    return [cat for cat in CATEGORIES if cat.env_vars]


def categorize_env_vars() -> dict[str, str]:
    """Create a mapping of env var to category id."""
    mapping: dict[str, str] = {}
    for cat in CATEGORIES:
        for env_var in cat.env_vars:
            mapping[env_var] = cat.id
    return mapping
