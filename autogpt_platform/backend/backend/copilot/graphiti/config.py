"""Configuration for Graphiti temporal knowledge graph integration."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from backend.util.clients import OPENROUTER_BASE_URL

_BACKEND_ROOT = Path(__file__).resolve().parents[3]


class GraphitiConfig(BaseSettings):
    """Configuration for Graphiti memory integration.

    All fields use the ``GRAPHITI_`` env-var prefix, e.g. ``GRAPHITI_ENABLED``.
    LLM/embedder keys fall back to the platform-wide OpenRouter and OpenAI keys
    when left empty so that operators don't need to manage separate credentials.
    """

    model_config = SettingsConfigDict(env_prefix="GRAPHITI_", extra="allow")

    # FalkorDB connection
    falkordb_host: str = Field(default="localhost")
    falkordb_port: int = Field(default=6380)
    falkordb_password: str = Field(default="")

    # LLM for entity extraction (used by graphiti-core during ingestion)
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="Model for entity extraction — must support structured output",
    )
    llm_base_url: str = Field(
        default="",
        description="Base URL for LLM API — empty falls back to OPENROUTER_BASE_URL",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for LLM — empty falls back to OPEN_ROUTER_API_KEY",
    )

    # Embedder (separate from LLM — embeddings go direct to OpenAI)
    embedder_model: str = Field(default="text-embedding-3-small")
    embedder_base_url: str = Field(
        default="",
        description="Base URL for embedder — empty uses OpenAI direct",
    )
    embedder_api_key: str = Field(
        default="",
        description="API key for embedder — empty falls back to OPENAI_API_KEY",
    )

    # Concurrency
    semaphore_limit: int = Field(
        default=5,
        description="Max concurrent LLM calls during ingestion (prevents rate limits)",
    )

    # Warm context
    context_max_facts: int = Field(default=20)
    context_timeout: float = Field(
        default=8.0,
        description="Seconds before warm context fetch is abandoned (needs headroom for FalkorDB cold connections)",
    )

    # Client cache
    client_cache_maxsize: int = Field(default=500)
    client_cache_ttl: int = Field(
        default=1800,
        description="TTL in seconds for cached Graphiti client instances (30 min)",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            file_secret_settings,
            DotEnvSettingsSource(settings_cls, env_file=_BACKEND_ROOT / ".env"),
            DotEnvSettingsSource(settings_cls, env_file=_BACKEND_ROOT / ".env.default"),
        )

    def resolve_llm_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        return os.getenv("OPEN_ROUTER_API_KEY", "")

    def resolve_llm_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        return OPENROUTER_BASE_URL

    def resolve_embedder_api_key(self) -> str:
        if self.embedder_api_key:
            return self.embedder_api_key
        return os.getenv("OPENAI_API_KEY", "")

    def resolve_embedder_base_url(self) -> str | None:
        if self.embedder_base_url:
            return self.embedder_base_url
        return None  # OpenAI SDK default


_graphiti_config: GraphitiConfig | None = None


def _get_config() -> GraphitiConfig:
    global _graphiti_config
    if _graphiti_config is None:
        _graphiti_config = GraphitiConfig()
    return _graphiti_config


# Backwards-compatible module-level attribute access.
# All internal code should use ``_get_config()`` to avoid import-time
# construction, but this keeps existing ``graphiti_config.xxx`` usage working.
class _LazyConfigProxy:
    def __getattr__(self, name: str):
        return getattr(_get_config(), name)


graphiti_config = _LazyConfigProxy()  # type: ignore[assignment]


async def is_enabled_for_user(user_id: str | None) -> bool:
    """Check if Graphiti memory is enabled for a specific user.

    Gated solely by LaunchDarkly flag ``graphiti-memory``
    (Flag.GRAPHITI_MEMORY).  When LD is not configured, defaults to False.
    """
    if not user_id:
        return False

    from backend.util.feature_flag import Flag, is_feature_enabled

    return await is_feature_enabled(
        Flag.GRAPHITI_MEMORY,
        user_id,
        default=False,
    )
