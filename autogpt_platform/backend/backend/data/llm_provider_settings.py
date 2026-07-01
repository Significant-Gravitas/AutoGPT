"""Persistence and encryption helpers for global LLM provider settings."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

SETTINGS_ID = "default"


class PersistedLlmProviderSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = SETTINGS_ID
    enabled: bool
    provider: str
    use_local: bool = Field(alias="useLocal")
    base_url: str | None = Field(alias="baseUrl")
    encrypted_api_key: str | None = Field(alias="encryptedApiKey")
    model: str
    title_model: str = Field(alias="titleModel")
    fast_standard_model: str = Field(alias="fastStandardModel")
    fast_advanced_model: str = Field(alias="fastAdvancedModel")
    thinking_standard_model: str = Field(alias="thinkingStandardModel")
    thinking_advanced_model: str = Field(alias="thinkingAdvancedModel")
    claude_agent_fallback_model: str = Field(alias="claudeAgentFallbackModel")
    request_timeout_s: float = Field(alias="requestTimeoutS")
    max_retries: int = Field(alias="maxRetries")
    local_request_timeout_s: float = Field(alias="localRequestTimeoutS")
    store_embedding_model: str = Field(alias="storeEmbeddingModel")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class LlmProviderSettingsWrite(BaseModel):
    enabled: bool
    provider: str
    use_local: bool
    base_url: str | None
    model: str
    title_model: str
    fast_standard_model: str
    fast_advanced_model: str
    thinking_standard_model: str
    thinking_advanced_model: str
    claude_agent_fallback_model: str
    request_timeout_s: float
    max_retries: int
    local_request_timeout_s: float
    store_embedding_model: str


async def get_llm_provider_settings() -> PersistedLlmProviderSettings | None:
    from backend.data.db import query_raw_with_schema

    rows = await query_raw_with_schema(
        'SELECT * FROM {schema_prefix}"LlmProviderSettings" WHERE "id" = $1',
        SETTINGS_ID,
        model=PersistedLlmProviderSettings,
    )
    return rows[0] if rows else None


async def upsert_llm_provider_settings(
    settings: LlmProviderSettingsWrite,
    *,
    encrypted_api_key: str | None,
) -> PersistedLlmProviderSettings:
    from backend.data.db import query_raw_with_schema

    rows = await query_raw_with_schema(
        """
        INSERT INTO {schema_prefix}"LlmProviderSettings" (
            "id", "enabled", "provider", "useLocal", "baseUrl",
            "encryptedApiKey", "model", "titleModel", "fastStandardModel",
            "fastAdvancedModel", "thinkingStandardModel",
            "thinkingAdvancedModel", "claudeAgentFallbackModel",
            "requestTimeoutS", "maxRetries", "localRequestTimeoutS",
            "storeEmbeddingModel", "updatedAt"
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
            $14, $15, $16, $17, CURRENT_TIMESTAMP
        )
        ON CONFLICT ("id") DO UPDATE SET
            "enabled" = EXCLUDED."enabled",
            "provider" = EXCLUDED."provider",
            "useLocal" = EXCLUDED."useLocal",
            "baseUrl" = EXCLUDED."baseUrl",
            "encryptedApiKey" = EXCLUDED."encryptedApiKey",
            "model" = EXCLUDED."model",
            "titleModel" = EXCLUDED."titleModel",
            "fastStandardModel" = EXCLUDED."fastStandardModel",
            "fastAdvancedModel" = EXCLUDED."fastAdvancedModel",
            "thinkingStandardModel" = EXCLUDED."thinkingStandardModel",
            "thinkingAdvancedModel" = EXCLUDED."thinkingAdvancedModel",
            "claudeAgentFallbackModel" = EXCLUDED."claudeAgentFallbackModel",
            "requestTimeoutS" = EXCLUDED."requestTimeoutS",
            "maxRetries" = EXCLUDED."maxRetries",
            "localRequestTimeoutS" = EXCLUDED."localRequestTimeoutS",
            "storeEmbeddingModel" = EXCLUDED."storeEmbeddingModel",
            "updatedAt" = CURRENT_TIMESTAMP
        RETURNING *
        """,
        SETTINGS_ID,
        settings.enabled,
        settings.provider,
        settings.use_local,
        settings.base_url,
        encrypted_api_key,
        settings.model,
        settings.title_model,
        settings.fast_standard_model,
        settings.fast_advanced_model,
        settings.thinking_standard_model,
        settings.thinking_advanced_model,
        settings.claude_agent_fallback_model,
        settings.request_timeout_s,
        settings.max_retries,
        settings.local_request_timeout_s,
        settings.store_embedding_model,
        model=PersistedLlmProviderSettings,
    )
    return rows[0]


def encrypt_llm_api_key(api_key: str) -> str:
    from backend.util.encryption import JSONCryptor

    return JSONCryptor().encrypt({"api_key": api_key})


def decrypt_llm_api_key(encrypted_api_key: str | None) -> str | None:
    if not encrypted_api_key:
        return None
    from backend.util.encryption import JSONCryptor

    value = JSONCryptor().decrypt(encrypted_api_key).get("api_key")
    return value if isinstance(value, str) and value else None
