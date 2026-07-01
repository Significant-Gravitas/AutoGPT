"""OpenAI-only configuration for database-backed embeddings."""

from pydantic import BaseModel, ConfigDict, Field, SecretStr

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = "openai"
    model: str = DEFAULT_EMBEDDING_MODEL
    api_key: SecretStr | None = Field(default=None, exclude=True)
    api_key_source: str = "OPENAI_INTERNAL_API_KEY"

    @property
    def key_present(self) -> bool:
        return self.api_key is not None


def resolve_embedding_config(
    *,
    openai_internal_api_key: str,
    model: str | None = None,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        model=(model or DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL,
        api_key=(
            SecretStr(openai_internal_api_key) if openai_internal_api_key else None
        ),
    )
