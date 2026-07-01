from backend.util.llm.embedding_config import resolve_embedding_config


def test_embedding_config_is_openai_only_and_masks_key() -> None:
    config = resolve_embedding_config(
        openai_internal_api_key="embedding-secret",
        model=None,
    )

    assert config.provider == "openai"
    assert config.model == "text-embedding-3-small"
    assert config.api_key_source == "OPENAI_INTERNAL_API_KEY"
    assert config.key_present is True
    assert "api_key" not in config.model_dump()
