"""Unit tests for the LLM registry module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pydantic

from backend.data.llm_registry.registry import (
    RegistryModel,
    RegistryModelCost,
    RegistryModelCreator,
    _build_schema_options,
    _record_to_registry_model,
    clear_registry_cache,
    get_all_model_slugs_for_validation,
    get_all_models,
    get_default_model_slug,
    get_enabled_models,
    get_model,
    get_schema_options,
    refresh_llm_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_record(**overrides):
    """Build a realistic mock Prisma LlmModel record."""
    provider = Mock()
    provider.name = "openai"
    provider.displayName = "OpenAI"

    record = Mock()
    record.slug = "openai/gpt-4o"
    record.displayName = "GPT-4o"
    record.description = "Latest GPT model"
    record.providerId = "provider-uuid"
    record.Provider = provider
    record.creatorId = "creator-uuid"
    record.Creator = None
    record.contextWindow = 128000
    record.maxOutputTokens = 16384
    record.priceTier = 2
    record.isEnabled = True
    record.isRecommended = False
    record.supportsTools = True
    record.supportsJsonOutput = True
    record.supportsReasoning = False
    record.supportsParallelToolCalls = True
    record.capabilities = {}
    record.metadata = {}
    record.Costs = []

    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def _make_registry_model(**kwargs) -> RegistryModel:
    """Build a minimal RegistryModel for testing registry-level functions."""
    from backend.blocks.llm import ModelMetadata

    defaults = dict(
        slug="openai/gpt-4o",
        display_name="GPT-4o",
        description=None,
        metadata=ModelMetadata(
            provider="openai",
            context_window=128000,
            max_output_tokens=16384,
            display_name="GPT-4o",
            provider_name="OpenAI",
            creator_name="Unknown",
            price_tier=2,
        ),
        capabilities={},
        extra_metadata={},
        provider_display_name="OpenAI",
        is_enabled=True,
        is_recommended=False,
    )
    defaults.update(kwargs)
    return RegistryModel(**defaults)


# ---------------------------------------------------------------------------
# _record_to_registry_model tests
# ---------------------------------------------------------------------------


def test_record_to_registry_model():
    """Happy-path: well-formed record produces a correct RegistryModel."""
    record = _make_mock_record()
    model = _record_to_registry_model(record)

    assert model.slug == "openai/gpt-4o"
    assert model.display_name == "GPT-4o"
    assert model.description == "Latest GPT model"
    assert model.provider_display_name == "OpenAI"
    assert model.is_enabled is True
    assert model.is_recommended is False
    assert model.supports_tools is True
    assert model.supports_json_output is True
    assert model.supports_reasoning is False
    assert model.supports_parallel_tool_calls is True
    assert model.metadata.provider == "openai"
    assert model.metadata.context_window == 128000
    assert model.metadata.max_output_tokens == 16384
    assert model.metadata.price_tier == 2
    assert model.creator is None
    assert model.costs == ()


def test_record_to_registry_model_missing_provider(caplog):
    """Record with no Provider relation falls back to providerId and logs a warning."""
    record = _make_mock_record(Provider=None, providerId="provider-uuid")
    with caplog.at_level("WARNING"):
        model = _record_to_registry_model(record)

    assert "no Provider" in caplog.text
    assert model.metadata.provider == "provider-uuid"
    assert model.provider_display_name == "provider-uuid"


def test_record_to_registry_model_missing_creator():
    """When Creator is None, creator_name defaults to 'Unknown' and creator field is None."""
    record = _make_mock_record(Creator=None)
    model = _record_to_registry_model(record)

    assert model.creator is None
    assert model.metadata.creator_name == "Unknown"


def test_record_to_registry_model_with_creator():
    """When Creator is present, it is parsed into RegistryModelCreator."""
    creator_mock = Mock()
    creator_mock.id = "creator-uuid"
    creator_mock.name = "openai"
    creator_mock.displayName = "OpenAI"
    creator_mock.description = "AI company"
    creator_mock.websiteUrl = "https://openai.com"
    creator_mock.logoUrl = "https://openai.com/logo.png"

    record = _make_mock_record(Creator=creator_mock)
    model = _record_to_registry_model(record)

    assert model.creator is not None
    assert isinstance(model.creator, RegistryModelCreator)
    assert model.creator.id == "creator-uuid"
    assert model.creator.display_name == "OpenAI"
    assert model.metadata.creator_name == "OpenAI"


def test_record_to_registry_model_null_max_output_tokens():
    """maxOutputTokens=None falls back to contextWindow."""
    record = _make_mock_record(maxOutputTokens=None, contextWindow=64000)
    model = _record_to_registry_model(record)

    assert model.metadata.max_output_tokens == 64000


def test_record_to_registry_model_invalid_price_tier(caplog):
    """Out-of-range priceTier is coerced to 1 and a warning is logged."""
    record = _make_mock_record(priceTier=99)
    with caplog.at_level("WARNING"):
        model = _record_to_registry_model(record)

    assert "out-of-range priceTier" in caplog.text
    assert model.metadata.price_tier == 1


def test_record_to_registry_model_with_costs():
    """Costs are parsed into RegistryModelCost tuples."""
    cost_mock = Mock()
    cost_mock.unit = "TOKENS"
    cost_mock.creditCost = 10
    cost_mock.credentialProvider = "openai"
    cost_mock.credentialId = None
    cost_mock.credentialType = None
    cost_mock.currency = "USD"
    cost_mock.metadata = {}

    record = _make_mock_record(Costs=[cost_mock])
    model = _record_to_registry_model(record)

    assert len(model.costs) == 1
    cost = model.costs[0]
    assert isinstance(cost, RegistryModelCost)
    assert cost.unit == "TOKENS"
    assert cost.credit_cost == 10
    assert cost.credential_provider == "openai"


# ---------------------------------------------------------------------------
# get_default_model_slug tests
# ---------------------------------------------------------------------------


def test_get_default_model_slug_recommended():
    """Recommended model is preferred over non-recommended enabled models."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {
        "openai/gpt-4o": _make_registry_model(
            slug="openai/gpt-4o", display_name="GPT-4o", is_recommended=False
        ),
        "openai/gpt-4o-recommended": _make_registry_model(
            slug="openai/gpt-4o-recommended",
            display_name="GPT-4o Recommended",
            is_recommended=True,
        ),
    }

    result = get_default_model_slug()
    assert result == "openai/gpt-4o-recommended"


def test_get_default_model_slug_fallback():
    """With no recommended model, falls back to first enabled (alphabetical)."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {
        "openai/gpt-4o": _make_registry_model(
            slug="openai/gpt-4o", display_name="GPT-4o", is_recommended=False
        ),
        "openai/gpt-3.5": _make_registry_model(
            slug="openai/gpt-3.5", display_name="GPT-3.5", is_recommended=False
        ),
    }

    result = get_default_model_slug()
    # Sorted alphabetically: GPT-3.5 < GPT-4o
    assert result == "openai/gpt-3.5"


def test_get_default_model_slug_empty():
    """Empty registry returns None."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {}

    result = get_default_model_slug()
    assert result is None


# ---------------------------------------------------------------------------
# _build_schema_options / get_schema_options tests
# ---------------------------------------------------------------------------


def test_build_schema_options():
    """Only enabled models appear, sorted case-insensitively."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {
        "openai/gpt-4o": _make_registry_model(
            slug="openai/gpt-4o", display_name="GPT-4o", is_enabled=True
        ),
        "openai/disabled": _make_registry_model(
            slug="openai/disabled", display_name="Disabled Model", is_enabled=False
        ),
        "openai/gpt-3.5": _make_registry_model(
            slug="openai/gpt-3.5", display_name="gpt-3.5", is_enabled=True
        ),
    }

    options = _build_schema_options()
    slugs = [o["value"] for o in options]

    # disabled model should be excluded
    assert "openai/disabled" not in slugs
    # only enabled models
    assert "openai/gpt-4o" in slugs
    assert "openai/gpt-3.5" in slugs
    # case-insensitive sort: "gpt-3.5" < "GPT-4o" (both lowercase: "gpt-3.5" < "gpt-4o")
    assert slugs.index("openai/gpt-3.5") < slugs.index("openai/gpt-4o")

    # Verify structure
    for option in options:
        assert "label" in option
        assert "value" in option
        assert "group" in option
        assert "description" in option


def test_get_schema_options_returns_copy():
    """Mutating the returned list does not affect the internal cache."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {
        "openai/gpt-4o": _make_registry_model(slug="openai/gpt-4o", display_name="GPT-4o"),
    }
    reg._schema_options = _build_schema_options()

    options = get_schema_options()
    original_length = len(options)
    options.append({"label": "Injected", "value": "evil/model", "group": "evil", "description": ""})

    # Internal state should be unchanged
    assert len(get_schema_options()) == original_length


# ---------------------------------------------------------------------------
# Pydantic frozen model tests
# ---------------------------------------------------------------------------


def test_registry_model_frozen():
    """Pydantic frozen=True should reject attribute assignment."""
    model = _make_registry_model()

    with pytest.raises((pydantic.ValidationError, TypeError)):
        model.slug = "changed/slug"  # type: ignore[misc]


def test_registry_model_cost_frozen():
    """RegistryModelCost is also frozen."""
    cost = RegistryModelCost(
        unit="TOKENS",
        credit_cost=5,
        credential_provider="openai",
    )
    with pytest.raises((pydantic.ValidationError, TypeError)):
        cost.unit = "RUN"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# refresh_llm_registry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_llm_registry():
    """Mock prisma find_many, verify cache is populated after refresh."""
    import backend.data.llm_registry.registry as reg

    record = _make_mock_record()
    mock_find_many = AsyncMock(return_value=[record])

    with patch("prisma.models.LlmModel.prisma") as mock_prisma_cls:
        mock_prisma_instance = Mock()
        mock_prisma_instance.find_many = mock_find_many
        mock_prisma_cls.return_value = mock_prisma_instance

        # Clear state first
        reg._dynamic_models = {}
        reg._schema_options = []

        await refresh_llm_registry()

    assert "openai/gpt-4o" in reg._dynamic_models
    model = reg._dynamic_models["openai/gpt-4o"]
    assert isinstance(model, RegistryModel)
    assert model.slug == "openai/gpt-4o"
    # Schema options should be populated too
    assert len(reg._schema_options) == 1
    assert reg._schema_options[0]["value"] == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# clear_registry_cache tests
# ---------------------------------------------------------------------------


def test_clear_registry_cache():
    """clear_registry_cache calls cache_clear on the cached fetch function."""
    import backend.data.llm_registry.registry as reg
    from unittest.mock import patch

    with patch.object(reg._fetch_registry_from_db, "cache_clear") as mock_clear:
        clear_registry_cache()
        mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# get_model / get_all_models / get_enabled_models / get_all_model_slugs tests
# ---------------------------------------------------------------------------


def test_get_model_found():
    """get_model returns the model when the slug exists in the registry."""
    import backend.data.llm_registry.registry as reg

    m = _make_registry_model(slug="openai/gpt-4o")
    reg._dynamic_models = {"openai/gpt-4o": m}

    result = get_model("openai/gpt-4o")

    assert result is m


def test_get_model_not_found():
    """get_model returns None for an unknown slug."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {}

    assert get_model("nonexistent/model") is None


def test_get_all_models_includes_disabled():
    """get_all_models returns all models, including disabled ones."""
    import backend.data.llm_registry.registry as reg

    enabled = _make_registry_model(slug="openai/gpt-4", is_enabled=True)
    disabled = _make_registry_model(slug="openai/old-model", is_enabled=False)
    reg._dynamic_models = {"openai/gpt-4": enabled, "openai/old-model": disabled}

    result = get_all_models()

    slugs = [m.slug for m in result]
    assert "openai/gpt-4" in slugs
    assert "openai/old-model" in slugs
    assert len(result) == 2


def test_get_enabled_models_excludes_disabled():
    """get_enabled_models returns only models where is_enabled=True."""
    import backend.data.llm_registry.registry as reg

    enabled = _make_registry_model(slug="openai/gpt-4", is_enabled=True)
    disabled = _make_registry_model(slug="openai/old-model", is_enabled=False)
    reg._dynamic_models = {"openai/gpt-4": enabled, "openai/old-model": disabled}

    result = get_enabled_models()

    assert len(result) == 1
    assert result[0].slug == "openai/gpt-4"


def test_get_all_model_slugs_for_validation():
    """get_all_model_slugs_for_validation returns only enabled model slugs."""
    import backend.data.llm_registry.registry as reg

    reg._dynamic_models = {
        "openai/gpt-4": _make_registry_model(slug="openai/gpt-4", is_enabled=True),
        "openai/old": _make_registry_model(slug="openai/old", is_enabled=False),
        "anthropic/claude": _make_registry_model(
            slug="anthropic/claude", is_enabled=True
        ),
    }

    result = get_all_model_slugs_for_validation()

    assert "openai/gpt-4" in result
    assert "anthropic/claude" in result
    assert "openai/old" not in result
    assert len(result) == 2


@pytest.mark.asyncio
async def test_refresh_llm_registry_error_is_reraised(mocker):
    """refresh_llm_registry re-raises exceptions after logging them."""
    mocker.patch(
        "backend.data.llm_registry.registry._fetch_registry_from_db",
        new=AsyncMock(side_effect=RuntimeError("DB unavailable")),
    )

    with pytest.raises(RuntimeError, match="DB unavailable"):
        await refresh_llm_registry()
