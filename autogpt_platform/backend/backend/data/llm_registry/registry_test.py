"""Unit tests for the in-process catalog view (registry.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogCreator,
    CatalogModel,
    CatalogModelCost,
    CatalogPayload,
    CatalogProvider,
)
from backend.data.llm_registry.registry import (
    RegistryModel,
    get_all_models,
    get_model,
    get_route,
    load_catalog,
)

CATALOG = get_catalog()


def _payload(models: list[CatalogModel], routing=None) -> CatalogPayload:
    return CatalogPayload(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        providers=[CatalogProvider(name="openai", display_name="OpenAI")],
        creators=[CatalogCreator(name="openai", display_name="OpenAI")],
        models=models,
        routing=routing or {},
    )


def _model(slug: str, **overrides) -> CatalogModel:
    kwargs = dict(
        slug=slug,
        display_name=slug.split("/")[-1].upper(),
        provider="openai",
        creator="openai",
        context_window=128000,
        max_output_tokens=16384,
        price_tier=2,
    )
    kwargs.update(overrides)
    return CatalogModel(**kwargs)


@pytest.fixture(autouse=True)
def isolated_l1():
    """Each test loads its own payload; restore the real catalog after."""
    yield
    load_catalog(CATALOG)


def test_load_catalog_builds_l1_from_payload():
    load_catalog(
        _payload(
            [_model("openai/gpt-a"), _model("openai/gpt-b", is_enabled=False)],
            routing={"copilot": {"thinking": {"standard": "openai/gpt-a"}}},
        )
    )

    assert {m.slug for m in get_all_models()} == {"openai/gpt-a", "openai/gpt-b"}
    enabled = [m.slug for m in get_all_models() if m.is_enabled]
    assert enabled == ["openai/gpt-a"]
    assert get_route("copilot", "thinking", "standard") == "openai/gpt-a"
    assert get_route("copilot", "fast", "standard") is None


def test_registry_model_carries_joined_display_data():
    load_catalog(
        _payload(
            [
                _model(
                    "openai/gpt-a",
                    cost=CatalogModelCost(run_credits=3, input_credits_per_1m=100),
                )
            ]
        )
    )
    model = get_model("openai/gpt-a")

    assert isinstance(model, RegistryModel)
    assert model.provider_display_name == "OpenAI"
    assert model.metadata.provider == "openai"
    assert model.metadata.price_tier == 2
    assert model.cost is not None and model.cost.run_credits == 3


def test_null_max_output_tokens_is_preserved():
    """None means "unknown/no published cap" — substituting context_window
    would overstate the limit and publish wrong data through the catalog
    endpoint."""
    load_catalog(_payload([_model("openai/gpt-a", max_output_tokens=None)]))
    model = get_model("openai/gpt-a")
    assert model is not None
    assert model.metadata.max_output_tokens is None


def test_unknown_provider_falls_back_to_name():
    payload = _payload([_model("other/x", provider="other")])
    load_catalog(payload)
    model = get_model("other/x")
    assert model is not None
    assert model.provider_display_name == "other"
    assert model.metadata.creator_name == "OpenAI"


def test_get_model_not_found():
    load_catalog(_payload([]))
    assert get_model("nonexistent/model") is None


def test_real_catalog_loads():
    """The shipped catalog file builds a complete L1."""
    load_catalog(CATALOG)

    assert len(get_all_models()) > 50
    assert any(m.is_recommended and m.is_enabled for m in get_all_models())
    # Cells ship empty (env stays authoritative until a cell is claimed) —
    # populated-cell behavior is covered by the seeded-payload tests above.
    assert get_route("copilot", "thinking", "standard") is None


def test_registry_metadata_stays_field_compatible_with_block_shape():
    """RegistryModelMetadata (router view) and ModelMetadata (block
    projection) are documented as field-compatible by design — enforce it
    so adding a field to one without the other fails here, not in prod."""
    from backend.data.llm_registry.llm_models import ModelMetadata
    from backend.data.llm_registry.registry import RegistryModelMetadata

    assert set(RegistryModelMetadata.model_fields) == set(ModelMetadata._fields)
