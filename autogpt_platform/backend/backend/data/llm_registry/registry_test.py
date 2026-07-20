"""Unit tests for the in-process catalog view (registry.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.data.llm_registry.catalog import CATALOG
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
    get_all_model_slugs_for_validation,
    get_all_models,
    get_default_model_slug,
    get_enabled_models,
    get_model,
    get_route,
    get_schema_options,
    load_catalog,
)


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
    assert [m.slug for m in get_enabled_models()] == ["openai/gpt-a"]
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
    assert model.creator is not None and model.creator.display_name == "OpenAI"
    assert model.metadata.provider == "openai"
    assert model.metadata.price_tier == 2
    assert model.cost is not None and model.cost.run_credits == 3


def test_null_max_output_tokens_falls_back_to_context_window():
    load_catalog(_payload([_model("openai/gpt-a", max_output_tokens=None)]))
    model = get_model("openai/gpt-a")
    assert model is not None
    assert model.metadata.max_output_tokens == 128000


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


def test_schema_options_enabled_only_sorted():
    load_catalog(
        _payload(
            [
                _model("openai/zeta", display_name="Zeta"),
                _model("openai/alpha", display_name="Alpha"),
                _model("openai/old", display_name="Old", is_enabled=False),
            ]
        )
    )
    options = get_schema_options()

    assert [o["value"] for o in options] == ["openai/alpha", "openai/zeta"]
    assert options[0]["group"] == "openai"


def test_default_model_prefers_recommended_enabled():
    load_catalog(
        _payload(
            [
                _model("openai/a", display_name="A"),
                _model("openai/b", display_name="B", is_recommended=True),
                _model(
                    "openai/c",
                    display_name="C",
                    is_recommended=True,
                    is_enabled=False,
                ),
            ]
        )
    )
    assert get_default_model_slug() == "openai/b"


def test_default_model_falls_back_to_first_enabled():
    load_catalog(
        _payload(
            [
                _model("openai/b", display_name="B"),
                _model("openai/a", display_name="A"),
            ]
        )
    )
    assert get_default_model_slug() == "openai/a"


def test_validation_slugs_enabled_only():
    load_catalog(
        _payload([_model("openai/a"), _model("openai/dead", is_enabled=False)])
    )
    assert get_all_model_slugs_for_validation() == ["openai/a"]


def test_real_catalog_loads():
    """The shipped catalog file builds a complete L1."""
    load_catalog(CATALOG)

    assert len(get_all_models()) > 50
    assert get_default_model_slug() is not None
    assert get_route("copilot", "thinking", "standard") is not None
    assert len(get_schema_options()) > 50
