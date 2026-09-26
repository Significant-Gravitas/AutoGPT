"""Validation tests for the catalog schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pydantic
import pytest

from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogModel,
    CatalogModelCost,
    CatalogPayload,
)


def _model_kwargs(**overrides):
    kwargs = dict(
        slug="openai/gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        context_window=128000,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_model_parses():
    m = CatalogModel(**_model_kwargs())
    assert m.price_tier == 1
    assert m.visibility == "GA"
    assert m.is_enabled is True
    assert m.cost is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("slug", "has spaces"),
        ("slug", ""),
        ("provider", "Has/Slash"),
        ("context_window", 0),
        ("context_window", -5),
        ("max_output_tokens", 0),
        ("price_tier", 0),
        ("price_tier", 4),
        ("display_name", ""),
        ("visibility", "SECRET"),
    ],
)
def test_invalid_model_fields_rejected(field, value):
    with pytest.raises(pydantic.ValidationError):
        CatalogModel(**_model_kwargs(**{field: value}))


# A valid base USD pair, so a USD cache price is only refused for its sign.
_USD_PAIR = {"provider_input_usd_per_1m": 1.0, "provider_output_usd_per_1m": 5.0}


@pytest.mark.parametrize(
    "field",
    [
        "run_credits",
        "input_credits_per_1m",
        "output_credits_per_1m",
        "cache_read_credits_per_1m",
        "cache_creation_credits_per_1m",
        "provider_input_usd_per_1m",
        "provider_output_usd_per_1m",
        "provider_cache_read_usd_per_1m",
        "provider_cache_creation_usd_per_1m",
    ],
)
def test_negative_costs_rejected(field):
    with pytest.raises(pydantic.ValidationError):
        CatalogModelCost.model_validate({**_USD_PAIR, field: -1})


@pytest.mark.parametrize(
    "field", ["provider_cache_read_usd_per_1m", "provider_cache_creation_usd_per_1m"]
)
def test_usd_cache_price_requires_the_base_usd_pair(field):
    with pytest.raises(pydantic.ValidationError, match="require"):
        CatalogModelCost.model_validate({field: 0.1})


def test_usd_cache_prices_accepted_alongside_the_base_pair():
    cost = CatalogModelCost(
        provider_input_usd_per_1m=1.0,
        provider_output_usd_per_1m=5.0,
        provider_cache_read_usd_per_1m=0.1,
        provider_cache_creation_usd_per_1m=1.25,
    )
    assert cost.provider_cache_read_usd_per_1m == 0.1
    assert cost.provider_cache_creation_usd_per_1m == 1.25


def test_routing_shape_accepts_nested_cells():
    payload = CatalogPayload(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        providers=[],
        creators=[],
        models=[],
        routing={"copilot": {"fast": {"standard": "openai/gpt-4o"}}},
    )
    assert payload.routing["copilot"]["fast"]["standard"] == "openai/gpt-4o"


def test_payload_model_count_cap():
    models = [CatalogModel(**_model_kwargs(slug=f"openai/m{i}")) for i in range(2001)]
    with pytest.raises(pydantic.ValidationError):
        CatalogPayload(
            schema_version=CATALOG_SCHEMA_VERSION,
            generated_at=datetime.now(timezone.utc),
            providers=[],
            creators=[],
            models=models,
        )
