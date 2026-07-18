"""Validation tests for the catalog payload schema + bundled catalog guard."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import resources

import pydantic
import pytest

from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogModel,
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
    assert m.kind == "CHAT"
    assert m.is_enabled is True


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
    ],
)
def test_invalid_model_fields_rejected(field, value):
    with pytest.raises(pydantic.ValidationError):
        CatalogModel(**_model_kwargs(**{field: value}))


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


def test_bundled_catalog_is_valid():
    """Forever-guard: the shipped catalog.json must parse and be referentially
    sound. If this fails, a catalog edit broke the file — fix the data, never
    the test."""
    raw = (
        resources.files("backend.data.llm_registry")
        .joinpath("catalog.json")
        .read_text(encoding="utf-8")
    )
    payload = CatalogPayload.model_validate_json(raw)

    assert payload.schema_version == CATALOG_SCHEMA_VERSION
    slugs = [m.slug for m in payload.models]
    assert len(slugs) == len(set(slugs)), "duplicate model slugs in catalog.json"
    provider_names = {p.name for p in payload.providers}
    creator_names = {c.name for c in payload.creators}
    for m in payload.models:
        assert m.provider in provider_names, f"{m.slug} references unknown provider"
        if m.creator:
            assert m.creator in creator_names, f"{m.slug} references unknown creator"
        if m.fallback_model_slug:
            assert m.fallback_model_slug in slugs
    assert len(payload.models) > 50, "catalog.json suspiciously small"
