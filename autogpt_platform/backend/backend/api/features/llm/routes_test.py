"""Tests for the public LLM catalog endpoint."""

from __future__ import annotations

import fastapi
import fastapi.testclient
import pytest

import backend.data.llm_registry.registry as reg
from backend.api.features.llm.routes import router
from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogModelCost,
)
from backend.data.llm_registry.registry import (
    RegistryModel,
    RegistryModelCreator,
    RegistryModelMetadata,
)

app = fastapi.FastAPI()
app.add_middleware(SecurityHeadersMiddleware)
app.include_router(router, prefix="/api/llm")
client = fastapi.testclient.TestClient(app)


def _model(slug: str, **overrides) -> RegistryModel:
    defaults = dict(
        slug=slug,
        display_name=slug.split("/")[-1].upper(),
        metadata=RegistryModelMetadata(
            provider="openai",
            context_window=128000,
            max_output_tokens=16384,
            display_name=slug.split("/")[-1].upper(),
            provider_name="OpenAI",
            creator_name="OpenAI",
            price_tier=2,
        ),
        provider_display_name="OpenAI",
        is_enabled=True,
        creator=RegistryModelCreator(name="openai", display_name="OpenAI"),
        cost=CatalogModelCost(run_credits=3, input_credits_per_1m=120.0),
    )
    defaults.update(overrides)
    return RegistryModel(**defaults)


@pytest.fixture(autouse=True)
def seeded_l1(mocker):
    """Seed the L1 cache directly and allow all rate-limit checks by default."""
    reg._dynamic_models = {
        "openai/gpt-a": _model("openai/gpt-a"),
        "openai/gpt-disabled": _model("openai/gpt-disabled", is_enabled=False),
        "openai/gpt-hidden": _model("openai/gpt-hidden", visibility="HIDDEN"),
        "openai/gpt-employees": _model("openai/gpt-employees", visibility="EMPLOYEES"),
    }
    mocker.patch(
        "backend.api.features.llm.routes.check_catalog_rate_limit",
        return_value=True,
    )
    yield
    reg._dynamic_models = {}


def test_catalog_returns_ga_models_only():
    resp = client.get("/api/llm/catalog")

    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == CATALOG_SCHEMA_VERSION
    slugs = [m["slug"] for m in body["models"]]
    assert slugs == ["openai/gpt-a", "openai/gpt-disabled"]
    assert "openai/gpt-hidden" not in slugs
    assert "openai/gpt-employees" not in slugs


def test_catalog_includes_disabled_ga_models_as_disabled():
    """Disabled GA models stay in the payload with is_enabled=false so
    consumers (e.g. the Phase B picker) can distinguish retired from
    never-existed."""
    resp = client.get("/api/llm/catalog")

    by_slug = {m["slug"]: m for m in resp.json()["models"]}
    assert by_slug["openai/gpt-disabled"]["is_enabled"] is False


def test_catalog_is_publicly_cacheable():
    resp = client.get("/api/llm/catalog")

    assert "public" in resp.headers["Cache-Control"]
    assert "max-age=300" in resp.headers["Cache-Control"]


def test_catalog_providers_and_creators_are_referenced():
    body = client.get("/api/llm/catalog").json()

    provider_names = {p["name"] for p in body["providers"]}
    creator_names = {c["name"] for c in body["creators"]}
    for m in body["models"]:
        assert m["provider"] in provider_names
        if m["creator"]:
            assert m["creator"] in creator_names


def test_catalog_requires_no_auth():
    """No Authorization header, no cookies — must still be a 200."""
    resp = client.get("/api/llm/catalog")
    assert resp.status_code == 200


def test_rate_limited_response_is_not_cacheable(mocker):
    mocker.patch(
        "backend.api.features.llm.routes.check_catalog_rate_limit",
        return_value=False,
    )

    resp = client.get("/api/llm/catalog")

    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"
    assert "no-store" in resp.headers["Cache-Control"]


def test_middleware_keeps_sibling_paths_uncacheable():
    """Only the exact catalog path is allowlisted."""

    @app.get("/api/llm/other")
    def other():
        return {"ok": True}

    resp = client.get("/api/llm/other")
    assert "no-store" in resp.headers["Cache-Control"]


def test_catalog_never_exposes_costs_or_routing():
    """Costs are cloud billing config and routing is per-deployment config —
    neither may appear in the public payload even though the L1 models carry
    costs (the seeded fixture sets them)."""
    reg._routes = {("copilot", "thinking", "standard"): "openai/gpt-a"}
    try:
        body = client.get("/api/llm/catalog").json()
    finally:
        reg._routes = {}

    assert body["routing"] == {}
    assert all(m["cost"] is None for m in body["models"])
