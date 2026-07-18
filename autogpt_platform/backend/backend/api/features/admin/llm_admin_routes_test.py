"""Route-layer tests for the LLM registry admin API.

Auth, validation, mapping, and audit wiring — the data layer is mocked here;
real-DB behavior is covered in backend/data/llm_registry/db_write_test.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.admin import llm_admin_routes
from backend.copilot.route_warnings import RouteWarning
from backend.data.llm_registry.catalog_model import (
    CATALOG_SCHEMA_VERSION,
    CatalogPayload,
)
from backend.data.llm_registry.db_routes import UnknownRouteModelError

app = fastapi.FastAPI()
app.include_router(llm_admin_routes.router, prefix="/api/admin/llm")
client = fastapi.testclient.TestClient(app)

unauthed_app = fastapi.FastAPI()
unauthed_app.include_router(llm_admin_routes.router, prefix="/api/admin/llm")
unauthed_client = fastapi.testclient.TestClient(unauthed_app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_admin):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def quiet_side_effects(mocker):
    """Mute cache refresh + audit persistence; expose the audit mock."""
    mocker.patch.object(
        llm_admin_routes.llm_registry, "refresh_runtime_caches", new=AsyncMock()
    )
    audit = mocker.patch.object(llm_admin_routes, "_audit", new=AsyncMock())
    return audit


def _mock_route_row(
    slug="openai/gpt-a", surface="copilot", mode="fast", tier="standard"
):
    row = Mock()
    row.id = "route-uuid"
    row.surface = surface
    row.mode = mode
    row.tier = tier
    row.modelSlug = slug
    row.updatedAt = datetime(2026, 7, 18, tzinfo=timezone.utc)
    return row


def test_requires_admin_auth():
    resp = unauthed_client.get("/api/admin/llm/routes")
    assert resp.status_code in (401, 403)


def test_set_route_unknown_model_404(mocker):
    mocker.patch.object(
        llm_admin_routes.db_routes,
        "set_route",
        new=AsyncMock(side_effect=UnknownRouteModelError("Model 'x' does not exist")),
    )
    resp = client.put(
        "/api/admin/llm/routes",
        json={"mode": "fast", "tier": "standard", "model_slug": "x"},
    )
    assert resp.status_code == 404


def test_set_route_disabled_model_422(mocker):
    mocker.patch.object(
        llm_admin_routes.db_routes,
        "set_route",
        new=AsyncMock(side_effect=ValueError("Model 'x' is disabled (kill switch)")),
    )
    resp = client.put(
        "/api/admin/llm/routes",
        json={"mode": "fast", "tier": "standard", "model_slug": "x"},
    )
    assert resp.status_code == 422


def test_set_route_returns_warnings_and_audits(mocker, quiet_side_effects):
    mocker.patch.object(
        llm_admin_routes.db_routes,
        "set_route",
        new=AsyncMock(
            return_value=(
                _mock_route_row(mode="thinking"),
                ["model 'openai/gpt-a' does not advertise reasoning support"],
            )
        ),
    )
    resp = client.put(
        "/api/admin/llm/routes",
        json={"mode": "thinking", "tier": "standard", "model_slug": "openai/gpt-a"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["route"]["model_slug"] == "openai/gpt-a"
    assert len(body["warnings"]) == 1
    quiet_side_effects.assert_called_once()
    assert quiet_side_effects.call_args.args[3] == "LLM_ROUTE_SET"


def test_set_route_none_clears_cell(mocker, quiet_side_effects):
    mocker.patch.object(
        llm_admin_routes.db_routes,
        "set_route",
        new=AsyncMock(return_value=(None, [])),
    )
    resp = client.put(
        "/api/admin/llm/routes",
        json={"mode": "fast", "tier": "standard", "model_slug": None},
    )

    assert resp.status_code == 200
    assert resp.json()["route"] is None
    assert quiet_side_effects.call_args.args[3] == "LLM_ROUTE_CLEARED"


def test_list_routes(mocker):
    mocker.patch.object(
        llm_admin_routes.db_routes,
        "list_routes",
        new=AsyncMock(return_value=[_mock_route_row()]),
    )
    resp = client.get("/api/admin/llm/routes")

    assert resp.status_code == 200
    assert resp.json()["routes"][0]["model_slug"] == "openai/gpt-a"


def test_route_warnings_endpoint(mocker):
    mocker.patch.object(
        llm_admin_routes,
        "get_route_warnings",
        new=AsyncMock(
            return_value=[
                RouteWarning(
                    slug="typo/model",
                    reason="unknown to the model registry",
                    count=42,
                    last_seen=datetime(2026, 7, 18, tzinfo=timezone.utc),
                    last_layer="ld",
                )
            ]
        ),
    )
    resp = client.get("/api/admin/llm/routes/warnings")

    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["slug"] == "typo/model"
    assert body[0]["count"] == 42


def test_catalog_export_endpoint(mocker):
    payload = CatalogPayload(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        providers=[],
        creators=[],
        models=[],
    )
    mocker.patch.object(
        llm_admin_routes.llm_registry,
        "export_catalog",
        new=AsyncMock(return_value=payload),
    )
    resp = client.get("/api/admin/llm/catalog/export")

    assert resp.status_code == 200
    assert resp.json()["schema_version"] == CATALOG_SCHEMA_VERSION


def _mock_model_record(slug="openai/gpt-a"):
    m = Mock()
    m.id = "model-uuid"
    m.slug = slug
    m.displayName = "GPT A"
    m.description = None
    m.providerId = "provider-uuid"
    m.creatorId = None
    m.contextWindow = 128000
    m.maxOutputTokens = 16384
    m.priceTier = 2
    m.isEnabled = True
    m.isRecommended = False
    m.kind = "CHAT"
    m.visibility = "GA"
    m.minSubscriptionTier = None
    m.fallbackModelSlug = None
    m.source = "LOCAL"
    m.catalogRemovedAt = None
    m.supportsTools = True
    m.supportsJsonOutput = True
    m.supportsReasoning = False
    m.supportsParallelToolCalls = False
    m.capabilities = {}
    m.metadata = {}
    m.createdAt = datetime(2026, 7, 18, tzinfo=timezone.utc)
    m.updatedAt = datetime(2026, 7, 18, tzinfo=timezone.utc)
    m.Creator = None
    m.Costs = []
    return m


def test_create_model_passes_new_columns(mocker, quiet_side_effects):
    provider = Mock()
    provider.id = "provider-uuid"
    provider.name = "openai"
    provider_prisma = MagicMock()
    provider_prisma.find_unique = AsyncMock(return_value=provider)
    mocker.patch("prisma.models.LlmProvider.prisma", return_value=provider_prisma)

    record = _mock_model_record()
    model_prisma = MagicMock()
    model_prisma.find_unique = AsyncMock(return_value=record)
    mocker.patch("prisma.models.LlmModel.prisma", return_value=model_prisma)

    create = mocker.patch.object(
        llm_admin_routes.db_write, "create_model", new=AsyncMock(return_value=record)
    )

    resp = client.post(
        "/api/admin/llm/models",
        json={
            "slug": "openai/gpt-a",
            "display_name": "GPT A",
            "provider_name": "openai",
            "context_window": 128000,
            "price_tier": 2,
            "visibility": "HIDDEN",
            "min_subscription_tier": "PRO",
        },
    )

    assert resp.status_code == 201
    kwargs = create.call_args.kwargs
    assert kwargs["visibility"] == "HIDDEN"
    assert kwargs["min_subscription_tier"] == "PRO"
    assert quiet_side_effects.call_args.args[3] == "LLM_MODEL_CREATED"


def test_create_model_unknown_provider_404(mocker):
    provider_prisma = MagicMock()
    provider_prisma.find_unique = AsyncMock(return_value=None)
    mocker.patch("prisma.models.LlmProvider.prisma", return_value=provider_prisma)

    resp = client.post(
        "/api/admin/llm/models",
        json={
            "slug": "openai/gpt-a",
            "display_name": "GPT A",
            "provider_name": "nope",
            "context_window": 128000,
            "price_tier": 2,
        },
    )
    assert resp.status_code == 404


def test_update_model_unknown_slug_404(mocker):
    model_prisma = MagicMock()
    model_prisma.find_unique = AsyncMock(return_value=None)
    mocker.patch("prisma.models.LlmModel.prisma", return_value=model_prisma)

    resp = client.patch("/api/admin/llm/models/openai/nope", json={"display_name": "X"})
    assert resp.status_code == 404


def test_toggle_recommended_model_guard(mocker):
    record = _mock_model_record()
    record.isRecommended = True
    model_prisma = MagicMock()
    model_prisma.find_unique = AsyncMock(return_value=record)
    mocker.patch("prisma.models.LlmModel.prisma", return_value=model_prisma)

    resp = client.post(
        "/api/admin/llm/models/openai/gpt-a/toggle",
        json={"is_enabled": False},
    )
    assert resp.status_code == 400
    assert "recommended" in resp.json()["detail"]
