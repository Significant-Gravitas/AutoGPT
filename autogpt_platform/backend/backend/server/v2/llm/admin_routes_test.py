"""Tests for LLM registry admin CRUD routes (admin_routes.py).

Covers provider, model, creator, migration CRUD endpoints.
All endpoints require admin authentication.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest

from backend.server.v2.llm.admin_routes import router as admin_router

admin_app = fastapi.FastAPI()
admin_app.include_router(admin_router)
admin_client = fastapi.testclient.TestClient(admin_app)


# ---------------------------------------------------------------------------
# Auth fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_admin):
    """Bypass JWT admin auth for all tests in this module."""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    admin_app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    admin_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_mock_provider(
    id: str = "prov-1",
    name: str = "openai",
    display_name: str = "OpenAI",
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    models: list | None = None,
) -> Mock:
    p = Mock()
    p.id = id
    p.name = name
    p.displayName = display_name
    p.description = description
    p.defaultCredentialProvider = default_credential_provider
    p.defaultCredentialId = default_credential_id
    p.defaultCredentialType = default_credential_type
    p.metadata = {}
    p.createdAt = _NOW
    p.updatedAt = _NOW
    p.Models = models if models is not None else []
    return p


def _make_mock_model(
    id: str = "model-1",
    slug: str = "gpt-4",
    display_name: str = "GPT-4",
    description: str | None = None,
    provider_id: str = "prov-1",
    creator_id: str | None = None,
    context_window: int = 128000,
    max_output_tokens: int | None = 4096,
    price_tier: int = 2,
    is_enabled: bool = True,
    is_recommended: bool = False,
    costs: list | None = None,
    creator: Mock | None = None,
) -> Mock:
    m = Mock()
    m.id = id
    m.slug = slug
    m.displayName = display_name
    m.description = description
    m.providerId = provider_id
    m.creatorId = creator_id
    m.contextWindow = context_window
    m.maxOutputTokens = max_output_tokens
    m.priceTier = price_tier
    m.isEnabled = is_enabled
    m.isRecommended = is_recommended
    m.supportsTools = False
    m.supportsJsonOutput = False
    m.supportsReasoning = False
    m.supportsParallelToolCalls = False
    m.capabilities = {}
    m.metadata = {}
    m.createdAt = _NOW
    m.updatedAt = _NOW
    m.Costs = costs or []
    m.Creator = creator
    return m


def _make_mock_creator(
    id: str = "creator-1",
    name: str = "openai",
    display_name: str = "OpenAI",
    description: str | None = None,
    website_url: str | None = None,
    logo_url: str | None = None,
    models: list | None = None,
) -> Mock:
    c = Mock()
    c.id = id
    c.name = name
    c.displayName = display_name
    c.description = description
    c.websiteUrl = website_url
    c.logoUrl = logo_url
    c.metadata = {}
    c.createdAt = _NOW
    c.updatedAt = _NOW
    c.Models = models if models is not None else []
    return c


def _make_mock_migration(
    id: str = "mig-1",
    source_slug: str = "gpt-3",
    target_slug: str = "gpt-4",
    node_count: int = 3,
    is_reverted: bool = False,
) -> dict:
    return {
        "id": id,
        "source_model_slug": source_slug,
        "target_model_slug": target_slug,
        "reason": "upgrade",
        "node_count": node_count,
        "custom_credit_cost": None,
        "is_reverted": is_reverted,
        "reverted_at": None,
        "created_at": _NOW.isoformat(),
    }


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------


def test_create_provider(mocker):
    """POST /llm/providers creates a provider and returns 201 with provider fields."""
    mock_provider = _make_mock_provider()
    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.create_provider = AsyncMock(return_value=mock_provider)
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post(
        "/llm/providers",
        json={"name": "openai", "display_name": "OpenAI"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "openai"
    assert data["display_name"] == "OpenAI"
    mock_db.create_provider.assert_called_once()
    mock_db.refresh_runtime_caches.assert_called_once()


def test_create_provider_validation_error(mocker):
    """db_write.create_provider raising ValueError returns 400."""
    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.create_provider = AsyncMock(side_effect=ValueError("duplicate name"))
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post(
        "/llm/providers",
        json={"name": "openai", "display_name": "OpenAI"},
    )

    assert response.status_code == 400
    assert "duplicate name" in response.json()["detail"]


def test_update_provider(mocker):
    """PATCH /llm/providers/{name} returns 200 with updated fields."""
    existing = _make_mock_provider()
    updated = _make_mock_provider(display_name="OpenAI Updated")

    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.update_provider = AsyncMock(return_value=updated)
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.patch(
        "/llm/providers/openai",
        json={"display_name": "OpenAI Updated"},
    )

    assert response.status_code == 200
    assert response.json()["display_name"] == "OpenAI Updated"


def test_update_provider_not_found(mocker):
    """PATCH returns 404 when the provider does not exist."""
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)
    mocker.patch("backend.server.v2.llm.admin_routes.db_write")

    response = admin_client.patch(
        "/llm/providers/nonexistent",
        json={"display_name": "X"},
    )

    assert response.status_code == 404


def test_delete_provider(mocker):
    """DELETE /llm/providers/{name} returns 204 on success."""
    existing = _make_mock_provider()
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.delete_provider = AsyncMock(return_value=True)
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.delete("/llm/providers/openai")

    assert response.status_code == 204


def test_delete_provider_not_found(mocker):
    """DELETE returns 404 when the provider does not exist."""
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)
    mocker.patch("backend.server.v2.llm.admin_routes.db_write")

    response = admin_client.delete("/llm/providers/ghost")

    assert response.status_code == 404


def test_delete_provider_has_models(mocker):
    """DELETE returns 400 when db_write raises ValueError (provider has models)."""
    existing = _make_mock_provider()
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.delete_provider = AsyncMock(
        side_effect=ValueError("Cannot delete provider — it has 2 model(s)")
    )
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.delete("/llm/providers/openai")

    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()


def test_create_provider_server_error(mocker):
    """POST /llm/providers returns 500 when an unexpected exception occurs."""
    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.create_provider = AsyncMock(side_effect=RuntimeError("unexpected"))
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post(
        "/llm/providers",
        json={"name": "openai", "display_name": "OpenAI"},
    )

    assert response.status_code == 500


def test_admin_list_providers(mocker):
    """GET /llm/admin/providers returns providers with model_count."""
    provider = _make_mock_provider(models=[_make_mock_model()])
    mocker.patch(
        "prisma.models.LlmProvider.prisma"
    ).return_value.find_many = AsyncMock(return_value=[provider])

    response = admin_client.get("/llm/admin/providers")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert len(providers) == 1
    assert providers[0]["model_count"] == 1
    assert providers[0]["name"] == "openai"


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------


def test_create_model(mocker):
    """POST /llm/models creates a model and returns 201 with model fields."""
    mock_provider = _make_mock_provider()
    mock_model = _make_mock_model()

    # Patch both provider lookups (by name, then by id fallback) and refetch
    prisma_models_mock = mocker.patch("prisma.models")
    prisma_models_mock.LlmProvider.prisma.return_value.find_unique = AsyncMock(
        return_value=mock_provider
    )
    prisma_models_mock.LlmModel.prisma.return_value.find_unique = AsyncMock(
        return_value=mock_model
    )
    prisma_models_mock.LlmModelCost.prisma.return_value.create = AsyncMock()

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.create_model = AsyncMock(return_value=mock_model)
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post(
        "/llm/models",
        json={
            "slug": "gpt-4",
            "display_name": "GPT-4",
            "provider_id": "openai",
            "context_window": 128000,
            "price_tier": 2,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["slug"] == "gpt-4"
    mock_db.create_model.assert_called_once()
    mock_db.refresh_runtime_caches.assert_called_once()


def test_create_model_provider_not_found(mocker):
    """POST /llm/models returns 404 when the provider is not found."""
    prisma_models_mock = mocker.patch("prisma.models")
    prisma_models_mock.LlmProvider.prisma.return_value.find_unique = AsyncMock(
        return_value=None
    )
    mocker.patch("backend.server.v2.llm.admin_routes.db_write")

    response = admin_client.post(
        "/llm/models",
        json={
            "slug": "gpt-4",
            "display_name": "GPT-4",
            "provider_id": "nonexistent",
            "context_window": 128000,
            "price_tier": 2,
        },
    )

    assert response.status_code == 404


def test_update_model(mocker):
    """PATCH /llm/models/{slug} returns 200 with updated model."""
    existing = _make_mock_model()
    updated = _make_mock_model(display_name="GPT-4 Turbo")

    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.update_model = AsyncMock(return_value=updated)
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.patch(
        "/llm/models/gpt-4",
        json={"display_name": "GPT-4 Turbo"},
    )

    assert response.status_code == 200
    assert response.json()["display_name"] == "GPT-4 Turbo"


def test_update_model_not_found(mocker):
    """PATCH returns 404 when the model slug does not exist."""
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)
    mocker.patch("backend.server.v2.llm.admin_routes.db_write")

    response = admin_client.patch("/llm/models/unknown-slug", json={})

    assert response.status_code == 404


def test_delete_model(mocker):
    """DELETE /llm/models/{slug} without replacement returns 200 with nodes_migrated=0."""
    existing = _make_mock_model()
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.delete_model = AsyncMock(
        return_value={
            "deleted_model_slug": "gpt-4",
            "deleted_model_display_name": "GPT-4",
            "replacement_model_slug": None,
            "nodes_migrated": 0,
        }
    )
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.delete("/llm/models/gpt-4")

    assert response.status_code == 200
    data = response.json()
    assert data["nodes_migrated"] == 0
    assert data["deleted_model_slug"] == "gpt-4"


def test_delete_model_with_migration(mocker):
    """DELETE with replacement_model_slug query param migrates nodes and returns count."""
    existing = _make_mock_model()
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.delete_model = AsyncMock(
        return_value={
            "deleted_model_slug": "gpt-3",
            "deleted_model_display_name": "GPT-3",
            "replacement_model_slug": "gpt-4",
            "nodes_migrated": 5,
        }
    )
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.delete(
        "/llm/models/gpt-3?replacement_model_slug=gpt-4"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["nodes_migrated"] == 5
    mock_db.delete_model.assert_called_once_with(
        model_id=existing.id, replacement_model_slug="gpt-4"
    )


def test_get_model_usage(mocker):
    """GET /llm/models/{slug}/usage returns node_count."""
    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.get_model_usage = AsyncMock(
        return_value={"model_slug": "gpt-4", "node_count": 7}
    )

    response = admin_client.get("/llm/models/gpt-4/usage")

    assert response.status_code == 200
    data = response.json()
    assert data["node_count"] == 7
    assert data["model_slug"] == "gpt-4"


def test_toggle_model_enable(mocker):
    """POST /llm/models/{slug}/toggle with is_enabled=True toggles model and returns 200."""
    existing = _make_mock_model(is_enabled=False)
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.toggle_model_with_migration = AsyncMock(
        return_value={"nodes_migrated": 0, "migrated_to_slug": None, "migration_id": None}
    )
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post(
        "/llm/models/gpt-4/toggle",
        json={"is_enabled": True},
    )

    assert response.status_code == 200
    assert response.json()["nodes_migrated"] == 0


def test_toggle_model_disable_with_migration(mocker):
    """Disabling with migrate_to_slug passes migration args and returns nodes_migrated."""
    existing = _make_mock_model(is_enabled=True)
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.toggle_model_with_migration = AsyncMock(
        return_value={
            "nodes_migrated": 3,
            "migrated_to_slug": "gpt-4-turbo",
            "migration_id": "mig-abc",
        }
    )
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post(
        "/llm/models/gpt-4/toggle",
        json={"is_enabled": False, "migrate_to_slug": "gpt-4-turbo"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["nodes_migrated"] == 3
    assert data["migration_id"] == "mig-abc"
    mock_db.toggle_model_with_migration.assert_called_once_with(
        model_id=existing.id,
        is_enabled=False,
        migrate_to_slug="gpt-4-turbo",
        migration_reason=None,
        custom_credit_cost=None,
    )


def test_toggle_model_not_found(mocker):
    """POST toggle returns 404 when the model slug does not exist."""
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)
    mocker.patch("backend.server.v2.llm.admin_routes.db_write")

    response = admin_client.post(
        "/llm/models/ghost-model/toggle",
        json={"is_enabled": True},
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_list_migrations(mocker):
    """GET /llm/migrations returns migrations list."""
    migrations = [_make_mock_migration(), _make_mock_migration(id="mig-2")]
    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.list_migrations = AsyncMock(return_value=migrations)

    response = admin_client.get("/llm/migrations")

    assert response.status_code == 200
    data = response.json()
    assert len(data["migrations"]) == 2
    assert data["migrations"][0]["id"] == "mig-1"


def test_revert_migration(mocker):
    """POST /llm/migrations/{id}/revert calls db_write and returns nodes_reverted."""
    mock_db = mocker.patch("backend.server.v2.llm.admin_routes.db_write")
    mock_db.revert_migration = AsyncMock(
        return_value={
            "migration_id": "mig-1",
            "source_model_slug": "gpt-3",
            "target_model_slug": "gpt-4",
            "nodes_reverted": 3,
            "nodes_already_changed": 0,
            "source_model_re_enabled": True,
        }
    )
    mock_db.refresh_runtime_caches = AsyncMock()

    response = admin_client.post("/llm/migrations/mig-1/revert")

    assert response.status_code == 200
    data = response.json()
    assert data["nodes_reverted"] == 3
    assert data["migration_id"] == "mig-1"
    mock_db.revert_migration.assert_called_once_with(
        migration_id="mig-1", re_enable_source_model=True
    )


# ---------------------------------------------------------------------------
# Creator CRUD
# ---------------------------------------------------------------------------


def test_create_creator(mocker):
    """POST /llm/creators returns 201 with creator fields."""
    mock_creator = _make_mock_creator()
    mocker.patch(
        "prisma.models.LlmModelCreator.prisma"
    ).return_value.create = AsyncMock(return_value=mock_creator)

    response = admin_client.post(
        "/llm/creators",
        json={"name": "openai", "display_name": "OpenAI"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "openai"
    assert data["display_name"] == "OpenAI"


def test_list_creators(mocker):
    """GET /llm/creators returns creators list."""
    creators = [_make_mock_creator(), _make_mock_creator(id="c-2", name="anthropic")]
    mocker.patch(
        "prisma.models.LlmModelCreator.prisma"
    ).return_value.find_many = AsyncMock(return_value=creators)

    response = admin_client.get("/llm/creators")

    assert response.status_code == 200
    data = response.json()
    assert len(data["creators"]) == 2


def test_update_creator(mocker):
    """PATCH /llm/creators/{name} returns 200 with updated creator."""
    existing = _make_mock_creator()
    updated = _make_mock_creator(display_name="OpenAI Corp")
    prisma_mock = mocker.patch("prisma.models.LlmModelCreator.prisma").return_value
    prisma_mock.find_unique = AsyncMock(return_value=existing)
    prisma_mock.update = AsyncMock(return_value=updated)

    response = admin_client.patch(
        "/llm/creators/openai",
        json={"display_name": "OpenAI Corp"},
    )

    assert response.status_code == 200
    assert response.json()["display_name"] == "OpenAI Corp"


def test_update_creator_not_found(mocker):
    """PATCH returns 404 when creator does not exist."""
    mocker.patch(
        "prisma.models.LlmModelCreator.prisma"
    ).return_value.find_unique = AsyncMock(return_value=None)

    response = admin_client.patch(
        "/llm/creators/nobody",
        json={"display_name": "Nobody"},
    )

    assert response.status_code == 404


def test_delete_creator_success(mocker):
    """DELETE /llm/creators/{name} returns 204 when creator has no models."""
    existing = _make_mock_creator(models=[])
    prisma_mock = mocker.patch("prisma.models.LlmModelCreator.prisma").return_value
    prisma_mock.find_unique = AsyncMock(return_value=existing)
    prisma_mock.delete = AsyncMock(return_value=existing)

    response = admin_client.delete("/llm/creators/openai")

    assert response.status_code == 204


def test_delete_creator_has_models(mocker):
    """DELETE returns 400 when the creator still has associated models."""
    existing = _make_mock_creator(models=[_make_mock_model()])
    mocker.patch(
        "prisma.models.LlmModelCreator.prisma"
    ).return_value.find_unique = AsyncMock(return_value=existing)

    response = admin_client.delete("/llm/creators/openai")

    assert response.status_code == 400
    assert "models" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Admin model list
# ---------------------------------------------------------------------------


def test_admin_list_models(mocker):
    """GET /llm/admin/models returns models list with creator and costs."""
    model = _make_mock_model()
    mocker.patch(
        "prisma.models.LlmModel.prisma"
    ).return_value.find_many = AsyncMock(return_value=[model])

    response = admin_client.get("/llm/admin/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["slug"] == "gpt-4"
