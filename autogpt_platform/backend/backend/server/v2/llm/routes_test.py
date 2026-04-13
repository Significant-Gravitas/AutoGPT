"""Tests for public read-only LLM registry routes (routes.py).

Covers:
- GET /llm/models  (enabled_only=True default, enabled_only=False)
- GET /llm/providers
"""

from unittest.mock import Mock

import fastapi
import fastapi.testclient
import pytest

from backend.server.v2.llm.routes import router

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


# ---------------------------------------------------------------------------
# Auth fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Bypass JWT auth for all tests in this module."""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cost(
    unit: str = "RUN",
    credit_cost: int = 10,
    credential_provider: str = "openai",
    credential_id: str | None = None,
    credential_type: str | None = None,
    currency: str | None = None,
    metadata: dict | None = None,
) -> Mock:
    cost = Mock()
    cost.unit = unit
    cost.credit_cost = credit_cost
    cost.credential_provider = credential_provider
    cost.credential_id = credential_id
    cost.credential_type = credential_type
    cost.currency = currency
    cost.metadata = metadata or {}
    return cost


def _make_mock_creator(
    id: str = "creator-1",
    name: str = "openai",
    display_name: str = "OpenAI",
    description: str | None = "AI company",
    website_url: str | None = "https://openai.com",
    logo_url: str | None = None,
) -> Mock:
    creator = Mock()
    creator.id = id
    creator.name = name
    creator.display_name = display_name
    creator.description = description
    creator.website_url = website_url
    creator.logo_url = logo_url
    return creator


def _make_mock_model(
    slug: str = "gpt-4",
    display_name: str = "GPT-4",
    description: str | None = "Latest GPT",
    provider_display_name: str = "OpenAI",
    is_enabled: bool = True,
    is_recommended: bool = False,
    capabilities: dict | None = None,
    provider_key: str = "openai",
    context_window: int = 128000,
    max_output_tokens: int | None = 4096,
    price_tier: int = 2,
    creator: Mock | None = None,
    costs: list | None = None,
) -> Mock:
    model = Mock()
    model.slug = slug
    model.display_name = display_name
    model.description = description
    model.provider_display_name = provider_display_name
    model.is_enabled = is_enabled
    model.is_recommended = is_recommended
    model.capabilities = capabilities or {}
    model.creator = creator
    model.costs = costs or []

    meta = Mock()
    meta.provider = provider_key
    meta.context_window = context_window
    meta.max_output_tokens = max_output_tokens
    meta.price_tier = price_tier
    model.metadata = meta

    return model


# ---------------------------------------------------------------------------
# GET /llm/models
# ---------------------------------------------------------------------------


def test_list_models_enabled_only(mocker):
    """Default enabled_only=True calls get_enabled_models and returns correct shape."""
    mock_model = _make_mock_model()
    mocker.patch(
        "backend.server.v2.llm.routes.get_enabled_models", return_value=[mock_model]
    )
    mocker.patch("backend.server.v2.llm.routes.get_all_models", return_value=[])

    response = client.get("/llm/models")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["models"]) == 1
    first = data["models"][0]
    assert first["slug"] == "gpt-4"
    assert first["display_name"] == "GPT-4"
    assert first["provider_name"] == "OpenAI"
    assert first["is_enabled"] is True
    assert first["is_recommended"] is False
    assert first["context_window"] == 128000
    assert first["price_tier"] == 2
    assert first["creator"] is None
    assert first["costs"] == []


def test_list_models_all(mocker):
    """enabled_only=false calls get_all_models instead of get_enabled_models."""
    mock_model = _make_mock_model(is_enabled=False, slug="gpt-3")
    mock_get_all = mocker.patch(
        "backend.server.v2.llm.routes.get_all_models", return_value=[mock_model]
    )
    mock_get_enabled = mocker.patch(
        "backend.server.v2.llm.routes.get_enabled_models", return_value=[]
    )

    response = client.get("/llm/models?enabled_only=false")

    assert response.status_code == 200
    mock_get_all.assert_called_once()
    mock_get_enabled.assert_not_called()
    data = response.json()
    assert data["total"] == 1
    assert data["models"][0]["slug"] == "gpt-3"


def test_list_models_empty(mocker):
    """Empty registry returns an empty models list with total=0."""
    mocker.patch("backend.server.v2.llm.routes.get_enabled_models", return_value=[])

    response = client.get("/llm/models")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["models"] == []


def test_list_models_with_creator(mocker):
    """Model with a creator surfaces creator fields in the response."""
    creator = _make_mock_creator()
    mock_model = _make_mock_model(creator=creator)
    mocker.patch(
        "backend.server.v2.llm.routes.get_enabled_models", return_value=[mock_model]
    )

    response = client.get("/llm/models")

    assert response.status_code == 200
    creator_data = response.json()["models"][0]["creator"]
    assert creator_data is not None
    assert creator_data["id"] == "creator-1"
    assert creator_data["name"] == "openai"
    assert creator_data["display_name"] == "OpenAI"
    assert creator_data["website_url"] == "https://openai.com"


def test_list_models_with_costs(mocker):
    """Model with costs surfaces cost entries in the response."""
    cost = _make_mock_cost(unit="TOKENS", credit_cost=5, credential_provider="openai")
    mock_model = _make_mock_model(costs=[cost])
    mocker.patch(
        "backend.server.v2.llm.routes.get_enabled_models", return_value=[mock_model]
    )

    response = client.get("/llm/models")

    assert response.status_code == 200
    costs_data = response.json()["models"][0]["costs"]
    assert len(costs_data) == 1
    assert costs_data[0]["unit"] == "TOKENS"
    assert costs_data[0]["credit_cost"] == 5
    assert costs_data[0]["credential_provider"] == "openai"


# ---------------------------------------------------------------------------
# GET /llm/providers
# ---------------------------------------------------------------------------


def test_list_providers(mocker):
    """Single provider groups its models correctly."""
    model_a = _make_mock_model(
        slug="gpt-4", display_name="GPT-4", provider_key="openai"
    )
    model_b = _make_mock_model(
        slug="gpt-3.5", display_name="GPT-3.5", provider_key="openai"
    )
    mocker.patch(
        "backend.server.v2.llm.routes.get_enabled_models",
        return_value=[model_a, model_b],
    )

    response = client.get("/llm/providers")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert len(providers) == 1
    provider = providers[0]
    assert provider["name"] == "openai"
    assert provider["display_name"] == "OpenAI"
    assert len(provider["models"]) == 2


def test_list_providers_multiple_providers(mocker):
    """Two different providers are both present and sorted alphabetically."""
    openai_model = _make_mock_model(
        slug="gpt-4",
        display_name="GPT-4",
        provider_key="openai",
        provider_display_name="OpenAI",
    )
    anthropic_model = _make_mock_model(
        slug="claude-3",
        display_name="Claude 3",
        provider_key="anthropic",
        provider_display_name="Anthropic",
    )
    mocker.patch(
        "backend.server.v2.llm.routes.get_enabled_models",
        return_value=[openai_model, anthropic_model],
    )

    response = client.get("/llm/providers")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert len(providers) == 2
    provider_names = [p["name"] for p in providers]
    # sorted alphabetically: anthropic before openai
    assert provider_names == sorted(provider_names)
    assert "anthropic" in provider_names
    assert "openai" in provider_names


def test_list_providers_empty(mocker):
    """Empty registry returns an empty providers list."""
    mocker.patch("backend.server.v2.llm.routes.get_enabled_models", return_value=[])

    response = client.get("/llm/providers")

    assert response.status_code == 200
    assert response.json()["providers"] == []
