from unittest.mock import patch

import pytest
from autogpt_libs.auth import get_optional_user_id
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.features.integrations.models import ProviderNamesResponse
from backend.api.features.integrations.router import router, settings
from backend.integrations.mcp_catalog import (
    get_connectable_mcp_catalog,
    get_mcp_catalog,
)
from backend.util.settings import BehaveAs

app = FastAPI()
app.include_router(router)
app.dependency_overrides[get_optional_user_id] = lambda: None


@pytest.mark.parametrize("behave_as", [BehaveAs.CLOUD, BehaveAs.LOCAL])
def test_provider_endpoint_only_exposes_connectable_catalog(behave_as: BehaveAs):
    with (
        patch.object(settings.config, "behave_as", behave_as),
        patch("backend.blocks.load_all_blocks"),
        patch(
            "backend.api.features.integrations.router.get_all_provider_names",
            return_value=["notion", "google"],
        ),
        patch(
            "backend.api.features.integrations.router.get_provider_description",
            return_value="Native integration",
        ),
        patch(
            "backend.api.features.integrations.router.get_supported_auth_types",
            return_value=["oauth2"],
        ),
    ):
        response = TestClient(app).get("/providers")

    assert response.status_code == 200
    providers = response.json()
    connectable = {
        entry.name
        for entry in get_connectable_mcp_catalog(settings.config.frontend_base_url)
    }
    assert {provider["name"] for provider in providers[2:]} == connectable
    assert len({provider["name"] for provider in providers}) == len(providers)
    assert "mcp_1password" not in {provider["name"] for provider in providers}
    assert "mcp_langfuse" in {provider["name"] for provider in providers}
    assert [provider["name"] for provider in providers[:2]] == ["notion", "google"]
    assert all(provider["mcp_server"] is None for provider in providers[:2])
    notion = next(
        provider for provider in providers if provider["name"] == "mcp_notion"
    )
    assert notion["display_name"] == "Notion"
    assert notion["supported_auth_types"] == []
    assert notion["mcp_server"]["server_url"] == "https://mcp.notion.com/mcp"
    assert notion["mcp_server"]["connection_mode"] == "hosted"


def test_catalog_entries_do_not_register_credential_providers():
    credential_providers = set(ProviderNamesResponse().providers)
    catalog_names = {entry.name for entry in get_mcp_catalog()}
    assert credential_providers.isdisjoint(catalog_names)


def test_provider_openapi_exposes_typed_mcp_metadata():
    schemas = app.openapi()["components"]["schemas"]
    metadata = schemas["MCPServerMetadata"]["properties"]
    assert metadata["connection_mode"]["enum"] == ["hosted", "custom", "unavailable"]
    assert metadata["auth_mode"]["enum"] == ["oauth", "token", "none", "unknown"]
    assert "mcp_server" in schemas["ProviderMetadata"]["properties"]
