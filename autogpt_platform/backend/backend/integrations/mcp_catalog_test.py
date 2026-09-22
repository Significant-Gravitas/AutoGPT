import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    MCPServerMetadata,
    get_mcp_catalog,
    parse_mcp_catalog,
)


def test_catalog_loads_validated_entries():
    assert get_mcp_catalog()


@pytest.mark.parametrize(
    "url",
    [
        "http://vendor.com/mcp",
        "https://localhost/mcp",
        "https://127.0.0.1/mcp",
        "https://192.168.1.1/mcp",
        "https://service.internal/mcp",
        "https://user:secret@vendor.com/mcp",
        "https://vendor.com/mcp?api_key=secret",
        "https://vendor.com/mcp#secret",
        "https://vendor.com:8443/mcp",
        "https://vendor.com/ mc p",
    ],
)
@pytest.mark.parametrize("field", ["server_url", "documentation_url"])
def test_catalog_rejects_unsafe_urls(url: str, field: str):
    data = get_mcp_catalog()[0].mcp_server.model_dump()
    data[field] = url
    with pytest.raises(ValidationError):
        MCPServerMetadata.model_validate(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"server_url": None},
        {"connection_mode": "custom"},
    ],
)
def test_catalog_rejects_misleading_connection_modes(changes: dict[str, str | None]):
    data = get_mcp_catalog()[0].mcp_server.model_dump()
    data.update(changes)
    with pytest.raises(ValidationError):
        MCPServerMetadata.model_validate(data)


def test_custom_presets_require_authentication_methods():
    data = get_mcp_catalog()[0].mcp_server.model_dump()
    data.update(server_url=None, connection_mode="custom")
    data.pop("auth_methods")
    with pytest.raises(ValidationError, match="auth_methods"):
        MCPServerMetadata.model_validate(data)


def test_catalog_rejects_duplicate_names():
    entry = get_mcp_catalog()[0].model_dump()
    with pytest.raises(ValueError, match="unique"):
        parse_mcp_catalog(json.dumps([entry, entry]))


def oauth_server_data():
    return {
        "server_url": "https://mcp.vendor.com/mcp",
        "documentation_url": "https://docs.vendor.com/mcp",
        "setup_instructions": "Sign in to the vendor.",
        "connection_mode": "hosted",
        "auth_methods": ["oauth"],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"auth_methods": []},
        {"auth_methods": ["oauth", "oauth"]},
        {"server_url": None, "connection_mode": "custom", "auth_methods": []},
        {"auth_methods": ["bearer"], "oauth_scopes": []},
    ],
)
def test_catalog_rejects_inconsistent_authentication(changes):
    with pytest.raises(ValidationError):
        MCPServerMetadata.model_validate({**oauth_server_data(), **changes})


@pytest.mark.parametrize("field", ["oauth_scopes", "oauth_write_scopes"])
@pytest.mark.parametrize(
    "scopes",
    [[""], ["read write"], ["read\twrite"], ["read\x00write"], ["read", "read"]],
)
def test_catalog_rejects_invalid_scope_tokens(field, scopes):
    data = {**oauth_server_data(), "oauth_scopes": []}
    data[field] = scopes
    with pytest.raises(ValidationError):
        MCPServerMetadata.model_validate(data)


def test_catalog_rejects_overlapping_scope_profiles():
    with pytest.raises(ValidationError):
        MCPServerMetadata.model_validate(
            {
                **oauth_server_data(),
                "oauth_scopes": ["read"],
                "oauth_write_scopes": ["read", "write"],
            }
        )


def test_catalog_preserves_empty_and_vendor_specific_scope_profiles():
    server = MCPServerMetadata.model_validate(
        {
            **oauth_server_data(),
            "oauth_scopes": [],
            "oauth_write_scopes": [
                "project:documents.write",
                "https://api.vendor.com/scope",
            ],
        }
    )
    assert server.oauth_scopes == []
    assert server.oauth_write_scopes == [
        "project:documents.write",
        "https://api.vendor.com/scope",
    ]


def catalog_entry(name, server):
    return MCPCatalogEntry.model_validate(
        {
            "name": name,
            "display_name": name,
            "description": "Vendor tools.",
            "mcp_server": server,
        }
    )


@pytest.mark.parametrize(
    "url,matched",
    [
        ("https://mcp.vendor.com/mcp", True),
        ("https://MCP.VENDOR.COM:443/mcp/", True),
        ("https://mcp.vendor.com/oauth", True),
        ("https://eu.vendor.com/mcp", True),
        ("https://mcp.vendor.com/other", False),
        ("https://mcp.vendor.com.evil.com/mcp", False),
        ("https://mcp.vendor.com@evil.com/mcp", False),
        ("https://user@mcp.vendor.com/mcp", False),
        ("https://mcp.vendor.com/mcp?tenant=other", False),
        ("https://mcp.vendor.com/mcp#other", False),
        ("http://mcp.vendor.com/mcp", False),
        ("https://mcp.vendor.com:8443/mcp", False),
    ],
)
def test_catalog_matches_only_complete_known_urls(url, matched):
    from backend.integrations.mcp_catalog import get_mcp_catalog_entry_for_url

    entry = catalog_entry(
        "mcp_vendor",
        {
            **oauth_server_data(),
            "oauth_server_url": "https://mcp.vendor.com/oauth",
            "server_url_options": [{"label": "EU", "url": "https://eu.vendor.com/mcp"}],
        },
    )
    with patch(
        "backend.integrations.mcp_catalog.get_mcp_catalog", return_value=(entry,)
    ):
        result = get_mcp_catalog_entry_for_url(url)

    assert result == (entry if matched else None)
