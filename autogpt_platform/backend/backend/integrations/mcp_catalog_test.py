import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    MCPServerMetadata,
    get_connectable_mcp_catalog,
    get_mcp_catalog,
    parse_mcp_catalog,
)


def test_catalog_contains_unique_official_entries():
    entries = json.loads(Path(__file__).with_name("mcp_catalog.json").read_text())
    assert entries
    assert len({entry["name"] for entry in entries}) == len(entries)
    assert all(entry["official"] is True for entry in entries)
    assert all(entry["name"].startswith("mcp_") for entry in entries)
    for entry in entries:
        server = entry["mcp_server"]
        assert urlsplit(server["documentation_url"]).scheme == "https"
        if server["connection_mode"] == "hosted":
            url = urlsplit(server["server_url"])
            assert url.scheme == "https"
            assert url.hostname and not url.username and not url.password
            assert not url.query and not url.fragment
            assert server["auth_mode"] != "unknown"
        else:
            assert server["server_url"] is None
    unavailable = {
        "mcp_1password",
        "mcp_aikido",
        "mcp_ahrefs",
        "mcp_asana",
        "mcp_slack",
        "mcp_browserbase",
        "mcp_google_drive",
    }
    assert all(
        entry["mcp_server"]["connection_mode"] == "unavailable"
        for entry in entries
        if entry["name"] in unavailable
    )
    assert unavailable <= {entry["name"] for entry in entries}


def test_catalog_loads_typed_entries_and_keeps_native_names_separate():
    entries = get_mcp_catalog()
    assert entries
    assert all(entry.name != entry.display_name.lower() for entry in entries)
    assert {entry.name for entry in entries} >= {"mcp_notion", "mcp_google_drive"}


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
        {"auth_mode": "unknown"},
        {"connection_mode": "unavailable"},
        {"connection_mode": "custom"},
    ],
)
def test_catalog_rejects_misleading_connection_modes(changes: dict[str, str | None]):
    data = get_mcp_catalog()[0].mcp_server.model_dump()
    data.update(changes)
    with pytest.raises(ValidationError):
        MCPServerMetadata.model_validate(data)


def test_custom_presets_require_documented_authentication():
    data = get_mcp_catalog()[0].mcp_server.model_dump()
    data.update(server_url=None, connection_mode="custom", auth_mode="unknown")
    with pytest.raises(ValidationError, match="documented auth"):
        MCPServerMetadata.model_validate(data)


def test_catalog_rejects_community_entries_and_duplicate_names():
    entry = get_mcp_catalog()[0].model_dump()
    with pytest.raises(ValueError, match="unique"):
        parse_mcp_catalog(json.dumps([entry, entry]))
    entry["official"] = False
    with pytest.raises(ValidationError):
        MCPCatalogEntry.model_validate(entry)


def test_documentation_connections_do_not_claim_account_access():
    entries = {entry.name: entry for entry in get_mcp_catalog()}
    for name in ("mcp_arize", "mcp_atlassian_forge", "mcp_aws_databases"):
        assert entries[name].mcp_server.auth_mode == "none"
        assert "documentation" in entries[name].description.lower()


def oauth_server_data():
    return {
        "server_url": "https://mcp.vendor.com/mcp",
        "documentation_url": "https://docs.vendor.com/mcp",
        "setup_instructions": "Sign in to the vendor.",
        "connection_mode": "hosted",
        "auth_mode": "oauth",
        "auth_methods": ["oauth"],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"auth_methods": []},
        {"auth_methods": ["bearer"]},
        {"auth_mode": "token", "auth_methods": ["oauth", "bearer"]},
        {"auth_mode": "none", "auth_methods": ["bearer", "none"]},
        {"auth_methods": ["oauth", "oauth"]},
        {"server_url": None, "connection_mode": "custom", "auth_methods": []},
        {"server_url": None, "connection_mode": "unavailable"},
        {"auth_mode": "token", "auth_methods": ["bearer"], "oauth_scopes": []},
        {
            "auth_mode": "token",
            "auth_methods": ["bearer"],
            "oauth_callback_mode": "loopback",
        },
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
            "official": True,
            "mcp_server": server,
        }
    )


@pytest.mark.parametrize(
    "frontend_url,local_oauth_available",
    [
        ("http://localhost:3000", True),
        ("https://LOCALHOST:3000", True),
        ("http://127.0.0.1:3000", True),
        ("http://[::1]:3000", True),
        ("https://app.agpt.co", False),
        ("https://localhost.example.com", False),
        ("https://localhost@app.example.com", False),
        (None, False),
    ],
)
def test_catalog_filters_oauth_by_actual_callback_hostname(
    frontend_url, local_oauth_available
):
    local = catalog_entry(
        "mcp_local", {**oauth_server_data(), "oauth_callback_mode": "loopback"}
    )
    global_entry = catalog_entry("mcp_global", oauth_server_data())
    unavailable = catalog_entry(
        "mcp_unavailable",
        {
            **oauth_server_data(),
            "server_url": None,
            "connection_mode": "unavailable",
            "auth_methods": [],
        },
    )
    with patch(
        "backend.integrations.mcp_catalog.get_mcp_catalog",
        return_value=(local, global_entry, unavailable),
    ):
        result = get_connectable_mcp_catalog(frontend_url)

    assert {entry.name for entry in result} == (
        {"mcp_local", "mcp_global"} if local_oauth_available else {"mcp_global"}
    )
    assert local.mcp_server.auth_methods == ["oauth"]


@pytest.mark.parametrize(
    "auth_mode,methods,expected_mode,expected_methods",
    [
        ("oauth", ["oauth", "bearer", "basic"], "token", ["bearer", "basic"]),
        ("token", ["bearer", "oauth"], "token", ["bearer"]),
        ("none", ["none", "oauth"], "none", ["none"]),
    ],
)
def test_catalog_filter_returns_valid_manual_fallback(
    auth_mode, methods, expected_mode, expected_methods
):
    entry = catalog_entry(
        "mcp_vendor",
        {
            **oauth_server_data(),
            "auth_mode": auth_mode,
            "auth_methods": methods,
            "oauth_callback_mode": "loopback",
            "oauth_server_url": "https://mcp.vendor.com/oauth",
            "oauth_scopes": ["read"],
            "oauth_write_scopes": ["write"],
        },
    )
    with patch(
        "backend.integrations.mcp_catalog.get_mcp_catalog", return_value=(entry,)
    ):
        result = get_connectable_mcp_catalog("https://app.agpt.co")

    assert len(result) == 1
    server = result[0].mcp_server
    assert server.auth_mode == expected_mode
    assert server.auth_methods == expected_methods
    assert server.server_url == "https://mcp.vendor.com/mcp"
    assert server.oauth_server_url is None
    assert server.oauth_scopes is None
    assert server.oauth_write_scopes == []
    assert server.oauth_callback_mode == "any"
    assert MCPCatalogEntry.model_validate(result[0].model_dump()) == result[0]
    assert entry.mcp_server.auth_methods == methods
    assert entry.mcp_server.oauth_scopes == ["read"]


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
