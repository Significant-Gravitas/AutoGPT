import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    MCPServerMetadata,
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
        "mcp_finances_plaid",
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
