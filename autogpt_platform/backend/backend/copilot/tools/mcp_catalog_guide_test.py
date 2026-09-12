import re
from collections import Counter
from unittest.mock import patch

import pytest

from backend.copilot.tools.get_mcp_guide import _load_guide
from backend.copilot.tools.skills import get_default_skill_with_body
from backend.integrations.mcp_catalog import get_connectable_mcp_catalog


@pytest.mark.parametrize(
    "frontend_url", ["https://platform.agpt.co", "http://localhost:3000"]
)
def test_both_guide_loaders_expose_supported_connections(frontend_url: str):
    with (
        patch("backend.copilot.tools.get_mcp_guide._GUIDE_CACHE", None),
        patch(
            "backend.integrations.mcp_guide.settings.config.frontend_base_url",
            frontend_url,
        ),
    ):
        legacy = _load_guide()
        skill = get_default_skill_with_body("mcp_tool_guide")
    assert skill is not None
    assert legacy == skill.body
    urls = re.findall(r"`(https://[^`\s]+)`", legacy)
    expected = {
        "https://mcp.linear.app/mcp",
        "https://mcp.stripe.com",
        "https://mcp.cloudflare.com/mcp",
    }
    for entry in get_connectable_mcp_catalog(frontend_url):
        assert entry.display_name in legacy
        server = entry.mcp_server
        expected.update(option.url for option in server.server_url_options)
        if server.server_url:
            expected.add(server.server_url)
        if server.oauth_server_url:
            expected.add(server.oauth_server_url)
    assert Counter(urls) == Counter(expected)
    assert "| Asana |" not in legacy
    assert "| Slack |" not in legacy
    assert "| HubSpot |" not in legacy
    assert "Search public Arize documentation" in legacy
