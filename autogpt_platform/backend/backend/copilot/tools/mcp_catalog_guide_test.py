import re
from collections import Counter

from backend.copilot.tools.get_mcp_guide import _load_guide
from backend.copilot.tools.skills import get_default_skill_with_body
from backend.integrations.mcp_catalog import get_connectable_mcp_catalog


def test_both_guide_loaders_expose_hosted_catalog_without_setup_only_entries():
    legacy = _load_guide()
    skill = get_default_skill_with_body("mcp_tool_guide")
    assert skill is not None
    assert legacy == skill.body
    urls = re.findall(r"`(https://[^`\s]+)`", legacy)
    hosted = {
        entry.mcp_server.server_url
        for entry in get_connectable_mcp_catalog()
        if entry.mcp_server.connection_mode == "hosted"
    }
    expected = hosted | {
        "https://mcp.linear.app/mcp",
        "https://mcp.stripe.com",
        "https://mcp.cloudflare.com/mcp",
    }
    assert Counter(urls) == Counter(expected)
    assert "| Asana |" not in legacy
    assert "| Slack |" not in legacy
    assert "| HubSpot |" not in legacy
    assert "Search public Arize documentation" in legacy
