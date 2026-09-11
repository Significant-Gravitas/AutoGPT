import re

from backend.copilot.tools.get_mcp_guide import _load_guide
from backend.copilot.tools.skills import get_default_skill_with_body


def test_both_guide_loaders_expose_hosted_catalog_without_setup_only_entries():
    legacy = _load_guide()
    skill = get_default_skill_with_body("mcp_tool_guide")
    assert skill is not None
    assert legacy == skill.body
    urls = re.findall(r"`(https://[^`\s]+)`", legacy)
    assert "https://mcp.calendly.com" in urls
    assert "https://mcp.atlassian.com/v2/mcp" in urls
    assert "https://mcp.linear.app/mcp" in urls
    assert "https://mcp.stripe.com" in urls
    assert "https://mcp.cloudflare.com/mcp" in urls
    assert urls.count("https://mcp.notion.com/mcp") == 1
    assert urls.count("https://aws-mcp.us-east-1.api.aws/mcp") == 1
    assert "| Asana |" not in legacy
    assert "| Slack |" not in legacy
    assert "| HubSpot |" not in legacy
    assert "Search public Arize documentation" in legacy
