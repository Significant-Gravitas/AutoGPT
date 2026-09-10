from backend.copilot.tools.get_mcp_guide import _load_guide
from backend.copilot.tools.skills import get_default_skill_with_body


def test_both_guide_loaders_expose_hosted_catalog_without_setup_only_entries():
    legacy = _load_guide()
    skill = get_default_skill_with_body("mcp_tool_guide")
    assert skill is not None
    assert legacy == skill.body
    assert "https://mcp.calendly.com" in legacy
    assert "https://mcp.atlassian.com/v2/mcp" in legacy
    assert "https://mcp.linear.app/mcp" in legacy
    assert "https://mcp.stripe.com" in legacy
    assert "https://mcp.cloudflare.com/mcp" in legacy
    assert legacy.count("https://mcp.notion.com/mcp") == 1
    assert legacy.count("https://aws-mcp.us-east-1.api.aws/mcp") == 1
    assert "| Asana |" not in legacy
    assert "| Slack |" not in legacy
    assert "| HubSpot |" not in legacy
    assert "Search public Arize documentation" in legacy
