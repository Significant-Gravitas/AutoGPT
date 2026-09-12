from unittest.mock import patch

from backend.copilot.tools.get_mcp_guide import _load_guide
from backend.copilot.tools.skills import get_default_skill_with_body
from backend.integrations.mcp_guide import MCP_CATALOG_MARKER


def test_both_guide_loaders_render_the_catalogue():
    with patch("backend.copilot.tools.get_mcp_guide._GUIDE_CACHE", None):
        legacy = _load_guide()
        skill = get_default_skill_with_body("mcp_tool_guide")

    assert skill is not None
    assert legacy == skill.body
    assert MCP_CATALOG_MARKER not in legacy
    assert "Additional official connections" in legacy
