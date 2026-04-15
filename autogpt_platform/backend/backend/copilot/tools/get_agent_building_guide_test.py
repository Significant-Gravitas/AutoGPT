"""Tests for GetAgentBuildingGuideTool."""

from backend.copilot.tools.get_agent_building_guide import _load_guide


def test_load_guide_returns_string():
    guide = _load_guide()
    assert isinstance(guide, str)
    assert len(guide) > 100


def test_load_guide_caches():
    guide1 = _load_guide()
    guide2 = _load_guide()
    assert guide1 is guide2
