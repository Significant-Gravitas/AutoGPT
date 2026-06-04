"""Tests for GetAgentBuildingGuideTool."""

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.get_agent_building_guide import (
    _TRIGGER_AGENTS_HEADING,
    GetAgentBuildingGuideTool,
    _load_guide,
    _strip_h3_section,
)
from backend.copilot.tools.models import ResponseType


def test_load_guide_returns_string():
    guide = _load_guide()
    assert isinstance(guide, str)
    assert len(guide) > 100


def test_load_guide_caches():
    guide1 = _load_guide()
    guide2 = _load_guide()
    assert guide1 is guide2


def test_guide_contains_trigger_agents_heading():
    """Sanity check: the heading sentinel used to gate the trigger-agents
    section must actually appear in the guide. If this fails, either the
    heading was renamed or the section was removed — update the sentinel
    in get_agent_building_guide.py to match."""
    guide = _load_guide()
    assert f"\n### {_TRIGGER_AGENTS_HEADING}" in guide


def test_strip_h3_section_preserves_following_sections():
    """A future H3 section appended after the gated one must survive
    the strip — that's the whole reason the gating is section-aware
    rather than strip-to-EOF."""
    guide = "\n".join(
        [
            "## Title",
            "",
            "### First Section",
            "",
            "alpha",
            "",
            "### Building Trigger Agents",
            "",
            "secret",
            "",
            "### Future Section",
            "",
            "beta",
        ]
    )
    stripped = _strip_h3_section(guide, "Building Trigger Agents")
    assert "alpha" in stripped
    assert "secret" not in stripped
    assert "Building Trigger Agents" not in stripped
    assert "### Future Section" in stripped
    assert "beta" in stripped


def test_strip_h3_section_no_op_when_heading_missing():
    """Stripping a missing heading is a no-op (defensive — guards
    against the heading being renamed without the sentinel updated)."""
    guide = "## Title\n\n### Only Section\n\ncontent"
    assert _strip_h3_section(guide, "Building Trigger Agents") == guide


def _make_session(user_id: str = "user-1") -> ChatSession:
    return ChatSession.new(user_id=user_id, dry_run=False)


@pytest.mark.asyncio
async def test_guide_includes_trigger_section_when_flag_enabled(mocker):
    mocker.patch(
        "backend.copilot.tools.get_agent_building_guide.is_feature_enabled",
        new=mocker.AsyncMock(return_value=True),
    )
    tool = GetAgentBuildingGuideTool()
    result = await tool._execute(user_id="user-1", session=_make_session())
    assert result.type == ResponseType.AGENT_BUILDER_GUIDE
    assert "Building Trigger Agents" in result.content  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_guide_strips_trigger_section_when_flag_disabled(mocker):
    mocker.patch(
        "backend.copilot.tools.get_agent_building_guide.is_feature_enabled",
        new=mocker.AsyncMock(return_value=False),
    )
    tool = GetAgentBuildingGuideTool()
    result = await tool._execute(user_id="user-1", session=_make_session())
    assert result.type == ResponseType.AGENT_BUILDER_GUIDE
    assert "Building Trigger Agents" not in result.content  # type: ignore[attr-defined]
    # Pre-trigger content (e.g. block-ID examples) must still be present.
    assert "AgentExecutorBlock" in result.content  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_guide_strips_trigger_section_when_no_user_id(mocker):
    """Anonymous CoPilot calls (no user_id) must not see the trigger
    section — there's no user context to evaluate the flag against, so
    default to off."""
    spy = mocker.patch(
        "backend.copilot.tools.get_agent_building_guide.is_feature_enabled",
        new=mocker.AsyncMock(return_value=True),
    )
    tool = GetAgentBuildingGuideTool()
    result = await tool._execute(user_id=None, session=_make_session())
    assert "Building Trigger Agents" not in result.content  # type: ignore[attr-defined]
    spy.assert_not_called()
