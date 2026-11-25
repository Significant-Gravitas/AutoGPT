"""Tests for the read_understanding tool."""

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.read_understanding import ReadUnderstandingTool
from backend.data.understanding import (
    LIST_FIELDS,
    BusinessUnderstanding,
    format_understanding_for_prompt,
)

_PAIN_POINTS = [f"pain-{i:03d}" for i in range(57)]


@pytest.fixture
def profile() -> BusinessUnderstanding:
    now = datetime.now(tz=timezone.utc)
    return BusinessUnderstanding(
        id="test-id",
        user_id="user-1",
        created_at=now,
        updated_at=now,
        pain_points=_PAIN_POINTS,
        current_software=["Slack"],
    )


@pytest.fixture
def session() -> MagicMock:
    session = MagicMock()
    session.session_id = "session-1"
    return session


@pytest.mark.asyncio
async def test_the_list_the_context_points_at_comes_back_whole(profile, session):
    pointer = re.search(
        r'`tool:(\w+)` with field "(\w+)"', format_understanding_for_prompt(profile)
    )
    assert pointer is not None
    tool = ReadUnderstandingTool()
    assert pointer.group(1) == tool.name

    result = await _read(tool, profile, session, field=pointer.group(2))

    assert result.lists == {"pain_points": _PAIN_POINTS}


@pytest.mark.asyncio
async def test_without_a_field_every_list_comes_back(profile, session):
    result = await _read(ReadUnderstandingTool(), profile, session)

    assert list(result.lists) == list(LIST_FIELDS)
    assert result.lists["pain_points"] == _PAIN_POINTS
    assert result.lists["current_software"] == ["Slack"]


async def _read(tool, profile, session, **kwargs):
    db = MagicMock()
    db.get_business_understanding = AsyncMock(return_value=profile)
    with patch(
        "backend.copilot.tools.read_understanding.understanding_db", return_value=db
    ):
        result = await tool._execute("user-1", session, **kwargs)
    db.get_business_understanding.assert_awaited_once_with("user-1")
    return result
