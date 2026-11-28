"""Tests for the add_understanding tool."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.add_understanding import AddUnderstandingTool
from backend.data.understanding import BusinessUnderstanding


@pytest.mark.asyncio
async def test_the_saved_profile_comes_back_with_each_lists_newest_items_only():
    pain_points = [f"pain-{i:03d}" for i in range(57)]
    now = datetime.now(tz=timezone.utc)
    saved = BusinessUnderstanding(
        id="test-id",
        user_id="user-1",
        created_at=now,
        updated_at=now,
        pain_points=pain_points,
    )
    db = MagicMock()
    db.upsert_business_understanding = AsyncMock(return_value=saved)
    session = MagicMock()
    session.session_id = "session-1"

    with patch(
        "backend.copilot.tools.add_understanding.understanding_db", return_value=db
    ):
        result = await AddUnderstandingTool()._execute(
            "user-1", session, pain_points=["pain-056"]
        )

    shown = result.current_understanding["pain_points"]
    assert shown[:-1] == pain_points[-20:]
    assert shown[-1].startswith("(and 37 more: `tool:read_understanding`")
