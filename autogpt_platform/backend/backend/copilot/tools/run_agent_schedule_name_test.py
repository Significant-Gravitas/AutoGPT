from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.run_agent import RunAgentTool

from ._test_data import make_session


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", "   ", "\t\n", "\u2003"])
async def test_blank_schedule_name_returns_error_before_side_effects(name: str):
    with (
        patch("backend.copilot.tools.run_agent.user_db") as users,
        patch(
            "backend.copilot.tools.run_agent.get_or_create_library_agent",
            new_callable=AsyncMock,
        ) as library,
        patch("backend.copilot.tools.run_agent.get_scheduler_client") as scheduler,
    ):
        response = await RunAgentTool()._schedule_agent(
            user_id="user-1",
            session=make_session("user-1"),
            graph=MagicMock(),
            graph_credentials={},
            inputs={},
            schedule_name=name,
            cron="0 9 * * *",
            timezone=None,
        )
    assert isinstance(response, ErrorResponse)
    assert response.message == "schedule_name is required for scheduled execution"
    users.assert_not_called()
    library.assert_not_awaited()
    scheduler.assert_not_called()
