from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.run_agent import RunAgentInput, RunAgentTool

_PATH = "backend.copilot.tools.run_agent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested,expected", [("UTC", "UTC"), ("Europe/London", "Europe/London")]
)
async def test_schedule_explicit_timezone(requested, expected):
    await _check_schedule_timezone(requested, expected)


@pytest.mark.asyncio
async def test_schedule_omitted_timezone_uses_account():
    await _check_schedule_timezone(RunAgentInput().timezone, "Asia/Calcutta")


async def _check_schedule_timezone(requested, expected):
    session = ChatSession.new("owner", dry_run=False)
    session.organization_id = "org"
    session.team_id = "team"
    scheduler = MagicMock(
        add_execution_schedule=AsyncMock(side_effect=RuntimeError("captured"))
    )
    with (
        patch(
            f"{_PATH}.get_or_create_library_agent", AsyncMock(return_value=MagicMock())
        ),
        patch(f"{_PATH}.emit_tool_display_name"),
        patch(
            f"{_PATH}.user_db",
            return_value=MagicMock(
                get_user_by_id=AsyncMock(
                    return_value=MagicMock(timezone="Asia/Calcutta")
                )
            ),
        ),
        patch(f"{_PATH}.get_scheduler_client", return_value=scheduler),
    ):
        with pytest.raises(RuntimeError, match="captured"):
            await RunAgentTool()._schedule_agent(
                "owner",
                session,
                MagicMock(id="graph"),
                {},
                {},
                "QA",
                "0 9 * * *",
                requested,
            )
    assert (
        scheduler.add_execution_schedule.await_args.kwargs["user_timezone"] == expected
    )
