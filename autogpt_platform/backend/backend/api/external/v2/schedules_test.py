"""v2 schedules show what the internal routes show, paused ones included."""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import APIKeyPermission
from pydantic import ValidationError

from backend.util.exceptions import NotFoundError

from .models import AgentRunScheduleCreateRequest
from .pagination import PageRequest
from .schedules import delete_schedule, list_all_schedules
from .tenancy import TenantContext

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)


def _schedule(schedule_id: str, *, paused: bool, expert_id: str | None = None):
    return Mock(
        id=schedule_id,
        organization_id="org-1",
        expert_id=expert_id,
        next_run_time="" if paused else "2026-10-10T09:00:00+00:00",
    )


@pytest.fixture
def scheduler(mocker: pytest_mock.MockFixture) -> Mock:
    client = Mock(
        get_graph_execution_schedules=AsyncMock(
            return_value=[
                _schedule("running", paused=False),
                _schedule("paused", paused=True),
                # Paused when its expert was archived; only a re-hire restores it.
                _schedule("archived", paused=True, expert_id="gone"),
            ]
        ),
        delete_schedule=AsyncMock(),
    )
    mocker.patch(
        "backend.api.external.v2.schedules.get_scheduler_client", return_value=client
    )
    mocker.patch(
        "backend.api.external.v2.schedules.get_user_team_ids",
        new_callable=AsyncMock,
        return_value=[],
    )
    mocker.patch(
        "backend.api.features.schedule_visibility.experts_db",
        return_value=Mock(active_expert_ids=AsyncMock(return_value=set())),
    )
    mocker.patch(
        "backend.api.external.v2.models.AgentRunSchedule.from_internal",
        side_effect=lambda s: s,
    )
    return client


async def test_paused_schedules_are_listed(scheduler: Mock) -> None:
    listed = await list_all_schedules(
        graph_id=None, page=PageRequest(limit=25), auth=_AUTH
    )

    assert [s.id for s in listed.items] == ["running", "paused"]
    assert scheduler.get_graph_execution_schedules.await_args.kwargs["include_paused"]


async def test_a_paused_schedule_can_be_deleted(scheduler: Mock) -> None:
    await delete_schedule(schedule_id="paused", auth=_AUTH)

    scheduler.delete_schedule.assert_awaited_once_with(
        schedule_id="paused", user_id="user-1"
    )


async def test_an_archived_experts_paused_schedule_stays_out_of_reach(
    scheduler: Mock,
) -> None:
    with pytest.raises(NotFoundError):
        await delete_schedule(schedule_id="archived", auth=_AUTH)

    scheduler.delete_schedule.assert_not_awaited()


def test_an_unknown_timezone_is_refused_before_it_reaches_the_scheduler() -> None:
    with pytest.raises(ValidationError, match="Unknown timezone"):
        AgentRunScheduleCreateRequest(
            graph_id="graph-1", name="daily", cron="0 9 * * *", timezone="Mars/Olympus"
        )

    request = AgentRunScheduleCreateRequest(
        graph_id="graph-1", name="daily", cron="0 9 * * *", timezone="Europe/Paris"
    )
    assert request.timezone == "Europe/Paris"
