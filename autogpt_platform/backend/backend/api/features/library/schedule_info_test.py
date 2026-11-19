"""Library schedule badges come from the schedule index when it can answer."""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock

from backend.api.features.library._schedule_info import _fetch_schedule_info

_EARLY = "2026-12-01T08:30:00+00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "from_index, listed",
    [
        ({"g1": _EARLY}, False),
        (None, True),
    ],
    ids=["index", "listing-fallback"],
)
async def test_each_graphs_earliest_next_run_is_shown(
    mocker: pytest_mock.MockFixture, from_index: dict | None, listed: bool
):
    client = Mock(
        get_graph_schedule_next_runs=AsyncMock(return_value=from_index),
        get_graph_execution_schedules=AsyncMock(
            return_value=[
                Mock(graph_id="g1", next_run_time="2026-12-01T10:00:00+01:00"),
                Mock(graph_id="g1", next_run_time=_EARLY),
            ]
        ),
    )
    mocker.patch(
        "backend.api.features.library._schedule_info.get_scheduler_client",
        return_value=client,
    )

    info = await _fetch_schedule_info("u1")

    assert info.next_runs == {"g1": _EARLY}
    assert client.get_graph_execution_schedules.await_count == int(listed)
