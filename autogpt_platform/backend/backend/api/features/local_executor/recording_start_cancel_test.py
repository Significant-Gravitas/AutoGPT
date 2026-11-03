import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.local_executor import routes
from backend.api.features.local_executor.models import RecordingStartRequest
from backend.copilot.tools.recording_models import RecordingSummary


def _shim(stop: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        capabilities=["recording"],
        recording=SimpleNamespace(
            start_with_consent=AsyncMock(return_value="rec-orphan"),
            effective_route_for=MagicMock(return_value="extract_then_cloud"),
            stop=stop,
        ),
        close_recording=MagicMock(),
    )


@pytest.mark.asyncio
async def test_a_cancelled_start_still_stops_the_recording() -> None:
    """Cancelled after the companion started recording, the request never
    hands its caller the id, so the route stops the recording before giving
    up rather than leave it running with no saved state."""
    stop = AsyncMock(return_value=RecordingSummary(recording_id="rec-orphan"))
    shim = _shim(stop)

    with (
        patch.object(routes, "_require_owned_shim", AsyncMock(return_value=shim)),
        patch.object(
            routes,
            "mark_recording_started",
            AsyncMock(side_effect=asyncio.CancelledError),
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await routes.start_session_recording(
                "session-1", RecordingStartRequest(), "owner-1"
            )

    stop.assert_awaited_once_with("rec-orphan")
    shim.close_recording.assert_called_once_with("rec-orphan")


@pytest.mark.asyncio
async def test_the_stop_survives_a_second_cancellation() -> None:
    """A cancelled scope can cancel every await that follows, so the stop
    runs on its own and finishes even if the request is cancelled again."""
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    async def slow_stop(recording_id: str) -> RecordingSummary:
        stop_started.set()
        await release_stop.wait()
        return RecordingSummary(recording_id=recording_id)

    stop = AsyncMock(side_effect=slow_stop)
    shim = _shim(stop)
    with (
        patch.object(routes, "_require_owned_shim", AsyncMock(return_value=shim)),
        patch.object(
            routes,
            "mark_recording_started",
            AsyncMock(side_effect=asyncio.CancelledError),
        ),
    ):
        request = asyncio.create_task(
            routes.start_session_recording(
                "session-1", RecordingStartRequest(), "owner-1"
            )
        )
        await asyncio.wait_for(stop_started.wait(), 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

    release_stop.set()
    await asyncio.wait_for(asyncio.gather(*routes._ROLLBACK_TASKS), 5)
    stop.assert_awaited_once_with("rec-orphan")
    shim.close_recording.assert_called_once_with("rec-orphan")
