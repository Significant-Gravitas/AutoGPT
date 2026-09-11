import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.local_executor.state import (
    _MAX_RECORDINGS_PER_SESSION,
    _STATE_TTL_SECONDS,
    _recording_order_key,
    _recordings_key,
    get_recording_state,
    list_recording_states,
    mark_recording_reviewed,
    mark_recording_started,
    mark_recording_stopped,
)
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis


@pytest.mark.asyncio
async def test_recording_lifecycle_is_shared_across_callers() -> None:
    redis = FakeRedis()
    with patch(
        "backend.api.features.local_executor.state.get_redis_async",
        AsyncMock(return_value=redis),
    ):
        await mark_recording_started(
            "session-1",
            "recording-1",
            mode="demonstration",
            interpretation_route="extract_then_cloud",
            channels=["floor"],
        )
        started = await get_recording_state("session-1", "recording-1")
        assert started is not None and started.status == "recording"

        await mark_recording_stopped(
            "session-1",
            "recording-1",
            summary={"recording_id": "recording-1", "step_count": 3},
        )
        await mark_recording_reviewed("session-1", "recording-1", step_count=2)

        reviewed = await get_recording_state("session-1", "recording-1")
        listed = await list_recording_states("session-1")

    assert reviewed is not None
    assert reviewed.status == "reviewed"
    assert reviewed.step_count == 2
    assert [state.recording_id for state in listed] == ["recording-1"]


@pytest.mark.asyncio
async def test_recording_state_prunes_oldest_hash_and_order_entries() -> None:
    redis = FakeRedis()
    clock = iter(float(value) for value in range(1_000))
    with (
        patch(
            "backend.api.features.local_executor.state.get_redis_async",
            AsyncMock(return_value=redis),
        ),
        patch(
            "backend.api.features.local_executor.state.time.time",
            side_effect=lambda: next(clock),
        ),
    ):
        for index in range(_MAX_RECORDINGS_PER_SESSION + 5):
            await mark_recording_started(
                "session-1",
                f"recording-{index}",
                mode="demonstration",
                interpretation_route="extract_then_cloud",
                channels=["floor"],
            )
        listed = await list_recording_states("session-1")

    retained_ids = {state.recording_id for state in listed}
    assert len(retained_ids) == _MAX_RECORDINGS_PER_SESSION
    assert retained_ids == {
        f"recording-{index}" for index in range(5, _MAX_RECORDINGS_PER_SESSION + 5)
    }
    assert set(redis.hashes[_recordings_key("session-1")]) == retained_ids
    assert set(redis.sorted_sets[_recording_order_key("session-1")]) == retained_ids


@pytest.mark.asyncio
async def test_concurrent_recording_state_writes_use_one_atomic_bounded_operation() -> (
    None
):
    redis = FakeRedis()
    with (
        patch.object(redis, "eval", wraps=redis.eval) as eval_mock,
        patch(
            "backend.api.features.local_executor.state.get_redis_async",
            AsyncMock(return_value=redis),
        ),
    ):
        await asyncio.gather(
            *(
                mark_recording_started(
                    "session-1",
                    f"recording-{index}",
                    mode="demonstration",
                    interpretation_route="extract_then_cloud",
                    channels=["floor"],
                )
                for index in range(_MAX_RECORDINGS_PER_SESSION * 2)
            )
        )

    state_key = _recordings_key("session-1")
    order_key = _recording_order_key("session-1")
    retained_hash_ids = set(redis.hashes[state_key])
    retained_order_ids = set(redis.sorted_sets[order_key])
    assert retained_hash_ids == retained_order_ids
    assert len(retained_hash_ids) == _MAX_RECORDINGS_PER_SESSION
    assert eval_mock.await_count == _MAX_RECORDINGS_PER_SESSION * 2
    assert all(
        "LOCAL_EXECUTOR_STORE_RECORDING_STATE" in call.args[0]
        for call in eval_mock.await_args_list
    )
    assert redis.expirations[state_key] == _STATE_TTL_SECONDS
    assert redis.expirations[order_key] == _STATE_TTL_SECONDS
