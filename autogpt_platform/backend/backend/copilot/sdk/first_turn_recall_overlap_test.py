"""A chat's first turn recalls its memories while it builds its query: the
setup doesn't wait for the recall, which is awaited where the query is
sent."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.copilot.response_model import StreamStart
from backend.copilot.sdk import service

from .retry_scenarios_test import TestFirstTurnWarmContextIsEphemeral as _FirstTurn
from .retry_scenarios_test import TestFollowUpWarmContextCallSite as _FollowUp

_SVC = "backend.copilot.sdk.service"
_BLOCK = "<temporal_context>Alice works on Atlas</temporal_context>"
_WAIT = 20


def _noting_the_build(building: asyncio.Event):
    """``inject_user_context`` that marks the first turn's query build."""
    inject = service.inject_user_context

    async def inject_noting_the_build(*args, **kwargs):
        building.set()
        return await inject(*args, **kwargs)

    return inject_noting_the_build


async def _run_first_turn(queries: list[str], extra: list, *, expect_error=False):
    await _FirstTurn()._run(
        _FirstTurn._first_turn("what is Alice working on"),
        _FollowUp._clients(queries),
        first_block=None,
        refresh=_FirstTurn._no_refresh,
        extra=extra,
        expect_error=expect_error,
    )


@pytest.mark.asyncio
async def test_a_first_turn_builds_its_query_while_its_memories_are_recalled():
    building = asyncio.Event()
    seen: list[str] = []

    async def recall(user_id, message, *, expert_id=None):
        try:
            await asyncio.wait_for(building.wait(), _WAIT)
            seen.append("recalled while the query was built")
        except TimeoutError:
            seen.append("the query waited for the recall")
        return _BLOCK

    queries: list[str] = []
    await _run_first_turn(
        queries,
        [
            (f"{_SVC}.fetch_warm_context", dict(new=recall)),
            (f"{_SVC}.inject_user_context", dict(new=_noting_the_build(building))),
        ],
    )

    assert seen == ["recalled while the query was built"]
    assert len(queries) == 1
    assert "Alice works on Atlas" in queries[0]


@pytest.mark.asyncio
async def test_a_first_turn_sends_its_query_only_once_its_memories_are_in():
    recalled = asyncio.Event()

    async def slow_recall(user_id, message, *, expert_id=None):
        # Long enough that a query sent without waiting goes out first.
        await asyncio.sleep(0.2)
        recalled.set()
        return _BLOCK

    queries: list[str] = []
    await _run_first_turn(
        queries, [(f"{_SVC}.fetch_warm_context", dict(new=slow_recall))]
    )

    assert recalled.is_set()
    assert len(queries) == 1
    assert queries[0].count("Alice works on Atlas") == 1


@pytest.mark.asyncio
async def test_a_first_turn_that_ends_before_its_query_stops_its_recall():
    building = asyncio.Event()
    outcome: list[str] = []

    async def stuck_recall(user_id, message, *, expert_id=None):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        return _BLOCK

    async def build_fails(*args, **kwargs):
        building.set()
        raise RuntimeError("message store down")

    queries: list[str] = []
    with pytest.raises(RuntimeError, match="message store down"):
        # Bounded: a setup that waited for the recall would never finish.
        await asyncio.wait_for(
            _run_first_turn(
                queries,
                [
                    (f"{_SVC}.fetch_warm_context", dict(new=stuck_recall)),
                    (f"{_SVC}.inject_user_context", dict(new=build_fails)),
                ],
            ),
            _WAIT * 5,
        )
    # The cancelled recall runs its except clause on the next loop pass.
    await asyncio.wait_for(_until(lambda: bool(outcome)), _WAIT)

    assert building.is_set()
    assert outcome == ["cancelled"]
    assert queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, asyncio.CancelledError, None],
    ids=["setup-error", "cancelled", "disconnect"],
)
async def test_setup_failure_cancels_recall_before_persistence_without_waiting(
    error_type: type[BaseException] | None,
):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    persisting = asyncio.Event()
    finish_recall = asyncio.Event()
    finish_persistence = asyncio.Event()
    recall_tasks: list[asyncio.Task] = []

    async def recall(*args, **kwargs):
        task = asyncio.current_task()
        assert task is not None
        recall_tasks.append(task)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
            await finish_recall.wait()

    async def setup_fails(*args, **kwargs):
        await asyncio.wait_for(started.wait(), _WAIT)
        assert error_type is not None
        raise error_type("setup failed")

    async def disconnect_at_start(*args, **kwargs):
        stream = service.stream_chat_completion_sdk(*args, **kwargs)
        try:
            async for event in stream:
                yield event
                if isinstance(event, StreamStart):
                    await asyncio.wait_for(started.wait(), _WAIT)
                    break
        finally:
            await stream.aclose()

    async def persist(*args, **kwargs):
        persisting.set()
        await finish_persistence.wait()

    queries: list[str] = []
    interruption = (
        (f"{_SVC}._build_system_prompt", dict(new=setup_fails))
        if error_type is not None
        else (
            f"{_FirstTurn.__module__}.stream_chat_completion_sdk",
            dict(new=disconnect_at_start),
        )
    )
    turn = asyncio.create_task(
        _run_first_turn(
            queries,
            [
                (f"{_SVC}._fetch_graphiti_context", dict(new=recall)),
                interruption,
                (f"{_SVC}.persist_unfolded_rows", dict(new=persist)),
            ],
        )
    )
    try:
        await asyncio.wait_for(persisting.wait(), _WAIT)
        await asyncio.wait_for(cancelled.wait(), _WAIT)
        assert queries == []
        assert not turn.done()
    finally:
        finish_recall.set()
        finish_persistence.set()
        for task in recall_tasks:
            task.cancel()
        outcomes = await asyncio.wait_for(
            asyncio.gather(turn, *recall_tasks, return_exceptions=True), _WAIT
        )

    if error_type is None:
        assert outcomes[0] is None
    else:
        assert isinstance(outcomes[0], error_type)


@pytest.mark.asyncio
async def test_failed_setup_and_persistence_stop_context_and_recall():
    started = [asyncio.Event() for _ in range(4)]
    stopped = [asyncio.Event() for _ in range(4)]
    reads: list[asyncio.Task] = []
    persisted = asyncio.Event()
    persist_attempts = 0

    def held_read(index: int):
        async def read(*args, **kwargs):
            task = asyncio.current_task()
            assert task is not None
            reads.append(task)
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped[index].set()

        return read

    async def stop_reads():
        for task in reads:
            task.cancel()
        await asyncio.gather(*reads, return_exceptions=True)

    async def setup_fails(*args, **kwargs):
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)), _WAIT
        )
        raise RuntimeError("setup failed with all context reads running")

    async def persist(*args, **kwargs):
        nonlocal persist_attempts
        persist_attempts += 1
        if persist_attempts == 1:
            raise RuntimeError("persistence unavailable after setup failed")
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in stopped)), _WAIT
            )
            persisted.set()
        finally:
            await stop_reads()

    queries: list[str] = []
    try:
        with pytest.raises(RuntimeError, match="persistence unavailable after setup"):
            await asyncio.wait_for(
                _run_first_turn(
                    queries,
                    [
                        (f"{_SVC}.build_session_context", dict(new=held_read(0))),
                        (f"{_SVC}.build_skills_context", dict(new=held_read(1))),
                        (
                            f"{_SVC}.first_message_expert_context",
                            dict(new=held_read(2)),
                        ),
                        (f"{_SVC}._fetch_graphiti_context", dict(new=held_read(3))),
                        (f"{_SVC}._build_system_prompt", dict(new=setup_fails)),
                        (f"{_SVC}.persist_unfolded_rows", dict(new=persist)),
                    ],
                ),
                _WAIT * 5,
            )
    finally:
        await stop_reads()

    assert persisted.is_set()
    assert queries == []


@pytest.mark.asyncio
async def test_a_follow_up_carries_its_own_refresh_not_a_first_turn_recall():
    recall = AsyncMock(return_value=_BLOCK)

    async def refresh(user_id, message, *, expert_id=None, force=False):
        return "<temporal_context>Bob works on Nova</temporal_context>"

    queries: list[str] = []
    await _FirstTurn()._run(
        _FirstTurn._second_turn(
            "what is Alice working on", "what do you know about Bob now"
        ),
        _FollowUp._clients(queries),
        first_block=None,
        refresh=refresh,
        extra=[(f"{_SVC}.fetch_warm_context", dict(new=recall))],
    )

    recall.assert_not_awaited()
    assert len(queries) == 1
    assert "Bob works on Nova" in queries[0]
    assert "Alice works on Atlas" not in queries[0]


async def _until(condition) -> None:
    while not condition():
        await asyncio.sleep(0)
