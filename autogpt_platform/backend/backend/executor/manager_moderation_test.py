"""A run waits a bounded time on each AutoMod check. A check that runs out of
time is cancelled, and the run goes on without it only with automod_fail_open."""

import asyncio
import inspect
import threading
import time
from collections.abc import Coroutine, Iterator
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.execution import ExecutionStatus
from backend.executor import manager
from backend.executor.automod import manager as automod_module
from backend.executor.automod.manager import AutoModManager
from backend.executor.automod.models import ModerationConfig
from backend.util.exceptions import ModerationError

Stage = Literal["input", "output"]


@pytest.fixture
def evaluation_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Stands in for the executor's node-evaluation loop, on its own thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10)
    loop.close()


@pytest.fixture
def db_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The database client the run's checks and node updates go through."""
    client = MagicMock()
    monkeypatch.setattr(manager, "get_db_async_client", lambda: client)
    return client


def _fail_open(
    monkeypatch: pytest.MonkeyPatch, fail_open: bool, *, enabled: bool = True
) -> MagicMock:
    config = MagicMock(enabled=enabled, fail_open=fail_open)
    automod = MagicMock(config=config, update_failed_nodes_for_moderation=AsyncMock())
    monkeypatch.setattr(manager, "automod_manager", automod)
    return automod


def _check_that_never_answers() -> (
    tuple[Coroutine[Any, Any, Exception | None], threading.Event, threading.Event]
):
    started = threading.Event()
    cancelled = threading.Event()

    async def check() -> Exception | None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return None

    return check(), started, cancelled


def _stopped(
    work: Coroutine[Any, Any, Any], started: threading.Event, cancelled: threading.Event
) -> bool:
    """Whether work the run gave up on was stopped rather than left running: it
    saw the cancel or, cancelled before its first step, never ran at all. From
    Python 3.13.10 a task is cancelled before that step when the run gave up on
    it before the loop got to it, as a loaded runner can make it."""
    for _ in range(1000):
        if cancelled.is_set():
            return True
        if not started.is_set() and (
            inspect.getcoroutinestate(work) == inspect.CORO_CLOSED
        ):
            return True
        time.sleep(0.01)
    return False


def _await(
    check: Coroutine[Any, Any, Exception | None],
    loop: asyncio.AbstractEventLoop,
    stage: Stage = "input",
) -> None:
    manager._await_moderation(
        check,
        loop,
        moderation_type=stage,
        graph_exec=MagicMock(user_id="user-1", graph_exec_id="exec-1"),
        log_metadata=MagicMock(),
    )


@pytest.mark.parametrize("stage", ["input", "output"])
def test_a_check_that_runs_out_of_time_fails_the_execution(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_loop: asyncio.AbstractEventLoop,
    db_client: MagicMock,
    stage: Stage,
) -> None:
    automod = _fail_open(monkeypatch, False)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    check, started, cancelled = _check_that_never_answers()

    with pytest.raises(ModerationError) as refused:
        _await(check, evaluation_loop, stage)

    assert refused.value.moderation_type == stage
    assert refused.value.graph_exec_id == "exec-1"
    assert _stopped(check, started, cancelled), "the timed-out check kept running"
    # Its nodes are failed as a refused check of the same kind fails them
    automod.update_failed_nodes_for_moderation.assert_awaited_once_with(
        db_client, "exec-1", stage
    )


def test_with_automod_off_a_run_never_waits_on_a_check(
    monkeypatch: pytest.MonkeyPatch, evaluation_loop: asyncio.AbstractEventLoop
) -> None:
    """Without AutoMod there is no moderation that could fail, so a busy
    evaluation loop must not hold the run up, let alone fail it."""
    _fail_open(monkeypatch, False, enabled=False)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    check, started, cancelled = _check_that_never_answers()

    _await(check, evaluation_loop)

    assert inspect.getcoroutinestate(check) == inspect.CORO_CLOSED
    assert not started.is_set(), "the check ran"


def test_with_fail_open_a_check_that_runs_out_of_time_is_skipped(
    monkeypatch: pytest.MonkeyPatch, evaluation_loop: asyncio.AbstractEventLoop
) -> None:
    automod = _fail_open(monkeypatch, True)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    check, started, cancelled = _check_that_never_answers()

    _await(check, evaluation_loop)

    assert _stopped(check, started, cancelled), "the timed-out check kept running"
    automod.update_failed_nodes_for_moderation.assert_not_called()


def test_a_check_the_loop_starts_after_the_wait_is_over_is_still_cancelled(
    monkeypatch: pytest.MonkeyPatch, evaluation_loop: asyncio.AbstractEventLoop
) -> None:
    """A loaded runner may get to a check only after the run has given up on
    it. The check must still be stopped, not left running: cancelled once it
    starts or, from Python 3.13.10, before it starts."""
    _fail_open(monkeypatch, True)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    loop_is_free = threading.Event()
    evaluation_loop.call_soon_threadsafe(loop_is_free.wait, 30)
    check, started, cancelled = _check_that_never_answers()

    try:
        _await(check, evaluation_loop)
        assert inspect.getcoroutinestate(check) == inspect.CORO_CREATED
    finally:
        loop_is_free.set()

    assert _stopped(check, started, cancelled), "the late check kept running"


def _automod_that_never_answers(
    monkeypatch: pytest.MonkeyPatch, *, fail_open: bool
) -> tuple[AutoModManager, threading.Event, threading.Event]:
    """The real AutoMod manager, switched on, whose request never comes back."""
    request_started = threading.Event()
    request_cancelled = threading.Event()

    async def never_answers(*args: object, **kwargs: object) -> object:
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise
        return None

    config = ModerationConfig(
        api_url="https://automod.example/api", api_key="key", fail_open=fail_open
    )
    monkeypatch.setattr(AutoModManager, "_load_config", lambda self: config)
    monkeypatch.setattr(
        automod_module, "is_feature_enabled", AsyncMock(return_value=True)
    )
    requests = MagicMock()
    requests.return_value.post = never_answers
    monkeypatch.setattr(automod_module, "Requests", requests)
    automod = AutoModManager()
    monkeypatch.setattr(manager, "automod_manager", automod)
    return automod, request_started, request_cancelled


@pytest.mark.parametrize("fail_open", [False, True], ids=["fail closed", "fail open"])
def test_when_automod_never_answers_the_run_follows_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_loop: asyncio.AbstractEventLoop,
    db_client: MagicMock,
    fail_open: bool,
) -> None:
    """End to end through the real AutoMod manager: its request hangs, so the
    run stops waiting, cancels the request and fails with its queued node
    failed as a refusal fails it, unless automod_fail_open lets it go on."""
    automod, request_started, request_cancelled = _automod_that_never_answers(
        monkeypatch, fail_open=fail_open
    )
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(manager, "send_async_execution_update", AsyncMock())
    db_client.get_graph = AsyncMock(
        return_value=MagicMock(
            nodes=[MagicMock(id="node-1", input_default={"prompt": "hello"})]
        )
    )
    db_client.get_node_executions = AsyncMock(
        return_value=[
            MagicMock(
                node_exec_id="node-exec-1",
                input_data={"prompt": "hello"},
                output_data={},
            )
        ]
    )
    db_client.update_node_execution_status = AsyncMock()
    graph_exec = MagicMock(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        graph_exec_id="exec-1",
        nodes_input_masks=None,
    )
    moderation = automod.moderate_graph_execution_inputs(
        db_client=db_client, graph_exec=graph_exec
    )

    if fail_open:
        _await(moderation, evaluation_loop)
        db_client.update_node_execution_status.assert_not_called()
    else:
        with pytest.raises(ModerationError):
            _await(moderation, evaluation_loop)
        update = db_client.update_node_execution_status.await_args
        assert update.args == ("node-exec-1",)
        assert update.kwargs["status"] == ExecutionStatus.FAILED

    assert _stopped(
        moderation, request_started, request_cancelled
    ), "the AutoMod request kept running"


@pytest.mark.parametrize("fail_open", [False, True])
def test_a_check_that_answers_in_time_decides(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_loop: asyncio.AbstractEventLoop,
    fail_open: bool,
) -> None:
    _fail_open(monkeypatch, fail_open)
    refusal = ModerationError(
        message="Execution failed due to input content moderation",
        user_id="user-1",
        graph_exec_id="exec-1",
        moderation_type="input",
    )

    async def refuse() -> Exception | None:
        return refusal

    async def approve() -> Exception | None:
        return None

    with pytest.raises(ModerationError) as refused:
        _await(refuse(), evaluation_loop)
    assert refused.value is refusal
    _await(approve(), evaluation_loop)
