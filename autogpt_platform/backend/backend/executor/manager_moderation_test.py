"""A run waits a bounded time on each AutoMod check. A check that runs out of
time is cancelled, and the run goes on without it only with automod_fail_open."""

import asyncio
import inspect
import threading
from collections.abc import Coroutine, Iterator
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

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


def _fail_open(
    monkeypatch: pytest.MonkeyPatch, fail_open: bool, *, enabled: bool = True
) -> None:
    config = MagicMock(enabled=enabled, fail_open=fail_open)
    monkeypatch.setattr(manager, "automod_manager", MagicMock(config=config))


def _check_that_never_answers() -> (
    tuple[Coroutine[Any, Any, Exception | None], threading.Event]
):
    cancelled = threading.Event()

    async def check() -> Exception | None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return None

    return check(), cancelled


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
    stage: Stage,
) -> None:
    _fail_open(monkeypatch, False)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    check, cancelled = _check_that_never_answers()

    with pytest.raises(ModerationError) as refused:
        _await(check, evaluation_loop, stage)

    assert refused.value.moderation_type == stage
    assert refused.value.graph_exec_id == "exec-1"
    assert cancelled.wait(timeout=10), "the timed-out check kept running"


def test_with_automod_off_a_run_never_waits_on_a_check(
    monkeypatch: pytest.MonkeyPatch, evaluation_loop: asyncio.AbstractEventLoop
) -> None:
    """Without AutoMod there is no moderation that could fail, so a busy
    evaluation loop must not hold the run up, let alone fail it."""
    _fail_open(monkeypatch, False, enabled=False)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    check, cancelled = _check_that_never_answers()

    _await(check, evaluation_loop)

    assert inspect.getcoroutinestate(check) == inspect.CORO_CLOSED
    assert not cancelled.is_set(), "the check ran"


def test_with_fail_open_a_check_that_runs_out_of_time_is_skipped(
    monkeypatch: pytest.MonkeyPatch, evaluation_loop: asyncio.AbstractEventLoop
) -> None:
    _fail_open(monkeypatch, True)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    check, cancelled = _check_that_never_answers()

    _await(check, evaluation_loop)

    assert cancelled.wait(timeout=10), "the timed-out check kept running"


def _automod_that_never_answers(
    monkeypatch: pytest.MonkeyPatch, *, fail_open: bool
) -> tuple[AutoModManager, threading.Event]:
    """The real AutoMod manager, switched on, whose request never comes back."""
    request_cancelled = threading.Event()

    async def never_answers(*args: object, **kwargs: object) -> object:
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
    return automod, request_cancelled


@pytest.mark.parametrize("fail_open", [False, True], ids=["fail closed", "fail open"])
def test_when_automod_never_answers_the_run_follows_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_loop: asyncio.AbstractEventLoop,
    fail_open: bool,
) -> None:
    """End to end through the real AutoMod manager: its request hangs, so the
    run stops waiting, cancels the request and fails, unless
    automod_fail_open lets it go on."""
    automod, request_cancelled = _automod_that_never_answers(
        monkeypatch, fail_open=fail_open
    )
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 0.2)
    db_client = MagicMock()
    db_client.get_graph = AsyncMock(
        return_value=MagicMock(
            nodes=[MagicMock(id="node-1", input_default={"prompt": "hello"})]
        )
    )
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
    else:
        with pytest.raises(ModerationError):
            _await(moderation, evaluation_loop)

    assert request_cancelled.wait(timeout=10), "the AutoMod request kept running"


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
