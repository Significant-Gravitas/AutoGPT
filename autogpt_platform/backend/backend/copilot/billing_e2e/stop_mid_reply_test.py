"""A message sent right after a Stop that reached a turn mid-reply is answered.

The worker's own entry point, ``CoPilotProcessor.execute``, runs SDK turns on
the bundled CLI against a fake provider that streams the first reply slowly,
on the processor's own loop thread. The cancel consumer's event is set while
that CLI is mid-reply, and the next turn starts as soon as the stopped one
ends. A CLI that outlives its turn refuses the next turn's session id
("Session ID ... is already in use"); one that waits out the SDK's close grace
holds the turn past the cancel route's confirmation window.
"""

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest
from aiohttp import web

from backend.api.features.chat.routes import _CANCEL_CONFIRM_TIMEOUT_SECONDS
from backend.copilot import rate_limit, usage_activation
from backend.copilot.config import ChatConfig
from backend.copilot.executor import processor
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.rate_limit import SubscriptionTier
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.cli_environment_e2e_test import _config, _require_cli
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.copilot.tree import root_envelope
from backend.copilot.turn_budget import TurnBudget
from backend.data import db_accessors

from .executor_turns import Executor, SdkTurnsViaExecutor
from .invoice import Route
from .ledger import PlatformLedger
from .provider import FakeProvider
from .trial_account import usage_state_of


@pytest.fixture(autouse=True)
def _paying_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user on a paid plan with no experts and memory off, none of it under
    test, and no background work a cancelled turn would stop."""
    store = MagicMock(get_subscription_trial=AsyncMock(return_value=None))
    monkeypatch.setattr(db_accessors, "credit_db", lambda: store)
    usage = MagicMock(
        get_usage_activation_state=AsyncMock(
            side_effect=lambda user_id: usage_state_of(
                user_id, SubscriptionTier.PRO, None
            )
        )
    )
    monkeypatch.setattr(usage_activation, "pro_activation_db", lambda: usage)
    monkeypatch.setattr(
        rate_limit,
        "get_redis_async",
        AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=None))),
    )
    monkeypatch.setattr(sdk_service, "_graphiti_enabled", lambda _user_id: False)
    monkeypatch.setattr(
        sdk_service, "_fetch_graphiti_context", AsyncMock(return_value=(False, ""))
    )
    monkeypatch.setattr(sdk_service, "_enqueue_graphiti_turn", AsyncMock())
    monkeypatch.setattr(sdk_service, "capture_chat_turn", AsyncMock())
    monkeypatch.setattr(
        sdk_service,
        "resolve_turn_budget",
        AsyncMock(return_value=TurnBudget(usd=100.0, limiting="task")),
    )
    monkeypatch.setattr(
        sdk_service, "first_message_expert_context", AsyncMock(return_value="")
    )
    monkeypatch.setattr(processor, "finish_background_execution", AsyncMock())


@pytest.fixture
def execution_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.mark.asyncio(loop_scope="function")
async def test_a_message_sent_right_after_a_stop_mid_reply_is_answered(
    ledger: PlatformLedger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_loop: asyncio.AbstractEventLoop,
):
    _require_cli()
    session = ChatSession.new("stop-mid-reply-e2e-user", dry_run=False)
    session.title = "Stop mid-reply"
    cwd = tmp_path / "copilot-cwd"
    async with _SlowFirstReply() as provider:
        executor = _StoppedExecutor(monkeypatch, _config(), session, execution_loop)
        turns = SdkTurnsViaExecutor(
            tmp_path, monkeypatch, provider, _config(), executor
        )
        try:
            executor.stop_when = provider.replying
            await _send(turns, session, "count to a thousand")
            left_running = _clis_in(cwd)
            executor.stop_when = None
            await _send(turns, session, "never mind, just say hi")
        finally:
            for process in _clis_in(cwd):
                process.kill()

    assert executor.stopped_mid_reply, "the CLI never started its reply"
    assert left_running == [], "the stopped turn ended with its CLI still running"
    # Within the window the cancel route waits, after which it completes the
    # turn itself and admits the next one.
    assert executor.completed_after is not None
    assert executor.completed_after < _CANCEL_CONFIRM_TIMEOUT_SECONDS
    # The next turn resumes the CLI session the Stop interrupted, and is answered.
    assert turns.resumed[-1] == (1, session.session_id)
    assert list(executor.outcomes.values()) == [CANCELLED_MESSAGE, None]


async def _send(turns: SdkTurnsViaExecutor, session: ChatSession, text: str) -> None:
    session.messages.append(ChatMessage(role="user", content=text))
    await asyncio.wait_for(turns(session, text), timeout=60)


class _StoppedExecutor(Executor):
    """The executor harness, running each turn through the worker's entry point
    and setting the turn's cancel event once ``stop_when`` is set."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ChatConfig,
        session: ChatSession,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(monkeypatch, config, session, use_sdk=True)
        self._processor.execution_loop = loop
        self.stop_when: asyncio.Event | None = None
        self.stopped_mid_reply = False
        self.completed_after: float | None = None
        # Each turn's outcome: the error it ended with, None if answered.
        self.outcomes: dict[str, str | None] = {}
        self._stopped_at: float | None = None

    async def run(self, session: ChatSession, prompt: str) -> None:
        entry = CoPilotExecutionEntry(
            session_id=session.session_id,
            turn_id=str(uuid.uuid4()),
            user_id=session.user_id,
            message=prompt,
            envelope=root_envelope(str(uuid.uuid4()), session_id=session.session_id),
        )
        cancel = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(self._processor.execute, entry, cancel, MagicMock())
        )
        stop = asyncio.ensure_future((self.stop_when or asyncio.Event()).wait())
        try:
            await asyncio.wait({worker, stop}, return_when=asyncio.FIRST_COMPLETED)
            if not worker.done():
                self.stopped_mid_reply = True
                self._stopped_at = time.monotonic()
        finally:
            # Set however this ends, so the worker thread never outlives the test.
            stop.cancel()
            cancel.set()
            await worker

    async def _completed(self, _session_id: str, **kwargs) -> None:
        if self._stopped_at is not None and self.completed_after is None:
            self.completed_after = time.monotonic() - self._stopped_at
        # The store keeps a turn's first outcome; the worker's fail-close after
        # it is a no-op there.
        self.outcomes.setdefault(kwargs["turn_id"], kwargs.get("error_message"))


class _SlowFirstReply(FakeProvider):
    """The fake provider, streaming its first reply a word at a time for a
    minute and every later one as usual."""

    def __init__(self) -> None:
        super().__init__()
        self.replying = asyncio.Event()

    async def _messages(self, request: web.Request, route: Route) -> web.StreamResponse:
        if self.replying.is_set():
            return await super()._messages(request, route)
        body = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            await response.write(_sse(_message_start(body["model"])))
            await response.write(
                _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            )
            for word in range(1200):
                delta = {"type": "text_delta", "text": f"{word} "}
                await response.write(
                    _sse({"type": "content_block_delta", "index": 0, "delta": delta})
                )
                self.replying.set()
                await asyncio.sleep(0.05)
        except ConnectionResetError:
            pass
        return response


def _message_start(model: str) -> dict:
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    }


def _sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _clis_in(cwd: Path) -> list[psutil.Process]:
    """The Claude Code CLIs still running in *cwd*: the SDK starts each one
    with ``stream-json`` I/O."""
    running = []
    for process in psutil.Process().children(recursive=True):
        try:
            if (
                "stream-json" in process.cmdline()
                and Path(process.cwd()) == cwd
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                running.append(process)
        except psutil.Error:
            continue
    return running
