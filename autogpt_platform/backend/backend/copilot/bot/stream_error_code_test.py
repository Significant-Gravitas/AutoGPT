"""An engine's error code reaches the bot through the executor and Redis.

The bot names why a turn failed from ``StreamError.code``. Every other test
of that hands the bot a built chunk; this one runs the real executor, the
real publish path and a real Redis stream between them.
"""

import logging
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import stream_registry
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamError
from backend.data import redis_client
from backend.platform_linking.models import ChatTurnHandle
from backend.util.testing import is_tcp_port_reachable

from .bot_backend import BotBackend, BotStreamError
from .failures import PROVIDER_LIMIT, categorize

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)


@pytest.fixture
async def turn():
    session_id, turn_id = f"bot-code-{uuid.uuid4().hex}", str(uuid.uuid4())
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    yield session_id, turn_id
    redis = await redis_client.get_redis_async()
    await redis.delete(stream_registry.get_session_meta_key(session_id))
    await redis.delete(stream_registry._get_turn_stream_key(turn_id))
    await redis.delete(stream_registry._get_turn_meta_key(turn_id))


@pytest.fixture(autouse=True)
def allowance_admits_every_turn(mocker):
    """The pickup budget gate has its own tests (processor_budget_gate_test)."""
    mocker.patch(
        "backend.copilot.executor.processor.refusal_for",
        AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def no_trial_attribution(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


async def _engine_failing_with(code: str):
    yield StreamError(errorText="You've hit your usage limit.", code=code)


async def _run_turn(session_id: str, turn_id: str, code: str) -> None:
    entry = CoPilotExecutionEntry(
        session_id=session_id,
        turn_id=turn_id,
        user_id="user-1",
        message="hi",
        is_user_message=True,
        request_arrival_at=0.0,
    )
    with (
        patch(
            "backend.copilot.executor.processor.ChatConfig",
            return_value=MagicMock(test_mode=True),
        ),
        patch(
            "backend.copilot.executor.processor.stream_chat_completion_dummy",
            side_effect=lambda **_: _engine_failing_with(code),
        ),
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=ChatSession.new("user-1", dry_run=False)),
        ),
    ):
        await CoPilotProcessor()._execute_async(
            entry,
            threading.Event(),
            MagicMock(),
            CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
        )


async def test_the_bot_names_the_failure_the_engine_reported(turn):
    session_id, turn_id = turn
    await _run_turn(session_id, turn_id, "usage_limit")

    with patch("backend.copilot.bot.bot_backend.get_platform_linking_manager_client"):
        bot = BotBackend()
    handle = ChatTurnHandle(session_id=session_id, turn_id=turn_id, user_id="user-1")
    with pytest.raises(BotStreamError) as raised:
        async for _ in bot.stream_turn(handle):
            pass

    assert raised.value.code == "usage_limit"
    assert categorize(raised.value) is PROVIDER_LIMIT
