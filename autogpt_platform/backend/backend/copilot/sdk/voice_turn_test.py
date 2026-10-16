"""A voice turn from the chat route to the CLI query: one stored user row, and
the voice instruction in what the model reads, once, after the user's words.

The inbound sanitizer strips ``<voice_turn>``. An instruction written into the
message before dispatch therefore never reached the model, and the engine's
duplicate check, comparing the stripped message with the saved row, missed and
stored the turn a second time. Drives the real ``schedule_chat_turn``, queue
entry, processor and ``stream_chat_completion_sdk`` up to the CLI query, with
the queue, the CLI client, the DB and Redis stubbed.
"""

import contextlib
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.executor import utils as executor_utils
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.processor_test import _make_log, _TrackedStream
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.prompting import VOICE_TURN_INSTRUCTION
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.continued_turn_query_test import _client
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk

_SVC = "backend.copilot.sdk.service"
_SESSION_ID = "test-session-id"
_USER_ID = "test-user"
_SPOKEN = "what did I run yesterday"


@pytest.fixture(autouse=True)
def ready_usage_state(mocker):
    """Dispatch captures the turn's trial attribution; nothing is on a trial."""
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


@pytest.mark.asyncio
async def test_a_voice_turn_stores_one_row_and_tells_the_model_once(
    tmp_path: Path,
):
    session = _session()
    client = _client()

    entry = await _dispatch(session)
    kwargs = await _engine_call(entry, session)
    await _run_engine(tmp_path, client, session, kwargs)

    assert [m.content for m in session.messages if m.role == "user"] == [
        "earlier",
        _SPOKEN,
    ]
    [query] = [call.args[0] for call in client.query.await_args_list]
    assert query.count(VOICE_TURN_INSTRUCTION) == 1
    assert query.rindex(_SPOKEN) < query.index(VOICE_TURN_INSTRUCTION)


@pytest.mark.asyncio
async def test_a_typed_turn_carries_no_voice_instruction(tmp_path: Path):
    session = _session()
    client = _client()

    entry = await _dispatch(session, voice=False)
    kwargs = await _engine_call(entry, session)
    await _run_engine(tmp_path, client, session, kwargs)

    [query] = [call.args[0] for call in client.query.await_args_list]
    assert "voice_turn" not in query


def _session() -> ChatSession:
    return ChatSession(
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="earlier", sequence=0),
            ChatMessage(role="assistant", content="ok", sequence=1),
        ],
    )


async def _dispatch(
    session: ChatSession, *, voice: bool = True
) -> CoPilotExecutionEntry:
    """The chat route's dispatch: save the row, then queue the turn."""
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    async def save(
        _session_id: str, message: ChatMessage, *, dedupe_content: bool = True
    ) -> ChatSession | None:
        # The chat route's dispatch dedupes a message against the last row.
        assert dedupe_content
        message.sequence = len(session.messages)
        session.messages.append(message)
        return session

    queue = MagicMock(publish_message=AsyncMock())
    with (
        patch.object(executor_utils, "acquire_turn_slot", new=acquire),
        patch.object(executor_utils, "admit_turn", new=AsyncMock()),
        patch("backend.copilot.model.append_and_save_message", new=save),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch("backend.copilot.stream_registry.create_session", new=AsyncMock()),
        patch(
            "backend.util.clients.get_async_copilot_queue",
            new=AsyncMock(return_value=queue),
        ),
    ):
        await executor_utils.schedule_chat_turn(
            session_id=_SESSION_ID, user_id=_USER_ID, message=_SPOKEN, voice=voice
        )
    return CoPilotExecutionEntry.model_validate_json(
        queue.publish_message.await_args.kwargs["message"]
    )


async def _engine_call(
    entry: CoPilotExecutionEntry, session: ChatSession
) -> dict[str, Any]:
    """What the processor hands the SDK engine for *entry*."""
    seen: dict[str, Any] = {}

    def engine(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    registry = "backend.copilot.executor.processor.stream_registry"
    with (
        patch(
            "backend.copilot.executor.processor.ChatConfig",
            return_value=ChatConfig(platform_route="openrouter", test_mode=False),
        ),
        patch(
            "backend.copilot.executor.processor.sdk_service.stream_chat_completion_sdk",
            side_effect=engine,
        ),
        patch(f"{registry}.stream_and_publish", return_value=_TrackedStream(events=[])),
        patch(f"{registry}.publish_chunk", new=AsyncMock()),
        patch(f"{registry}.mark_session_completed", new=AsyncMock()),
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=session),
        ),
        # The pickup budget gate has its own tests (processor_budget_gate_test).
        patch(
            "backend.copilot.executor.processor.refusal_for",
            new=AsyncMock(return_value=None),
        ),
    ):
        await CoPilotProcessor()._execute_async(
            entry, threading.Event(), MagicMock(), _make_log()
        )
    return seen


async def _run_engine(
    tmp_path: Path, client: MagicMock, session: ChatSession, kwargs: dict[str, Any]
) -> None:
    """The real engine, from the processor's call up to the CLI query."""
    transcript = build_test_transcript([("user", "earlier"), ("assistant", "ok")])
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    patches = _make_sdk_patches(session, transcript, None, lambda *a, **k: cm) + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (f"{_SVC}.build_turn_budget_block", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.build_skills_update_notice", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
        (f"{_SVC}._rebuild_first_turn_context", dict(new=AsyncMock(return_value=[]))),
        # Only a second user row, the duplicate this file guards, reaches it.
        (f"{_SVC}.persist_session_safe", dict(side_effect=_persisted)),
    ]
    with contextlib.ExitStack() as stack:
        for target, patch_kwargs in patches:
            stack.enter_context(patch(target, **patch_kwargs))
        async for _ in stream_chat_completion_sdk(**kwargs):
            pass


async def _persisted(session: ChatSession, _log_prefix: str = "") -> ChatSession:
    """Save an appended row as the DB would: it gets the next sequence."""
    session.messages[-1].sequence = len(session.messages) - 1
    return session
