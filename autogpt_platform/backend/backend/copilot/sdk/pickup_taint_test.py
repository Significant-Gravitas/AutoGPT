"""Keep going and the resume at the budget reset run with the paused turn's
taint while the API, the scheduler and the executor run different versions.

During a rolling deploy either side can be the older one:

- an executor from before the field writes its pause marker without
  ``turn_paused_tainted``: the route and the scheduler read that as tainted;
- an API or a scheduler from before the field dispatches the pickup without
  the taint: the processor reads it again off the chat's latest pause.

Each case drives the real dispatch (``continue_session`` and
``_dispatch_at_reset_turn`` now, ``schedule_chat_turn`` and ``schedule_turn``
with the arguments an older caller passes), the queue entry, the processor and
``stream_chat_completion_sdk`` up to the CLI query, with the queue, the CLI
client, the DB and Redis stubbed. A send at the reset is the user's own words
and stays untainted either way.
"""

import contextlib
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.chat import continuation as continuation_routes
from backend.copilot.at_reset import AtResetKind
from backend.copilot.constants import COPILOT_SYSTEM_PREFIX
from backend.copilot.context import get_current_envelope
from backend.copilot.continuation import KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE
from backend.copilot.db import PaginatedMessages
from backend.copilot.executor import utils as executor_utils
from backend.copilot.executor.utils import CoPilotExecutionEntry, ScheduledTurnOrigin
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk import tool_adapter
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.continued_turn_query_test import _client
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.sdk.voice_turn_test import _engine_call
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.tree import TurnEnvelope
from backend.copilot.trial_cost_context import TrialCostContext
from backend.copilot.turn_pause import PAUSE_KEY, TurnPause, append_pause_marker
from backend.executor import scheduler
from backend.executor.scheduler import CopilotTurnJobArgs

_SVC = "backend.copilot.sdk.service"
_SESSION_ID = "test-session-id"
_USER_ID = "test-user"
_SEND = "also check the invoices"
_PAUSE = TurnPause(reason="daily", resets_at=datetime(2026, 10, 16, tzinfo=UTC))

Dispatch = Callable[[ChatSession], Awaitable[None]]


@pytest.fixture(autouse=True)
def ready_usage_state(mocker):
    """Nothing here is on a trial."""
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


# What dispatches the pickup: this version, and one from before the field.
async def _keep_going(session: ChatSession) -> None:
    with (
        patch.object(
            continuation_routes,
            "_validate_and_get_writable_session",
            new=AsyncMock(return_value=session),
        ),
        patch.object(continuation_routes, "enforce_model_entitlement", new=AsyncMock()),
        patch.object(continuation_routes, "_admit", new=AsyncMock()),
        patch.object(continuation_routes, "_refuse_if_busy", new=AsyncMock()),
        patch.object(
            continuation_routes,
            "get_chat_session",
            new=AsyncMock(return_value=session),
        ),
        patch.object(continuation_routes, "cancel_resume_at_reset_in_background"),
        patch.object(continuation_routes, "_announce", new=AsyncMock()),
    ):
        await continuation_routes.continue_session(
            _SESSION_ID,
            MagicMock(headers={}),
            request=None,
            user_id=_USER_ID,
            ctx=MagicMock(org_id=None, team_id=None),
        )


async def _keep_going_from_an_older_api(session: ChatSession) -> None:
    await executor_utils.schedule_chat_turn(
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        message=KEEP_GOING_NOTE,
        require_fresh_slot=True,
        dedupe_content=False,
        is_user_message=False,
        session_origin=session.metadata.origin,
        llm_auth_provider=session.metadata.llm_auth_provider,
        llm_credential_id=session.metadata.llm_credential_id,
        permissions=resolve_session_permissions(session),
    )


async def _at_reset(session: ChatSession, kind: AtResetKind) -> None:
    args = CopilotTurnJobArgs(
        user_id=_USER_ID,
        session_id=_SESSION_ID,
        message=_SEND if kind == "send" else "",
        purpose="send" if kind == "send" else "resume",
    )
    chat = PaginatedMessages(
        messages=list(session.messages),
        has_more=False,
        oldest_sequence=0,
        session=session,
    )
    await scheduler._dispatch_at_reset_turn(args, kind, chat)


async def _resume(session: ChatSession) -> None:
    await _at_reset(session, "resume")


async def _send(session: ChatSession) -> None:
    await _at_reset(session, "send")


async def _from_an_older_scheduler(
    session: ChatSession, *, message: str, is_user_message: bool
) -> None:
    await executor_utils.schedule_turn(
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        turn_id=str(uuid.uuid4()),
        message=message,
        is_user_message=is_user_message,
        tool_call_id="scheduled_followup",
        tool_name="schedule_followup",
        unattended=True,
        llm_auth_provider=session.metadata.llm_auth_provider,
        llm_credential_id=session.metadata.llm_credential_id,
        permissions=resolve_session_permissions(session),
        scheduled=ScheduledTurnOrigin(schedule_id="resume-at-reset"),
    )


async def _resume_from_an_older_scheduler(session: ChatSession) -> None:
    await _from_an_older_scheduler(
        session, message=RESUMED_AT_RESET_NOTE, is_user_message=False
    )


async def _send_from_an_older_scheduler(session: ChatSession) -> None:
    await _from_an_older_scheduler(session, message=_SEND, is_user_message=True)


# The paused turn's marker: this version's, and one from before the field.
def _marker(tainted: bool) -> ChatMessage:
    holder = _session([])
    append_pause_marker(holder, _PAUSE, tainted=tainted)
    return holder.messages[-1]


def _marker_from_an_older_executor() -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=f"{COPILOT_SYSTEM_PREFIX} {_PAUSE.summary}",
        metadata={PAUSE_KEY: _PAUSE.as_part()},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatch",
    [_keep_going, _resume],
    ids=["keep-going", "reset-resume"],
)
async def test_a_pickup_of_a_marker_from_an_older_executor_runs_tainted(
    tmp_path: Path, dispatch: Dispatch
):
    session = _session([_marker_from_an_older_executor()])

    entry, ran_with = await _pick_up(tmp_path, session, dispatch)

    assert entry.envelope is not None and entry.envelope.tainted
    assert ran_with is not None and ran_with.tainted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatch",
    [_keep_going_from_an_older_api, _resume_from_an_older_scheduler],
    ids=["keep-going", "reset-resume"],
)
@pytest.mark.parametrize(
    "tainted", [True, False], ids=["paused-tainted", "paused-untainted"]
)
async def test_a_pickup_an_older_caller_queued_runs_with_the_paused_turns_taint(
    tmp_path: Path, dispatch: Dispatch, tainted: bool
):
    session = _session([_marker(tainted)])

    entry, ran_with = await _pick_up(tmp_path, session, dispatch)

    assert entry.envelope is not None and not entry.envelope.tainted
    assert ran_with is not None and ran_with.tainted is tainted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatch",
    [_keep_going, _resume],
    ids=["keep-going", "reset-resume"],
)
@pytest.mark.parametrize(
    "tainted", [True, False], ids=["paused-tainted", "paused-untainted"]
)
async def test_a_pickup_runs_with_the_paused_turns_taint(
    tmp_path: Path, dispatch: Dispatch, tainted: bool
):
    session = _session([_marker(tainted)])

    entry, ran_with = await _pick_up(tmp_path, session, dispatch)

    assert entry.envelope is not None and entry.envelope.tainted is tainted
    assert ran_with is not None and ran_with.tainted is tainted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatch",
    [_send, _send_from_an_older_scheduler],
    ids=["this-version", "older-scheduler"],
)
@pytest.mark.parametrize(
    "marker",
    [lambda: _marker(True), _marker_from_an_older_executor],
    ids=["tainted-marker", "older-marker"],
)
async def test_a_send_at_the_reset_runs_untainted(
    tmp_path: Path, dispatch: Dispatch, marker: Callable[[], ChatMessage]
):
    session = _session([marker()])

    entry, ran_with = await _pick_up(tmp_path, session, dispatch)

    assert entry.message == _SEND
    assert ran_with is not None and not ran_with.tainted


def _session(tail: list[ChatMessage]) -> ChatSession:
    """A web chat whose last turn paused: rows the paused turn saved, then
    *tail* (its marker)."""
    session = ChatSession(
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="Build the report"),
            ChatMessage(role="assistant", content="Two of three sections done."),
            *tail,
        ],
    )
    for sequence, message in enumerate(session.messages):
        message.sequence = sequence
    return session


async def _pick_up(
    tmp_path: Path, session: ChatSession, dispatch: Dispatch
) -> tuple[CoPilotExecutionEntry, TurnEnvelope | None]:
    """Dispatch the pickup, run what it queued through the processor and the
    engine; returns the queue entry and the envelope the CLI query ran under."""
    entry = await _queued(session, dispatch)
    kwargs = await _engine_call(entry, session)
    return entry, await _run_engine(tmp_path, session, kwargs)


async def _queued(session: ChatSession, dispatch: Dispatch) -> CoPilotExecutionEntry:
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    async def save(
        _session_id: str, message: ChatMessage, *, dedupe_content: bool = True
    ) -> ChatSession | None:
        message.sequence = len(session.messages)
        session.messages.append(message)
        return session

    queue = MagicMock(publish_message=AsyncMock())
    with (
        patch.object(executor_utils, "acquire_turn_slot", new=acquire),
        patch.object(executor_utils, "check_inflight_cap", new=AsyncMock()),
        patch.object(executor_utils, "admit_turn", new=AsyncMock()),
        patch.object(
            executor_utils,
            "capture_cost_context",
            new=AsyncMock(
                return_value=TrialCostContext(user_id=_USER_ID, trial_id=None)
            ),
        ),
        patch("backend.copilot.model.append_and_save_message", new=save),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch("backend.copilot.stream_registry.create_session", new=AsyncMock()),
        patch(
            "backend.util.clients.get_async_copilot_queue",
            new=AsyncMock(return_value=queue),
        ),
    ):
        await dispatch(session)
    return CoPilotExecutionEntry.model_validate_json(
        queue.publish_message.await_args.kwargs["message"]
    )


async def _run_engine(
    tmp_path: Path, session: ChatSession, kwargs: dict[str, Any]
) -> TurnEnvelope | None:
    """The real engine, from the processor's call up to the CLI query; returns
    the envelope the turn's tools would read when the query went out."""
    seen: list[TurnEnvelope | None] = []

    async def query(*_args, **_kwargs) -> None:
        seen.append(get_current_envelope())

    client = _client()
    client.query = AsyncMock(side_effect=query)
    transcript = build_test_transcript(
        [("user", "Build the report"), ("assistant", "Two of three sections done.")]
    )
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
        (f"{_SVC}.persist_session_safe", dict(side_effect=_persisted)),
        # The shared patches stub this out; the real one is what puts the
        # turn's envelope where its tools read it.
        (f"{_SVC}.set_execution_context", dict(new=tool_adapter.set_execution_context)),
    ]
    with contextlib.ExitStack() as stack:
        for target, patch_kwargs in patches:
            stack.enter_context(patch(target, **patch_kwargs))
        async for _ in stream_chat_completion_sdk(**kwargs):
            pass
    assert seen, "the turn never reached the CLI query"
    return seen[0]


async def _persisted(session: ChatSession, _log_prefix: str = "") -> ChatSession:
    session.messages[-1].sequence = len(session.messages) - 1
    return session
