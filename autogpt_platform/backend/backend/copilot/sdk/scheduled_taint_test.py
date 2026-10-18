"""A follow-up fires with the taint of the turn that scheduled it.

The chat it was set up in was started on the web, so only the schedule itself
can carry the scheduling turn's taint to the turn that fires later, in that
chat or in a fresh one. Drives the real ``schedule_followup`` tool inside a
turn, the scheduler's ``add_copilot_turn_schedule`` over an in-memory jobstore,
the fire from the job's stored kwargs (``_execute_copilot_turn``, outside any
turn), the dispatch to the queue, the processor and
``stream_chat_completion_sdk`` up to the CLI query, with the queue, the CLI
client, the DB and Redis stubbed.
"""

import asyncio
import contextlib
import contextvars
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from backend.copilot.context import get_current_envelope, set_execution_context
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk import tool_adapter
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.continued_turn_query_test import _client
from backend.copilot.sdk.pickup_taint_test import _queued
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.sdk.voice_turn_test import _engine_call
from backend.copilot.tools.schedule_followup import (
    ScheduleCreatedResponse,
    ScheduleFollowupTool,
)
from backend.copilot.tree import TurnEnvelope, root_envelope
from backend.executor.scheduler import Jobstores, Scheduler, _execute_copilot_turn

_SVC = "backend.copilot.sdk.service"
_SCHEDULER = "backend.executor.scheduler"
_TOOL = "backend.copilot.tools.schedule_followup"
_USER_ID = "test-user"
_FOLLOWUP = "Check whether the deploy finished"
_HISTORY = [
    ("user", "Ship the release"),
    ("assistant", "Deploy started; I'll check on it later."),
]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("same_chat", [True, False], ids=["this-chat", "fresh-chat"])
@pytest.mark.parametrize(
    "tainted", [True, False], ids=["tainted-turn", "untainted-turn"]
)
async def test_a_followup_fires_with_the_taint_of_the_turn_that_scheduled_it(
    tmp_path: Path, same_chat: bool, tainted: bool
):
    chat = _web_chat()
    fired_into = chat if same_chat else _fresh_chat()

    job = await _schedule(chat, tainted=tainted, same_chat=same_chat)
    entry = await _queued(fired_into, lambda _s: _fire(job, chat, fired_into))
    ran_with = await _run_to_query(
        tmp_path,
        fired_into,
        await _engine_call(entry, fired_into),
        history=_HISTORY if same_chat else [],
    )

    assert job["tainted"] is tainted
    assert entry.envelope is not None and entry.envelope.tainted is tainted
    assert ran_with is not None and ran_with.tainted is tainted


def _web_chat() -> ChatSession:
    chat = ChatSession(
        session_id="web-chat-id",
        user_id=_USER_ID,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role=role, content=content, sequence=sequence)
            for sequence, (role, content) in enumerate(_HISTORY)
        ],
    )
    assert chat.metadata.source_platform is None
    return chat


def _fresh_chat() -> ChatSession:
    chat = ChatSession.new(_USER_ID, dry_run=False)
    chat.session_id = "fresh-chat-id"
    chat.metadata.origin = "automation"
    return chat


async def _schedule(
    chat: ChatSession, *, tainted: bool, same_chat: bool
) -> dict[str, Any]:
    """The tool call, inside a turn whose envelope is *tainted* or not; returns
    the kwargs the scheduler stored on the job it made."""
    service = Scheduler(register_system_tasks=False)
    service.scheduler = BackgroundScheduler(
        jobstores={Jobstores.EXECUTION.value: MemoryJobStore()}
    )
    service.scheduler.start(paused=True)

    async def add(**kwargs):
        return service.add_copilot_turn_schedule(**kwargs)

    users = MagicMock(get_user_by_id=AsyncMock(return_value=MagicMock(timezone="UTC")))
    turn = contextvars.copy_context()
    turn.run(
        set_execution_context,
        _USER_ID,
        chat,
        envelope=root_envelope("scheduling-turn", tainted=tainted),
    )
    try:
        with (
            patch(
                f"{_TOOL}.is_followups_feature_enabled",
                new=AsyncMock(return_value=True),
            ),
            patch(f"{_TOOL}.user_db", return_value=users),
            patch(
                f"{_TOOL}.get_scheduler_client",
                return_value=MagicMock(add_copilot_turn_schedule=add),
            ),
            patch("backend.executor.schedule_events.record_schedule_created"),
        ):
            result = await asyncio.get_running_loop().create_task(
                ScheduleFollowupTool()._execute(
                    user_id=_USER_ID,
                    session=chat,
                    message=_FOLLOWUP,
                    delay_seconds=1200,
                    **({"session_id": chat.session_id} if same_chat else {}),
                ),
                context=turn,
            )
        assert isinstance(result, ScheduleCreatedResponse)
        [job] = service.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
        return dict(job.kwargs)
    finally:
        service.scheduler.shutdown(wait=False)


async def _fire(job: dict[str, Any], chat: ChatSession, fresh: ChatSession) -> None:
    # The scheduler fires outside any turn: nothing is inherited from one.
    assert get_current_envelope() is None
    with (
        patch(f"{_SCHEDULER}.get_chat_session", new=AsyncMock(return_value=chat)),
        patch(f"{_SCHEDULER}.create_chat_session", new=AsyncMock(return_value=fresh)),
        patch(
            f"{_SCHEDULER}.resolve_default_chat_route",
            new=AsyncMock(return_value=("platform", None)),
        ),
        patch(
            f"{_SCHEDULER}._refuse_if_turn_may_not_start",
            new=AsyncMock(return_value=False),
        ),
        patch(f"{_SCHEDULER}.product_analytics"),
    ):
        await _execute_copilot_turn(**job)


async def _run_to_query(
    tmp_path: Path,
    session: ChatSession,
    kwargs: dict[str, Any],
    *,
    history: list[tuple[str, str]],
) -> TurnEnvelope | None:
    """The real engine, from the processor's call up to the CLI query; returns
    the envelope the turn's tools would read when the query went out."""
    seen: list[TurnEnvelope | None] = []

    async def query(*_args, **_kwargs) -> None:
        seen.append(get_current_envelope())

    client = _client()
    client.query = AsyncMock(side_effect=query)
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    transcript = build_test_transcript(history) if history else ""
    patches = _make_sdk_patches(session, transcript, None, lambda *a, **k: cm) + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (f"{_SVC}.build_turn_budget_block", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.build_skills_update_notice", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
        (f"{_SVC}._first_turn_context_blocks", dict(new=AsyncMock(return_value=[]))),
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
