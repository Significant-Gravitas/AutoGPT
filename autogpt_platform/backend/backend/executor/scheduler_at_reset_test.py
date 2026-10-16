"""The scheduler's side of resume and send at the budget reset: the job's
id and shape, and what a due one does (``copilot/at_reset.py`` decides)."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.at_reset import at_reset_schedule_id, reset_jitter_seconds
from backend.copilot.continuation import RESUMED_AT_RESET_NOTE
from backend.copilot.db import PaginatedMessages
from backend.copilot.executor import utils as executor_utils
from backend.copilot.executor.utils import ScheduledTurnOrigin
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
    ChatSessionMetadata,
)
from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    RateLimitExceeded,
    SubscriptionTier,
    UsageWindow,
)
from backend.copilot.trial_cost_context import TrialCostContext
from backend.copilot.turn_pause import PAUSE_KEY, TurnPause, append_pause_marker
from backend.executor.scheduler import (
    _MAX_AT_RESET_RETRIES,
    CopilotTurnJobArgs,
    Jobstores,
    Scheduler,
    _execute_copilot_turn,
)

_SCHEDULER = "backend.executor.scheduler"
_AT_RESET = "backend.copilot.at_reset"
USER = "user-1"
SESSION = "0d3f7a52-61c4-4f0e-a8b2-9c1e5d7f3a64"
SET_AT = datetime(2026, 10, 14, 15, 0, tzinfo=UTC)
DAILY_RESET = datetime(2026, 10, 15, tzinfo=UTC)
WEEKLY_RESET = datetime(2026, 10, 19, tzinfo=UTC)


def _args(kind: str = "resume", **overrides) -> CopilotTurnJobArgs:
    payload = {
        "schedule_id": f"at-reset-{kind}-{SESSION}",
        "user_id": USER,
        "session_id": SESSION,
        "message": "also check the invoices" if kind == "send" else "Resume.",
        "run_at": SET_AT + timedelta(hours=9),
        "user_timezone": "UTC",
        "purpose": kind,
        "created_at": SET_AT,
        **overrides,
    }
    return CopilotTurnJobArgs.model_validate(payload)


def _chat(*rows: ChatMessage, expert_id: str | None = None) -> PaginatedMessages:
    session = ChatSessionInfo(
        session_id=SESSION,
        user_id=USER,
        usage=[],
        started_at=SET_AT - timedelta(hours=1),
        updated_at=SET_AT,
        expert_id=expert_id,
        metadata=ChatSessionMetadata(llm_auth_provider="platform"),
    )
    return PaginatedMessages(
        messages=list(rows), has_more=False, oldest_sequence=0, session=session
    )


async def _fire(args: CopilotTurnJobArgs) -> None:
    await _execute_copilot_turn(**args.model_dump(mode="json"))


@pytest.fixture(autouse=True)
def no_cancels():
    """Cancels are remembered in process: no test sees another's."""
    with patch.dict(f"{_SCHEDULER}._at_reset_cancels", clear=True):
        yield


@pytest.fixture
def fire_env():
    """A due at-reset turn on an idle, platform-route chat with budget left."""
    store = MagicMock(get_latest_user_message_in_session=AsyncMock(return_value=None))
    client = MagicMock(
        add_copilot_turn_schedule=AsyncMock(), delete_schedule=AsyncMock()
    )
    limits = (3_000_000, 6_000_000, SubscriptionTier.PRO)
    with (
        patch(
            f"{_SCHEDULER}.load_at_reset_chat", AsyncMock(return_value=_chat())
        ) as load,
        patch(f"{_SCHEDULER}.schedule_turn", AsyncMock()) as dispatch,
        patch(f"{_SCHEDULER}.get_scheduler_client", return_value=client) as factory,
        patch(f"{_AT_RESET}.chat_db", return_value=store),
        patch(f"{_AT_RESET}.is_turn_in_flight", AsyncMock(return_value=False)) as busy,
        patch(f"{_AT_RESET}.get_global_rate_limits", AsyncMock(return_value=limits)),
        patch(f"{_AT_RESET}.check_rate_limit", AsyncMock()) as gate,
        patch(f"{_AT_RESET}.get_usage_status", AsyncMock()) as usage,
    ):
        yield SimpleNamespace(
            load=load,
            dispatch=dispatch,
            client=client,
            factory=factory,
            busy=busy,
            gate=gate,
            usage=usage,
        )


@pytest.fixture
def live_scheduler():
    """The service over an in-memory jobstore, paused so nothing fires."""
    service = Scheduler(register_system_tasks=False)
    service.scheduler = BackgroundScheduler(
        jobstores={Jobstores.EXECUTION.value: MemoryJobStore()}
    )
    service.scheduler.start(paused=True)
    yield service
    service.scheduler.shutdown(wait=False)


def test_a_row_from_before_the_field_loads_as_a_followup():
    legacy = CopilotTurnJobArgs.model_validate(
        {"user_id": USER, "session_id": SESSION, "message": "check CI"}
    )
    assert legacy.purpose == "followup"
    assert legacy.created_at is None
    roundtrip = CopilotTurnJobArgs.model_validate(_args("send").model_dump(mode="json"))
    assert roundtrip.purpose == "send"
    assert roundtrip.created_at == SET_AT


def test_an_at_reset_job_is_named_for_what_it_does_and_not_recorded():
    service = Scheduler(register_system_tasks=False)
    job = MagicMock(id="job", next_run_time=None)
    job.name = "job"
    with (
        patch.object(service, "_persist_schedule", return_value=job) as persist,
        patch("backend.executor.schedule_events.record_schedule_created") as record,
    ):
        service.add_copilot_turn_schedule(
            user_id=USER,
            session_id=SESSION,
            message="Resume.",
            run_at=SET_AT,
            user_timezone="UTC",
            purpose="resume",
            schedule_id=at_reset_schedule_id("resume", SESSION),
        )
        record.assert_not_called()
        service.add_copilot_turn_schedule(
            user_id=USER, session_id=SESSION, message="check CI", run_at=SET_AT
        )
        record.assert_called_once()
    at_reset, followup = (c.kwargs for c in persist.call_args_list)
    assert at_reset["job_args"].schedule_id == f"at-reset-resume-{SESSION}"
    assert at_reset["job_args"].purpose == "resume"
    assert at_reset["job_args"].created_at is not None
    assert at_reset["name"] == "Resume when the budget resets"
    assert followup["job_args"].purpose == "followup"
    assert followup["job_args"].schedule_id != at_reset["job_args"].schedule_id


def test_setting_it_again_replaces_the_earlier_job(live_scheduler):
    job_id = at_reset_schedule_id("send", SESSION)
    first = datetime.now(UTC) + timedelta(hours=3)
    for run_at, message in ((first, "draft"), (first + timedelta(days=1), "final")):
        live_scheduler.add_copilot_turn_schedule(
            user_id=USER,
            session_id=SESSION,
            message=message,
            run_at=run_at,
            user_timezone="UTC",
            purpose="send",
            schedule_id=job_id,
        )
    jobs = live_scheduler.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value)
    assert [job.id for job in jobs] == [job_id]
    assert jobs[0].kwargs["message"] == "final"
    assert jobs[0].next_run_time == first + timedelta(days=1)


def test_cancel_leaves_the_jobs_schedule_index_row(live_scheduler):
    # A row with no job is skipped on read, while the chat's next turn at reset
    # reuses the id: deleting the row could hide a job set again meanwhile.
    live_scheduler._schedule_index = MagicMock()
    live_scheduler.add_copilot_turn_schedule(
        user_id=USER,
        session_id=SESSION,
        message="Resume.",
        run_at=datetime.now(UTC) + timedelta(hours=3),
        user_timezone="UTC",
        purpose="resume",
        schedule_id=at_reset_schedule_id("resume", SESSION),
    )

    assert live_scheduler.cancel_at_reset_turn(USER, SESSION, "resume") is True
    live_scheduler._schedule_index.delete_many.assert_not_called()


def test_cancel_removes_only_the_owners_job_and_is_idempotent(live_scheduler):
    live_scheduler.add_copilot_turn_schedule(
        user_id=USER,
        session_id=SESSION,
        message="Resume.",
        run_at=datetime.now(UTC) + timedelta(hours=3),
        user_timezone="UTC",
        purpose="resume",
        schedule_id=at_reset_schedule_id("resume", SESSION),
    )
    assert (
        live_scheduler.cancel_at_reset_turn("someone-else", SESSION, "resume") is False
    )
    assert live_scheduler.cancel_at_reset_turn(USER, SESSION, "send") is False
    assert live_scheduler.cancel_at_reset_turn(USER, SESSION, "resume") is True
    assert live_scheduler.scheduler.get_jobs(jobstore=Jobstores.EXECUTION.value) == []
    assert live_scheduler.cancel_at_reset_turn(USER, SESSION, "resume") is False


@pytest.mark.asyncio
async def test_a_resume_the_user_already_picked_up_is_dropped(fire_env):
    newer = ChatMessage(
        role="user", content="do this instead", created_at=SET_AT + timedelta(minutes=5)
    )
    fire_env.load.return_value = _chat(newer)
    await _fire(_args("resume"))
    fire_env.dispatch.assert_not_awaited()
    fire_env.client.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, message, as_user",
    [
        ("resume", RESUMED_AT_RESET_NOTE, False),
        ("send", "also check the invoices", True),
    ],
    ids=["resume-fires-the-note-not-as-the-user", "send-fires-the-users-words"],
)
async def test_it_fires_into_its_chat(fire_env, kind, message, as_user):
    await _fire(_args(kind))
    sent = fire_env.dispatch.await_args.kwargs
    assert (sent["message"], sent["is_user_message"]) == (message, as_user)
    assert sent["session_id"] == SESSION
    assert sent["unattended"] is True
    assert sent["llm_auth_provider"] == "platform"
    assert sent["scheduled"] == ScheduledTurnOrigin(
        schedule_id=f"at-reset-{kind}-{SESSION}"
    )


@pytest.mark.asyncio
async def test_a_busy_chat_puts_it_back_in_five_minutes_under_its_own_id(fire_env):
    fire_env.busy.return_value = True
    before = datetime.now(UTC)
    await _fire(_args("send", cap_retry_count=2))
    fire_env.dispatch.assert_not_awaited()
    readded = fire_env.client.add_copilot_turn_schedule.await_args.kwargs
    assert readded["schedule_id"] == f"at-reset-send-{SESSION}"
    assert readded["purpose"] == "send"
    assert readded["message"] == "also check the invoices"
    assert readded["created_at"] == SET_AT
    assert readded["cap_retry_count"] == 3
    assert timedelta(minutes=5) <= readded["run_at"] - before < timedelta(minutes=6)


@pytest.mark.asyncio
async def test_a_put_back_lost_in_transit_is_repeated(fire_env):
    # Dropped, it would lose the user's turn; under its own id it is safe to
    # repeat on a connection the scheduler had just closed.
    fire_env.busy.return_value = True
    await _fire(_args("send"))
    fire_env.client.add_copilot_turn_schedule.assert_awaited_once()
    fire_env.factory.assert_called_once_with(should_retry=True)


@pytest.mark.asyncio
async def test_retries_are_bounded(fire_env):
    fire_env.busy.return_value = True
    await _fire(_args("send", cap_retry_count=_MAX_AT_RESET_RETRIES))
    fire_env.client.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_still_out_of_budget_moves_it_to_the_next_reset(fire_env):
    fire_env.gate.side_effect = RateLimitExceeded("daily", DAILY_RESET)
    fire_env.usage.return_value = CoPilotUsageStatus(
        daily=UsageWindow(used=3_000_000, limit=3_000_000, resets_at=DAILY_RESET),
        weekly=UsageWindow(used=0, limit=6_000_000, resets_at=WEEKLY_RESET),
        tier=SubscriptionTier.PRO,
    )
    await _fire(_args("resume", cap_retry_count=4))
    fire_env.dispatch.assert_not_awaited()
    moved = fire_env.client.add_copilot_turn_schedule.await_args.kwargs
    jitter = timedelta(seconds=reset_jitter_seconds(SESSION))
    assert moved["run_at"] == DAILY_RESET + timedelta(seconds=180) + jitter
    assert moved["schedule_id"] == f"at-reset-resume-{SESSION}"
    assert moved["cap_retry_count"] == 0


@pytest.mark.asyncio
async def test_a_refused_dispatch_puts_it_back(fire_env):
    fire_env.dispatch.side_effect = ConcurrentTurnLimitError("cap")
    await _fire(_args("send"))
    readded = fire_env.client.add_copilot_turn_schedule.await_args.kwargs
    assert readded["purpose"] == "send"
    assert readded["cap_retry_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [None, _chat(expert_id="expert-2")])
async def test_a_gone_or_rescoped_chat_removes_the_job(fire_env, chat):
    fire_env.load.return_value = chat
    await _fire(_args("resume"))
    fire_env.dispatch.assert_not_awaited()
    fire_env.client.delete_schedule.assert_awaited_once_with(
        schedule_id=f"at-reset-resume-{SESSION}", user_id=USER
    )


@pytest.mark.asyncio
async def test_an_unreadable_expert_puts_it_back_as_itself(fire_env):
    fire_env.load.return_value = _chat(expert_id="expert-1")
    with patch(
        f"{_SCHEDULER}._expert_scope_status", AsyncMock(return_value="unavailable")
    ):
        await _fire(_args("resume", expert_id="expert-1"))
    readded = fire_env.client.add_copilot_turn_schedule.await_args.kwargs
    assert readded["purpose"] == "resume"
    assert readded["schedule_id"] == f"at-reset-resume-{SESSION}"
    assert readded["expert_id"] == "expert-1"


@pytest.mark.asyncio
async def test_followups_keep_their_own_path(fire_env):
    followup = CopilotTurnJobArgs(
        schedule_id="sched-1", user_id=USER, session_id=SESSION, message="check CI"
    )
    with patch(
        f"{_SCHEDULER}.get_chat_session",
        AsyncMock(return_value=MagicMock(expert_id=None)),
    ):
        await _fire(followup)
    fire_env.load.assert_not_awaited()
    assert fire_env.dispatch.await_args.kwargs["message"] == "check CI"


@pytest.mark.asyncio
async def test_a_job_cancelled_while_it_runs_neither_fires_nor_comes_back(
    fire_env, live_scheduler
):
    # The cancel finds no job (it is out of the store while it runs) but is
    # remembered, so the run that was deciding stands down either way.
    assert live_scheduler.cancel_at_reset_turn(USER, SESSION, "send") is False
    await _fire(_args("send", created_at=datetime.now(UTC) - timedelta(minutes=1)))
    fire_env.dispatch.assert_not_awaited()

    fire_env.busy.return_value = True
    await _fire(_args("send", created_at=datetime.now(UTC) - timedelta(minutes=1)))
    fire_env.client.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_send_set_again_after_a_cancel_still_fires(fire_env, live_scheduler):
    live_scheduler.cancel_at_reset_turn(USER, SESSION, "send")
    await _fire(_args("send", created_at=datetime.now(UTC) + timedelta(seconds=1)))
    fire_env.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_builder_chat_keeps_its_tool_limits(fire_env):
    chat = _chat()
    chat.session.metadata = ChatSessionMetadata(
        llm_auth_provider="platform", builder_graph_id="graph-1"
    )
    fire_env.load.return_value = chat
    await _fire(_args("resume"))
    permissions = fire_env.dispatch.await_args.kwargs["permissions"]
    assert permissions is not None and permissions.tools


def _job_args(service: Scheduler, kind: str) -> CopilotTurnJobArgs:
    job = service.scheduler.get_job(
        at_reset_schedule_id(kind, SESSION), jobstore=Jobstores.EXECUTION.value
    )
    return CopilotTurnJobArgs(**job.kwargs)


def _add(service: Scheduler, kind: str, message: str, created_at=None) -> None:
    service.add_copilot_turn_schedule(
        user_id=USER,
        session_id=SESSION,
        message=message,
        run_at=datetime.now(UTC) + timedelta(hours=3),
        user_timezone="UTC",
        purpose=kind,
        schedule_id=at_reset_schedule_id(kind, SESSION),
        created_at=created_at,
    )


def test_a_send_put_back_joins_one_written_while_it_ran(live_scheduler):
    # The user set a new send while the old one was out of the store,
    # deciding; putting the old one back must not drop what they wrote.
    _add(live_scheduler, "send", "also the changelog")
    _add(
        live_scheduler,
        "send",
        "draft the launch post",
        created_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    args = _job_args(live_scheduler, "send")
    assert args.message == "draft the launch post\n\nalso the changelog"


def test_a_put_back_repeated_after_its_first_try_landed_adds_nothing(
    live_scheduler,
):
    # The put-back call is retried when its answer is lost in transit, which
    # can happen after the scheduler already put it back.
    _add(live_scheduler, "send", "also the changelog")
    set_at = datetime.now(UTC) - timedelta(minutes=1)
    for _ in range(2):
        _add(live_scheduler, "send", "draft the launch post", created_at=set_at)
    args = _job_args(live_scheduler, "send")
    assert args.message == "draft the launch post\n\nalso the changelog"


def test_a_resume_put_back_leaves_one_set_while_it_ran(live_scheduler):
    _add(live_scheduler, "resume", "Resume.")
    newer = _job_args(live_scheduler, "resume").created_at
    _add(
        live_scheduler,
        "resume",
        "Resume.",
        created_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    assert _job_args(live_scheduler, "resume").created_at == newer


def test_a_put_back_with_nothing_newer_is_itself(live_scheduler):
    set_at = datetime.now(UTC) - timedelta(minutes=1)
    _add(live_scheduler, "send", "draft the launch post", created_at=set_at)
    args = _job_args(live_scheduler, "send")
    assert (args.message, args.created_at) == ("draft the launch post", set_at)


@pytest.fixture
def queued(fire_env):
    """``fire_env`` with the real dispatch, down to the queue: what the
    executor would pick up is ``queued.entry()``."""
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    queue = MagicMock(publish_message=AsyncMock())
    cost = TrialCostContext(user_id=USER, trial_id=None)
    with (
        patch(f"{_SCHEDULER}.schedule_turn", executor_utils.schedule_turn),
        patch.object(executor_utils, "check_inflight_cap", AsyncMock()),
        patch.object(executor_utils, "acquire_turn_slot", acquire),
        patch.object(executor_utils, "admit_turn", AsyncMock()),
        patch.object(
            executor_utils, "capture_cost_context", AsyncMock(return_value=cost)
        ),
        patch("backend.copilot.stream_registry.create_session", AsyncMock()),
        patch(
            "backend.util.clients.get_async_copilot_queue",
            AsyncMock(return_value=queue),
        ),
    ):

        def entry() -> executor_utils.CoPilotExecutionEntry:
            call = queue.publish_message.await_args
            assert call is not None
            return executor_utils.CoPilotExecutionEntry.model_validate_json(
                call.kwargs["message"]
            )

        fire_env.entry = entry
        yield fire_env


def _paused_rows(*, tainted: bool) -> list[ChatMessage]:
    session = ChatSession.new(USER, dry_run=False)
    session.messages = [ChatMessage(role="user", content="Build the report")]
    append_pause_marker(session, TurnPause(reason="daily"), tainted=tainted)
    return session.messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tainted",
    [True, False],
    ids=["paused-turn-from-a-bot", "paused-turn-from-the-owner"],
)
async def test_a_resume_starts_with_the_paused_turns_taint(queued, tainted: bool):
    """A web chat, so only the paused turn's own record can say a bot's
    message started the work the resume carries on."""
    queued.load.return_value = _chat(*_paused_rows(tainted=tainted))

    await _fire(_args("resume"))

    entry = queued.entry()
    assert entry.message == RESUMED_AT_RESET_NOTE
    assert entry.envelope is not None
    assert entry.envelope.tainted is tainted


@pytest.mark.asyncio
async def test_a_resume_on_a_marker_from_before_the_field_continues_tainted(
    queued,
):
    """An executor from before the field wrote it, so how the paused turn ran
    is unknown: the resume runs as if it was tainted."""
    old = ChatMessage(
        role="assistant",
        content="Paused: this task is out of today's budget.",
        metadata={PAUSE_KEY: TurnPause(reason="daily").as_part()},
    )
    queued.load.return_value = _chat(
        ChatMessage(role="user", content="Build the report"), old
    )

    await _fire(_args("resume"))

    entry = queued.entry()
    assert entry.envelope is not None
    assert entry.envelope.tainted is True


@pytest.mark.asyncio
async def test_a_send_is_the_users_own_words_whoever_started_the_task(queued):
    queued.load.return_value = _chat(*_paused_rows(tainted=True))

    await _fire(_args("send"))

    entry = queued.entry()
    assert entry.message == "also check the invoices"
    assert entry.envelope is not None
    assert entry.envelope.tainted is False
