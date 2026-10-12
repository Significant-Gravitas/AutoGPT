from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import ExpertRoutine
from backend.api.features.experts.routines import (
    RoutineNotFoundError,
    RoutineUnansweredAsksError,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.routines import (
    ListRoutinesTool,
    RoutineResponse,
    RoutinesResponse,
    ScheduleRoutineTool,
)

_PATH = "backend.copilot.tools.routines"
_SCOPE = "backend.copilot.tools.expert_scope"


def _routine(**overrides) -> ExpertRoutine:
    return ExpertRoutine(
        **{
            "id": "routine-1",
            "expert_id": "expert-a",
            "key": "queue-sweep",
            "title": "Sweep the queue",
            "prompt": "Read the queue and stage a draft per item.",
            "crons": ["H 9 * * 1-5"],
            "asks": ["Where is the queue?"],
            "session_mode": "THREAD",
            "source": "TEMPLATE",
            "enabled": False,
            **overrides,
        }
    )


@pytest.fixture
def experts():
    db = MagicMock()
    db.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: (
            MagicMock(id=expert_id) if expert_id in {"expert-a", "expert-b"} else None
        )
    )
    db.list_routines = AsyncMock(return_value=[_routine()])
    db.enable_routine = AsyncMock(
        return_value=_routine(enabled=True, customized=True, crons=["17 9 * * 1-5"])
    )
    db.disable_routine = AsyncMock(return_value=_routine())
    db.create_routine = AsyncMock(
        return_value=_routine(id="routine-2", key=None, source="OWNER", customized=True)
    )
    with (
        patch(f"{_PATH}.experts_db", return_value=db),
        patch(f"{_SCOPE}.experts_db", return_value=db),
    ):
        yield db


def _session(expert_id: str | None) -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, expert_id=expert_id)


async def test_an_expert_lists_its_own_standing_work(experts):
    result = await ListRoutinesTool()._execute("user-1", _session("expert-a"))
    assert isinstance(result, RoutinesResponse)
    assert [r.key for r in result.routines] == ["queue-sweep"]
    experts.list_routines.assert_awaited_once_with("user-1", "expert-a")


async def test_autopilot_naming_nobody_lists_the_accounts_own_routines(experts):
    """Otto is the default assistant, not a row in Expert, so its standing work
    hangs off the owner. Refusing here — as the expert-only version did — is
    what sent the model to ``schedule_followup`` instead."""
    result = await ListRoutinesTool()._execute("user-1", _session(None))
    assert isinstance(result, RoutinesResponse)
    assert result.expert_id is None
    experts.list_routines.assert_awaited_once_with("user-1", None)


async def test_autopilot_may_still_name_one_of_its_experts(experts):
    await ListRoutinesTool()._execute("user-1", _session(None), expert_id="expert-b")
    experts.list_routines.assert_awaited_once_with("user-1", "expert-b")


async def test_switching_one_on_passes_the_owners_answers_through(experts):
    """The whole point of the round trip: what reaches the database is the
    prompt and cadence the owner agreed, not the template's proposal."""
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the support queue.",
        crons=["0 10 * * 1-5"],
        grants_credentials=True,
    )
    assert isinstance(result, RoutineResponse)
    assert result.routine.enabled is True
    kwargs = experts.enable_routine.await_args.kwargs
    assert kwargs["prompt"] == "Read the support queue."
    assert kwargs["crons"] == ["0 10 * * 1-5"]
    assert kwargs["grants_credentials"] is True


async def test_an_omitted_grant_leaves_the_routine_where_it_was(experts):
    """Omitting the flag must not read as permission, and must not read as
    revocation either: rewording a routine the owner already bound to their
    inbox should not quietly unbind it. ``None`` is "leave it alone"."""
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
    )
    assert experts.enable_routine.await_args.kwargs["grants_credentials"] is None


async def test_pinning_defaults_to_the_chat_the_routine_was_set_up_in(experts):
    """THREAD is the default because a daily routine firing into the chat the
    user is having now buries it; PINNED is for when they asked for that."""
    session = _session("expert-a")
    await ScheduleRoutineTool()._execute(
        "user-1",
        session,
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
        session_mode="PINNED",
    )
    kwargs = experts.enable_routine.await_args.kwargs
    assert kwargs["session_mode"] == "PINNED"
    assert kwargs["pinned_session_id"] == session.session_id


async def test_pinning_can_name_another_chat_the_owner_holds(experts):
    """The one thing ``schedule_followup`` could do that a routine could not:
    land in a conversation other than the one being had now."""
    target = MagicMock(expert_id="expert-a")
    with patch(f"{_PATH}.get_chat_session", AsyncMock(return_value=target)):
        await ScheduleRoutineTool()._execute(
            "user-1",
            _session("expert-a"),
            routine_id="routine-1",
            enabled=True,
            prompt="Read the queue.",
            session_mode="PINNED",
            session_id="other-session",
        )
    assert experts.enable_routine.await_args.kwargs["pinned_session_id"] == (
        "other-session"
    )


async def test_a_chat_in_another_experts_scope_cannot_be_pinned(experts):
    """A routine aimed at another persona's thread would write that expert's
    memory under this one's prompt, every run, unattended."""
    with patch(
        f"{_PATH}.get_chat_session",
        AsyncMock(return_value=MagicMock(expert_id="expert-b")),
    ):
        result = await ScheduleRoutineTool()._execute(
            "user-1",
            _session("expert-a"),
            routine_id="routine-1",
            enabled=True,
            session_mode="PINNED",
            session_id="someone-elses",
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "session_not_found"
    experts.enable_routine.assert_not_awaited()


async def test_a_chat_that_is_not_the_callers_cannot_be_pinned(experts):
    with patch(f"{_PATH}.get_chat_session", AsyncMock(return_value=None)):
        result = await ScheduleRoutineTool()._execute(
            "user-1",
            _session("expert-a"),
            routine_id="routine-1",
            enabled=True,
            session_mode="PINNED",
            session_id="not-mine",
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "session_not_found"


async def test_unanswered_asks_come_back_as_the_questions_to_ask(experts):
    """The model has to be told what is missing, or it cannot close the gap
    with the user — a bare refusal would just stall."""
    experts.enable_routine.side_effect = RoutineUnansweredAsksError(
        "'Sweep the queue' needs answers before it can run: Where is the queue?"
    )
    result = await ScheduleRoutineTool()._execute(
        "user-1", _session("expert-a"), routine_id="routine-1", enabled=True
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "unanswered_asks"
    assert "Where is the queue?" in result.message


async def test_switching_one_off_keeps_the_row(experts):
    result = await ScheduleRoutineTool()._execute(
        "user-1", _session("expert-a"), routine_id="routine-1", enabled=False
    )
    assert isinstance(result, RoutineResponse)
    assert result.routine.enabled is False
    experts.disable_routine.assert_awaited_once_with("user-1", "expert-a", "routine-1")
    experts.enable_routine.assert_not_awaited()


async def test_an_expert_records_standing_work_it_agreed_in_conversation(experts):
    """No routine_id means there was no proposal — a raised expert has no
    template, so this is the only way it gets any standing work at all."""
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        enabled=False,
        title="Morning triage",
        prompt="Read overnight tickets and rank them.",
        crons=["H 8 * * 1-5"],
        session_mode="FRESH",
    )
    assert isinstance(result, RoutineResponse)
    kwargs = experts.create_routine.await_args.kwargs
    assert kwargs["title"] == "Morning triage"
    assert kwargs["crons"] == ["H 8 * * 1-5"]
    assert kwargs["session_mode"] == "FRESH"
    experts.enable_routine.assert_not_awaited()


async def test_work_the_owner_dictated_reaches_what_they_reach(experts):
    """The provenance rule, at the point it is decided: a prompt the owner
    spoke is not a template, so it is not muted like one."""
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        enabled=False,
        title="Morning triage",
        prompt="Read overnight tickets.",
        crons=["H 8 * * 1-5"],
    )
    kwargs = experts.create_routine.await_args.kwargs
    assert kwargs["grants_credentials"] is True


async def test_the_owner_can_still_ask_for_one_that_touches_nothing(experts):
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        enabled=False,
        title="Morning triage",
        prompt="Read overnight tickets.",
        crons=["H 8 * * 1-5"],
        grants_credentials=False,
    )
    assert experts.create_routine.await_args.kwargs["grants_credentials"] is False


async def test_a_one_shot_is_recorded_as_a_time_not_a_cadence(experts):
    """ "Check the deploy in an hour" gets a row like anything else, so it can
    be listed, reworded and cancelled rather than living only as a job."""
    before = datetime.now(timezone.utc)
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        enabled=True,
        title="Check the deploy",
        prompt="Check whether the deploy finished.",
        delay_seconds=3600,
    )
    kwargs = experts.create_routine.await_args.kwargs
    assert kwargs["crons"] is None
    assert kwargs["run_at"] is not None
    assert (kwargs["run_at"] - before).total_seconds() == pytest.approx(3600, abs=30)


async def test_a_delay_shorter_than_the_followup_minimum_is_refused(experts):
    """Kept identical to ``schedule_followup`` so the two cannot disagree about
    what the soonest deferral is."""
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        enabled=True,
        title="Too soon",
        prompt="Now-ish.",
        delay_seconds=5,
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_schedule"
    experts.create_routine.assert_not_awaited()


async def test_creating_and_switching_on_together_enables_the_new_row(experts):
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        enabled=True,
        title="Morning triage",
        prompt="Read overnight tickets.",
        crons=["H 8 * * 1-5"],
    )
    experts.create_routine.assert_awaited_once()
    # The id enabled is the one just created, never the caller's.
    assert experts.enable_routine.await_args.args[2] == "routine-2"
    # The cadence was resolved onto the new row a moment ago; re-sending it
    # would re-validate the same values and, for a one-shot, recompute the
    # delay against a later now.
    assert experts.enable_routine.await_args.kwargs["crons"] is None
    assert experts.enable_routine.await_args.kwargs["run_at"] is None


async def test_a_call_naming_neither_a_routine_nor_a_title_is_refused(experts):
    result = await ScheduleRoutineTool()._execute(
        "user-1", _session("expert-a"), enabled=True
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_routine"
    experts.create_routine.assert_not_awaited()
    experts.enable_routine.assert_not_awaited()


async def test_an_expert_cannot_reach_a_teammates_routines(experts):
    """``resolve_routine_owner``'s rule, which these tools inherit: an expert
    acts on itself and may not name another."""
    result = await ListRoutinesTool()._execute(
        "user-1", _session("expert-a"), expert_id="expert-b"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "access_denied"
    experts.list_routines.assert_not_awaited()


async def test_an_expert_that_is_not_on_the_account_is_not_found(experts):
    result = await ListRoutinesTool()._execute(
        "user-1", _session(None), expert_id="expert-z"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "expert_not_found"


async def test_a_routine_id_from_another_account_is_not_found(experts):
    experts.enable_routine.side_effect = RoutineNotFoundError("routine-1")
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Mine now.",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "routine_not_found"


async def test_an_invalid_cadence_is_reported_rather_than_raised(experts):
    experts.enable_routine.side_effect = ValueError("bad cron")
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        crons=["not a cron"],
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_schedule"


async def test_an_unauthenticated_call_is_refused_before_any_lookup(experts):
    result = await ScheduleRoutineTool()._execute(
        None, _session("expert-a"), routine_id="routine-1", enabled=True
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"
    experts.enable_routine.assert_not_awaited()


async def test_the_confirmation_states_what_the_routine_may_reach(experts):
    """The model repeats this back, so it has to carry the two things an owner
    would want to check: when it runs, and whether it can touch their accounts."""
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
    )
    assert isinstance(result, RoutineResponse)
    assert "17 9 * * 1-5" in result.message
    assert "reaches nothing outside the platform" in result.message


async def test_the_confirmation_of_a_one_shot_names_its_time(experts):
    experts.enable_routine.return_value = _routine(
        enabled=True,
        crons=[],
        run_at=datetime(2026, 9, 20, 14, 30, tzinfo=timezone.utc),
    )
    result = await ScheduleRoutineTool()._execute(
        "user-1", _session("expert-a"), routine_id="routine-1", enabled=True
    )
    assert isinstance(result, RoutineResponse)
    assert "2026-09-20 14:30" in result.message


async def test_autopilot_pinning_an_experts_routine_must_name_that_experts_chat(
    experts,
):
    """Personal AutoPilot is in no expert scope, so "this chat" is the wrong
    default when it manages an expert's routine: the chat it would pin belongs
    to Otto, and the fire path refuses it on every run — deleting the schedule
    and leaving a routine that says it is on with nothing behind it."""
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        expert_id="expert-a",
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
        session_mode="PINNED",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "session_required"
    experts.enable_routine.assert_not_awaited()


async def test_autopilot_may_pin_an_experts_routine_to_that_experts_chat(experts):
    with patch(
        f"{_PATH}.get_chat_session",
        AsyncMock(return_value=MagicMock(expert_id="expert-a")),
    ):
        await ScheduleRoutineTool()._execute(
            "user-1",
            _session(None),
            expert_id="expert-a",
            routine_id="routine-1",
            enabled=True,
            prompt="Read the queue.",
            session_mode="PINNED",
            session_id="an-expert-a-chat",
        )
    assert experts.enable_routine.await_args.kwargs["pinned_session_id"] == (
        "an-expert-a-chat"
    )


async def test_autopilot_pinning_its_own_routine_still_defaults_to_this_chat(experts):
    """The account's own standing work is in the account's own scope, so the
    default is right there and only there."""
    session = _session(None)
    await ScheduleRoutineTool()._execute(
        "user-1",
        session,
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
        session_mode="PINNED",
    )
    assert experts.enable_routine.await_args.kwargs["pinned_session_id"] == (
        session.session_id
    )


async def test_autopilot_can_manage_an_experts_thread_routine_without_a_session(
    experts,
):
    """The regression the PINNED scope check introduced. THREAD has no chat to
    name, so requiring one refused the ordinary case outright: personal
    AutoPilot switching on an expert's routine, which is most of the flow."""
    result = await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        expert_id="expert-a",
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
    )
    assert isinstance(result, RoutineResponse)
    experts.enable_routine.assert_awaited_once()
    assert experts.enable_routine.await_args.kwargs["pinned_session_id"] is None


async def test_autopilot_can_create_an_experts_fresh_routine_without_a_session(
    experts,
):
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session(None),
        expert_id="expert-a",
        enabled=False,
        title="Morning triage",
        prompt="Read overnight tickets.",
        crons=["H 8 * * 1-5"],
        session_mode="FRESH",
    )
    experts.create_routine.assert_awaited_once()
    assert experts.create_routine.await_args.kwargs["session_id"] is None


async def test_omitting_the_mode_does_not_re_pin_a_routine_to_this_chat(experts):
    """An omitted `session_mode` means "leave it as it is". Answering with the
    caller's chat would move an already-pinned routine into whatever
    conversation happened to reword it."""
    await ScheduleRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Reworded, nothing else.",
    )
    assert experts.enable_routine.await_args.kwargs["pinned_session_id"] is None
