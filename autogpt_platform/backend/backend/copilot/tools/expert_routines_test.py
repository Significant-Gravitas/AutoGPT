from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import ExpertRoutine
from backend.api.features.experts.routines import (
    RoutineNotFoundError,
    RoutineUnansweredAsksError,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools.expert_routines import (
    ExpertRoutineResponse,
    ExpertRoutinesResponse,
    ListExpertRoutinesTool,
    SetExpertRoutineTool,
)
from backend.copilot.tools.models import ErrorResponse

_PATH = "backend.copilot.tools.expert_routines"
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
        return_value=_routine(id="routine-2", key=None, customized=True)
    )
    with (
        patch(f"{_PATH}.experts_db", return_value=db),
        patch(f"{_SCOPE}.experts_db", return_value=db),
    ):
        yield db


def _session(expert_id: str | None) -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, expert_id=expert_id)


async def test_an_expert_lists_its_own_standing_work(experts):
    result = await ListExpertRoutinesTool()._execute("user-1", _session("expert-a"))
    assert isinstance(result, ExpertRoutinesResponse)
    assert [r.key for r in result.routines] == ["queue-sweep"]
    experts.list_routines.assert_awaited_once_with("user-1", "expert-a")


async def test_switching_one_on_passes_the_owners_answers_through(experts):
    """The whole point of the round trip: what reaches the database is the
    prompt and cadence the owner agreed, not the template's proposal."""
    result = await SetExpertRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the support queue.",
        crons=["0 10 * * 1-5"],
        grants_credentials=True,
    )
    assert isinstance(result, ExpertRoutineResponse)
    assert result.routine.enabled is True
    kwargs = experts.enable_routine.await_args.kwargs
    assert kwargs["prompt"] == "Read the support queue."
    assert kwargs["crons"] == ["0 10 * * 1-5"]
    assert kwargs["grants_credentials"] is True


async def test_a_routine_reaches_nothing_unless_the_owner_says_so(experts):
    """Omitting the flag must not read as permission. A seeded routine that
    quietly gained the owner's connections would be the whole safety story
    undone by a default."""
    await SetExpertRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
    )
    assert experts.enable_routine.await_args.kwargs["grants_credentials"] is False


async def test_this_chat_is_only_offered_as_a_home_when_asked_for(experts):
    """THREAD is the default because a daily routine firing into the chat the
    user is having now buries it."""
    await SetExpertRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
        session_mode="HERE",
    )
    kwargs = experts.enable_routine.await_args.kwargs
    assert kwargs["session_mode"] == "HERE"
    assert kwargs["here_session_id"] is not None


async def test_unanswered_asks_come_back_as_the_questions_to_ask(experts):
    """The model has to be told what is missing, or it cannot close the gap
    with the user — a bare refusal would just stall."""
    experts.enable_routine.side_effect = RoutineUnansweredAsksError(
        "'Sweep the queue' needs answers before it can run: Where is the queue?"
    )
    result = await SetExpertRoutineTool()._execute(
        "user-1", _session("expert-a"), routine_id="routine-1", enabled=True
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "unanswered_asks"
    assert "Where is the queue?" in result.message


async def test_switching_one_off_keeps_the_row(experts):
    result = await SetExpertRoutineTool()._execute(
        "user-1", _session("expert-a"), routine_id="routine-1", enabled=False
    )
    assert isinstance(result, ExpertRoutineResponse)
    assert result.routine.enabled is False
    experts.disable_routine.assert_awaited_once_with("user-1", "expert-a", "routine-1")
    experts.enable_routine.assert_not_awaited()


async def test_an_expert_records_standing_work_it_agreed_in_conversation(experts):
    """No routine_id means there was no proposal — a raised expert has no
    template, so this is the only way it gets any standing work at all."""
    result = await SetExpertRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        enabled=False,
        title="Morning triage",
        prompt="Read overnight tickets and rank them.",
        crons=["H 8 * * 1-5"],
        session_mode="FRESH",
    )
    assert isinstance(result, ExpertRoutineResponse)
    kwargs = experts.create_routine.await_args.kwargs
    assert kwargs["title"] == "Morning triage"
    assert kwargs["crons"] == ["H 8 * * 1-5"]
    assert kwargs["session_mode"] == "FRESH"
    experts.enable_routine.assert_not_awaited()


async def test_creating_and_switching_on_together_enables_the_new_row(experts):
    await SetExpertRoutineTool()._execute(
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


async def test_a_call_naming_neither_a_routine_nor_a_title_is_refused(experts):
    result = await SetExpertRoutineTool()._execute(
        "user-1", _session("expert-a"), enabled=True
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_routine"
    experts.create_routine.assert_not_awaited()
    experts.enable_routine.assert_not_awaited()


async def test_an_expert_cannot_reach_a_teammates_routines(experts):
    """``resolve_target_expert``'s rule, which these tools inherit: an expert
    acts on itself and may not name another."""
    result = await ListExpertRoutinesTool()._execute(
        "user-1", _session("expert-a"), expert_id="expert-b"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "access_denied"
    experts.list_routines.assert_not_awaited()


async def test_autopilot_must_name_the_expert_it_means(experts):
    result = await ListExpertRoutinesTool()._execute("user-1", _session(None))
    assert isinstance(result, ErrorResponse)
    assert result.error == "expert_required"


async def test_a_routine_id_from_another_account_is_not_found(experts):
    experts.enable_routine.side_effect = RoutineNotFoundError("routine-1")
    result = await SetExpertRoutineTool()._execute(
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
    result = await SetExpertRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        crons=["not a cron"],
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_cron"


async def test_an_unauthenticated_call_is_refused_before_any_lookup(experts):
    result = await SetExpertRoutineTool()._execute(
        None, _session("expert-a"), routine_id="routine-1", enabled=True
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"
    experts.enable_routine.assert_not_awaited()


async def test_the_confirmation_states_what_the_routine_may_reach(experts):
    """The model repeats this back, so it has to carry the two things an owner
    would want to check: when it runs, and whether it can touch their accounts."""
    result = await SetExpertRoutineTool()._execute(
        "user-1",
        _session("expert-a"),
        routine_id="routine-1",
        enabled=True,
        prompt="Read the queue.",
    )
    assert isinstance(result, ExpertRoutineResponse)
    assert "17 9 * * 1-5" in result.message
    assert "reaches nothing outside the platform" in result.message
