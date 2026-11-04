import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.tasks import dispatch, dispatch_finish
from backend.api.features.tasks.models import (
    ActivityKind,
    Actor,
    AssigneeType,
    Task,
    TaskStatus,
)


def _task(assignee_type: AssigneeType, **extra) -> Task:
    assignee = SimpleNamespace(type=assignee_type, id=None, name="Otto")
    return cast(
        Task,
        SimpleNamespace(
            id="task-1",
            key="T-2",
            title="Hire a developer and a marketer",
            work_session_id=None,
            assignee=assignee,
            status=TaskStatus.IN_PROGRESS,
            **extra,
        ),
    )


def _store(creator: Actor) -> MagicMock:
    store = MagicMock()
    store.list_activity = AsyncMock(
        return_value=[SimpleNamespace(kind=ActivityKind.CREATED, actor=creator)]
    )
    store.set_work_session = AsyncMock()
    return store


@pytest.mark.parametrize(
    "creator,assignee,origin",
    [
        (Actor.user("user-1"), AssigneeType.OTTO, "interactive"),
        (Actor.otto(), AssigneeType.OTTO, "automation"),
        (Actor.user("user-1"), AssigneeType.EXPERT, "automation"),
    ],
)
async def test_only_a_task_the_user_wrote_for_otto_gets_a_chat_that_can_hire(
    creator, assignee, origin
):
    created = SimpleNamespace(session_id="chat-1")
    with (
        patch.object(dispatch, "tasks_store", return_value=_store(creator)),
        patch.object(
            dispatch, "create_chat_session", AsyncMock(return_value=created)
        ) as create,
    ):
        await dispatch._ensure_work_session("user-1", _task(assignee))

    assert create.await_args is not None
    assert create.await_args.kwargs["origin"] == origin


async def test_a_chat_the_user_allowed_can_hire_whoever_wrote_the_task():
    store = _store(Actor.otto())
    store.list_activity.return_value.append(
        SimpleNamespace(
            kind=ActivityKind.ACTION_TAKEN,
            actor=Actor.user("user-1"),
            data={"action": "allow"},
        )
    )
    created = SimpleNamespace(session_id="chat-1")
    with (
        patch.object(dispatch, "tasks_store", return_value=store),
        patch.object(
            dispatch, "create_chat_session", AsyncMock(return_value=created)
        ) as create,
    ):
        await dispatch._ensure_work_session("user-1", _task(AssigneeType.OTTO))

    assert create.await_args is not None
    assert create.await_args.kwargs["origin"] == "interactive"


def test_a_turn_reads_which_calls_failed():
    error = json.dumps({"type": "error", "message": "Hiring isn't allowed."})
    messages = [
        SimpleNamespace(
            role="assistant",
            tool_calls=[
                {"id": "a", "function": {"name": "hire_expert"}},
                {"id": "b", "function": {"name": "web_search"}},
            ],
        ),
        SimpleNamespace(role="tool", tool_call_id="a", content=error),
        SimpleNamespace(role="tool", tool_call_id="b", content="{}"),
    ]

    assert dispatch_finish._turn_calls(messages) == [  # type: ignore[arg-type]
        ("hire_expert", "Hiring isn't allowed."),
        ("web_search", None),
    ]


async def test_a_turn_that_hit_a_permission_wall_leaves_a_blocker():
    error = json.dumps({"type": "error", "message": "Hiring isn't allowed here."})
    session = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="go", tool_calls=None),
            SimpleNamespace(
                role="assistant",
                tool_calls=[{"id": "a", "function": {"name": "hire_expert"}}],
            ),
            SimpleNamespace(role="tool", tool_call_id="a", content=error),
        ]
    )
    store = MagicMock()
    store.contract_fill_from_turn = AsyncMock(return_value=[])
    turn = SimpleNamespace(data={"start_message_count": 1})
    with (
        patch.object(
            dispatch_finish, "get_chat_session", AsyncMock(return_value=session)
        ),
        patch.object(dispatch_finish, "tasks_store", return_value=store),
    ):
        await dispatch_finish._fill_contract(
            "user-1", _task(AssigneeType.OTTO), "chat-1", "completed", turn  # type: ignore[arg-type]
        )

    args = store.contract_fill_from_turn.await_args.args
    assert args[4:] == ("Hiring isn't allowed here.", "hire_expert")


def test_the_kickoff_names_the_task_it_starts():
    assert dispatch.kickoff_metadata(_task(AssigneeType.OTTO)) == {
        "kind": "task_kickoff",
        "task_id": "task-1",
        "task_key": "T-2",
        "task_title": "Hire a developer and a marketer",
    }


async def test_an_otto_turn_that_stops_on_hire_cards_waits_for_the_user():
    proposal = json.dumps({"type": "expert_change_proposed", "message": "..."})
    session = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="You are assigned task T-2"),
            SimpleNamespace(role="tool", content=proposal),
            SimpleNamespace(role="assistant", content="Approve the hires above."),
        ]
    )
    store = MagicMock()
    store.set_task_status = AsyncMock(return_value="blocked-task")
    store.add_activity = AsyncMock(return_value="note")
    with (
        patch.object(
            dispatch_finish, "get_chat_session", AsyncMock(return_value=session)
        ),
        patch.object(dispatch_finish, "tasks_store", return_value=store),
        patch.object(dispatch_finish, "publish_task_event", AsyncMock()),
    ):
        waiting = await dispatch_finish._ended_waiting_on_user("chat-1", "user-1")
        settled = await dispatch_finish._settle_unfinished(
            "user-1", _task(AssigneeType.OTTO), "completed", waiting
        )

    assert waiting is True
    assert settled == "blocked-task"
    assert store.set_task_status.await_args.args[2] == TaskStatus.BLOCKED


async def test_a_quiet_otto_turn_leaves_the_task_open():
    store = MagicMock()
    store.set_task_status = AsyncMock()
    task = _task(AssigneeType.OTTO)
    with patch.object(dispatch_finish, "tasks_store", return_value=store):
        settled = await dispatch_finish._settle_unfinished(
            "user-1", task, "completed", False
        )

    assert settled is task
    store.set_task_status.assert_not_awaited()


async def test_a_task_over_budget_says_so_on_its_card():
    task = _task(
        AssigneeType.EXPERT,
        budget_cap_usd=None,
        metrics=SimpleNamespace(cost_usd=7.17, turns=1),
    )
    store = MagicMock()
    store.set_live = AsyncMock()
    store.contract_update_task = AsyncMock(return_value="update")
    publish = AsyncMock()
    with (
        patch.object(dispatch, "tasks_store", return_value=store),
        patch.object(dispatch, "publish_contract_update", publish),
    ):
        started = await dispatch.start_task_turn("user-1", task, "go on")

    assert started is False
    blocker = store.contract_update_task.await_args.args[2]["blocker"]
    assert blocker["cause"] == "limit"
    assert blocker["what_stopped_it"].startswith(
        "Paused: this task reached its budget of $2.00 (spent $7.17)."
    )
    publish.assert_awaited_once_with("user-1", "update")
