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


def _settle_store() -> MagicMock:
    store = MagicMock()
    update = SimpleNamespace(task="settled-task")
    store.contract_ask = AsyncMock(return_value=update)
    store.contract_update_task = AsyncMock(return_value=update)
    store.set_task_status = AsyncMock()
    return store


async def _settle(task: Task, status: str, waiting: bool, reply: str | None):
    store = _settle_store()
    with (
        patch.object(dispatch_finish, "tasks_store", return_value=store),
        patch.object(dispatch_finish, "publish_contract_update", AsyncMock()),
    ):
        settled = await dispatch_finish._settle_unfinished(
            "user-1", task, status, waiting, reply
        )
    return settled, store


async def test_an_otto_turn_that_stops_on_hire_cards_asks_the_user():
    proposal = json.dumps({"type": "expert_change_proposed", "message": "..."})
    session = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="You are assigned task T-2"),
            SimpleNamespace(role="tool", content=proposal),
            SimpleNamespace(role="assistant", content="Approve the hires above."),
        ]
    )
    with patch.object(
        dispatch_finish, "get_chat_session", AsyncMock(return_value=session)
    ):
        waiting, reply = await dispatch_finish._turn_ending("chat-1", "user-1")
    settled, store = await _settle(
        _task(AssigneeType.OTTO, raw_status="running"), "completed", waiting, reply
    )

    assert waiting is True
    assert settled == "settled-task"
    assert store.contract_ask.await_args.args[2] == "Approve the hires above."
    store.set_task_status.assert_not_awaited()


async def test_a_quiet_turn_finishes_the_task_with_its_last_words():
    task = _task(AssigneeType.EXPERT, raw_status="running")

    settled, store = await _settle(
        task, "completed", False, "## Pick\n**Linear** fits a team of 8. It is fast."
    )

    assert settled == "settled-task"
    assert store.contract_update_task.await_args.args[2] == {
        "result": {"summary": "Linear fits a team of 8."}
    }


async def test_a_quiet_turn_with_nothing_said_fails_with_a_reason():
    _, store = await _settle(
        _task(AssigneeType.EXPERT, raw_status="running"), "completed", False, None
    )

    blocker = store.contract_update_task.await_args.args[2]["blocker"]
    assert blocker["what_stopped_it"]
    assert blocker["next_actions"] == ["retry_failed_part"]


async def test_a_task_waiting_on_its_subtasks_stays_open():
    task = _task(AssigneeType.OTTO, raw_status="waiting_on_subtasks")

    settled, store = await _settle(task, "completed", False, "Started 3 sub-tasks.")

    assert settled is task
    store.contract_update_task.assert_not_awaited()


async def test_the_work_chat_is_named_after_its_task():
    created = SimpleNamespace(session_id="chat-1")
    title = AsyncMock(return_value=True)
    with (
        patch.object(dispatch, "tasks_store", return_value=_store(Actor.otto())),
        patch.object(dispatch, "create_chat_session", AsyncMock(return_value=created)),
        patch.object(dispatch, "update_session_title", title),
    ):
        await dispatch._ensure_work_session("user-1", _task(AssigneeType.EXPERT))

    title.assert_awaited_once_with(
        "chat-1", "user-1", "Hire a developer and a marketer", only_if_empty=True
    )


def test_the_kickoff_says_how_every_turn_ends():
    message = dispatch.kickoff_message(
        _task(AssigneeType.EXPERT, description=None, parent_key=None)
    )

    assert "blocker" in message and "Never send a result" in message
    assert message.count("Hire a developer and a marketer") == 1


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


async def test_a_reply_in_the_chat_closes_the_ask_the_chat_opened():
    ask = SimpleNamespace(id="ask-1", why=dispatch_finish.CHAT_WAIT_WHY)
    task = _task(AssigneeType.EXPERT, ask=ask)
    task.status = TaskStatus.BLOCKED
    store = MagicMock()
    store.answer_input = AsyncMock()
    store.get_task = AsyncMock(return_value="reopened")
    with patch.object(dispatch_finish, "tasks_store", return_value=store):
        reopened = await dispatch_finish._reopen_after_chat_reply("user-1", task)

    assert reopened == "reopened"
    assert store.answer_input.await_args.args[3] == "ask-1"


async def test_an_ask_the_worker_opened_stays_open():
    ask = SimpleNamespace(id="ask-1", why="Needed to pick a vendor.")
    task = _task(AssigneeType.EXPERT, ask=ask)
    task.status = TaskStatus.BLOCKED
    store = MagicMock()
    store.answer_input = AsyncMock()
    with patch.object(dispatch_finish, "tasks_store", return_value=store):
        assert await dispatch_finish._reopen_after_chat_reply("user-1", task) is task

    store.answer_input.assert_not_awaited()
