from datetime import UTC, datetime

from backend.api.features.tasks.inbox import _dedupe, _event_steps, inbox_state
from backend.api.features.tasks.models import ContractState, InboxState, InboxStep

AT = datetime(2026, 11, 3, tzinfo=UTC)


def test_each_state_lands_in_one_inbox_bucket():
    assert inbox_state(ContractState.WORKING) == InboxState.RUNNING
    assert inbox_state(ContractState.NEEDS_YOU) == InboxState.WAITING
    assert inbox_state(ContractState.STUCK) == InboxState.FAILED
    assert inbox_state(ContractState.STOPPED) == InboxState.FAILED
    assert inbox_state(ContractState.DONE) == InboxState.DONE


def test_turn_expands_into_tool_steps_then_summary():
    event = {
        "kind": "turn",
        "actorName": "Maria",
        "body": None,
        "createdAt": AT,
        "data": {
            "status": "completed",
            "tool_calls": 2,
            "tool_call_cards": [
                {"name": "web_search", "title": "Web search", "status": "completed"},
                {"name": "read_workspace_file", "status": "error"},
            ],
        },
    }
    steps = _event_steps(event)
    assert [s.label for s in steps] == [
        "Web search",
        "read_workspace_file",
        "Maria worked on it",
    ]
    assert steps[1].status == "error"


def test_user_comment_reads_as_you():
    event = {"kind": "comment", "actorName": "Me", "body": "ship it", "createdAt": AT}
    assert _event_steps(event)[0].label == "You commented: ship it"


def test_repeated_steps_are_dropped():
    steps = [InboxStep(label=x, time=AT, kind="system") for x in ["a", "a", "b", "a"]]
    assert [s.label for s in _dedupe(steps)] == ["a", "b", "a"]


def test_a_turn_summary_reads_as_its_first_plain_sentence():
    event = {
        "kind": "turn",
        "actorName": "Maria",
        "body": None,
        "createdAt": AT,
        "data": {"status": "completed", "summary": "**Use Linear.** It is fast."},
    }
    assert _event_steps(event)[-1].label == "Use Linear."


def test_a_bare_status_change_is_not_a_step():
    event = {
        "kind": "status_changed",
        "actorName": "System",
        "body": None,
        "createdAt": AT,
        "data": {"from": "todo", "to": "in_progress"},
    }
    assert _event_steps(event) == []
