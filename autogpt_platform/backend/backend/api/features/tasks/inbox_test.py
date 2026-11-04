from datetime import UTC, datetime

from backend.api.features.tasks.inbox import _dedupe, _event_steps, inbox_state
from backend.api.features.tasks.models import InboxState, InboxStep, TaskStatus

AT = datetime(2026, 11, 3, tzinfo=UTC)


def test_inbox_state_maps_statuses():
    assert inbox_state(TaskStatus.IN_PROGRESS, None) == InboxState.RUNNING
    assert inbox_state(TaskStatus.BLOCKED, "completed") == InboxState.WAITING
    assert inbox_state(TaskStatus.BLOCKED, "failed") == InboxState.FAILED
    assert inbox_state(TaskStatus.CANCELLED, None) == InboxState.FAILED
    assert inbox_state(TaskStatus.DONE, "failed") == InboxState.DONE


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
        "Maria finished a turn · 2 tool calls",
    ]
    assert steps[1].status == "error"


def test_user_comment_reads_as_you():
    event = {"kind": "comment", "actorName": "Me", "body": "ship it", "createdAt": AT}
    assert _event_steps(event)[0].label == "You commented: ship it"


def test_repeated_steps_are_dropped():
    steps = [InboxStep(label=x, time=AT, kind="system") for x in ["a", "a", "b", "a"]]
    assert [s.label for s in _dedupe(steps)] == ["a", "b", "a"]
