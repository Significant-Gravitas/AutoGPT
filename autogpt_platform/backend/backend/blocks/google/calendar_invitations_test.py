"""Unit tests for the Respond To Event block's request building."""

import pytest

from backend.blocks.google import calendar_invitations
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._calendar_api import TEST_EVENT_RESOURCE
from backend.blocks.google.calendar_invitations import GoogleCalendarRespondToEventBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError

INVITATION = {
    **TEST_EVENT_RESOURCE,
    "organizer": {"email": "alex@example.com"},
    "attendees": [
        {"email": "alex@example.com", "organizer": True, "responseStatus": "accepted"},
        {"email": "me@example.com", "self": True, "responseStatus": "needsAction"},
    ],
}


class _FakeEvents:
    """Records events().get/patch calls and answers with the invitation."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def events(self):
        return self

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return self

    def patch(self, **kwargs):
        self.calls.append(("patch", kwargs))
        return self

    def execute(self):
        return INVITATION


def _respond_input(**fields) -> GoogleCalendarRespondToEventBlock.Input:
    return GoogleCalendarRespondToEventBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "event_id": INVITATION["id"],
            "response": "accepted",
            **fields,
        }
    )


def test_reply_changes_only_my_own_response():
    body = GoogleCalendarRespondToEventBlock()._reply_body(
        INVITATION, _respond_input(response="tentative", comment="Might be late")
    )
    assert body == {
        "attendeesOmitted": True,
        "attendees": [
            {
                "email": "me@example.com",
                "responseStatus": "tentative",
                "comment": "Might be late",
            }
        ],
    }


def test_reply_without_a_comment_leaves_the_comment_alone():
    body = GoogleCalendarRespondToEventBlock()._reply_body(INVITATION, _respond_input())
    assert body["attendees"] == [
        {"email": "me@example.com", "responseStatus": "accepted"}
    ]


def test_reply_needs_me_on_the_guest_list():
    event = {**INVITATION, "attendees": INVITATION["attendees"][:1]}
    with pytest.raises(BlockExecutionError, match="guest list"):
        GoogleCalendarRespondToEventBlock()._reply_body(event, _respond_input())


@pytest.mark.asyncio
async def test_respond_reads_the_event_then_sends_only_my_reply(monkeypatch):
    service = _FakeEvents()
    monkeypatch.setattr(
        calendar_invitations, "build_calendar_service", lambda credentials: service
    )
    outputs = [
        output
        async for output in GoogleCalendarRespondToEventBlock().run(
            _respond_input(response="declined", notify_organizer=False),
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        )
    ]
    assert [name for name, _ in service.calls] == ["get", "patch"]
    patch = service.calls[1][1]
    assert patch["sendUpdates"] == "none"
    assert patch["body"]["attendeesOmitted"] is True
    assert patch["body"]["attendees"][0]["responseStatus"] == "declined"
    assert outputs[0][0] == "event"
