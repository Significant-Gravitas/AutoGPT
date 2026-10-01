"""Regression tests for bugs fixed in the Gmail blocks in gmail.py.

The blocks' own test_input/test_mock cases mock the API calls away, so
these drive the real code with a fake Gmail service.
"""

import pytest

from backend.blocks.google.gmail import (
    GmailGetThreadBlock,
    GmailReadBlock,
    GmailRemoveLabelBlock,
)

READONLY = "https://www.googleapis.com/auth/gmail.readonly"


class _Request:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Resource:
    def __init__(self, gmail_service: "_FakeGmail", name: str):
        self._gmail = gmail_service
        self._name = name

    def __getattr__(self, method: str):
        def call(**kwargs):
            key = f"{self._name}.{method}"
            self._gmail.calls.append((key, kwargs))
            return _Request(self._gmail.responses[key])

        return call


class _FakeGmail:
    """Canned responses keyed by "resource.method", e.g. "messages.get"."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def users(self):
        return self

    def messages(self):
        return _Resource(self, "messages")

    def threads(self):
        return _Resource(self, "threads")

    def labels(self):
        return _Resource(self, "labels")

    def called(self, key: str) -> list[dict]:
        return [kwargs for name, kwargs in self.calls if name == key]


def _message_with_headers(headers: dict[str, str]) -> dict:
    return {
        "id": "m1",
        "threadId": "t1",
        "labelIds": ["INBOX"],
        "snippet": "",
        "sizeEstimate": 100,
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": name, "value": value} for name, value in headers.items()
            ],
            "body": {"size": 0},
        },
    }


# "undisclosed-recipients:;" and a doubled comma parse to a blank address on
# every Python; a missing header does too on Python 3.13 (and 3.12.6+, 3.11.10+).
_ODD_RECIPIENTS = _message_with_headers(
    {
        "From": "Ann Lee <ann@example.com>",
        "To": "undisclosed-recipients:;",
        "Cc": "bob@example.com,, cara@example.com",
        "Subject": "Minutes",
    }
)


@pytest.mark.asyncio
async def test_read_drops_blank_recipients():
    fake = _FakeGmail(
        {
            "messages.list": {"messages": [{"id": "m1"}]},
            "messages.get": _ODD_RECIPIENTS,
        }
    )
    [email] = await GmailReadBlock()._read_emails(fake, "", 1, [READONLY])
    assert (email.to, email.cc, email.bcc) == (
        [],
        ["bob@example.com", "cara@example.com"],
        [],
    )


@pytest.mark.asyncio
async def test_get_thread_drops_blank_recipients():
    fake = _FakeGmail(
        {"threads.get": {"id": "t1", "historyId": "1", "messages": [_ODD_RECIPIENTS]}}
    )
    thread = await GmailGetThreadBlock()._get_thread(fake, "t1", [READONLY])
    [email] = thread["messages"]
    assert (email["to"], email["cc"], email["bcc"]) == (
        [],
        ["bob@example.com", "cara@example.com"],
        [],
    )


_LABELS = {
    "labels": [{"id": "UNREAD", "name": "UNREAD"}, {"id": "Label_7", "name": "Todo"}]
}


@pytest.mark.asyncio
async def test_remove_label_with_an_unknown_name_matches_its_output_schema():
    fake = _FakeGmail({"labels.list": _LABELS})
    result = await GmailRemoveLabelBlock()._remove_label(fake, "m1", "Tod")
    assert result == {"status": "Label not found", "label_id": ""}
    assert GmailRemoveLabelBlock.Output.validate_field("result", result) is None
    assert fake.called("messages.modify") == []


@pytest.mark.asyncio
async def test_remove_label_reports_success_when_no_labels_are_left():
    fake = _FakeGmail(
        {
            "labels.list": _LABELS,
            "messages.get": {"id": "m1", "threadId": "t1", "labelIds": ["UNREAD"]},
            # Gmail leaves out labelIds when a message has none left.
            "messages.modify": {"id": "m1", "threadId": "t1"},
        }
    )
    result = await GmailRemoveLabelBlock()._remove_label(fake, "m1", "UNREAD")
    assert result == {"status": "Label removed successfully", "label_id": "UNREAD"}
    assert fake.called("messages.modify") == [
        {"userId": "me", "id": "m1", "body": {"removeLabelIds": ["UNREAD"]}}
    ]


@pytest.mark.asyncio
async def test_remove_label_reports_a_label_that_was_not_applied():
    fake = _FakeGmail(
        {
            "labels.list": _LABELS,
            "messages.get": {"id": "m1", "threadId": "t1", "labelIds": ["INBOX"]},
            "messages.modify": {"id": "m1", "threadId": "t1", "labelIds": ["INBOX"]},
        }
    )
    result = await GmailRemoveLabelBlock()._remove_label(fake, "m1", "Todo")
    assert result == {
        "status": "Label already removed or not applied",
        "label_id": "Label_7",
    }
    assert fake.called("messages.modify") == []
