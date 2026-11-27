"""Regression tests for bugs fixed in the Gmail blocks in gmail.py.

The blocks' own test_input/test_mock cases mock the API calls away, so
these drive the real code with a fake Gmail service.
"""

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google.gmail import (
    GmailAddLabelBlock,
    GmailGetThreadBlock,
    GmailReadBlock,
    GmailRemoveLabelBlock,
)

READONLY = "https://www.googleapis.com/auth/gmail.readonly"


class _Request:
    def __init__(self, result):
        self._result = result

    def execute(self):
        result = self._result() if callable(self._result) else self._result
        if isinstance(result, Exception):
            raise result
        return result


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
    """Canned responses keyed by "resource.method", e.g. "messages.get".

    A response can be an exception to raise, or a function returning the
    response, for a call whose answer changes between calls.
    """

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
    "labels": [
        {"id": "UNREAD", "name": "UNREAD", "type": "system"},
        {"id": "TRASH", "name": "TRASH", "type": "system"},
        {"id": "Label_7", "name": "Todo", "type": "user"},
    ]
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


def _http_error(status: int, message: str) -> HttpError:
    return HttpError(
        httplib2.Response({"status": status}),
        f'{{"error": {{"code": {status}, "message": "{message}"}}}}'.encode(),
    )


def _conflict() -> HttpError:
    return _http_error(409, "Label name exists or conflicts")


_MODIFIED = {"id": "m1", "threadId": "t1", "labelIds": ["INBOX", "Label_7"]}
_LATER = {"id": "Label_9", "name": "Later", "type": "user"}


@pytest.mark.asyncio
async def test_add_label_uses_an_existing_label_whatever_its_case():
    fake = _FakeGmail({"labels.list": _LABELS, "messages.modify": _MODIFIED})
    result = await GmailAddLabelBlock()._add_label(fake, "m1", "todo")
    assert result == {"status": "Label added successfully", "label_id": "Label_7"}
    assert fake.called("labels.create") == []
    assert fake.called("messages.modify") == [
        {"userId": "me", "id": "m1", "body": {"addLabelIds": ["Label_7"]}}
    ]


@pytest.mark.asyncio
async def test_remove_label_finds_a_label_whatever_its_case():
    fake = _FakeGmail(
        {
            "labels.list": _LABELS,
            "messages.get": {"id": "m1", "threadId": "t1", "labelIds": ["Label_7"]},
            "messages.modify": {"id": "m1", "threadId": "t1"},
        }
    )
    result = await GmailRemoveLabelBlock()._remove_label(fake, "m1", "TODO")
    assert result == {"status": "Label removed successfully", "label_id": "Label_7"}


@pytest.mark.asyncio
async def test_add_label_uses_the_label_when_creating_it_conflicts():
    # Another run creates "Later" between the lookup and the create of "later".
    lists = iter([_LABELS, {"labels": [*_LABELS["labels"], _LATER]}])
    fake = _FakeGmail(
        {
            "labels.list": lambda: next(lists),
            "labels.create": _conflict(),
            "messages.modify": _MODIFIED,
        }
    )
    result = await GmailAddLabelBlock()._add_label(fake, "m1", "later")
    assert result == {"status": "Label added successfully", "label_id": "Label_9"}
    assert fake.called("messages.modify") == [
        {"userId": "me", "id": "m1", "body": {"addLabelIds": ["Label_9"]}}
    ]


@pytest.mark.asyncio
async def test_add_label_conflict_with_no_matching_label_is_a_plain_error(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeGmail({"labels.list": _LABELS, "labels.create": _conflict()})
    block = GmailAddLabelBlock()
    monkeypatch.setattr(block, "_build_service", lambda *args, **kwargs: fake)
    input_data = GmailAddLabelBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
        message_id="m1",
        label_name="Chats",
    )
    # A ValueError reaches the user as a block execution error with this
    # message, where the raw HttpError became an unknown error.
    with pytest.raises(ValueError, match='label "Chats"'):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
    assert fake.called("messages.modify") == []


@pytest.mark.asyncio
async def test_add_label_passes_other_create_errors_through():
    denied = _http_error(403, "Insufficient Permission")
    fake = _FakeGmail({"labels.list": _LABELS, "labels.create": denied})
    with pytest.raises(HttpError) as raised:
        await GmailAddLabelBlock()._add_label(fake, "m1", "Later")
    assert raised.value is denied
    assert len(fake.called("labels.list")) == 1


@pytest.mark.asyncio
async def test_system_labels_still_need_their_exact_name():
    fake = _FakeGmail({"labels.list": _LABELS, "labels.create": _conflict()})
    with pytest.raises(ValueError, match='label "trash"'):
        await GmailAddLabelBlock()._add_label(fake, "m1", "trash")
    removed = await GmailRemoveLabelBlock()._remove_label(fake, "m1", "unread")
    assert removed == {"status": "Label not found", "label_id": ""}
    assert fake.called("messages.modify") == []
