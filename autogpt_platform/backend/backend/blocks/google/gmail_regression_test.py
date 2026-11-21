"""Regression tests for bugs fixed in the Gmail blocks in gmail.py.

The blocks' own test_input/test_mock cases mock the API calls away, so
these drive the real code with a fake Gmail service.
"""

import base64
import email
from email import policy
from unittest.mock import MagicMock

import pytest

from backend.blocks.google.gmail import (
    GmailCreateDraftBlock,
    GmailDraftReplyBlock,
    GmailForwardBlock,
    GmailGetThreadBlock,
    GmailReadBlock,
    GmailRemoveLabelBlock,
    GmailReplyBlock,
    GmailSendBlock,
    _build_reply_message,
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


_SUBJECT_LIMIT = 998


@pytest.mark.parametrize(
    "block_cls",
    [
        GmailSendBlock,
        GmailCreateDraftBlock,
        GmailReplyBlock,
        GmailDraftReplyBlock,
        GmailForwardBlock,
    ],
)
def test_a_subject_over_the_header_limit_is_refused_before_the_run(block_cls):
    block = block_cls()
    data = {**block.test_input, "subject": "a\n" * (_SUBJECT_LIMIT // 2)}
    assert block.input_schema.validate_data(data) is None

    data["subject"] += "a"

    assert block.input_schema.validate_data(data) == (
        f"subject: must have at most {_SUBJECT_LIMIT} characters"
    )


_LONG_SUBJECT = "word " * 2000


def _sent_subject(raw: str) -> str:
    message = email.message_from_bytes(
        base64.urlsafe_b64decode(raw), policy=policy.default
    )
    return str(message["Subject"])


@pytest.mark.asyncio
async def test_a_reply_bounds_the_subject_it_takes_from_the_original():
    """The original's subject was written by its sender, not the workflow."""
    parent = _message_with_headers(
        {"From": "ann@example.com", "Subject": _LONG_SUBJECT}
    )
    input_data = GmailReplyBlock.Input.model_construct(
        threadId="t1",
        parentMessageId="m1",
        to=[],
        cc=[],
        bcc=[],
        replyAll=False,
        subject="",
        body="Thanks",
        content_type=None,
        attachments=[],
    )

    raw, _ = await _build_reply_message(
        _FakeGmail({"messages.get": parent}), input_data, MagicMock()
    )

    subject = _sent_subject(raw)
    assert subject.startswith("Re: word word")
    assert len(subject) <= _SUBJECT_LIMIT


@pytest.mark.asyncio
async def test_a_forward_bounds_the_subject_it_takes_from_the_original():
    fake = _FakeGmail(
        {
            "messages.get": _message_with_headers(
                {"From": "ann@example.com", "Subject": _LONG_SUBJECT}
            ),
            "messages.send": {"id": "m2", "threadId": "t1"},
        }
    )
    input_data = GmailForwardBlock.Input.model_construct(
        messageId="m1",
        to=["bob@example.com"],
        cc=[],
        bcc=[],
        subject="",
        forwardMessage="",
        includeAttachments=False,
        content_type=None,
    )

    await GmailForwardBlock()._forward_message(fake, input_data, MagicMock())

    (sent,) = fake.called("messages.send")
    subject = _sent_subject(sent["body"]["raw"])
    assert subject.startswith("Fwd: word word")
    assert len(subject) <= _SUBJECT_LIMIT
