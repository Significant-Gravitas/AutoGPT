"""Unit tests for the Gmail blocks' request building, parsing and errors.

The blocks' own test_input/test_mock cases mock the API calls away; these
cover what those mocks skip, using a fake Gmail service.
"""

import base64
import itertools
from email.utils import getaddresses

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google import gmail
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._gmail_api import (
    Attachment,
    Email,
    GmailChangeResult,
    GmailDraft,
    GmailTarget,
    clean_label_name,
    email_from_message,
    gmail_error,
    message_format,
    to_change_result,
)
from backend.blocks.google.gmail import GmailGetThreadBlock, GmailReadBlock
from backend.blocks.google.gmail_labels import (
    GmailCreateLabelBlock,
    GmailUpdateLabelsBlock,
    LabelColor,
    LabelListVisibility,
    label_body,
)
from backend.blocks.google.gmail_messages import (
    GmailGetMessageBlock,
    GmailListDraftsBlock,
)
from backend.blocks.google.gmail_organize import (
    GmailMarkAsReadBlock,
    GmailSpamBlock,
    GmailTrashBlock,
    TrashAction,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

READONLY = "https://www.googleapis.com/auth/gmail.readonly"
METADATA = "https://www.googleapis.com/auth/gmail.metadata"

# A missing To/Cc/Bcc header parses to [""] on Pythons with the strict address
# parser (3.13, 3.12.6+, 3.11.10+) and to [] on older ones. Pinned as-is.
MISSING_HEADER = [addr.strip() for _, addr in getaddresses([""])]


class _Request:
    def __init__(self, result):
        self._result = result

    def execute(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Resource:
    def __init__(self, gmail_service: "_FakeGmail", name: str):
        self._gmail = gmail_service
        self._name = name

    def attachments(self):
        return _Resource(self._gmail, "attachments")

    def __getattr__(self, method: str):
        def call(**kwargs):
            key = f"{self._name}.{method}"
            self._gmail.calls.append((key, kwargs))
            result = self._gmail.responses[key]
            return _Request(result(**kwargs) if callable(result) else result)

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

    def drafts(self):
        return _Resource(self, "drafts")

    def labels(self):
        return _Resource(self, "labels")

    def called(self, key: str) -> list[dict]:
        return [kwargs for name, kwargs in self.calls if name == key]


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def _raw_message() -> dict:
    """A received email as messages.get(format="full") returns it."""
    return {
        "id": "18f3c5a2b4d6e8f0",
        "threadId": "18f3c5a2b4d6e8f0",
        "labelIds": ["INBOX", "UNREAD", "IMPORTANT"],
        "snippet": "Hi Bob, the Q3 numbers are attached.",
        "sizeEstimate": 48213,
        "historyId": "645006",
        "payload": {
            "mimeType": "multipart/mixed",
            "filename": "",
            "headers": [
                {"name": "From", "value": "Ann Lee <ann@example.com>"},
                {
                    "name": "To",
                    "value": '"Stone, Bob" <bob@example.com>, cara@example.com',
                },
                {"name": "Subject", "value": "Q3 numbers"},
                {"name": "Date", "value": "Thu, 25 Sep 2026 09:00:00 +0100"},
                {"name": "Message-ID", "value": "<CAB123@mail.example.com>"},
            ],
            "body": {"size": 0},
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "filename": "",
                    "body": {"size": 0},
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "filename": "",
                            "body": {
                                "size": 41,
                                "data": _b64(
                                    "Hi Bob,\r\n\r\nThe Q3 numbers are attached.\r\n"
                                ),
                            },
                        },
                        {
                            "mimeType": "text/html",
                            "filename": "",
                            "body": {
                                "size": 52,
                                "data": _b64(
                                    "<p>Hi Bob,</p><p>The Q3 numbers are attached.</p>"
                                ),
                            },
                        },
                    ],
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "q3.pdf",
                    "body": {"attachmentId": "ANGjdJ8q3", "size": 45000},
                },
            ],
        },
    }


EXPECTED_EMAIL = Email(
    threadId="18f3c5a2b4d6e8f0",
    labelIds=["INBOX", "UNREAD", "IMPORTANT"],
    id="18f3c5a2b4d6e8f0",
    subject="Q3 numbers",
    snippet="Hi Bob, the Q3 numbers are attached.",
    from_="ann@example.com",
    to=["bob@example.com", "cara@example.com"],
    cc=MISSING_HEADER,
    bcc=MISSING_HEADER,
    date="Thu, 25 Sep 2026 09:00:00 +0100",
    body="Hi Bob,\r\n\r\nThe Q3 numbers are attached.\r\n",
    sizeEstimate=48213,
    attachments=[
        Attachment(
            filename="q3.pdf",
            content_type="application/pdf",
            size=45000,
            attachment_id="ANGjdJ8q3",
        )
    ],
)


def _http_error(status: int, reason: str) -> HttpError:
    content = f'{{"error": {{"code": {status}, "message": "{reason}"}}}}'.encode()
    return HttpError(httplib2.Response({"status": status}), content)


async def _collect(outputs) -> list[tuple[str, object]]:
    return [output async for output in outputs]


def _use_fake(monkeypatch, block, fake: _FakeGmail):
    monkeypatch.setattr(block, "_build_service", lambda *args, **kwargs: fake)
    return block


# --- Parsing: one mapping for Gmail Read, Get Thread and Get Message ---


def test_missing_header_quirk_is_one_of_the_known_shapes():
    assert MISSING_HEADER in ([], [""])


def test_models_still_import_from_gmail_module():
    assert gmail.Email is Email
    assert gmail.Attachment is Attachment


@pytest.mark.asyncio
async def test_read_block_parses_a_message_as_before():
    fake = _FakeGmail(
        {
            "messages.list": {"messages": [{"id": "18f3c5a2b4d6e8f0"}]},
            "messages.get": _raw_message(),
        }
    )
    emails = await GmailReadBlock()._read_emails(fake, "from:ann", 5, [READONLY])
    assert emails == [EXPECTED_EMAIL]
    assert fake.called("messages.list") == [
        {"userId": "me", "maxResults": 5, "q": "from:ann"}
    ]


@pytest.mark.asyncio
async def test_get_thread_block_parses_its_messages_as_before():
    fake = _FakeGmail(
        {
            "threads.get": {
                "id": "18f3c5a2b4d6e8f0",
                "historyId": "645006",
                "messages": [_raw_message()],
            }
        }
    )
    thread = await GmailGetThreadBlock()._get_thread(
        fake, "18f3c5a2b4d6e8f0", [READONLY]
    )
    assert thread["messages"] == [EXPECTED_EMAIL.model_dump()]


@pytest.mark.asyncio
async def test_get_message_block_parses_the_same_email(monkeypatch):
    fake = _FakeGmail({"messages.get": _raw_message()})
    block = _use_fake(monkeypatch, GmailGetMessageBlock(), fake)
    outputs = await _collect(
        block.run(
            GmailGetMessageBlock.Input.model_validate(
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "message_id": " 18f3c5a2b4d6e8f0 ",
                }
            ),
            credentials=TEST_CREDENTIALS,
        )
    )
    assert outputs == [("email", EXPECTED_EMAIL)]
    assert fake.called("messages.get") == [
        {"userId": "me", "id": "18f3c5a2b4d6e8f0", "format": "full"}
    ]


def test_email_from_message_fills_gaps():
    email = email_from_message(
        {"id": "m1", "payload": {"headers": [{"name": "FROM", "value": "a@x.com"}]}},
        body="",
        attachments=[],
        thread_id="t1",
    )
    assert (email.threadId, email.subject, email.from_) == (
        "t1",
        "No Subject",
        "a@x.com",
    )
    assert (email.labelIds, email.snippet, email.sizeEstimate) == ([], "", 0)


@pytest.mark.parametrize(
    "scopes, expected",
    [
        ([READONLY], "full"),
        (None, "full"),
        ([READONLY, METADATA.upper()], "metadata"),
    ],
)
def test_message_format_follows_the_metadata_scope(scopes, expected):
    assert message_format(scopes) == expected


# --- Get Message ---


def _message_input(**fields) -> GmailGetMessageBlock.Input:
    return GmailGetMessageBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )


@pytest.mark.asyncio
async def test_get_message_by_message_id_header(monkeypatch):
    fake = _FakeGmail(
        {
            "messages.list": {"messages": [{"id": "18f3c5a2b4d6e8f0"}]},
            "messages.get": _raw_message(),
        }
    )
    block = _use_fake(monkeypatch, GmailGetMessageBlock(), fake)
    outputs = await _collect(
        block.run(
            _message_input(message_id="<CAB123@mail.example.com>"),
            credentials=TEST_CREDENTIALS,
        )
    )
    assert outputs == [("email", EXPECTED_EMAIL)]
    assert fake.called("messages.list") == [
        {
            "userId": "me",
            "q": "rfc822msgid:CAB123@mail.example.com",
            "maxResults": 1,
            "includeSpamTrash": True,
        }
    ]
    assert fake.called("messages.get")[0]["id"] == "18f3c5a2b4d6e8f0"


@pytest.mark.asyncio
async def test_get_message_by_unknown_message_id_header(monkeypatch):
    fake = _FakeGmail({"messages.list": {"resultSizeEstimate": 0}})
    block = _use_fake(monkeypatch, GmailGetMessageBlock(), fake)
    with pytest.raises(BlockExecutionError, match="No email in this Gmail account"):
        await _collect(
            block.run(
                _message_input(message_id="nope@example.com"),
                credentials=TEST_CREDENTIALS,
            )
        )
    assert fake.called("messages.get") == []


@pytest.mark.asyncio
async def test_get_message_by_draft_id(monkeypatch):
    draft = {"id": "r-506", "message": _raw_message()}
    fake = _FakeGmail({"drafts.get": draft})
    block = _use_fake(monkeypatch, GmailGetMessageBlock(), fake)
    credentials = TEST_CREDENTIALS.model_copy(update={"scopes": [READONLY, METADATA]})
    outputs = await _collect(
        block.run(_message_input(draft_id="r-506"), credentials=credentials)
    )
    assert outputs == [("email", EXPECTED_EMAIL), ("draft_id", "r-506")]
    assert fake.called("drafts.get") == [
        {"userId": "me", "id": "r-506", "format": "metadata"}
    ]


@pytest.mark.parametrize(
    "fields", [{}, {"message_id": "  "}, {"message_id": "m1", "draft_id": "r-1"}]
)
@pytest.mark.asyncio
async def test_get_message_needs_exactly_one_id(monkeypatch, fields):
    block = _use_fake(monkeypatch, GmailGetMessageBlock(), _FakeGmail({}))
    with pytest.raises(BlockInputError, match="either a message ID or a draft ID"):
        await _collect(
            block.run(_message_input(**fields), credentials=TEST_CREDENTIALS)
        )


@pytest.mark.asyncio
async def test_get_message_maps_a_missing_draft(monkeypatch):
    fake = _FakeGmail(
        {"drafts.get": _http_error(404, "Requested entity was not found.")}
    )
    block = _use_fake(monkeypatch, GmailGetMessageBlock(), fake)
    with pytest.raises(BlockExecutionError, match="couldn't find that draft"):
        await _collect(
            block.run(_message_input(draft_id="r-404"), credentials=TEST_CREDENTIALS)
        )


# --- List Drafts ---


@pytest.mark.asyncio
async def test_list_drafts_reads_each_draft_and_pages(monkeypatch):
    fake = _FakeGmail(
        {
            "drafts.list": {
                "drafts": [
                    {"id": "r-1", "message": {"id": "m1", "threadId": "m1"}},
                    {"id": "r-2", "message": {"id": "m2", "threadId": "m2"}},
                ],
                "nextPageToken": "page-2",
            },
            "drafts.get": lambda **kwargs: {
                "id": kwargs["id"],
                "message": _raw_message(),
            },
        }
    )
    block = _use_fake(monkeypatch, GmailListDraftsBlock(), fake)
    input_data = GmailListDraftsBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "query": "to:bob@example.com",
            "max_results": 2,
            "page_token": "page-1",
        }
    )
    outputs = await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))

    drafts = [
        GmailDraft(id="r-1", email=EXPECTED_EMAIL),
        GmailDraft(id="r-2", email=EXPECTED_EMAIL),
    ]
    assert outputs == [
        ("drafts", drafts),
        ("draft", drafts[0]),
        ("draft", drafts[1]),
        ("next_page_token", "page-2"),
    ]
    assert fake.called("drafts.list") == [
        {
            "userId": "me",
            "maxResults": 2,
            "q": "to:bob@example.com",
            "pageToken": "page-1",
        }
    ]
    assert [call["id"] for call in fake.called("drafts.get")] == ["r-1", "r-2"]


@pytest.mark.asyncio
async def test_list_drafts_with_no_drafts(monkeypatch):
    fake = _FakeGmail({"drafts.list": {"resultSizeEstimate": 0}})
    block = _use_fake(monkeypatch, GmailListDraftsBlock(), fake)
    input_data = GmailListDraftsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT}
    )
    outputs = await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))
    assert outputs == [("drafts", [])]
    assert fake.called("drafts.list") == [{"userId": "me", "maxResults": 20}]


# --- Trash, spam and read state ---


def _thread_resource(*label_sets: list[str]) -> dict:
    return {
        "id": "t1",
        "messages": [
            {"id": f"m{index}", "threadId": "t1", "labelIds": labels}
            for index, labels in enumerate(label_sets)
        ],
    }


@pytest.mark.parametrize(
    "target, action, expected_call",
    [
        (GmailTarget.MESSAGE, TrashAction.TRASH, "messages.trash"),
        (GmailTarget.MESSAGE, TrashAction.RESTORE, "messages.untrash"),
        (GmailTarget.THREAD, TrashAction.TRASH, "threads.trash"),
        (GmailTarget.THREAD, TrashAction.RESTORE, "threads.untrash"),
    ],
)
def test_trash_calls_the_matching_endpoint(target, action, expected_call):
    fake = _FakeGmail({expected_call: {"id": "x1"}})
    GmailTrashBlock._trash(fake, target, "x1", action)
    assert fake.calls == [(expected_call, {"userId": "me", "id": "x1"})]


@pytest.mark.asyncio
async def test_trash_thread_reports_the_labels_of_all_its_messages(monkeypatch):
    fake = _FakeGmail({"threads.trash": _thread_resource(["TRASH", "SENT"], ["TRASH"])})
    block = _use_fake(monkeypatch, GmailTrashBlock(), fake)
    input_data = GmailTrashBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "message_or_thread_id": "t1",
            "target": "thread",
        }
    )
    outputs = await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))
    assert outputs == [
        (
            "result",
            GmailChangeResult(id="t1", thread_id="t1", label_ids=["TRASH", "SENT"]),
        )
    ]


@pytest.mark.parametrize(
    "block_cls, fields, add, remove",
    [
        (GmailSpamBlock, {}, ["SPAM"], ["INBOX"]),
        (GmailSpamBlock, {"action": "not_spam"}, ["INBOX"], ["SPAM"]),
        (GmailMarkAsReadBlock, {}, [], ["UNREAD"]),
        (GmailMarkAsReadBlock, {"mark_as": "unread"}, ["UNREAD"], []),
    ],
)
@pytest.mark.asyncio
async def test_spam_and_read_state_modify_the_right_labels(
    monkeypatch, block_cls, fields, add, remove
):
    fake = _FakeGmail({"threads.modify": _thread_resource(["INBOX"])})
    block = _use_fake(monkeypatch, block_cls(), fake)
    input_data = block_cls.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "message_or_thread_id": "t1",
            "target": "thread",
            **fields,
        }
    )
    await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))
    assert fake.called("threads.modify") == [
        {
            "userId": "me",
            "id": "t1",
            "body": {"addLabelIds": add, "removeLabelIds": remove},
        }
    ]


@pytest.mark.asyncio
async def test_changes_need_an_id(monkeypatch):
    block = _use_fake(monkeypatch, GmailMarkAsReadBlock(), _FakeGmail({}))
    input_data = GmailMarkAsReadBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "message_or_thread_id": " "}
    )
    with pytest.raises(BlockInputError, match="Give the ID"):
        await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))


def test_change_result_for_a_message():
    result = to_change_result(
        {"id": "m1", "threadId": "t1", "labelIds": ["INBOX"]}, GmailTarget.MESSAGE
    )
    assert result == GmailChangeResult(id="m1", thread_id="t1", label_ids=["INBOX"])


# --- Update Labels ---

_LABELS = [
    {"id": "INBOX", "name": "INBOX", "type": "system"},
    {"id": "IMPORTANT", "name": "IMPORTANT", "type": "system"},
    {"id": "Label_7", "name": "Clients/Acme", "type": "user"},
    {"id": "Label_8", "name": "Projects", "type": "user"},
]


def _label_service(modify_result: dict | None = None) -> _FakeGmail:
    new_ids = itertools.count(100)
    return _FakeGmail(
        {
            "labels.list": {"labels": [dict(label) for label in _LABELS]},
            "labels.create": lambda **kwargs: {
                "id": f"Label_{next(new_ids)}",
                "type": "user",
                **kwargs["body"],
            },
            "messages.modify": modify_result or {"id": "m1", "threadId": "t1"},
            "threads.modify": {"id": "t1", "messages": []},
            "messages.get": {"id": "m1", "threadId": "t1", "labelIds": ["INBOX"]},
        }
    )


def test_update_labels_resolves_ids_names_and_system_labels():
    fake = _label_service()
    _, created = GmailUpdateLabelsBlock._update_labels(
        fake,
        GmailTarget.MESSAGE,
        "m1",
        ["Label_8", "clients/acme", "starred", "Clients/Acme"],
        ["inbox"],
    )
    assert created == []
    assert fake.called("labels.create") == []
    assert fake.called("messages.modify") == [
        {
            "userId": "me",
            "id": "m1",
            "body": {
                "addLabelIds": ["Label_8", "Label_7", "STARRED"],
                "removeLabelIds": ["INBOX"],
            },
        }
    ]


def test_update_labels_creates_missing_user_labels_with_their_parents():
    fake = _label_service()
    _, created = GmailUpdateLabelsBlock._update_labels(
        fake, GmailTarget.MESSAGE, "m1", ["Projects/Alpha", "Travel/2026/Rome"], []
    )
    assert created == ["Projects/Alpha", "Travel", "Travel/2026", "Travel/2026/Rome"]
    assert [call["body"]["name"] for call in fake.called("labels.create")] == created
    add_ids = fake.called("messages.modify")[0]["body"]["addLabelIds"]
    assert add_ids == ["Label_100", "Label_103"]


def test_update_labels_never_creates_system_labels():
    fake = _label_service()
    GmailUpdateLabelsBlock._update_labels(
        fake, GmailTarget.THREAD, "t1", ["unread", "CATEGORY_PROMOTIONS"], []
    )
    assert fake.called("labels.create") == []
    assert fake.called("threads.modify")[0]["body"]["addLabelIds"] == [
        "UNREAD",
        "CATEGORY_PROMOTIONS",
    ]


def test_update_labels_ignores_unknown_labels_to_remove():
    fake = _label_service()
    resource, created = GmailUpdateLabelsBlock._update_labels(
        fake, GmailTarget.MESSAGE, "m1", [], ["Newsletters"]
    )
    assert fake.called("messages.modify") == []
    assert fake.called("messages.get") == [
        {"userId": "me", "id": "m1", "format": "minimal"}
    ]
    assert (resource["labelIds"], created) == (["INBOX"], [])


@pytest.mark.parametrize(
    "add, remove, message",
    [
        ([], [], "at least one label"),
        ([" ", "/"], [], "at least one label"),
        (["Clients/Acme"], ["clients/acme"], "same label: Clients/Acme"),
    ],
)
@pytest.mark.asyncio
async def test_update_labels_checks_its_input(monkeypatch, add, remove, message):
    block = _use_fake(monkeypatch, GmailUpdateLabelsBlock(), _FakeGmail({}))
    input_data = GmailUpdateLabelsBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "message_or_thread_id": "m1",
            "add_labels": add,
            "remove_labels": remove,
        }
    )
    with pytest.raises(BlockInputError, match=message):
        await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))


@pytest.mark.asyncio
async def test_update_labels_outputs_created_labels_only_when_any(monkeypatch):
    fake = _label_service({"id": "m1", "threadId": "t1", "labelIds": ["Label_7"]})
    block = _use_fake(monkeypatch, GmailUpdateLabelsBlock(), fake)
    input_data = GmailUpdateLabelsBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "message_or_thread_id": "m1",
            "add_labels": ["Clients/Acme"],
        }
    )
    outputs = await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))
    assert outputs == [
        ("result", GmailChangeResult(id="m1", thread_id="t1", label_ids=["Label_7"]))
    ]


# --- Create Label ---


def _create_input(**fields) -> GmailCreateLabelBlock.Input:
    return GmailCreateLabelBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "name": "x", **fields}
    )


def test_label_body_maps_color_and_visibility():
    assert label_body(_create_input()) == {
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
    }
    assert label_body(
        _create_input(
            color=LabelColor.DARK_GREEN,
            show_in_label_list=LabelListVisibility.SHOW_IF_UNREAD,
            show_in_message_list=False,
        )
    ) == {
        "labelListVisibility": "labelShowIfUnread",
        "messageListVisibility": "hide",
        "color": {"backgroundColor": "#076239", "textColor": "#ffffff"},
    }


def test_create_label_returns_an_existing_label_without_creating():
    fake = _label_service()
    label, created = GmailCreateLabelBlock._create_label(fake, "clients/ACME", {}, True)
    assert (label["id"], created) == ("Label_7", False)
    assert fake.called("labels.create") == []


def test_create_label_creates_missing_parents_first():
    fake = _label_service()
    body = {"labelListVisibility": "labelShow", "messageListVisibility": "show"}
    label, created = GmailCreateLabelBlock._create_label(
        fake, "Projects/Alpha/Sprint 1", body, True
    )
    assert created is True
    assert [call["body"] for call in fake.called("labels.create")] == [
        {"name": "Projects/Alpha"},
        {**body, "name": "Projects/Alpha/Sprint 1"},
    ]
    assert label["name"] == "Projects/Alpha/Sprint 1"


def test_create_label_can_skip_parents():
    fake = _label_service()
    GmailCreateLabelBlock._create_label(fake, "Travel/Rome", {}, False)
    assert [call["body"]["name"] for call in fake.called("labels.create")] == [
        "Travel/Rome"
    ]


@pytest.mark.parametrize(
    "raw, cleaned",
    [(" Projects / Alpha ", "Projects/Alpha"), ("Receipts", "Receipts"), (" / ", "")],
)
def test_clean_label_name(raw, cleaned):
    assert clean_label_name(raw) == cleaned


@pytest.mark.asyncio
async def test_create_label_needs_a_name(monkeypatch):
    block = _use_fake(monkeypatch, GmailCreateLabelBlock(), _FakeGmail({}))
    with pytest.raises(BlockInputError, match="Give the label a name"):
        await _collect(
            block.run(_create_input(name=" / "), credentials=TEST_CREDENTIALS)
        )


# --- Errors ---


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (404, "Requested entity was not found.", "couldn't find that thread"),
        (400, "Invalid id value", "isn't a valid Gmail thread ID"),
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (500, "Backend Error", "Gmail API error 500: Backend Error"),
    ],
)
def test_gmail_error_messages(status: int, reason: str, expected: str):
    error = gmail_error(_http_error(status, reason), "block", "id", "thread")
    assert expected in str(error)
