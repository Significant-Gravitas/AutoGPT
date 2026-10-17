"""Unit tests for the Google Chat blocks' request building, parsing and errors.

The blocks' own test_input/test_mock cases mock the API calls away; these
cover what those mocks skip.
"""

import json
from datetime import datetime, timedelta, timezone

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._chat_api import (
    ChatSpaceType,
    chat_error,
    parse_space_name,
    quote_filter_value,
    resolve_thread,
    rfc3339,
    to_chat_message,
    to_chat_space,
    to_user_name,
)
from backend.blocks.google.chat_direct_messages import (
    GoogleChatFindDirectMessageBlock,
    GoogleChatStartDirectMessageBlock,
)
from backend.blocks.google.chat_message_search import (
    GoogleChatSearchMessagesBlock,
    build_search_filter,
)
from backend.blocks.google.chat_messages import (
    GoogleChatListMessagesBlock,
    GoogleChatSendMessageBlock,
    build_list_filter,
    takes_thread_replies,
)
from backend.blocks.google.chat_spaces import (
    GoogleChatFindGroupChatsBlock,
    GoogleChatListSpacesBlock,
    GoogleChatSearchSpacesBlock,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

SPACE = "spaces/AAQAl4nchPl"
THREAD = f"{SPACE}/threads/Pq4Rs"


class _Request:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeChat:
    """Records calls made through service.spaces()[.messages()].<method>(...)."""

    def __init__(self, result: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._result = result or {}
        self._resource = "spaces"

    def spaces(self):
        self._resource = "spaces"
        return self

    def messages(self):
        self._resource = "spaces.messages"
        return self

    def __getattr__(self, method: str):
        def call(**kwargs):
            self.calls.append((f"{self._resource}.{method}", kwargs))
            return _Request(self._result)

        return call


def _http_error(status: int, message: str) -> HttpError:
    content = json.dumps({"error": {"code": status, "message": message}}).encode()
    return HttpError(httplib2.Response({"status": status}), content)


async def _run(block, **fields) -> list[tuple[str, object]]:
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )
    return [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]


# Parsing helpers


@pytest.mark.parametrize(
    "value, expected",
    [
        (SPACE, SPACE),
        ("AAQAl4nchPl", SPACE),
        (" https://chat.google.com/room/AAQAl4nchPl?cls=11 ", SPACE),
        ("https://chat.google.com/dm/DMdAna1234", "spaces/DMdAna1234"),
        ("https://mail.google.com/mail/u/0/#chat/space/AAAAAAAAA", "spaces/AAAAAAAAA"),
        (f"{SPACE}/messages/Pq4Rs.Pq4Rs", SPACE),
        ("   ", ""),
    ],
)
def test_parse_space_name(value: str, expected: str):
    assert parse_space_name(value) == expected


def test_resolve_thread_accepts_ids_and_names_in_the_space():
    assert resolve_thread("Pq4Rs", SPACE, "block", "id") == THREAD
    assert resolve_thread(f" {THREAD} ", SPACE, "block", "id") == THREAD


def test_resolve_thread_rejects_a_thread_from_another_space():
    with pytest.raises(BlockInputError, match="different conversation"):
        resolve_thread("spaces/OTHER/threads/Pq4Rs", SPACE, "block", "id")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("dana@example.com", "users/dana@example.com"),
        (" 112233445566778899 ", "users/112233445566778899"),
        ("users/112233445566778899", "users/112233445566778899"),
    ],
)
def test_to_user_name(value: str, expected: str):
    assert to_user_name(value) == expected


def test_rfc3339_reads_times_without_a_zone_as_utc():
    assert rfc3339(datetime(2026, 9, 1, 8, 30)) == "2026-09-01T08:30:00+00:00"
    eastern = timezone(timedelta(hours=-4))
    assert rfc3339(datetime(2026, 9, 1, tzinfo=eastern)) == "2026-09-01T00:00:00-04:00"


def test_quote_filter_value_escapes_quotes_and_backslashes():
    assert quote_filter_value('Q4 "launch" \\ plan') == '"Q4 \\"launch\\" \\\\ plan"'


# Response parsing


def test_to_chat_space_maps_types_counts_and_details():
    space = to_chat_space(
        {
            "name": SPACE,
            "displayName": "Launch planning",
            "spaceType": "GROUP_CHAT",
            "spaceUri": "https://chat.google.com/room/AAQAl4nchPl",
            "membershipCount": {"joinedGroupCount": 1},
            "spaceDetails": {"description": "Q4"},
        }
    )
    assert space.space_type == "group_chat"
    assert space.member_count == 0
    assert space.description == "Q4"

    bare = to_chat_space({"name": SPACE, "spaceType": "SOMETHING_NEW"})
    assert (bare.space_type, bare.member_count, bare.display_name) == (
        "something_new",
        None,
        "",
    )


def test_to_chat_message_maps_sender_thread_and_attachments():
    message = to_chat_message(
        {
            "name": f"{SPACE}/messages/Pq4Rs.Pq4Rs",
            "sender": {"name": "users/app", "displayName": "Bot", "type": "BOT"},
            "text": "Report attached",
            "thread": {"name": THREAD},
            "threadReply": True,
            "attachment": [
                {
                    "name": f"{SPACE}/messages/Pq4Rs.Pq4Rs/attachments/A1",
                    "contentName": "report.pdf",
                    "contentType": "application/pdf",
                    "driveDataRef": {"driveFileId": "1AbCdEf"},
                }
            ],
        }
    )
    assert message.space_id == SPACE
    assert message.thread_id == THREAD
    assert message.is_thread_reply is True
    assert message.sender is not None and message.sender.type == "app"
    assert message.attachments[0].file_name == "report.pdf"
    assert message.attachments[0].drive_file_id == "1AbCdEf"

    minimal = to_chat_message({"name": f"{SPACE}/messages/X"})
    assert (minimal.sender, minimal.thread_id, minimal.text) == (None, None, "")


# Filters


def test_build_list_filter_combines_time_range_and_thread():
    assert build_list_filter("", None, None) == ""
    query = build_list_filter(
        THREAD,
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 24, 12, tzinfo=timezone.utc),
    )
    assert query == (
        'create_time > "2026-09-01T00:00:00+00:00" AND '
        'create_time < "2026-09-24T12:00:00+00:00" AND '
        f"thread.name = {THREAD}"
    )


def _search_input(**fields) -> GoogleChatSearchMessagesBlock.Input:
    return GoogleChatSearchMessagesBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )


def test_build_search_filter_combines_every_filter():
    query = build_search_filter(
        _search_input(
            keywords=' "launch checklist" ',
            space="https://chat.google.com/room/AAQAl4nchPl",
            sender="dana@example.com",
            created_after=datetime(2026, 9, 1, tzinfo=timezone.utc),
            created_before=datetime(2026, 9, 24, 12, tzinfo=timezone.utc),
            space_type=ChatSpaceType.SPACE,
            space_name_contains="Launch plan",
            mentions_me=True,
            unread_only=True,
            has_link=True,
            has_attachment=True,
        )
    )
    assert query == (
        '"launch checklist" AND space.name = "spaces/AAQAl4nchPl" AND '
        'sender.name = "users/dana@example.com" AND '
        'create_time >= "2026-09-01T00:00:00+00:00" AND '
        'create_time < "2026-09-24T12:00:00+00:00" AND '
        'space.space_type = "SPACE" AND space.display_name:"Launch plan" AND '
        "annotations.user_mentions.user.name:users/me AND is_unread() AND "
        "has_link() AND attachment:*"
    )


def test_build_search_filter_is_empty_without_filters():
    assert build_search_filter(_search_input(keywords="  ")) == ""


# Requests


def test_list_spaces_request_filters_by_type():
    service = _FakeChat()
    GoogleChatListSpacesBlock._list_spaces(
        service,
        space_type=ChatSpaceType.DIRECT_MESSAGE,
        page_size=50,
        page_token="tok",
    )
    GoogleChatListSpacesBlock._list_spaces(
        service, space_type=ChatSpaceType.ANY, page_size=100, page_token=""
    )
    assert service.calls == [
        (
            "spaces.list",
            {
                "pageSize": 50,
                "filter": 'space_type = "DIRECT_MESSAGE"',
                "pageToken": "tok",
            },
        ),
        ("spaces.list", {"pageSize": 100}),
    ]


def test_search_spaces_request_searches_named_spaces_without_admin_access():
    service = _FakeChat()
    GoogleChatSearchSpacesBlock._search_spaces(
        service, name_query="launch plan", page_size=25
    )
    assert service.calls == [
        (
            "spaces.search",
            {
                "query": 'space_type = "SPACE" AND display_name:"launch plan"',
                "pageSize": 25,
                "useAdminAccess": False,
            },
        )
    ]


def test_find_group_chats_request_asks_for_full_space_details():
    service = _FakeChat()
    GoogleChatFindGroupChatsBlock._find_group_chats(
        service, users=["users/a@example.com"], page_size=10, page_token=""
    )
    assert service.calls == [
        (
            "spaces.findGroupChats",
            {
                "users": ["users/a@example.com"],
                "pageSize": 10,
                "spaceView": "SPACE_VIEW_EXPANDED",
            },
        )
    ]


def test_direct_message_requests():
    service = _FakeChat()
    GoogleChatFindDirectMessageBlock._find_direct_message(
        service, "users/dana@example.com"
    )
    GoogleChatStartDirectMessageBlock._set_up_direct_message(
        service, "users/dana@example.com"
    )
    assert service.calls == [
        ("spaces.findDirectMessage", {"name": "users/dana@example.com"}),
        (
            "spaces.setup",
            {
                "body": {
                    "space": {"spaceType": "DIRECT_MESSAGE", "singleUserBotDm": False},
                    "memberships": [
                        {"member": {"name": "users/dana@example.com", "type": "HUMAN"}}
                    ],
                }
            },
        ),
    ]


def test_list_messages_request_orders_and_filters():
    service = _FakeChat()
    GoogleChatListMessagesBlock._list_messages(
        service,
        space=SPACE,
        query=f"thread.name = {THREAD}",
        newest_first=True,
        page_size=25,
        page_token="",
    )
    GoogleChatListMessagesBlock._list_messages(
        service, space=SPACE, query="", newest_first=False, page_size=5, page_token="t"
    )
    assert service.calls == [
        (
            "spaces.messages.list",
            {
                "parent": SPACE,
                "pageSize": 25,
                "orderBy": "createTime DESC",
                "filter": f"thread.name = {THREAD}",
            },
        ),
        (
            "spaces.messages.list",
            {
                "parent": SPACE,
                "pageSize": 5,
                "orderBy": "createTime ASC",
                "pageToken": "t",
            },
        ),
    ]


def test_search_messages_request_searches_every_space():
    service = _FakeChat()
    GoogleChatSearchMessagesBlock._search_messages(
        service, query="is_unread()", page_size=25, page_token="tok"
    )
    assert service.calls == [
        (
            "spaces.messages.search",
            {
                "parent": "spaces/-",
                "body": {"filter": "is_unread()", "pageSize": 25, "pageToken": "tok"},
            },
        )
    ]


def test_send_request_replies_in_thread_or_fails():
    service = _FakeChat()
    GoogleChatSendMessageBlock._send(
        service, space=SPACE, text="**Done**", thread=THREAD, markdown=True
    )
    GoogleChatSendMessageBlock._send(
        service, space=SPACE, text="*Done*", thread="", markdown=False
    )
    assert service.calls == [
        (
            "spaces.messages.create",
            {
                "parent": SPACE,
                "body": {
                    "text": "**Done**",
                    "markupSyntax": "MARKUP_SYNTAX_MARKDOWN",
                    "thread": {"name": THREAD},
                },
                "messageReplyOption": "REPLY_MESSAGE_OR_FAIL",
            },
        ),
        ("spaces.messages.create", {"parent": SPACE, "body": {"text": "*Done*"}}),
    ]


# Block behaviour


@pytest.mark.parametrize(
    "space, expected",
    [
        ({"spaceType": "SPACE", "spaceThreadingState": "THREADED_MESSAGES"}, True),
        ({"spaceType": "SPACE", "spaceThreadingState": "GROUPED_MESSAGES"}, True),
        ({"spaceType": "SPACE", "spaceThreadingState": "UNTHREADED_MESSAGES"}, False),
        ({"spaceType": "DIRECT_MESSAGE"}, False),
        ({"spaceType": "GROUP_CHAT"}, False),
    ],
)
def test_takes_thread_replies(space: dict, expected: bool):
    assert takes_thread_replies(space) is expected


@pytest.mark.asyncio
async def test_send_refuses_a_thread_reply_before_sending(monkeypatch):
    block = GoogleChatSendMessageBlock()
    sent: list[dict] = []
    monkeypatch.setattr(
        block,
        "_get_space",
        lambda service, space: {"name": space, "spaceType": "DIRECT_MESSAGE"},
    )
    monkeypatch.setattr(block, "_send", lambda *a, **kw: sent.append(kw) or {})
    with pytest.raises(BlockInputError, match="doesn't take thread replies"):
        await _run(block, space="spaces/DMdAna1234", text="Hi", thread_id="T1")
    assert sent == []


@pytest.mark.asyncio
async def test_send_reports_a_missing_thread(monkeypatch):
    block = GoogleChatSendMessageBlock()
    monkeypatch.setattr(
        block, "_get_space", lambda service, space: {"spaceType": "SPACE"}
    )

    def fail(*args, **kwargs):
        raise _http_error(404, "Thread not found.")

    monkeypatch.setattr(block, "_send", fail)
    with pytest.raises(BlockExecutionError, match="couldn't find that thread"):
        await _run(block, space=SPACE, text="Hi", thread_id=THREAD)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected", [("   ", "Write the message"), ("x" * 32_001, "32,000-byte")]
)
async def test_send_checks_the_text(text: str, expected: str):
    with pytest.raises(BlockInputError, match=expected):
        await _run(GoogleChatSendMessageBlock(), space=SPACE, text=text)


@pytest.mark.asyncio
async def test_search_messages_needs_a_filter():
    with pytest.raises(BlockInputError, match="at least one keyword or filter"):
        await _run(GoogleChatSearchMessagesBlock())


@pytest.mark.asyncio
async def test_find_group_chats_dedupes_people(monkeypatch):
    block = GoogleChatFindGroupChatsBlock()
    seen: list[list[str]] = []

    def find(service, *, users, page_size, page_token):
        seen.append(users)
        return {}

    monkeypatch.setattr(block, "_find_group_chats", find)
    outputs = await _run(
        block, people=["a@example.com", " a@example.com", "", "users/123"]
    )
    assert seen == [["users/a@example.com", "users/123"]]
    assert outputs == [("spaces", [])]

    with pytest.raises(BlockInputError, match="between 1 and 49"):
        await _run(block, people=[f"p{i}@example.com" for i in range(50)])


@pytest.mark.asyncio
async def test_find_direct_message_explains_a_missing_dm(monkeypatch):
    block = GoogleChatFindDirectMessageBlock()

    def fail(service, user):
        raise _http_error(404, "Direct message not found.")

    monkeypatch.setattr(block, "_find_direct_message", fail)
    with pytest.raises(BlockExecutionError, match="Start Direct Message"):
        await _run(block, person="dana@example.com")


@pytest.mark.asyncio
async def test_empty_inputs_are_rejected():
    with pytest.raises(BlockInputError, match="words from the space"):
        await _run(GoogleChatSearchSpacesBlock(), name_query=" ")
    with pytest.raises(BlockInputError, match="email address or Chat user ID"):
        await _run(GoogleChatStartDirectMessageBlock(), person=" ")
    with pytest.raises(BlockInputError, match="space ID"):
        await _run(GoogleChatListMessagesBlock(), space=" ")


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (
            404,
            "Google Chat app not found. To create a Chat app, you must turn on "
            "the Chat API and configure the app in the Google Cloud console.",
            "needs a configured Chat app",
        ),
        (
            403,
            "Google Chat API has not been used in project 123 before or it is disabled.",
            "isn't enabled",
        ),
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (404, "Requested entity was not found.", "couldn't find that conversation"),
        (400, "Invalid filter.", "rejected the request: Invalid filter."),
        (500, "Backend Error", "Google Chat API error 500"),
    ],
)
def test_chat_error_messages(status: int, reason: str, expected: str):
    assert expected in str(chat_error(_http_error(status, reason), "block", "id"))


def test_chat_error_uses_the_block_not_found_message():
    error = chat_error(_http_error(404, "Not found."), "b", "i", not_found="No DM yet.")
    assert str(error) == "No DM yet."
