import asyncio
from datetime import datetime

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockInputError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._chat_api import (
    CHAT_MESSAGES_CREATE_SCOPE,
    CHAT_MESSAGES_READONLY_SCOPE,
    CHAT_SPACES_READONLY_SCOPE,
    ChatMessage,
    build_chat_service,
    chat_error,
    require_space,
    resolve_thread,
    rfc3339,
    to_chat_message,
)

MAX_MESSAGE_BYTES = 32_000

_TEST_MESSAGE_RESOURCE = {
    "name": "spaces/AAQAl4nchPl/messages/Pq4Rs.Pq4Rs",
    "sender": {
        "name": "users/112233445566778899",
        "displayName": "Dana Lee",
        "email": "dana@example.com",
        "type": "HUMAN",
    },
    "createTime": "2026-09-24T16:05:00.123456Z",
    "text": "The launch checklist is ready for review.",
    "thread": {"name": "spaces/AAQAl4nchPl/threads/Pq4Rs"},
    "space": {"name": "spaces/AAQAl4nchPl"},
}
_TEST_SENT_RESOURCE = {
    "name": "spaces/AAQAl4nchPl/messages/Xy7Zw.Xy7Zw",
    "sender": {"name": "users/998877665544332211", "type": "HUMAN"},
    "createTime": "2026-09-25T10:00:00.000000Z",
    "text": "Thanks, reviewing now.",
    "thread": {"name": "spaces/AAQAl4nchPl/threads/Pq4Rs"},
    "threadReply": True,
    "space": {"name": "spaces/AAQAl4nchPl"},
}
_TEST_MESSAGE = to_chat_message(_TEST_MESSAGE_RESOURCE)

_NO_THREAD_REPLIES = (
    "This conversation doesn't take thread replies (the Chat API only replies in "
    "threads in named spaces), so nothing was sent. Leave thread_id empty to post "
    "to the conversation instead."
)
_THREAD_NOT_FOUND = (
    "Google Chat couldn't find that thread in this conversation, so nothing was "
    "sent. Check thread_id, or leave it empty to post to the conversation instead."
)


class GoogleChatListMessagesBlock(Block):
    """Read the messages in one Google Chat conversation."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_MESSAGES_READONLY_SCOPE]
        )
        space: str = SchemaField(
            description="The conversation to read: its space ID (spaces/...) or a Google Chat link",
            placeholder="spaces/AAAAxxxxxxx",
        )
        thread_id: str = SchemaField(
            description="Only messages in this thread (spaces/.../threads/...)",
            default="",
        )
        created_after: datetime | None = SchemaField(
            description="Only messages sent after this time",
            default=None,
        )
        created_before: datetime | None = SchemaField(
            description="Only messages sent before this time",
            default=None,
        )
        newest_first: bool = SchemaField(
            description="Return the newest messages first",
            default=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of messages to return",
            default=25,
            ge=1,
            le=1000,
            advanced=False,
        )
        page_token: str = SchemaField(
            description="Page token from a previous call, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        messages: list[ChatMessage] = SchemaField(
            description="Messages, newest first unless that is turned off"
        )
        message: ChatMessage = SchemaField(description="Each message")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more messages"
        )

    def __init__(self):
        super().__init__(
            id="5d45f2f4-46b9-43c3-864a-ea49143df60e",
            description=(
                "Read the messages in a Google Chat space, group chat or direct "
                "message, optionally only one thread or a time range. Returns each "
                "message's text, sender, time and thread."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatListMessagesBlock.Input,
            output_schema=GoogleChatListMessagesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "space": "https://chat.google.com/room/AAQAl4nchPl?cls=11",
                "max_results": 1,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("messages", [_TEST_MESSAGE]),
                ("message", _TEST_MESSAGE),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_list_messages": lambda *args, **kwargs: {
                    "messages": [_TEST_MESSAGE_RESOURCE],
                    "nextPageToken": "next-page",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        space = require_space(input_data.space, self.name, self.id)
        thread = (
            resolve_thread(input_data.thread_id, space, self.name, self.id)
            if input_data.thread_id.strip()
            else ""
        )
        query = build_list_filter(
            thread, input_data.created_after, input_data.created_before
        )
        service = build_chat_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._list_messages,
                service,
                space=space,
                query=query,
                newest_first=input_data.newest_first,
                page_size=input_data.max_results,
                page_token=input_data.page_token,
            )
        except HttpError as e:
            raise chat_error(e, self.name, self.id) from e

        messages = [to_chat_message(m) for m in result.get("messages", [])]
        yield "messages", messages
        for message in messages:
            yield "message", message
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _list_messages(
        service,
        *,
        space: str,
        query: str,
        newest_first: bool,
        page_size: int,
        page_token: str,
    ) -> dict:
        params: dict = {
            "parent": space,
            "pageSize": page_size,
            "orderBy": "createTime DESC" if newest_first else "createTime ASC",
        }
        if query:
            params["filter"] = query
        if page_token:
            params["pageToken"] = page_token
        return service.spaces().messages().list(**params).execute()


class GoogleChatSendMessageBlock(Block):
    """Send a Google Chat message as the user, optionally as a thread reply."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_MESSAGES_CREATE_SCOPE, CHAT_SPACES_READONLY_SCOPE]
        )
        space: str = SchemaField(
            description=(
                "Where to send it: a space ID (spaces/...) or a Google Chat link. "
                "Get a direct message's ID from Google Chat Start Direct Message."
            ),
            placeholder="spaces/AAAAxxxxxxx",
        )
        text: str = SchemaField(
            description=(
                "The message, up to 32,000 bytes. Markdown works for bold, italics, "
                "code, links and lists. Mention someone with "
                '`<chat-user data-email="name@example.com">`.'
            ),
        )
        thread_id: str = SchemaField(
            description=(
                "Reply in this thread (spaces/.../threads/...) instead of starting "
                "a new one. The Chat API only takes thread replies in named spaces."
            ),
            default="",
            advanced=False,
        )
        markdown: bool = SchemaField(
            description=(
                "Read the text as Markdown. Turn off to use Google Chat's own "
                "formatting instead: `*bold*`, `_italic_`, `<users/123>` mentions."
            ),
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message: ChatMessage = SchemaField(
            description="The sent message, with its ID and thread for follow-up replies"
        )

    def __init__(self):
        super().__init__(
            id="cd91f313-7cf5-49e5-9457-fc1ebddef36e",
            description=(
                "Send a Google Chat message as the user to a space, group chat or "
                "direct message, or reply in a thread of a named space. Returns the "
                "sent message."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatSendMessageBlock.Input,
            output_schema=GoogleChatSendMessageBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            is_irreversible_action=True,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "space": "spaces/AAQAl4nchPl",
                "text": "Thanks, reviewing now.",
                "thread_id": "spaces/AAQAl4nchPl/threads/Pq4Rs",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("message", to_chat_message(_TEST_SENT_RESOURCE))],
            test_mock={
                "_get_space": lambda *args, **kwargs: {
                    "name": "spaces/AAQAl4nchPl",
                    "spaceType": "SPACE",
                    "spaceThreadingState": "THREADED_MESSAGES",
                },
                "_send": lambda *args, **kwargs: _TEST_SENT_RESOURCE,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        space = require_space(input_data.space, self.name, self.id)
        self._check_text(input_data.text)
        thread = (
            resolve_thread(input_data.thread_id, space, self.name, self.id)
            if input_data.thread_id.strip()
            else ""
        )
        service = build_chat_service(credentials)
        if thread:
            await self._require_thread_replies(service, space)
        try:
            sent = await asyncio.to_thread(
                self._send,
                service,
                space=space,
                text=input_data.text,
                thread=thread,
                markdown=input_data.markdown,
            )
        except HttpError as e:
            not_found = _THREAD_NOT_FOUND if thread else None
            raise chat_error(e, self.name, self.id, not_found=not_found) from e
        yield "message", to_chat_message(sent)

    def _check_text(self, text: str) -> None:
        if not text.strip():
            message = "Write the message text."
        elif len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
            message = (
                "The message is over Google Chat's 32,000-byte limit. Shorten it "
                "or split it into several messages."
            )
        else:
            return
        raise BlockInputError(message=message, block_name=self.name, block_id=self.id)

    async def _require_thread_replies(self, service, space: str) -> None:
        """Fail before sending: without thread support, Google posts a new thread."""
        try:
            details = await asyncio.to_thread(self._get_space, service, space)
        except HttpError as e:
            raise chat_error(e, self.name, self.id) from e
        if not takes_thread_replies(details):
            raise BlockInputError(
                message=_NO_THREAD_REPLIES, block_name=self.name, block_id=self.id
            )

    @staticmethod
    def _get_space(service, space: str) -> dict:
        return service.spaces().get(name=space).execute()

    @staticmethod
    def _send(service, *, space: str, text: str, thread: str, markdown: bool) -> dict:
        body: dict = {"text": text}
        if markdown:
            body["markupSyntax"] = "MARKUP_SYNTAX_MARKDOWN"
        params: dict = {"parent": space, "body": body}
        if thread:
            body["thread"] = {"name": thread}
            params["messageReplyOption"] = "REPLY_MESSAGE_OR_FAIL"
        return service.spaces().messages().create(**params).execute()


def build_list_filter(
    thread: str, created_after: datetime | None, created_before: datetime | None
) -> str:
    """Combine the list block's thread and time filters into a Chat API filter."""
    clauses: list[str] = []
    if created_after:
        clauses.append(f'create_time > "{rfc3339(created_after)}"')
    if created_before:
        clauses.append(f'create_time < "{rfc3339(created_before)}"')
    if thread:
        clauses.append(f"thread.name = {thread}")
    return " AND ".join(clauses)


def takes_thread_replies(space: dict) -> bool:
    """Whether the Chat API can post a thread reply in this space."""
    return (
        space.get("spaceType") == "SPACE"
        and space.get("spaceThreadingState") != "UNTHREADED_MESSAGES"
    )
