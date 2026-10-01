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
    API_SPACE_TYPES,
    CHAT_MESSAGES_READONLY_SCOPE,
    CHAT_READ_STATE_READONLY_SCOPE,
    CHAT_SPACES_READONLY_SCOPE,
    ChatMessage,
    ChatSpaceType,
    build_chat_service,
    chat_error,
    parse_space_name,
    quote_filter_value,
    rfc3339,
    to_chat_message,
    to_user_name,
)

_TEST_RESULT_RESOURCE = {
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
_TEST_RESULT = to_chat_message(_TEST_RESULT_RESOURCE)


class GoogleChatSearchMessagesBlock(Block):
    """Search Google Chat messages across the user's conversations."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [
                CHAT_MESSAGES_READONLY_SCOPE,
                CHAT_SPACES_READONLY_SCOPE,
                CHAT_READ_STATE_READONLY_SCOPE,
            ]
        )
        keywords: str = SchemaField(
            description=(
                "Words the messages must contain. Put a phrase in double quotes, "
                'e.g. "launch checklist".'
            ),
            default="",
            advanced=False,
        )
        unread_only: bool = SchemaField(
            description="Only messages the user hasn't read", default=False
        )
        mentions_me: bool = SchemaField(
            description="Only messages that @mention the user", default=False
        )
        space: str = SchemaField(
            description="Only messages in this conversation (space ID or Google Chat link)",
            default="",
        )
        sender: str = SchemaField(
            description="Only messages from this person (email address or users/...)",
            default="",
        )
        created_after: datetime | None = SchemaField(
            description="Only messages sent at or after this time", default=None
        )
        created_before: datetime | None = SchemaField(
            description="Only messages sent before this time", default=None
        )
        space_type: ChatSpaceType = SchemaField(
            description="Only messages in conversations of this kind",
            default=ChatSpaceType.ANY,
        )
        space_name_contains: str = SchemaField(
            description=(
                "Only messages in spaces whose name contains this text. Google "
                "searches the 5 best-matching spaces."
            ),
            default="",
        )
        has_link: bool = SchemaField(
            description="Only messages that contain a link", default=False
        )
        has_attachment: bool = SchemaField(
            description="Only messages with an attachment", default=False
        )
        max_results: int = SchemaField(
            description="Maximum number of messages to return",
            default=25,
            ge=1,
            le=100,
        )
        page_token: str = SchemaField(
            description="Page token from a previous search, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        messages: list[ChatMessage] = SchemaField(
            description="Matching messages, newest first"
        )
        message: ChatMessage = SchemaField(description="Each matching message")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="3a287e23-3e79-4f7f-b1e5-35a7baba3a47",
            description=(
                "Search Google Chat messages across every conversation the user is "
                "in, by keywords, sender, conversation, time, unread status, "
                "mentions, links or attachments. Returns each message's text, "
                "sender, time, conversation and thread."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatSearchMessagesBlock.Input,
            output_schema=GoogleChatSearchMessagesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "keywords": '"launch checklist"',
                "unread_only": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("messages", [_TEST_RESULT]), ("message", _TEST_RESULT)],
            test_mock={
                "_search_messages": lambda *args, **kwargs: {
                    "results": [{"message": _TEST_RESULT_RESOURCE}]
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        query = build_search_filter(input_data)
        if not query:
            raise BlockInputError(
                message="Give at least one keyword or filter to search for.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_chat_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._search_messages,
                service,
                query=query,
                page_size=input_data.max_results,
                page_token=input_data.page_token,
            )
        except HttpError as e:
            raise chat_error(e, self.name, self.id) from e

        messages = [
            to_chat_message(r["message"])
            for r in result.get("results", [])
            if "message" in r
        ]
        yield "messages", messages
        for message in messages:
            yield "message", message
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _search_messages(
        service, *, query: str, page_size: int, page_token: str
    ) -> dict:
        body: dict = {"filter": query, "pageSize": page_size}
        if page_token:
            body["pageToken"] = page_token
        return (
            service.spaces().messages().search(parent="spaces/-", body=body).execute()
        )


def build_search_filter(input_data: GoogleChatSearchMessagesBlock.Input) -> str:
    """Combine the search block's inputs into one Chat API search query."""
    clauses: list[str] = []
    if keywords := input_data.keywords.strip():
        clauses.append(keywords)
    if space := parse_space_name(input_data.space):
        clauses.append(f"space.name = {quote_filter_value(space)}")
    if sender := input_data.sender.strip():
        clauses.append(f"sender.name = {quote_filter_value(to_user_name(sender))}")
    if input_data.created_after:
        clauses.append(f'create_time >= "{rfc3339(input_data.created_after)}"')
    if input_data.created_before:
        clauses.append(f'create_time < "{rfc3339(input_data.created_before)}"')
    if input_data.space_type != ChatSpaceType.ANY:
        api_type = API_SPACE_TYPES[input_data.space_type]
        clauses.append(f'space.space_type = "{api_type}"')
    if space_name := input_data.space_name_contains.strip():
        clauses.append(f"space.display_name:{quote_filter_value(space_name)}")
    if input_data.mentions_me:
        clauses.append("annotations.user_mentions.user.name:users/me")
    if input_data.unread_only:
        clauses.append("is_unread()")
    if input_data.has_link:
        clauses.append("has_link()")
    if input_data.has_attachment:
        clauses.append("attachment:*")
    return " AND ".join(clauses)
