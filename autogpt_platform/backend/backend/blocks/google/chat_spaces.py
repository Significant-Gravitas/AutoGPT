import asyncio

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
    CHAT_MEMBERSHIPS_READONLY_SCOPE,
    CHAT_SPACES_READONLY_SCOPE,
    ChatSpace,
    ChatSpaceType,
    build_chat_service,
    chat_error,
    quote_filter_value,
    to_chat_space,
    to_user_name,
)

_TEST_SPACE_RESOURCE = {
    "name": "spaces/AAQAl4nchPl",
    "displayName": "Launch planning",
    "spaceType": "SPACE",
    "spaceUri": "https://chat.google.com/room/AAQAl4nchPl?cls=11",
    "membershipCount": {"joinedDirectHumanUserCount": 8},
    "spaceDetails": {"description": "Coordinating the Q4 launch"},
    "lastActiveTime": "2026-09-24T16:05:00Z",
}
_TEST_GROUP_CHAT_RESOURCE = {
    "name": "spaces/GCtRio5678",
    "spaceType": "GROUP_CHAT",
    "spaceUri": "https://chat.google.com/room/GCtRio5678?cls=11",
    "membershipCount": {"joinedDirectHumanUserCount": 3},
    "lastActiveTime": "2026-09-22T14:40:00Z",
}
_TEST_SPACE = to_chat_space(_TEST_SPACE_RESOURCE)
_TEST_GROUP_CHAT = to_chat_space(_TEST_GROUP_CHAT_RESOURCE)


class GoogleChatListSpacesBlock(Block):
    """List the Google Chat conversations the user is a member of."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_SPACES_READONLY_SCOPE]
        )
        space_type: ChatSpaceType = SchemaField(
            description="Only conversations of this kind",
            default=ChatSpaceType.ANY,
            advanced=False,
        )
        max_results: int = SchemaField(
            description="Maximum number of conversations to return",
            default=100,
            ge=1,
            le=1000,
        )
        page_token: str = SchemaField(
            description="Page token from a previous call, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        spaces: list[ChatSpace] = SchemaField(
            description="Conversations the user is in"
        )
        space: ChatSpace = SchemaField(description="Each conversation")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="ff5ea590-59d6-4da8-86f1-e7442e26be0e",
            description=(
                "List the Google Chat conversations the user is in (named spaces, "
                "group chats and direct messages) with their IDs, names, types "
                "and member counts."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatListSpacesBlock.Input,
            output_schema=GoogleChatListSpacesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("spaces", [_TEST_SPACE, _TEST_GROUP_CHAT]),
                ("space", _TEST_SPACE),
                ("space", _TEST_GROUP_CHAT),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_list_spaces": lambda *args, **kwargs: {
                    "spaces": [_TEST_SPACE_RESOURCE, _TEST_GROUP_CHAT_RESOURCE],
                    "nextPageToken": "next-page",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_chat_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._list_spaces,
                service,
                space_type=input_data.space_type,
                page_size=input_data.max_results,
                page_token=input_data.page_token,
            )
        except HttpError as e:
            raise chat_error(e, self.name, self.id) from e

        spaces = [to_chat_space(s) for s in result.get("spaces", [])]
        yield "spaces", spaces
        for space in spaces:
            yield "space", space
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _list_spaces(
        service, *, space_type: ChatSpaceType, page_size: int, page_token: str
    ) -> dict:
        params: dict = {"pageSize": page_size}
        if space_type != ChatSpaceType.ANY:
            params["filter"] = f'space_type = "{API_SPACE_TYPES[space_type]}"'
        if page_token:
            params["pageToken"] = page_token
        return service.spaces().list(**params).execute()


class GoogleChatSearchSpacesBlock(Block):
    """Find named Google Chat spaces by words in their name."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_SPACES_READONLY_SCOPE]
        )
        name_query: str = SchemaField(
            description=(
                "Words from the space's name. Each word matches the start of a "
                "word in the name, in any order: 'launch plan' finds 'Q4 Launch "
                "Planning'."
            ),
            placeholder="launch plan",
        )
        max_results: int = SchemaField(
            description="Maximum number of spaces to return",
            default=25,
            ge=1,
            le=100,
        )

    class Output(BlockSchemaOutput):
        spaces: list[ChatSpace] = SchemaField(description="Matching spaces")
        space: ChatSpace = SchemaField(description="Each matching space")

    def __init__(self):
        super().__init__(
            id="d0c7ed29-607a-4a61-add3-24c4217f4bb6",
            description=(
                "Find named Google Chat spaces the user is in by words in the "
                "space name, and get their IDs. Direct messages and group chats "
                "have no name: use Find Direct Message or Find Group Chats."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatSearchSpacesBlock.Input,
            output_schema=GoogleChatSearchSpacesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "name_query": "launch plan",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("spaces", [_TEST_SPACE]), ("space", _TEST_SPACE)],
            test_mock={
                "_search_spaces": lambda *args, **kwargs: {
                    "results": [{"space": _TEST_SPACE_RESOURCE}]
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        name_query = input_data.name_query.strip()
        if not name_query:
            raise BlockInputError(
                message="Give some words from the space's name to search for.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_chat_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._search_spaces,
                service,
                name_query=name_query,
                page_size=input_data.max_results,
            )
        except HttpError as e:
            raise chat_error(e, self.name, self.id) from e

        spaces = [
            to_chat_space(r["space"]) for r in result.get("results", []) if "space" in r
        ]
        yield "spaces", spaces
        for space in spaces:
            yield "space", space

    @staticmethod
    def _search_spaces(service, *, name_query: str, page_size: int) -> dict:
        query = (
            f'space_type = "SPACE" AND display_name:{quote_filter_value(name_query)}'
        )
        return (
            service.spaces()
            .search(query=query, pageSize=page_size, useAdminAccess=False)
            .execute()
        )


class GoogleChatFindGroupChatsBlock(Block):
    """Find group chats made up of the user and exactly the given people."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_MEMBERSHIPS_READONLY_SCOPE, CHAT_SPACES_READONLY_SCOPE]
        )
        people: list[str] = SchemaField(
            description=(
                "Email addresses or Chat user IDs (users/...) of everyone else in "
                "the group chat, not including you. Up to 49."
            ),
        )
        max_results: int = SchemaField(
            description="Maximum number of group chats to return",
            default=10,
            ge=1,
            le=30,
        )
        page_token: str = SchemaField(
            description="Page token from a previous call, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        spaces: list[ChatSpace] = SchemaField(description="Matching group chats")
        space: ChatSpace = SchemaField(description="Each matching group chat")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="886854b1-52d7-4f99-940b-e1dbbeaa9fec",
            description=(
                "Find Google Chat group chats whose members are exactly the user "
                "plus the people you list, by email address or user ID, and get "
                "their IDs for reading or sending messages."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatFindGroupChatsBlock.Input,
            output_schema=GoogleChatFindGroupChatsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "people": ["dana@example.com", "sam@example.com"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("spaces", [_TEST_GROUP_CHAT]),
                ("space", _TEST_GROUP_CHAT),
            ],
            test_mock={
                "_find_group_chats": lambda *args, **kwargs: {
                    "spaces": [_TEST_GROUP_CHAT_RESOURCE]
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        users = list(
            dict.fromkeys(to_user_name(p) for p in input_data.people if p.strip())
        )
        if not 1 <= len(users) <= 49:
            raise BlockInputError(
                message="List between 1 and 49 people to look for, not counting yourself.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_chat_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._find_group_chats,
                service,
                users=users,
                page_size=input_data.max_results,
                page_token=input_data.page_token,
            )
        except HttpError as e:
            raise chat_error(e, self.name, self.id) from e

        spaces = [to_chat_space(s) for s in result.get("spaces", [])]
        yield "spaces", spaces
        for space in spaces:
            yield "space", space
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _find_group_chats(
        service, *, users: list[str], page_size: int, page_token: str
    ) -> dict:
        params: dict = {
            "users": users,
            "pageSize": page_size,
            "spaceView": "SPACE_VIEW_EXPANDED",
        }
        if page_token:
            params["pageToken"] = page_token
        return service.spaces().findGroupChats(**params).execute()
