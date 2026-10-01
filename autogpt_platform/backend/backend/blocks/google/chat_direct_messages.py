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
    CHAT_SPACES_CREATE_SCOPE,
    CHAT_SPACES_READONLY_SCOPE,
    ChatSpace,
    build_chat_service,
    chat_error,
    to_chat_space,
    to_user_name,
)

_TEST_DM_RESOURCE = {
    "name": "spaces/DMdAna1234",
    "spaceType": "DIRECT_MESSAGE",
    "spaceUri": "https://chat.google.com/dm/DMdAna1234?cls=11",
    "membershipCount": {"joinedDirectHumanUserCount": 2},
    "lastActiveTime": "2026-09-23T09:12:00Z",
}
_TEST_DM = to_chat_space(_TEST_DM_RESOURCE)
_PERSON_DESCRIPTION = "The person's email address or Chat user ID (users/...)"


class GoogleChatFindDirectMessageBlock(Block):
    """Find the user's existing direct message with one person."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_SPACES_READONLY_SCOPE]
        )
        person: str = SchemaField(
            description=_PERSON_DESCRIPTION, placeholder="colleague@example.com"
        )

    class Output(BlockSchemaOutput):
        space: ChatSpace = SchemaField(description="The direct message conversation")

    def __init__(self):
        super().__init__(
            id="3ca82d77-0f72-4fde-b626-ace4d12df7b7",
            description=(
                "Find the user's existing Google Chat direct message with a person, "
                "by email address or user ID, and get its ID for reading or "
                "sending messages."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatFindDirectMessageBlock.Input,
            output_schema=GoogleChatFindDirectMessageBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "person": "dana@example.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("space", _TEST_DM)],
            test_mock={
                "_find_direct_message": lambda *args, **kwargs: _TEST_DM_RESOURCE
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        person = require_person(input_data.person, self.name, self.id)
        service = build_chat_service(credentials)
        try:
            space = await asyncio.to_thread(
                self._find_direct_message, service, to_user_name(person)
            )
        except HttpError as e:
            not_found = (
                f"You don't have a Google Chat direct message with {person} yet, or "
                "Google Chat can't find them. Use Google Chat Start Direct Message "
                "to start one."
            )
            raise chat_error(e, self.name, self.id, not_found=not_found) from e
        yield "space", to_chat_space(space)

    @staticmethod
    def _find_direct_message(service, user: str) -> dict:
        return service.spaces().findDirectMessage(name=user).execute()


class GoogleChatStartDirectMessageBlock(Block):
    """Get the user's direct message with a person, creating it if needed."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CHAT_SPACES_CREATE_SCOPE]
        )
        person: str = SchemaField(
            description=_PERSON_DESCRIPTION, placeholder="colleague@example.com"
        )

    class Output(BlockSchemaOutput):
        space: ChatSpace = SchemaField(
            description="The direct message conversation, existing or new"
        )

    def __init__(self):
        super().__init__(
            id="d4bc949b-b43d-4601-90a3-3528e9d9647e",
            description=(
                "Open a Google Chat direct message with a person, by email address "
                "or user ID: returns the existing conversation, or creates an empty "
                "one. Nothing is sent. Pass the result to Google Chat Send Message."
            ),
            categories={BlockCategory.COMMUNICATION},
            input_schema=GoogleChatStartDirectMessageBlock.Input,
            output_schema=GoogleChatStartDirectMessageBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "person": "dana@example.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("space", _TEST_DM)],
            test_mock={
                "_set_up_direct_message": lambda *args, **kwargs: _TEST_DM_RESOURCE
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        person = require_person(input_data.person, self.name, self.id)
        service = build_chat_service(credentials)
        try:
            space = await asyncio.to_thread(
                self._set_up_direct_message, service, to_user_name(person)
            )
        except HttpError as e:
            not_found = f"Google Chat can't find {person}."
            raise chat_error(e, self.name, self.id, not_found=not_found) from e
        yield "space", to_chat_space(space)

    @staticmethod
    def _set_up_direct_message(service, user: str) -> dict:
        body = {
            "space": {"spaceType": "DIRECT_MESSAGE", "singleUserBotDm": False},
            "memberships": [{"member": {"name": user, "type": "HUMAN"}}],
        }
        return service.spaces().setup(body=body).execute()


def require_person(value: str, block_name: str, block_id: str) -> str:
    person = value.strip()
    if not person:
        raise BlockInputError(
            message="Give the person's email address or Chat user ID.",
            block_name=block_name,
            block_id=block_id,
        )
    return person
