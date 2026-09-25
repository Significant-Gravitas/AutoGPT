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

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._people_api import (
    DIRECTORY_READONLY_SCOPE,
    PERSON_FIELDS,
    GooglePerson,
    build_people_service,
    people_error,
    require_query,
    to_person,
)

DOMAIN_PROFILE_SOURCE = "DIRECTORY_SOURCE_TYPE_DOMAIN_PROFILE"
DOMAIN_CONTACT_SOURCE = "DIRECTORY_SOURCE_TYPE_DOMAIN_CONTACT"

_TEST_COLLEAGUE = {
    "resourceName": "people/104683264128812345678",
    "names": [
        {
            "metadata": {"primary": True},
            "displayName": "Bob Lee",
            "givenName": "Bob",
            "familyName": "Lee",
        }
    ],
    "emailAddresses": [{"metadata": {"primary": True}, "value": "bob.lee@example.com"}],
    "organizations": [{"metadata": {"primary": True}, "title": "Engineering Manager"}],
}


class GoogleContactsSearchDirectoryBlock(Block):
    """Search the user's Google Workspace directory."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [DIRECTORY_READONLY_SCOPE]
        )
        query: str = SchemaField(
            description=(
                "Name or email address to look for. Matches the start of words, "
                "so 'Ali' finds 'Alice Smith'."
            )
        )
        include_shared_contacts: bool = SchemaField(
            description=(
                "Also search contacts the Workspace admin shared with the whole "
                "organization, not just colleagues' directory profiles"
            ),
            default=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of people to return",
            default=10,
            ge=1,
            le=500,
        )
        page_token: str = SchemaField(
            description="Page token from a previous search, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        people: list[GooglePerson] = SchemaField(
            description="Matching colleagues and shared contacts"
        )
        person: GooglePerson = SchemaField(description="Each matching person")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        colleague = to_person(_TEST_COLLEAGUE)
        super().__init__(
            id="a873efcc-2133-4039-809c-0bf952585788",
            description=(
                "Search the user's Google Workspace directory for colleagues and "
                "contacts shared with the organization, by name or email address. "
                "Returns names, email addresses, phone numbers, job titles and "
                "photos. Only works for Google Workspace (work or school) accounts."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.COMMUNICATION},
            input_schema=GoogleContactsSearchDirectoryBlock.Input,
            output_schema=GoogleContactsSearchDirectoryBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "query": "Bob"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("people", [colleague]),
                ("person", colleague),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_search_directory": lambda *args, **kwargs: {
                    "people": [_TEST_COLLEAGUE],
                    "nextPageToken": "next-page",
                    "totalSize": 12,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        query = require_query(input_data.query, self.name, self.id)
        sources = [DOMAIN_PROFILE_SOURCE]
        if input_data.include_shared_contacts:
            sources.append(DOMAIN_CONTACT_SOURCE)
        service = build_people_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._search_directory,
                service,
                query=query,
                sources=sources,
                page_size=input_data.max_results,
                page_token=input_data.page_token,
            )
        except HttpError as e:
            raise people_error(
                e, self.name, self.id, access="the Workspace directory"
            ) from e

        people = [to_person(person) for person in result.get("people", [])]
        yield "people", people
        for person in people:
            yield "person", person
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _search_directory(
        service, *, query: str, sources: list[str], page_size: int, page_token: str
    ) -> dict:
        params: dict = {
            "query": query,
            "readMask": PERSON_FIELDS,
            "sources": sources,
            "pageSize": page_size,
        }
        if page_token:
            params["pageToken"] = page_token
        return service.people().searchDirectoryPeople(**params).execute()
