import asyncio
import re

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
    CONTACTS_READONLY_SCOPE,
    ORGANIZATION_READ_SCOPE,
    OTHER_CONTACT_FIELDS,
    OTHER_CONTACTS_READONLY_SCOPE,
    PERSON_FIELDS,
    PROFILE_FIELDS,
    USERINFO_EMAIL_SCOPE,
    USERINFO_PROFILE_SCOPE,
    GooglePerson,
    SearchFn,
    build_people_service,
    people_error,
    primary_item,
    require_query,
    search_with_warmup,
    to_person,
)

_PRIMARY = {"primary": True}
_TEST_CONTACT = {
    "resourceName": "people/c7243562183412345678",
    "names": [
        {
            "metadata": _PRIMARY,
            "displayName": "Alice Smith",
            "givenName": "Alice",
            "familyName": "Smith",
        }
    ],
    "emailAddresses": [{"metadata": _PRIMARY, "value": "alice@example.com"}],
    "phoneNumbers": [
        {
            "metadata": _PRIMARY,
            "value": "(202) 555-0143",
            "canonicalForm": "+12025550143",
        }
    ],
    "organizations": [
        {"metadata": _PRIMARY, "name": "Example Corp", "title": "Head of Sales"}
    ],
    "photos": [
        {"metadata": _PRIMARY, "url": "https://lh3.googleusercontent.com/cm/alice"}
    ],
}
_TEST_OTHER_CONTACT = {
    "resourceName": "otherContacts/c4428953157612345678",
    "names": [{"metadata": _PRIMARY, "displayName": "Alice Jones"}],
    "emailAddresses": [{"metadata": _PRIMARY, "value": "alice.jones@example.org"}],
}
_TEST_PROFILE = {
    "resourceName": "people/112233445566778899001",
    "names": [
        {
            "metadata": _PRIMARY,
            "displayName": "Sam Taylor",
            "givenName": "Sam",
            "familyName": "Taylor",
        }
    ],
    "emailAddresses": [{"metadata": _PRIMARY, "value": "sam@example.com"}],
    "photos": [
        {"metadata": _PRIMARY, "url": "https://lh3.googleusercontent.com/a/sam"}
    ],
    "organizations": [
        {"metadata": _PRIMARY, "name": "Example Corp", "title": "Operations Lead"}
    ],
    "locales": [{"metadata": _PRIMARY, "value": "en-GB"}],
}


class GoogleContactsSearchBlock(Block):
    """Search the user's saved Google contacts, and optionally other contacts."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CONTACTS_READONLY_SCOPE, OTHER_CONTACTS_READONLY_SCOPE]
        )
        query: str = SchemaField(
            description=(
                "Name, email address, phone number or company to look for. "
                "Matches the start of words, so 'Ali' finds 'Alice Smith'."
            )
        )
        include_other_contacts: bool = SchemaField(
            description=(
                "Also search 'Other contacts': people the user has emailed but "
                "never saved. These have only a name, email and phone number."
            ),
            default=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of people to return (Google allows up to 30)",
            default=10,
            ge=1,
            le=30,
        )

    class Output(BlockSchemaOutput):
        contacts: list[GooglePerson] = SchemaField(
            description="Matching people, saved contacts first"
        )
        contact: GooglePerson = SchemaField(description="Each matching person")

    def __init__(self):
        saved, other = to_person(_TEST_CONTACT), to_person(_TEST_OTHER_CONTACT)
        super().__init__(
            id="2ae8e943-d5c9-4cb0-b8af-8ae38d7b85cc",
            description=(
                "Search the user's Google Contacts by name, email address, phone "
                "number or company, optionally including 'Other contacts' (people "
                "they have emailed but never saved). Returns names, email "
                "addresses, phone numbers, companies, job titles and photos."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.COMMUNICATION},
            input_schema=GoogleContactsSearchBlock.Input,
            output_schema=GoogleContactsSearchBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "query": "Alice"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("contacts", [saved, other]),
                ("contact", saved),
                ("contact", other),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: (
                    [_TEST_CONTACT],
                    [_TEST_OTHER_CONTACT],
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        query = require_query(input_data.query, self.name, self.id)
        service = build_people_service(credentials)
        try:
            saved, other = await asyncio.to_thread(
                self._search,
                service,
                query=query,
                page_size=input_data.max_results,
                include_other_contacts=input_data.include_other_contacts,
            )
        except HttpError as e:
            raise people_error(e, self.name, self.id, access="its contacts") from e

        contacts = merge_people(
            [to_person(person) for person in saved],
            [to_person(person) for person in other],
            input_data.max_results,
        )
        yield "contacts", contacts
        for contact in contacts:
            yield "contact", contact

    @staticmethod
    def _search(
        service, *, query: str, page_size: int, include_other_contacts: bool
    ) -> tuple[list[dict], list[dict]]:
        searches: list[SearchFn] = [
            lambda q: search_saved_contacts(service, q, page_size)
        ]
        if include_other_contacts:
            searches.append(lambda q: search_other_contacts(service, q, page_size))
        found = search_with_warmup(searches, query)
        return found[0], (found[1] if include_other_contacts else [])


class GoogleContactsGetMyProfileBlock(Block):
    """Get the connected Google account's own profile."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [USERINFO_PROFILE_SCOPE, USERINFO_EMAIL_SCOPE, ORGANIZATION_READ_SCOPE]
        )

    class Output(BlockSchemaOutput):
        profile: GooglePerson = SchemaField(description="The user's profile")
        name: str = SchemaField(description="The user's display name")
        email: str = SchemaField(description="The user's primary email address")
        locale: str = SchemaField(
            description=(
                "The user's language as a BCP 47 tag, such as en-GB, when Google "
                "shares it"
            )
        )

    def __init__(self):
        profile = to_person(_TEST_PROFILE)
        super().__init__(
            id="f5246f08-0818-4c16-bab4-18870eabc8a2",
            description=(
                "Get the connected Google account's own profile: name, email "
                "address, photo, company and job title, plus language when "
                "Google shares it."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.COMMUNICATION},
            input_schema=GoogleContactsGetMyProfileBlock.Input,
            output_schema=GoogleContactsGetMyProfileBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("profile", profile),
                ("name", "Sam Taylor"),
                ("email", "sam@example.com"),
                ("locale", "en-GB"),
            ],
            test_mock={"_get_profile": lambda *args, **kwargs: _TEST_PROFILE},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_people_service(credentials)
        try:
            result = await asyncio.to_thread(self._get_profile, service)
        except HttpError as e:
            raise people_error(
                e, self.name, self.id, access="its profile and work details"
            ) from e

        profile = to_person(result)
        yield "profile", profile
        if profile.name:
            yield "name", profile.name
        if profile.email:
            yield "email", profile.email
        if locale := primary_item(result.get("locales")).get("value"):
            yield "locale", locale

    @staticmethod
    def _get_profile(service) -> dict:
        return (
            service.people()
            .get(resourceName="people/me", personFields=PROFILE_FIELDS)
            .execute()
        )


def search_saved_contacts(service, query: str, page_size: int) -> list[dict]:
    response = (
        service.people()
        .searchContacts(query=query, readMask=PERSON_FIELDS, pageSize=page_size)
        .execute()
    )
    return _matched_people(response)


def search_other_contacts(service, query: str, page_size: int) -> list[dict]:
    response = (
        service.otherContacts()
        .search(query=query, readMask=OTHER_CONTACT_FIELDS, pageSize=page_size)
        .execute()
    )
    return _matched_people(response)


def merge_people(
    first: list[GooglePerson], extra: list[GooglePerson], limit: int
) -> list[GooglePerson]:
    """Add the extra people who share no email or phone with an earlier one."""
    merged = list(first)
    seen = {key for person in first for key in _identity_keys(person)}
    for person in extra:
        keys = _identity_keys(person)
        if keys & seen:
            continue
        merged.append(person)
        seen |= keys
    return merged[:limit]


def _identity_keys(person: GooglePerson) -> set[str]:
    emails = {email.lower() for email in person.emails}
    phones = {re.sub(r"\D", "", phone) for phone in person.phones}
    return (emails | phones) - {""}


def _matched_people(response: dict) -> list[dict]:
    return [
        result["person"] for result in response.get("results", []) if "person" in result
    ]
