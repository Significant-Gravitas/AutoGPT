"""Unit tests for the Google Contacts blocks' requests, parsing and merging.

The blocks' own test_input/test_mock cases mock the API calls away; these
cover what those mocks skip.
"""

import json
from typing import Any

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks._base import BlockOutput
from backend.blocks.google import _people_api, contacts
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._people_api import (
    OTHER_CONTACT_FIELDS,
    PERSON_FIELDS,
    PROFILE_FIELDS,
    SEARCH_RETRY_DELAY_SECONDS,
    GooglePerson,
    people_error,
    to_person,
)
from backend.blocks.google.contacts import (
    GoogleContactsGetMyProfileBlock,
    GoogleContactsSearchBlock,
    merge_people,
)
from backend.util.exceptions import BlockInputError

PRIMARY = {"primary": True}
ALICE = {
    "resourceName": "people/c1",
    "names": [{"metadata": PRIMARY, "displayName": "Alice Smith"}],
    "emailAddresses": [{"metadata": PRIMARY, "value": "alice@example.com"}],
}
ALICE_JONES = {
    "resourceName": "otherContacts/c2",
    "names": [{"metadata": PRIMARY, "displayName": "Alice Jones"}],
    "emailAddresses": [{"metadata": PRIMARY, "value": "alice.jones@example.org"}],
}


def test_to_person_puts_primary_values_first():
    person = to_person(
        {
            "resourceName": "people/c1",
            "names": [
                {"displayName": "Al Smith"},
                {
                    "metadata": PRIMARY,
                    "displayName": "Alice Smith",
                    "givenName": "Alice",
                    "familyName": "Smith",
                },
            ],
            "emailAddresses": [
                {"value": "alice@home.example"},
                {"metadata": PRIMARY, "value": "alice@work.example"},
                {"value": "ALICE@work.example"},
            ],
            "phoneNumbers": [
                {"value": "020 7946 0000"},
                {
                    "metadata": PRIMARY,
                    "value": "(202) 555-0143",
                    "canonicalForm": "+12025550143",
                },
            ],
            "organizations": [
                {"name": "Old Co", "title": "Intern"},
                {"metadata": PRIMARY, "name": "Example Corp", "title": "Head of Sales"},
            ],
            "photos": [
                {
                    "metadata": PRIMARY,
                    "url": "https://lh3.example/avatar",
                    "default": True,
                },
                {"url": "https://lh3.example/alice"},
            ],
        }
    )
    assert person == GooglePerson(
        resource_name="people/c1",
        name="Alice Smith",
        given_name="Alice",
        family_name="Smith",
        email="alice@work.example",
        emails=["alice@work.example", "alice@home.example"],
        phones=["+12025550143", "020 7946 0000"],
        organization="Example Corp",
        title="Head of Sales",
        photo_url="https://lh3.example/alice",
    )


def test_to_person_handles_missing_fields_and_default_avatars():
    person = to_person(
        {
            "resourceName": "otherContacts/c2",
            "emailAddresses": [{"value": "bob@example.com"}],
            "photos": [{"url": "https://lh3.example/avatar", "default": True}],
        }
    )
    assert person == GooglePerson(
        resource_name="otherContacts/c2",
        email="bob@example.com",
        emails=["bob@example.com"],
    )


def test_merge_people_keeps_saved_contacts_first_and_drops_duplicates():
    alice = GooglePerson(
        resource_name="people/c1",
        emails=["alice@example.com"],
        phones=["+12025550143"],
    )
    extra = [
        GooglePerson(resource_name="otherContacts/c2", emails=["ALICE@example.com"]),
        GooglePerson(resource_name="otherContacts/c3", phones=["+1 (202) 555-0143"]),
        GooglePerson(resource_name="otherContacts/c4", emails=["bob@example.com"]),
        GooglePerson(resource_name="otherContacts/c5", emails=["bob@example.com"]),
        GooglePerson(resource_name="otherContacts/c6"),
    ]
    merged = merge_people([alice], extra, limit=10)
    assert [person.resource_name for person in merged] == [
        "people/c1",
        "otherContacts/c4",
        "otherContacts/c6",
    ]


def test_merge_people_applies_the_limit_to_the_merged_list():
    saved = [
        GooglePerson(resource_name=f"people/c{i}", emails=[f"s{i}@example.com"])
        for i in range(3)
    ]
    other = [GooglePerson(resource_name="otherContacts/c9", emails=["o@example.com"])]
    merged = merge_people(saved, other, limit=3)
    assert [person.resource_name for person in merged] == [
        "people/c0",
        "people/c1",
        "people/c2",
    ]


class _Request:
    def __init__(self, result: dict):
        self._result = result

    def execute(self) -> dict:
        return self._result


class _FakePeopleService:
    """Stands in for the People API client and records every request.

    ``saved`` and ``other`` hold the people each non-empty contact search
    returns, in order. Empty (warm-up) searches return nothing.
    """

    def __init__(
        self,
        saved: list[list[dict]] | None = None,
        other: list[list[dict]] | None = None,
    ):
        self.calls: list[tuple[str, dict]] = []
        self._answers = {"searchContacts": saved or [], "search": other or []}

    def people(self):
        return self

    def otherContacts(self):
        return self

    def searchContacts(self, **kwargs) -> _Request:
        return self._search("searchContacts", kwargs)

    def search(self, **kwargs) -> _Request:
        return self._search("search", kwargs)

    def get(self, **kwargs) -> _Request:
        self.calls.append(("get", kwargs))
        return _Request({"resourceName": "people/me"})

    def _search(self, method: str, kwargs: dict) -> _Request:
        self.calls.append((method, kwargs))
        if not kwargs["query"]:
            return _Request({})
        people = self._answers[method].pop(0)
        return _Request({"results": [{"person": person} for person in people]})


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(_people_api.time, "sleep", recorded.append)
    return recorded


def test_search_warms_up_both_sources_before_searching(sleeps: list[float]):
    service = _FakePeopleService(saved=[[ALICE]], other=[[ALICE_JONES]])
    found = GoogleContactsSearchBlock._search(
        service, query="Ali", page_size=5, include_other_contacts=True
    )
    assert found == ([ALICE], [ALICE_JONES])
    assert service.calls == [
        ("searchContacts", {"query": "", "readMask": PERSON_FIELDS, "pageSize": 5}),
        ("search", {"query": "", "readMask": OTHER_CONTACT_FIELDS, "pageSize": 5}),
        ("searchContacts", {"query": "Ali", "readMask": PERSON_FIELDS, "pageSize": 5}),
        ("search", {"query": "Ali", "readMask": OTHER_CONTACT_FIELDS, "pageSize": 5}),
    ]
    assert sleeps == []


def test_search_repeats_once_after_a_pause_when_no_one_is_found(
    sleeps: list[float],
):
    service = _FakePeopleService(saved=[[], [ALICE]], other=[[], [ALICE_JONES]])
    found = GoogleContactsSearchBlock._search(
        service, query="Ali", page_size=10, include_other_contacts=True
    )
    assert found == ([ALICE], [ALICE_JONES])
    assert sleeps == [SEARCH_RETRY_DELAY_SECONDS]
    assert [(method, kw["query"]) for method, kw in service.calls[4:]] == [
        ("searchContacts", "Ali"),
        ("search", "Ali"),
    ]


def test_search_does_not_wait_when_one_source_finds_people(sleeps: list[float]):
    service = _FakePeopleService(saved=[[ALICE]], other=[[]])
    found = GoogleContactsSearchBlock._search(
        service, query="Ali", page_size=10, include_other_contacts=True
    )
    assert found == ([ALICE], [])
    assert sleeps == []
    assert len(service.calls) == 4


def test_search_gives_up_after_one_retry(sleeps: list[float]):
    service = _FakePeopleService(saved=[[], []])
    found = GoogleContactsSearchBlock._search(
        service, query="Zed", page_size=10, include_other_contacts=False
    )
    assert found == ([], [])
    assert sleeps == [SEARCH_RETRY_DELAY_SECONDS]
    assert [method for method, _ in service.calls] == ["searchContacts"] * 3


@pytest.mark.asyncio
async def test_contacts_search_merges_other_contacts(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
):
    duplicate = {
        "resourceName": "otherContacts/c9",
        "emailAddresses": [{"value": "ALICE@example.com"}],
    }
    service = _FakePeopleService(saved=[[ALICE]], other=[[duplicate, ALICE_JONES]])
    monkeypatch.setattr(contacts, "build_people_service", lambda credentials: service)
    block = GoogleContactsSearchBlock()
    input_data = GoogleContactsSearchBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "query": " Alice "}
    )
    outputs = await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))
    expected = [to_person(ALICE), to_person(ALICE_JONES)]
    assert outputs == [
        ("contacts", expected),
        ("contact", expected[0]),
        ("contact", expected[1]),
    ]
    assert {kw["query"] for _, kw in service.calls} == {"", "Alice"}


@pytest.mark.asyncio
async def test_contacts_search_leaves_other_contacts_alone_when_off(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
):
    service = _FakePeopleService(saved=[[ALICE]])
    monkeypatch.setattr(contacts, "build_people_service", lambda credentials: service)
    input_data = GoogleContactsSearchBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "query": "Alice",
            "include_other_contacts": False,
        }
    )
    outputs = await _collect(
        GoogleContactsSearchBlock().run(input_data, credentials=TEST_CREDENTIALS)
    )
    assert outputs[0] == ("contacts", [to_person(ALICE)])
    assert {method for method, _ in service.calls} == {"searchContacts"}


@pytest.mark.asyncio
async def test_contacts_search_rejects_a_blank_query():
    input_data = GoogleContactsSearchBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "query": "   "}
    )
    with pytest.raises(BlockInputError, match="search for"):
        await _collect(
            GoogleContactsSearchBlock().run(input_data, credentials=TEST_CREDENTIALS)
        )


def test_profile_request_asks_for_the_signed_in_user():
    service = _FakePeopleService()
    GoogleContactsGetMyProfileBlock._get_profile(service)
    assert service.calls == [
        ("get", {"resourceName": "people/me", "personFields": PROFILE_FIELDS})
    ]


@pytest.mark.asyncio
async def test_profile_skips_outputs_google_did_not_return(
    monkeypatch: pytest.MonkeyPatch,
):
    block = GoogleContactsGetMyProfileBlock()
    monkeypatch.setattr(contacts, "build_people_service", lambda credentials: None)
    monkeypatch.setattr(
        block,
        "_get_profile",
        lambda service: {
            "resourceName": "people/1",
            "emailAddresses": [{"value": "sam@example.com"}],
        },
    )
    input_data = GoogleContactsGetMyProfileBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT}
    )
    outputs = await _collect(block.run(input_data, credentials=TEST_CREDENTIALS))
    assert [name for name, _ in outputs] == ["profile", "email"]


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (400, "Must be a G Suite domain user.", "Google Workspace"),
        (429, "Quota exceeded for quota metric 'Read requests'", "API error 429"),
    ],
)
def test_people_error_messages(status: int, reason: str, expected: str):
    error = people_error(_http_error(status, reason), "block", "id", access="x")
    assert expected in str(error)


def _http_error(status: int, reason: str, api_status: str = "") -> HttpError:
    content = json.dumps(
        {"error": {"code": status, "message": reason, "status": api_status}}
    ).encode()
    return HttpError(httplib2.Response({"status": status}), content)


async def _collect(outputs: BlockOutput) -> list[tuple[str, Any]]:
    return [output async for output in outputs]
