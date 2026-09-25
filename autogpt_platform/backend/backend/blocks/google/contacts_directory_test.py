"""Unit tests for the Google Workspace directory search block's requests.

The block's own test_input/test_mock case mocks the API call away; these
cover what that mock skips.
"""

import json
from typing import Any

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks._base import BlockOutput
from backend.blocks.google import contacts_directory
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._people_api import PERSON_FIELDS, to_person
from backend.blocks.google.contacts_directory import (
    DOMAIN_CONTACT_SOURCE,
    DOMAIN_PROFILE_SOURCE,
    GoogleContactsSearchDirectoryBlock,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

COLLEAGUE = {
    "resourceName": "people/1046832641288",
    "names": [{"metadata": {"primary": True}, "displayName": "Bob Lee"}],
    "emailAddresses": [{"metadata": {"primary": True}, "value": "bob.lee@example.com"}],
}


class _Request:
    def __init__(self, result: dict):
        self._result = result

    def execute(self) -> dict:
        return self._result


class _FakeDirectoryService:
    """Stands in for the People API client and records directory searches."""

    def __init__(self, result: dict | None = None):
        self.calls: list[dict] = []
        self._result = result or {}

    def people(self):
        return self

    def searchDirectoryPeople(self, **kwargs) -> _Request:
        self.calls.append(kwargs)
        return _Request(self._result)


def _input(**fields) -> GoogleContactsSearchDirectoryBlock.Input:
    return GoogleContactsSearchDirectoryBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "include_shared, sources",
    [
        (True, [DOMAIN_PROFILE_SOURCE, DOMAIN_CONTACT_SOURCE]),
        (False, [DOMAIN_PROFILE_SOURCE]),
    ],
)
async def test_directory_search_request(
    monkeypatch: pytest.MonkeyPatch, include_shared: bool, sources: list[str]
):
    service = _FakeDirectoryService({"people": [COLLEAGUE]})
    monkeypatch.setattr(
        contacts_directory, "build_people_service", lambda credentials: service
    )
    input_data = _input(
        query=" Bob ",
        include_shared_contacts=include_shared,
        max_results=25,
        page_token="page-2",
    )
    outputs = await _collect(
        GoogleContactsSearchDirectoryBlock().run(
            input_data, credentials=TEST_CREDENTIALS
        )
    )
    assert service.calls == [
        {
            "query": "Bob",
            "readMask": PERSON_FIELDS,
            "sources": sources,
            "pageSize": 25,
            "pageToken": "page-2",
        }
    ]
    assert outputs == [
        ("people", [to_person(COLLEAGUE)]),
        ("person", to_person(COLLEAGUE)),
    ]


def test_directory_search_sends_a_page_token_only_when_given():
    service = _FakeDirectoryService()
    GoogleContactsSearchDirectoryBlock._search_directory(
        service,
        query="Bob",
        sources=[DOMAIN_PROFILE_SOURCE],
        page_size=10,
        page_token="",
    )
    assert "pageToken" not in service.calls[0]


@pytest.mark.asyncio
async def test_directory_search_rejects_a_blank_query():
    with pytest.raises(BlockInputError, match="search for"):
        await _collect(
            GoogleContactsSearchDirectoryBlock().run(
                _input(query="   "), credentials=TEST_CREDENTIALS
            )
        )


@pytest.mark.asyncio
async def test_directory_search_explains_personal_accounts(
    monkeypatch: pytest.MonkeyPatch,
):
    block = GoogleContactsSearchDirectoryBlock()
    monkeypatch.setattr(
        contacts_directory, "build_people_service", lambda credentials: None
    )

    def refuse(*args, **kwargs):
        raise _http_error(400, "Must be a G Suite domain user.", "FAILED_PRECONDITION")

    monkeypatch.setattr(block, "_search_directory", refuse)
    with pytest.raises(BlockExecutionError, match="personal Google account"):
        await _collect(block.run(_input(query="Bob"), credentials=TEST_CREDENTIALS))


def _http_error(status: int, reason: str, api_status: str) -> HttpError:
    content = json.dumps(
        {"error": {"code": status, "message": reason, "status": api_status}}
    ).encode()
    return HttpError(httplib2.Response({"status": status}), content)


async def _collect(outputs: BlockOutput) -> list[tuple[str, Any]]:
    return [output async for output in outputs]
