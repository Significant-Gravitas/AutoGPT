from unittest import mock

from backend.data.model import HostScopedCredentials

from ..pagination import PageRequest, encode_page_cursor
from . import helpers

_FIELDS = 50


async def test_a_page_of_requirements_builds_each_credential_once() -> None:
    """A page of two builds two requirements, and one CredentialInfo per stored
    credential however many requirements it matches."""
    schema = {
        "properties": {
            f"http-h{i}.credentials": {
                "credentials_provider": ["http"],
                "credentials_types": ["host_scoped"],
            }
            for i in range(_FIELDS)
        }
    }
    stored = [
        HostScopedCredentials(provider="http", host=f"c{i}.example.com")
        for i in range(10)
    ]

    with (
        mock.patch.object(
            helpers.creds_manager.store,
            "get_all_creds",
            new=mock.AsyncMock(return_value=stored),
        ),
        mock.patch.object(
            helpers.CredentialInfo,
            "from_internal",
            wraps=helpers.CredentialInfo.from_internal,
        ) as built,
    ):
        page = await helpers.get_credential_requirements(
            schema, "user-1", PageRequest(limit=2, cursor=encode_page_cursor(2))
        )

    assert [r.field_name for r in page.items] == [
        "http-h2.credentials",
        "http-h3.credentials",
    ]
    assert page.total_count == _FIELDS
    assert page.next_cursor == encode_page_cursor(3)
    for requirement in page.items:
        assert [c.id for c in requirement.matching_credentials] == [
            c.id for c in stored
        ]
    assert built.call_count == len(stored)
