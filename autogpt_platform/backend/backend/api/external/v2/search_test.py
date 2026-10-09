"""A search answers with what a direct read of each result would."""

from unittest.mock import AsyncMock

import pytest_mock
from fastapi import Response
from prisma.enums import APIKeyPermission

from .models import SearchContentType
from .pagination import PageRequest
from .search import search
from .tenancy import TenantContext

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-a",
)


def _row(content_type: str, content_id: str) -> dict:
    return {
        "content_type": content_type,
        "content_id": content_id,
        "searchable_text": "text",
        "metadata": None,
        "updated_at": None,
        "combined_score": 1.0,
    }


async def test_library_agents_outside_the_organization_are_left_out(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The index records the user, not the organization, of a library agent."""
    mocker.patch("backend.api.external.v2.search.enforce", new_callable=AsyncMock)
    mocker.patch(
        "backend.api.external.v2.search.unified_hybrid_search",
        new_callable=AsyncMock,
        return_value=(
            [
                _row("LIBRARY_AGENT", "mine-in-a"),
                _row("LIBRARY_AGENT", "mine-in-b"),
                _row("LIBRARY_AGENT", "untagged"),
                _row("LIBRARY_AGENT", "deleted"),
                _row("BLOCK", "block-1"),
            ],
            5,
        ),
    )
    mocker.patch(
        "backend.api.external.v2.search.library_db.get_library_agent_organizations",
        new_callable=AsyncMock,
        return_value={"mine-in-a": "org-a", "mine-in-b": "org-b", "untagged": None},
    )

    found = await search(
        response=Response(),
        query="report",
        content_types=[SearchContentType.LIBRARY_AGENT, SearchContentType.BLOCK],
        category=None,
        page=PageRequest(limit=25),
        auth=_AUTH,
    )

    assert [r.content_id for r in found.items] == ["mine-in-a", "untagged", "block-1"]
