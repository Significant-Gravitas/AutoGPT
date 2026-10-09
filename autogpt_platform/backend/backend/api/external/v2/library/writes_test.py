"""Library writes stay inside the organization the credential acts in.

Library rows are the user's, and a lookup by id finds the user's row in any of
their organizations; every folder a write names, and every row it creates,
must be the credential's organization's.
"""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.util.exceptions import NotFoundError

from ..models import (
    LibraryAgentUpdateRequest,
    LibraryFolderCreateRequest,
    LibraryFolderMoveRequest,
)
from ..tenancy import TenantContext
from .agents import fork_library_agent, update_library_agent
from .folders import create_folder, move_folder

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-a",
    team_id="team-a",
)


def _folder(organization_id: str | None) -> Mock:
    return Mock(organization_id=organization_id)


@pytest.fixture
def folders(mocker: pytest_mock.MockFixture) -> dict[str, AsyncMock]:
    """Folder f-a is org A's, f-b org B's; the writes are recorded."""
    by_id = {"f-a": _folder("org-a"), "f-b": _folder("org-b")}
    mocker.patch(
        "backend.api.features.library.db.get_folder",
        new_callable=AsyncMock,
        side_effect=lambda folder_id, user_id: by_id[folder_id],
    )
    mocker.patch("backend.api.external.v2.library.folders.LibraryFolder.from_internal")
    mocker.patch("backend.api.external.v2.library.agents.LibraryAgent.from_internal")
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=AsyncMock,
        return_value=Mock(organization_id="org-a"),
    )
    return {
        name: mocker.patch(
            f"backend.api.features.library.db.{name}", new_callable=AsyncMock
        )
        for name in (
            "create_folder",
            "move_folder",
            "update_library_agent",
            "fork_library_agent",
        )
    }


async def test_a_new_folder_is_tagged_with_the_credentials_tenant(
    folders: dict[str, AsyncMock],
) -> None:
    await create_folder(
        LibraryFolderCreateRequest(name="Reports", parent_id="f-a"), auth=_AUTH
    )

    kwargs = folders["create_folder"].await_args.kwargs
    assert (kwargs["organization_id"], kwargs["team_id"]) == ("org-a", "team-a")


async def test_a_folder_cannot_be_created_under_another_organizations(
    folders: dict[str, AsyncMock],
) -> None:
    with pytest.raises(NotFoundError):
        await create_folder(
            LibraryFolderCreateRequest(name="Reports", parent_id="f-b"), auth=_AUTH
        )
    folders["create_folder"].assert_not_awaited()


async def test_a_folder_cannot_be_moved_under_another_organizations(
    folders: dict[str, AsyncMock],
) -> None:
    with pytest.raises(NotFoundError):
        await move_folder(
            LibraryFolderMoveRequest(target_parent_id="f-b"),
            folder_id="f-a",
            auth=_AUTH,
        )
    folders["move_folder"].assert_not_awaited()


async def test_an_agent_cannot_be_filed_in_another_organizations_folder(
    folders: dict[str, AsyncMock],
) -> None:
    with pytest.raises(NotFoundError):
        await update_library_agent(
            LibraryAgentUpdateRequest(folder_id="f-b"), agent_id="agent-1", auth=_AUTH
        )
    folders["update_library_agent"].assert_not_awaited()


async def test_a_fork_lands_in_the_credentials_tenant(
    folders: dict[str, AsyncMock],
) -> None:
    await fork_library_agent(agent_id="agent-1", auth=_AUTH)

    kwargs = folders["fork_library_agent"].await_args.kwargs
    assert (kwargs["organization_id"], kwargs["team_id"]) == ("org-a", "team-a")
