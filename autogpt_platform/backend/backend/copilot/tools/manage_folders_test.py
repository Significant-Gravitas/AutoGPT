"""Tests for folder management copilot tools."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library import model as library_model
from backend.copilot.tools.manage_folders import (
    CreateFolderTool,
    DeleteFolderTool,
    ListFoldersTool,
    MoveAgentsToFolderTool,
    MoveFolderTool,
    UpdateFolderTool,
)
from backend.copilot.tools.models import (
    AgentsMovedToFolderResponse,
    ErrorResponse,
    FolderCreatedResponse,
    FolderDeletedResponse,
    FolderListResponse,
    FolderMovedResponse,
    FolderUpdatedResponse,
)

from ._test_data import make_session

_TEST_USER_ID = "test-user-folders"
_NOW = datetime.now(UTC)


def _make_folder(
    id: str = "folder-1",
    name: str = "My Folder",
    parent_id: str | None = None,
    icon: str | None = None,
    color: str | None = None,
    agent_count: int = 0,
    subfolder_count: int = 0,
) -> library_model.LibraryFolder:
    return library_model.LibraryFolder(
        id=id,
        user_id=_TEST_USER_ID,
        name=name,
        icon=icon,
        color=color,
        parent_id=parent_id,
        created_at=_NOW,
        updated_at=_NOW,
        agent_count=agent_count,
        subfolder_count=subfolder_count,
    )


def _make_tree(
    id: str = "folder-1",
    name: str = "Root",
    children: list[library_model.LibraryFolderTree] | None = None,
) -> library_model.LibraryFolderTree:
    return library_model.LibraryFolderTree(
        id=id,
        user_id=_TEST_USER_ID,
        name=name,
        created_at=_NOW,
        updated_at=_NOW,
        children=children or [],
    )


def _make_library_agent(id: str = "agent-1", name: str = "Test Agent"):
    agent = MagicMock()
    agent.id = id
    agent.name = name
    return agent


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── CreateFolderTool ──


@pytest.fixture
def create_tool():
    return CreateFolderTool()


@pytest.mark.asyncio
async def test_create_folder_missing_name(create_tool, session):
    result = await create_tool._execute(user_id=_TEST_USER_ID, session=session, name="")
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_name"


@pytest.mark.asyncio
async def test_create_folder_none_name(create_tool, session):
    result = await create_tool._execute(
        user_id=_TEST_USER_ID, session=session, name=None
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_name"


@pytest.mark.asyncio
async def test_create_folder_success(create_tool, session):
    folder = _make_folder(name="New Folder")
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.create_folder = AsyncMock(return_value=folder)
        result = await create_tool._execute(
            user_id=_TEST_USER_ID, session=session, name="New Folder"
        )

    assert isinstance(result, FolderCreatedResponse)
    assert result.folder.name == "New Folder"
    assert "New Folder" in result.message


@pytest.mark.asyncio
async def test_create_folder_db_error(create_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.create_folder = AsyncMock(
            side_effect=Exception("db down")
        )
        result = await create_tool._execute(
            user_id=_TEST_USER_ID, session=session, name="Folder"
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "create_folder_failed"


# ── ListFoldersTool ──


@pytest.fixture
def list_tool():
    return ListFoldersTool()


@pytest.mark.asyncio
async def test_list_folders_by_parent(list_tool, session):
    folders = [_make_folder(id="f1", name="A"), _make_folder(id="f2", name="B")]
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.list_folders = AsyncMock(return_value=folders)
        result = await list_tool._execute(
            user_id=_TEST_USER_ID, session=session, parent_id="parent-1"
        )

    assert isinstance(result, FolderListResponse)
    assert result.count == 2
    assert len(result.folders) == 2


@pytest.mark.asyncio
async def test_list_folders_tree(list_tool, session):
    tree = [
        _make_tree(id="r1", name="Root", children=[_make_tree(id="c1", name="Child")])
    ]
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.get_folder_tree = AsyncMock(return_value=tree)
        result = await list_tool._execute(user_id=_TEST_USER_ID, session=session)

    assert isinstance(result, FolderListResponse)
    assert result.count == 2  # root + child
    assert result.tree is not None
    assert len(result.tree) == 1


@pytest.mark.asyncio
async def test_list_folders_tree_with_agents_includes_root(list_tool, session):
    tree = [_make_tree(id="r1", name="Root")]
    raw_map = {"r1": [{"id": "a1", "name": "Foldered", "description": "In folder"}]}
    root_raw = [{"id": "a2", "name": "Loose Agent", "description": "At root"}]
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.get_folder_tree = AsyncMock(return_value=tree)
        mock_lib.return_value.get_folder_agents_map = AsyncMock(return_value=raw_map)
        mock_lib.return_value.get_root_agent_summaries = AsyncMock(
            return_value=root_raw
        )
        result = await list_tool._execute(
            user_id=_TEST_USER_ID, session=session, include_agents=True
        )

    assert isinstance(result, FolderListResponse)
    assert result.root_agents is not None
    assert len(result.root_agents) == 1
    assert result.root_agents[0].name == "Loose Agent"
    assert result.tree is not None
    assert result.tree[0].agents is not None
    assert result.tree[0].agents[0].name == "Foldered"
    mock_lib.return_value.get_root_agent_summaries.assert_awaited_once_with(
        _TEST_USER_ID
    )


@pytest.mark.asyncio
async def test_list_folders_tree_without_agents_no_root(list_tool, session):
    tree = [_make_tree(id="r1", name="Root")]
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.get_folder_tree = AsyncMock(return_value=tree)
        result = await list_tool._execute(
            user_id=_TEST_USER_ID, session=session, include_agents=False
        )

    assert isinstance(result, FolderListResponse)
    assert result.root_agents is None


@pytest.mark.asyncio
async def test_list_folders_db_error(list_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.get_folder_tree = AsyncMock(
            side_effect=Exception("timeout")
        )
        result = await list_tool._execute(user_id=_TEST_USER_ID, session=session)

    assert isinstance(result, ErrorResponse)
    assert result.error == "list_folders_failed"


# ── UpdateFolderTool ──


@pytest.fixture
def update_tool():
    return UpdateFolderTool()


@pytest.mark.asyncio
async def test_update_folder_missing_id(update_tool, session):
    result = await update_tool._execute(
        user_id=_TEST_USER_ID, session=session, folder_id=""
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_folder_id"


@pytest.mark.asyncio
async def test_update_folder_none_id(update_tool, session):
    result = await update_tool._execute(
        user_id=_TEST_USER_ID, session=session, folder_id=None
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_folder_id"


@pytest.mark.asyncio
async def test_update_folder_success(update_tool, session):
    folder = _make_folder(name="Renamed")
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.update_folder = AsyncMock(return_value=folder)
        result = await update_tool._execute(
            user_id=_TEST_USER_ID, session=session, folder_id="folder-1", name="Renamed"
        )

    assert isinstance(result, FolderUpdatedResponse)
    assert result.folder.name == "Renamed"


@pytest.mark.asyncio
async def test_update_folder_db_error(update_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.update_folder = AsyncMock(
            side_effect=Exception("not found")
        )
        result = await update_tool._execute(
            user_id=_TEST_USER_ID, session=session, folder_id="folder-1", name="X"
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "update_folder_failed"


# ── MoveFolderTool ──


@pytest.fixture
def move_tool():
    return MoveFolderTool()


@pytest.mark.asyncio
async def test_move_folder_missing_id(move_tool, session):
    result = await move_tool._execute(
        user_id=_TEST_USER_ID, session=session, folder_id=""
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_folder_id"


@pytest.mark.asyncio
async def test_move_folder_to_parent(move_tool, session):
    folder = _make_folder(name="Moved")
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.move_folder = AsyncMock(return_value=folder)
        result = await move_tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            folder_id="folder-1",
            target_parent_id="parent-1",
        )

    assert isinstance(result, FolderMovedResponse)
    assert "subfolder" in result.message


@pytest.mark.asyncio
async def test_move_folder_to_root(move_tool, session):
    folder = _make_folder(name="Moved")
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.move_folder = AsyncMock(return_value=folder)
        result = await move_tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            folder_id="folder-1",
            target_parent_id=None,
        )

    assert isinstance(result, FolderMovedResponse)
    assert "root level" in result.message


@pytest.mark.asyncio
async def test_move_folder_db_error(move_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.move_folder = AsyncMock(side_effect=Exception("circular"))
        result = await move_tool._execute(
            user_id=_TEST_USER_ID, session=session, folder_id="folder-1"
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "move_folder_failed"


# ── DeleteFolderTool ──


@pytest.fixture
def delete_tool():
    return DeleteFolderTool()


@pytest.mark.asyncio
async def test_delete_folder_missing_id(delete_tool, session):
    result = await delete_tool._execute(
        user_id=_TEST_USER_ID, session=session, folder_id=""
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_folder_id"


@pytest.mark.asyncio
async def test_delete_folder_success(delete_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.delete_folder = AsyncMock(return_value=None)
        result = await delete_tool._execute(
            user_id=_TEST_USER_ID, session=session, folder_id="folder-1"
        )

    assert isinstance(result, FolderDeletedResponse)
    assert result.folder_id == "folder-1"
    assert "root level" in result.message


@pytest.mark.asyncio
async def test_delete_folder_db_error(delete_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.delete_folder = AsyncMock(
            side_effect=Exception("permission denied")
        )
        result = await delete_tool._execute(
            user_id=_TEST_USER_ID, session=session, folder_id="folder-1"
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "delete_folder_failed"


# ── MoveAgentsToFolderTool ──


@pytest.fixture
def move_agents_tool():
    return MoveAgentsToFolderTool()


@pytest.mark.asyncio
async def test_move_agents_missing_ids(move_agents_tool, session):
    result = await move_agents_tool._execute(
        user_id=_TEST_USER_ID, session=session, agent_ids=[]
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_ids"


@pytest.mark.asyncio
async def test_move_agents_success(move_agents_tool, session):
    agents = [
        _make_library_agent(id="a1", name="Agent Alpha"),
        _make_library_agent(id="a2", name="Agent Beta"),
    ]
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.bulk_move_agents_to_folder = AsyncMock(
            return_value=agents
        )
        result = await move_agents_tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_ids=["a1", "a2"],
            folder_id="folder-1",
        )

    assert isinstance(result, AgentsMovedToFolderResponse)
    assert result.count == 2
    assert result.agent_names == ["Agent Alpha", "Agent Beta"]
    assert "Agent Alpha" in result.message
    assert "Agent Beta" in result.message


@pytest.mark.asyncio
async def test_move_agents_to_root(move_agents_tool, session):
    agents = [_make_library_agent(id="a1", name="Agent One")]
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.bulk_move_agents_to_folder = AsyncMock(
            return_value=agents
        )
        result = await move_agents_tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_ids=["a1"],
            folder_id=None,
        )

    assert isinstance(result, AgentsMovedToFolderResponse)
    assert "root level" in result.message


@pytest.mark.asyncio
async def test_move_agents_db_error(move_agents_tool, session):
    with patch("backend.copilot.tools.manage_folders.library_db") as mock_lib:
        mock_lib.return_value.bulk_move_agents_to_folder = AsyncMock(
            side_effect=Exception("folder not found")
        )
        result = await move_agents_tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_ids=["a1"],
            folder_id="bad-folder",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "move_agents_failed"
