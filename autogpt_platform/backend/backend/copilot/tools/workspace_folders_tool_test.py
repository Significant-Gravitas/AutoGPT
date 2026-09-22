"""``list_workspace_files`` reports the folders beside the files.

Without them a chat can only reach what it already knows the id of: the
listing defaults to the current conversation, and the user's uploads live at
the workspace root or in a folder.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_files import (
    _MAX_RECURSIVE_FOLDERS,
    ListWorkspaceFilesTool,
    WorkspaceFileListResponse,
    _folder_subtree,
)
from backend.data.workspace_folder import WorkspaceFolder
from backend.data.workspace_scope import WorkspaceScope
from backend.util.workspace import WorkspaceManager

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _folder(folder_id: str, name: str, parent_id: str | None = None, files: int = 0):
    return WorkspaceFolder(
        id=folder_id,
        workspace_id="ws-1",
        name=name,
        parent_id=parent_id,
        created_at=NOW,
        updated_at=NOW,
        file_count=files,
    )


TREE = [
    _folder("invoices", "Invoices", files=2),
    _folder("2026", "2026", parent_id="invoices", files=5),
    _folder("q1", "Q1", parent_id="2026", files=7),
    _folder("photos", "Photos", files=1),
]


@pytest.fixture
def db():
    db = MagicMock()
    db.list_workspace_files = AsyncMock(return_value=[])
    db.count_workspace_files = AsyncMock(return_value=0)
    folders = MagicMock()
    folders.list_workspace_folders = AsyncMock(return_value=TREE)
    manager = WorkspaceManager("user-1", "ws-1", "sess-1")
    with (
        patch("backend.util.workspace.workspace_db", return_value=db),
        patch(
            "backend.util.workspace.workspace_folder_db",
            return_value=folders,
        ),
        patch(
            "backend.copilot.tools.workspace_files.get_workspace_manager",
            new=AsyncMock(return_value=manager),
        ),
    ):
        yield db


def _session() -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.session_id = "sess-1"
    return session


async def _list(db, **kwargs) -> WorkspaceFileListResponse:
    result = await ListWorkspaceFilesTool()._execute("user-1", _session(), **kwargs)
    assert isinstance(result, WorkspaceFileListResponse)
    return result


async def test_a_bare_listing_reports_the_root_folders(db):
    result = await _list(db)

    assert [f.folder_id for f in result.folders] == ["invoices", "photos"]
    assert [f.file_count for f in result.folders] == [2, 1]
    assert "[folder] Invoices (2 file(s)), folder_id=invoices" in result.message


async def test_folder_id_lists_that_folder_and_its_child_folders(db):
    result = await _list(db, folder_id="invoices")

    assert [f.folder_id for f in result.folders] == ["2026"]
    assert "in folder Invoices:" in result.message
    assert db.list_workspace_files.call_args.kwargs["folder_id"] == "invoices"
    assert db.count_workspace_files.call_args.kwargs["folder_id"] == "invoices"


async def test_recursive_spans_the_whole_subtree(db):
    result = await _list(db, folder_id="invoices", recursive=True)

    assert sorted(db.list_workspace_files.call_args.kwargs["folder_ids"]) == [
        "2026",
        "invoices",
        "q1",
    ]
    # The count uses the same axis, so "... and N more" cannot outrun the page.
    assert sorted(db.count_workspace_files.call_args.kwargs["folder_ids"]) == [
        "2026",
        "invoices",
        "q1",
    ]
    assert [f.folder_id for f in result.folders] == ["2026"]


async def test_an_empty_folder_id_is_refused_rather_than_listing_everything(db):
    """An empty string is not "no folder": the manager reads it as present and
    drops the current-session filter, then the query reads it as false and drops
    the folder filter, so the model would get the whole workspace."""
    result = await ListWorkspaceFilesTool()._execute("user-1", _session(), folder_id="")

    assert isinstance(result, ErrorResponse)
    db.list_workspace_files.assert_not_awaited()


async def test_recursive_is_ignored_without_a_folder(db):
    await _list(db, recursive=True)
    assert db.list_workspace_files.call_args.kwargs["folder_id"] is None


def test_the_subtree_walk_is_bounded_and_says_what_it_left_out():
    chain = [_folder("f0", "f0")] + [
        _folder(f"f{i}", f"f{i}", parent_id=f"f{i - 1}")
        for i in range(1, _MAX_RECURSIVE_FOLDERS + 10)
    ]
    subtree, unsearched = _folder_subtree(chain, "f0")

    assert len(subtree) == _MAX_RECURSIVE_FOLDERS
    assert unsearched == len(chain) - _MAX_RECURSIVE_FOLDERS == 10
    # Nearest first, so what is dropped is the deepest rather than an
    # arbitrary set the caller cannot name.
    assert subtree[:3] == ["f0", "f1", "f2"]


def test_the_subtree_walk_terminates_on_a_cycle():
    """The move endpoint refuses one, but data predating it must not hang."""
    looped = [
        _folder("a", "a", parent_id="b"),
        _folder("b", "b", parent_id="a"),
    ]
    subtree, unsearched = _folder_subtree(looped, "a")

    assert sorted(subtree) == ["a", "b"]
    assert unsearched == 0


@pytest.mark.parametrize(
    "scope, sees_folders",
    [
        (WorkspaceScope(expert_id="exp-1", session_ids=["sess-1"]), False),
        (WorkspaceScope(expert_id="exp-1", reads_user_files=True), True),
        (None, True),
    ],
)
async def test_folders_are_listed_only_for_a_scope_that_reads_user_files(
    db, scope, sees_folders
):
    manager = WorkspaceManager("user-1", "ws-1", "sess-1", scope=scope)
    with patch(
        "backend.copilot.tools.workspace_files.get_workspace_manager",
        new=AsyncMock(return_value=manager),
    ):
        result = await _list(db)

    assert bool(result.folders) is sees_folders
    assert ("[folder]" in result.message) is sees_folders
