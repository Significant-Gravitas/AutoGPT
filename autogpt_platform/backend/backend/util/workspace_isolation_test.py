"""Expert-scoped WorkspaceManager: every path- and ID-based operation must
stay inside the expert's resolved grants."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_test import _make_workspace_file

SCOPE = WorkspaceScope(
    expert_id="expert-a",
    session_ids=["expert-a", "expert-a-old"],
    delegated_session_ids=["sub-1"],
)


@pytest.fixture
def db():
    db = MagicMock()
    db.get_workspace_file = AsyncMock(
        return_value=_make_workspace_file(path="/sessions/expert-b/private.txt")
    )
    db.get_workspace_file_by_path = AsyncMock(return_value=_make_workspace_file())
    db.list_workspace_files = AsyncMock(return_value=[])
    db.count_workspace_files = AsyncMock(return_value=0)
    db.soft_delete_workspace_file = AsyncMock()
    db.get_workspace_total_size = AsyncMock(return_value=0)
    with patch("backend.util.workspace.workspace_db", return_value=db):
        yield db


@pytest.fixture
def storage():
    storage = AsyncMock()
    storage.retrieve.return_value = b"allowed"
    with patch("backend.util.workspace.get_workspace_storage", return_value=storage):
        yield storage


def _expert_manager() -> WorkspaceManager:
    return WorkspaceManager("user-1", "ws-1", "expert-a", scope=SCOPE)


def _skills_manager() -> WorkspaceManager:
    """Session-less manager, as the skills registry builds it."""
    return WorkspaceManager("user-1", "ws-1", None, scope=SCOPE)


@pytest.mark.parametrize(
    "operation", ["read_file_by_id", "get_file_info", "delete_file", "get_download_url"]
)
async def test_foreign_file_id_is_denied(db, storage, operation: str):
    with pytest.raises(WorkspaceAccessDeniedError):
        await getattr(_expert_manager(), operation)("foreign-file")
    storage.retrieve.assert_not_awaited()
    storage.delete.assert_not_awaited()
    storage.get_download_url.assert_not_awaited()
    db.soft_delete_workspace_file.assert_not_awaited()


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/expert-b/private.txt",
        "/sessions/personal/private.txt",
        "/sessions/expert-a-older/private.txt",
        "/sessions/expert-a/../expert-b/private.txt",
    ],
)
async def test_explicit_paths_cannot_escape_scope(db, path: str):
    manager = _expert_manager()
    with pytest.raises(WorkspaceAccessDeniedError):
        await manager.read_file(path)
    with pytest.raises(WorkspaceAccessDeniedError):
        await manager.get_file_info_by_path(path)
    with pytest.raises(WorkspaceAccessDeniedError):
        await manager.write_file(b"x", "private.txt", path=path, overwrite=True)
    db.get_workspace_file_by_path.assert_not_awaited()


@pytest.mark.parametrize(
    "path",
    ["/skills/autopilot/SKILL.md", "/skills/autopilot/references/file.txt"],
)
async def test_skills_registry_is_readable_but_not_writable(db, storage, path: str):
    assert await _skills_manager().read_file(path) == b"allowed"
    with pytest.raises(WorkspaceAccessDeniedError):
        await _skills_manager().write_file(b"x", "f", path=path, overwrite=True)


@pytest.mark.parametrize(
    "path",
    [
        "relative.txt",
        "/sessions/expert-a-old/file.txt",
        "/sessions/sub-1/result.json",
    ],
)
async def test_own_and_delegated_paths_are_readable(db, storage, path: str):
    assert await _expert_manager().read_file(path) == b"allowed"


@pytest.mark.parametrize(
    "path",
    ["/experts/expert-a/skills/mine/SKILL.md", "/root-file.txt"],
)
async def test_paths_outside_any_grant_are_denied(db, path: str):
    with pytest.raises(WorkspaceAccessDeniedError):
        await _skills_manager().read_file(path)
    db.get_workspace_file_by_path.assert_not_awaited()


async def test_delegated_session_is_not_writable(db):
    with pytest.raises(WorkspaceAccessDeniedError):
        await _expert_manager().write_file(
            b"x", "f", path="/sessions/sub-1/result.json", overwrite=True
        )


async def test_listing_and_counts_are_filtered_inside_the_query(db):
    manager = _expert_manager()
    await manager.list_files(include_all_sessions=True, limit=1, offset=4)
    await manager.get_file_count(include_all_sessions=True)
    list_kwargs = db.list_workspace_files.call_args.kwargs
    assert list_kwargs["allowed_path_prefixes"] == SCOPE.read_prefixes
    assert list_kwargs["path_prefix"] is None
    assert list_kwargs["offset"] == 4
    count_kwargs = db.count_workspace_files.call_args.kwargs
    assert count_kwargs["allowed_path_prefixes"] == SCOPE.read_prefixes


async def test_owner_without_scope_keeps_full_access(db, storage):
    manager = WorkspaceManager("user-1", "ws-1", "personal")
    assert await manager.read_file("/sessions/expert-b/private.txt") == b"allowed"
    assert await manager.read_file_by_id("foreign-file") == b"allowed"
    await manager.list_files(include_all_sessions=True)
    assert db.list_workspace_files.call_args.kwargs["allowed_path_prefixes"] is None
