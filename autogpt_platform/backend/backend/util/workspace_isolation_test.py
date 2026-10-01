"""Expert-scoped WorkspaceManager: every path- and ID-based operation must
stay inside the expert's resolved grants."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_test import _make_workspace_file

SCOPE = WorkspaceScope(
    expert_id="expert-a",
    owns_skills_folder=True,
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


@pytest.fixture(autouse=True)
def write_infrastructure():
    """A completed ``write_file`` reaches the quota lookup, whose DatabaseManager
    RPC has no server here and retries for ~100 × 30 s instead of failing."""
    with (
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=250 * 1024 * 1024,
        ),
    ):
        yield


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
    [
        "/experts/expert-a/skills/mine/SKILL.md",
        "/experts/expert-a/skills/mine/references/file.txt",
    ],
)
async def test_own_skills_folder_is_readable(db, storage, path: str):
    assert await _skills_manager().read_file(path) == b"allowed"


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
    [
        "/skills/autopilot/SKILL.md",
        "/skills/autopilot/references/file.txt",
        "/experts/expert-b/skills/theirs/SKILL.md",
        "/root-file.txt",
    ],
)
async def test_paths_outside_any_grant_are_denied(db, path: str):
    with pytest.raises(WorkspaceAccessDeniedError):
        await _skills_manager().read_file(path)
    with pytest.raises(WorkspaceAccessDeniedError):
        await _skills_manager().write_file(b"x", "f", path=path, overwrite=True)
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


async def test_requested_prefixes_are_narrowed_to_the_scope(db):
    manager = _expert_manager()
    await manager.list_files(
        include_all_sessions=True,
        allowed_path_prefixes=["/sessions/expert-a/", "/sessions/expert-b/"],
    )
    assert db.list_workspace_files.call_args.kwargs["allowed_path_prefixes"] == [
        "/sessions/expert-a/"
    ]
    await manager.get_file_count(
        include_all_sessions=True, allowed_path_prefixes=["/sessions/expert-b/"]
    )
    assert db.count_workspace_files.call_args.kwargs["allowed_path_prefixes"] == []


async def test_owner_without_scope_lists_the_requested_prefixes(db):
    manager = WorkspaceManager("user-1", "ws-1", "personal")
    await manager.list_files(
        include_all_sessions=True, allowed_path_prefixes=["/sessions/expert-b/"]
    )
    assert db.list_workspace_files.call_args.kwargs["allowed_path_prefixes"] == [
        "/sessions/expert-b/"
    ]


# What ``resolve_expert_workspace_scope`` builds for a live hired expert: the
# same grants plus read access to the owner's own files.
USER_FILES_SCOPE = SCOPE.model_copy(update={"reads_user_files": True})


def _user_files_manager() -> WorkspaceManager:
    return WorkspaceManager("user-1", "ws-1", "expert-a", scope=USER_FILES_SCOPE)


async def test_a_user_file_is_readable_by_id(db, storage):
    db.get_workspace_file = AsyncMock(
        return_value=_make_workspace_file(path="/quarterly-report.pdf")
    )
    manager = _user_files_manager()

    assert await manager.read_file_by_id("root-upload") == b"allowed"
    assert (await manager.get_file_info("root-upload")) is not None
    await manager.get_download_url("root-upload")


async def test_a_user_file_is_never_writable(db, storage):
    db.get_workspace_file = AsyncMock(
        return_value=_make_workspace_file(path="/quarterly-report.pdf")
    )

    with pytest.raises(WorkspaceAccessDeniedError):
        await _user_files_manager().delete_file("root-upload")
    db.soft_delete_workspace_file.assert_not_awaited()

    # A session-less manager takes the path as written, so this is the shape
    # that actually asks to overwrite the user's own file.
    session_less = WorkspaceManager("user-1", "ws-1", None, scope=USER_FILES_SCOPE)
    with pytest.raises(WorkspaceAccessDeniedError):
        await session_less.write_file(
            b"x", "quarterly-report.pdf", path="/quarterly-report.pdf", overwrite=True
        )


async def test_a_root_path_written_from_a_session_lands_in_that_session(db, storage):
    """``_resolve_path`` re-homes a bare path under the current conversation,
    so an expert cannot address the user's own file by path at all — reaching
    one is by file_id, which is what the tool descriptions now say."""
    db.create_workspace_file = AsyncMock(
        return_value=_make_workspace_file(
            path="/sessions/expert-a/quarterly-report.pdf"
        )
    )

    await _user_files_manager().write_file(
        b"x", "quarterly-report.pdf", path="/quarterly-report.pdf", overwrite=True
    )

    assert (
        db.create_workspace_file.call_args.kwargs["path"]
        == "/sessions/expert-a/quarterly-report.pdf"
    )


async def test_another_experts_conversation_stays_denied_with_the_grant(db, storage):
    with pytest.raises(WorkspaceAccessDeniedError):
        await _user_files_manager().read_file_by_id("foreign-file")
    storage.retrieve.assert_not_awaited()


async def test_the_query_widens_to_user_files_only_with_the_grant(db):
    await _user_files_manager().list_files(include_all_sessions=True)
    assert db.list_workspace_files.call_args.kwargs["include_user_files"] is True
    await _user_files_manager().get_file_count(include_all_sessions=True)
    assert db.count_workspace_files.call_args.kwargs["include_user_files"] is True

    await _expert_manager().list_files(include_all_sessions=True)
    assert db.list_workspace_files.call_args.kwargs["include_user_files"] is False


async def test_a_folder_listing_lifts_the_current_session_default(db):
    """A folder is a workspace-wide axis: keeping the session prefix on would
    intersect it to nothing for any file outside this conversation."""
    manager = _user_files_manager()
    await manager.list_files(folder_id="fld-1")
    list_kwargs = db.list_workspace_files.call_args.kwargs
    assert list_kwargs["path_prefix"] is None
    assert list_kwargs["folder_id"] == "fld-1"
    assert list_kwargs["allowed_path_prefixes"] == USER_FILES_SCOPE.read_prefixes

    await manager.list_files()
    assert (
        db.list_workspace_files.call_args.kwargs["path_prefix"] == "/sessions/expert-a/"
    )
