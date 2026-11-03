"""A chat in an organization reaches every file its person keeps there, by ID,
by path and in listings: a chat's own folder is where it starts, not a wall.
What stays a wall is the organization and team, which filter every query, under
a live lease, for a chat that still belongs to them."""

import gc
import warnings
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.db_accessors import LiveResourceAccessRevoked
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_test import _make_workspace_file

_SCOPE = {"organization_id": "org-1", "team_id": "team-1", "user_global_config": False}
_OTHER_CHATS_FILE = _make_workspace_file(
    id="file-b", path="/sessions/chat-b/notes.txt"
).model_copy(
    update={"session_id": "chat-b", "organization_id": "org-1", "team_id": "team-1"}
)


class _Guard:
    async def run(self, action):
        return await action


@pytest.fixture
def leases():
    calls: list[tuple] = []

    @asynccontextmanager
    async def lease(*args):
        calls.append(args)
        yield _Guard()

    with patch("backend.util.workspace.live_resource_lease", lease):
        yield calls


@pytest.fixture
def session_check():
    check = AsyncMock()
    with patch("backend.util.workspace.require_exact_chat_session_scope", check):
        yield check


@pytest.fixture
def db():
    db = MagicMock()
    db.get_workspace_file = AsyncMock(return_value=_OTHER_CHATS_FILE)
    db.get_workspace_file_by_path = AsyncMock(return_value=_OTHER_CHATS_FILE)
    db.list_workspace_files = AsyncMock(return_value=[_OTHER_CHATS_FILE])
    with patch("backend.util.workspace.workspace_db", return_value=db):
        yield db


@pytest.fixture
def storage():
    storage = AsyncMock()
    storage.retrieve.return_value = b"notes"
    with patch("backend.util.workspace.get_workspace_storage", return_value=storage):
        yield storage


def _chat_a() -> WorkspaceManager:
    return WorkspaceManager(
        "user-1", "ws-1", "chat-a", organization_id="org-1", team_id="team-1"
    )


async def test_a_chat_reads_another_chats_file_by_id(
    db, storage, leases, session_check
):
    assert await _chat_a().read_file_by_id("file-b") == b"notes"

    db.get_workspace_file.assert_awaited_once_with("file-b", "ws-1", **_SCOPE)
    assert leases == [("user-1", "org-1", "team-1", "execute")]
    session_check.assert_awaited_once_with("chat-a", "user-1", "org-1", "team-1")


async def test_a_chat_reads_another_chats_file_by_path(
    db, storage, leases, session_check
):
    path = "/sessions/chat-b/notes.txt"

    assert await _chat_a().read_file(path) == b"notes"

    db.get_workspace_file_by_path.assert_awaited_once_with("ws-1", path, **_SCOPE)
    session_check.assert_awaited_once()


async def test_listing_every_chat_keeps_the_org_and_team_filters(
    db, leases, session_check
):
    files = await _chat_a().list_files(include_all_sessions=True)

    assert files == [_OTHER_CHATS_FILE]
    query = db.list_workspace_files.await_args.kwargs
    assert query["path_prefix"] is None
    assert (query["organization_id"], query["team_id"]) == ("org-1", "team-1")
    assert leases == [("user-1", "org-1", "team-1", "execute")]


async def test_a_listing_still_starts_in_the_chats_own_folder(
    db, leases, session_check
):
    await _chat_a().list_files()

    assert db.list_workspace_files.await_args.kwargs["path_prefix"] == (
        "/sessions/chat-a/"
    )


async def test_a_chat_that_left_the_org_reads_nothing(db, storage, leases):
    revoked = AsyncMock(side_effect=LiveResourceAccessRevoked("revoked"))

    with patch("backend.util.workspace.require_exact_chat_session_scope", revoked):
        with pytest.raises(LiveResourceAccessRevoked):
            await _chat_a().read_file_by_id("file-b")

    db.get_workspace_file.assert_not_awaited()
    storage.retrieve.assert_not_awaited()


async def test_a_refused_read_is_closed_rather_than_left_unawaited(db, leases):
    # A fresh exception, so no traceback outlives the read and keeps it alive.
    revoked = AsyncMock(side_effect=LiveResourceAccessRevoked)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch("backend.util.workspace.require_exact_chat_session_scope", revoked):
            with pytest.raises(LiveResourceAccessRevoked):
                await _chat_a().read_file_by_id("file-b")
        gc.collect()

    assert not [w for w in caught if "never awaited" in str(w.message)]
