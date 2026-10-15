"""A shared run stops serving once it is deleted or its chat share ends.

The public share page and its file links must answer a deleted run, or a
run shared only through a chat that is no longer shared, with the same 404
as an unknown token.  The tests run the real routes and data layer over a
small in-memory stand-in for the four Prisma tables involved, so they pin
behaviour rather than the shape of a query.
"""

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.enums import SharedVia
from prisma.models import AgentGraphExecution, ChatLinkedShare, SharedExecutionFile
from starlette.responses import Response

import backend.data.db as db_module
from backend.api.features.graph_executions.routes import router
from backend.data.execution import delete_graph_execution
from backend.data.workspace import WorkspaceFile
from backend.util.exceptions import DatabaseError

app = FastAPI()
app.include_router(router, prefix="/api")

OWNER = "user-owner"
OTHER_USER = "user-other"
TOKEN = "550e8400-e29b-41d4-a716-446655440000"
OTHER_TOKEN = "7d444840-9dc0-11d1-b245-5ffdce74fad2"
UNKNOWN_TOKEN = "9b2f6c1e-0000-4000-8000-000000000000"
FILE_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
OTHER_FILE_ID = "6ba7b811-9dad-11d1-80b4-00c04fd430c8"


class _FakeShareDb:
    """Rows for AgentGraphExecution, SharedExecutionFile, ChatLinkedShare
    and ChatSession, with the subset of Prisma's where-clause semantics the
    share code uses: equality, ``OR``, ``is`` and ``some``."""

    def __init__(self) -> None:
        self.executions: dict[str, dict[str, Any]] = {}
        self.files: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []
        self.sessions: dict[str, dict[str, Any]] = {}

    def share_run(
        self,
        execution_id: str,
        *,
        user_id: str,
        token: str,
        file_id: str,
        shared_via: SharedVia = SharedVia.USER,
    ) -> None:
        self.executions[execution_id] = {
            "id": execution_id,
            "userId": user_id,
            "isDeleted": False,
            "isShared": True,
            "shareToken": token,
            "sharedAt": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "sharedVia": shared_via,
            "executionStatus": "COMPLETED",
            "createdAt": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        self.files.append(
            {"executionId": execution_id, "fileId": file_id, "shareToken": token}
        )

    def link_to_chat(self, execution_id: str, session_id: str, shared: bool) -> None:
        self.sessions[session_id] = {"id": session_id, "isShared": shared}
        self.links.append({"sessionId": session_id, "executionId": execution_id})

    def execution_client(self) -> MagicMock:
        client = MagicMock()
        client.find_first = AsyncMock(side_effect=self._find_execution)
        client.update_many = AsyncMock(side_effect=self._update_executions)
        client.delete_many = AsyncMock(side_effect=self._delete_executions)
        return client

    def file_client(self) -> MagicMock:
        client = MagicMock()
        client.find_first = AsyncMock(side_effect=self._find_file)
        client.delete_many = AsyncMock(
            side_effect=lambda where: self._remove(self.files, where)
        )
        return client

    def link_client(self) -> MagicMock:
        client = MagicMock()
        client.delete_many = AsyncMock(
            side_effect=lambda where: self._remove(self.links, where)
        )
        return client

    def _execution_view(self, row: dict[str, Any]) -> dict[str, Any]:
        links = [
            {**link, "Session": self.sessions[link["sessionId"]]}
            for link in self.links
            if link["executionId"] == row["id"]
        ]
        return {**row, "ChatLinkedShares": links}

    def _file_view(self, row: dict[str, Any]) -> dict[str, Any]:
        execution = self.executions.get(row["executionId"])
        return {**row, "Execution": execution and self._execution_view(execution)}

    async def _find_execution(self, where: dict, include: Any = None) -> Any:
        for row in self.executions.values():
            if _matches(self._execution_view(row), where):
                graph = SimpleNamespace(name="Shared agent", description=None)
                return SimpleNamespace(**row, AgentGraph=graph, NodeExecutions=[])
        return None

    async def _update_executions(self, where: dict, data: dict) -> int:
        matched = [
            row
            for row in self.executions.values()
            if _matches(self._execution_view(row), where)
        ]
        for row in matched:
            row.update(data)
        return len(matched)

    async def _delete_executions(self, where: dict) -> int:
        gone = {
            row["id"]
            for row in self.executions.values()
            if _matches(self._execution_view(row), where)
        }
        for execution_id in gone:
            del self.executions[execution_id]
        # The schema cascades both tables on an execution's hard delete.
        self.files = [f for f in self.files if f["executionId"] not in gone]
        self.links = [link for link in self.links if link["executionId"] not in gone]
        return len(gone)

    async def _find_file(self, where: dict) -> Any:
        for row in self.files:
            if _matches(self._file_view(row), where):
                return SimpleNamespace(**row)
        return None

    @staticmethod
    def _remove(rows: list[dict[str, Any]], where: dict) -> int:
        kept = [row for row in rows if not _matches(row, where)]
        removed = len(rows) - len(kept)
        rows[:] = kept
        return removed


def _matches(row: dict[str, Any] | None, where: dict[str, Any]) -> bool:
    if row is None:
        return False
    for key, condition in where.items():
        if key == "OR":
            if not any(_matches(row, branch) for branch in condition):
                return False
        elif key == "AND":
            if not all(_matches(row, branch) for branch in condition):
                return False
        elif isinstance(condition, dict) and "is" in condition:
            if not _matches(row.get(key), condition["is"]):
                return False
        elif isinstance(condition, dict) and "some" in condition:
            if not any(_matches(item, condition["some"]) for item in row[key]):
                return False
        elif isinstance(condition, dict):
            raise AssertionError(f"fake does not support filter {key}={condition}")
        elif row.get(key) != condition:
            return False
    return True


class _TxStub:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        return None


def _workspace_file(file_id: str) -> WorkspaceFile:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return WorkspaceFile(
        id=file_id,
        workspace_id="ws-1",
        created_at=now,
        updated_at=now,
        name="report.pdf",
        path="/report.pdf",
        storage_path="local://uploads/report.pdf",
        mime_type="application/pdf",
        size_bytes=4,
        checksum=None,
        is_deleted=False,
        deleted_at=None,
        metadata={},
    )


async def _download_response(file: WorkspaceFile, *, inline: bool = False) -> Response:
    return Response(content=b"%PDF", media_type="application/pdf")


@pytest.fixture
def fake_db() -> Iterator[_FakeShareDb]:
    fake = _FakeShareDb()
    with (
        patch.object(
            AgentGraphExecution, "prisma", return_value=fake.execution_client()
        ),
        patch.object(SharedExecutionFile, "prisma", return_value=fake.file_client()),
        patch.object(ChatLinkedShare, "prisma", return_value=fake.link_client()),
        patch.object(
            db_module, "prisma", MagicMock(tx=MagicMock(return_value=_TxStub()))
        ),
        patch(
            "backend.api.features.graph_executions.routes.get_workspace_file_by_id",
            new=AsyncMock(side_effect=_workspace_file),
        ),
        patch(
            "backend.api.features.graph_executions.routes.create_file_download_response",
            side_effect=_download_response,
        ),
    ):
        yield fake


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _page(client: TestClient, token: str) -> Any:
    return client.get(f"/api/public/shared/{token}")


def _file(client: TestClient, token: str, file_id: str) -> Any:
    return client.get(f"/api/public/shared/{token}/files/{file_id}/download")


def _assert_like_unknown_token(page: Any, file: Any, client: TestClient) -> None:
    unknown_page = _page(client, UNKNOWN_TOKEN)
    unknown_file = _file(client, UNKNOWN_TOKEN, FILE_ID)
    assert (page.status_code, page.content) == (404, unknown_page.content)
    assert (file.status_code, file.content) == (404, unknown_file.content)


def test_a_live_shared_run_serves_its_page_and_file(fake_db, client):
    fake_db.share_run("exec-1", user_id=OWNER, token=TOKEN, file_id=FILE_ID)

    assert _page(client, TOKEN).status_code == 200
    assert _file(client, TOKEN, FILE_ID).content == b"%PDF"


def test_a_run_soft_deleted_before_the_fix_stops_serving_its_file(fake_db, client):
    """Rows left by the old delete path: deleted, still marked shared, with
    the file allowlist intact.  The read path alone has to refuse them."""
    fake_db.share_run("exec-1", user_id=OWNER, token=TOKEN, file_id=FILE_ID)
    fake_db.executions["exec-1"]["isDeleted"] = True

    _assert_like_unknown_token(
        _page(client, TOKEN), _file(client, TOKEN, FILE_ID), client
    )


@pytest.mark.asyncio
async def test_deleting_a_shared_run_ends_its_share(fake_db, client):
    fake_db.share_run("exec-1", user_id=OWNER, token=TOKEN, file_id=FILE_ID)
    fake_db.link_to_chat("exec-1", "chat-1", shared=True)

    await delete_graph_execution(graph_exec_id="exec-1", user_id=OWNER)

    _assert_like_unknown_token(
        _page(client, TOKEN), _file(client, TOKEN, FILE_ID), client
    )
    run = fake_db.executions["exec-1"]
    assert run["isDeleted"] is True
    assert (run["isShared"], run["shareToken"], run["sharedVia"]) == (
        False,
        None,
        None,
    )
    assert fake_db.files == []
    assert fake_db.links == []


@pytest.mark.asyncio
async def test_deleting_a_run_leaves_other_users_shares_alone(fake_db, client):
    fake_db.share_run("exec-1", user_id=OWNER, token=TOKEN, file_id=FILE_ID)
    fake_db.share_run(
        "exec-2", user_id=OTHER_USER, token=OTHER_TOKEN, file_id=OTHER_FILE_ID
    )
    fake_db.link_to_chat("exec-2", "chat-2", shared=True)

    await delete_graph_execution(graph_exec_id="exec-1", user_id=OWNER)
    with pytest.raises(DatabaseError):
        await delete_graph_execution(graph_exec_id="exec-2", user_id=OWNER)

    assert fake_db.executions["exec-2"]["isShared"] is True
    assert fake_db.executions["exec-2"]["isDeleted"] is False
    assert [f["fileId"] for f in fake_db.files] == [OTHER_FILE_ID]
    assert [link["executionId"] for link in fake_db.links] == ["exec-2"]
    assert _page(client, OTHER_TOKEN).status_code == 200
    assert _file(client, OTHER_TOKEN, OTHER_FILE_ID).status_code == 200


@pytest.mark.asyncio
async def test_hard_deleting_a_shared_run_ends_its_share(fake_db, client):
    fake_db.share_run("exec-1", user_id=OWNER, token=TOKEN, file_id=FILE_ID)

    await delete_graph_execution(
        graph_exec_id="exec-1", user_id=OWNER, soft_delete=False
    )

    _assert_like_unknown_token(
        _page(client, TOKEN), _file(client, TOKEN, FILE_ID), client
    )


def test_a_chat_shared_run_stops_serving_once_no_shared_chat_links_it(fake_db, client):
    """A chat deleted while its share cascade failed leaves the run marked
    shared with no chat left to link it."""
    fake_db.share_run(
        "exec-1",
        user_id=OWNER,
        token=TOKEN,
        file_id=FILE_ID,
        shared_via=SharedVia.CHAT_LINK,
    )

    _assert_like_unknown_token(
        _page(client, TOKEN), _file(client, TOKEN, FILE_ID), client
    )


def test_a_chat_shared_run_stops_serving_once_its_chat_is_unshared(fake_db, client):
    fake_db.share_run(
        "exec-1",
        user_id=OWNER,
        token=TOKEN,
        file_id=FILE_ID,
        shared_via=SharedVia.CHAT_LINK,
    )
    fake_db.link_to_chat("exec-1", "chat-1", shared=False)

    _assert_like_unknown_token(
        _page(client, TOKEN), _file(client, TOKEN, FILE_ID), client
    )


def test_a_chat_shared_run_serves_while_its_chat_is_shared(fake_db, client):
    fake_db.share_run(
        "exec-1",
        user_id=OWNER,
        token=TOKEN,
        file_id=FILE_ID,
        shared_via=SharedVia.CHAT_LINK,
    )
    fake_db.link_to_chat("exec-1", "chat-1", shared=True)

    assert _page(client, TOKEN).status_code == 200
    assert _file(client, TOKEN, FILE_ID).status_code == 200
