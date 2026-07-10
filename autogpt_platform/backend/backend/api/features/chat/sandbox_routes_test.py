"""Tests for the copilot sandbox IDE REST endpoints and porcelain parsing."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from e2b import FileType

from backend.api.features.chat import sandbox_routes
from backend.api.features.chat.sandbox_routes import _parse_porcelain
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(sandbox_routes.router)


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    """Mirror the production NotFoundError → 404 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Override auth + the feature-flag guard for all tests in this module."""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[sandbox_routes.sandbox_flag_dependency] = lambda: None
    yield
    app.dependency_overrides.clear()


def _setup_sandbox(
    mocker: pytest_mock.MockerFixture, *, session_exists: bool = True
) -> MagicMock:
    """Patch session lookup, config and sandbox acquisition; return the sandbox mock."""
    mocker.patch(
        "backend.api.features.chat.sandbox_routes.get_chat_session_metadata",
        new_callable=AsyncMock,
        return_value=MagicMock() if session_exists else None,
    )
    cfg = MagicMock()
    cfg.e2b_active = True
    cfg.e2b_api_key = "test-key"
    cfg.e2b_sandbox_timeout = 420
    cfg.e2b_sandbox_template = "base"
    cfg.e2b_sandbox_on_timeout = "pause"
    mocker.patch(
        "backend.api.features.chat.sandbox_routes.ChatConfig", return_value=cfg
    )

    sandbox = MagicMock()
    sandbox.files.list = AsyncMock(return_value=[])
    sandbox.files.read = AsyncMock(return_value=bytearray(b""))
    sandbox.files.write = AsyncMock()
    sandbox.commands.run = AsyncMock()
    mocker.patch(
        "backend.api.features.chat.sandbox_routes.get_or_create_sandbox",
        new_callable=AsyncMock,
        return_value=sandbox,
    )
    return sandbox


def test_get_tree_happy_path(mocker: pytest_mock.MockerFixture):
    sandbox = _setup_sandbox(mocker)
    file_entry = MagicMock(path="/home/user/a.py", type=FileType.FILE)
    file_entry.name = "a.py"
    dir_entry = MagicMock(path="/home/user/src", type=FileType.DIR)
    dir_entry.name = "src"
    sandbox.files.list.return_value = [file_entry, dir_entry]

    resp = client.get("/sessions/sess-1/sandbox/tree")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    # Dirs sort before files.
    assert [e["name"] for e in entries] == ["src", "a.py"]
    assert entries[0]["type"] == "dir"
    assert entries[1]["path"] == "a.py"


def test_get_tree_path_escape_rejected(mocker: pytest_mock.MockerFixture):
    _setup_sandbox(mocker)
    resp = client.get("/sessions/sess-1/sandbox/tree", params={"path": "../etc"})
    assert resp.status_code == 400


def test_get_tree_unknown_session_404(mocker: pytest_mock.MockerFixture):
    _setup_sandbox(mocker, session_exists=False)
    resp = client.get("/sessions/missing/sandbox/tree")
    assert resp.status_code == 404


def test_get_file_happy_path(mocker: pytest_mock.MockerFixture):
    sandbox = _setup_sandbox(mocker)
    sandbox.files.read.return_value = bytearray(b"print('hi')")
    resp = client.get("/sessions/sess-1/sandbox/file", params={"path": "a.py"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "print('hi')"
    assert body["truncated"] is False


def test_get_file_truncates_large_content(mocker: pytest_mock.MockerFixture):
    sandbox = _setup_sandbox(mocker)
    sandbox.files.read.return_value = bytearray(b"x" * (1024 * 1024 + 10))
    resp = client.get("/sessions/sess-1/sandbox/file", params={"path": "big.txt"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["truncated"] is True
    assert len(body["content"]) == 1024 * 1024


def test_write_file_uses_resolved_path(mocker: pytest_mock.MockerFixture):
    sandbox = _setup_sandbox(mocker)
    resp = client.put(
        "/sessions/sess-1/sandbox/file",
        json={"path": "dir/a.py", "content": "hello"},
    )
    assert resp.status_code == 200
    sandbox.files.write.assert_awaited_once_with("/home/user/dir/a.py", "hello")
    assert resp.json()["content"] == "hello"


def test_changes_non_repo_returns_empty(mocker: pytest_mock.MockerFixture):
    sandbox = _setup_sandbox(mocker)
    sandbox.commands.run.return_value = MagicMock(
        stdout="", stderr="not a git repo", exit_code=1
    )
    resp = client.get("/sessions/sess-1/sandbox/changes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_git_repo"] is False
    assert body["files"] == []


def test_diff_untracked_file_has_empty_original(mocker: pytest_mock.MockerFixture):
    sandbox = _setup_sandbox(mocker)
    # git show HEAD:<path> fails for an untracked file.
    sandbox.commands.run.return_value = MagicMock(
        stdout="", stderr="fatal: path not in HEAD", exit_code=128
    )
    sandbox.files.read.return_value = bytearray(b"new content")
    resp = client.get("/sessions/sess-1/sandbox/diff", params={"path": "new.py"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["original"] == ""
    assert body["modified"] == "new content"


def test_parse_porcelain_variants():
    parsed = _parse_porcelain(
        " M foo.py\n"
        "A  new.py\n"
        " D gone.py\n"
        "?? untracked.py\n"
        "R  old.py -> new.py\n"
    )
    assert [(f.status, f.path) for f in parsed] == [
        ("M", "foo.py"),
        ("A", "new.py"),
        ("D", "gone.py"),
        ("?", "untracked.py"),
        ("R", "new.py"),
    ]


def test_parse_porcelain_empty():
    assert _parse_porcelain("") == []
