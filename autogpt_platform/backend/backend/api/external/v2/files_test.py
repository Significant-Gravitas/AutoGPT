"""Uploads and the file listing must address the same storage.

Before this, `POST /files/upload` wrote to the temporary cloud bucket while
every other file endpoint read the persistent workspace, so an uploaded file
never appeared in the list and had no id to download or delete.
"""

import io
import mimetypes
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest import mock

import pytest
import pytest_mock
from fastapi import HTTPException, UploadFile
from prisma.enums import APIKeyPermission
from starlette.datastructures import Headers

from backend.api.external.v2.files import get_file, list_files, upload_file
from backend.api.external.v2.pagination import PageRequest
from backend.api.external.v2.tenancy import TenantContext
from backend.data.workspace import Workspace, WorkspaceFile

USER_ID = "user-1"
ORG_ID = "org-1"
WORKSPACE_ID = "ws-1"


async def test_an_uploaded_file_is_listed_and_addressable(
    workspace: dict[str, WorkspaceFile],
) -> None:
    """The whole of item 1.9: upload, then find the same file by the id it returned."""
    uploaded = await upload_file(
        file=_upload("report.csv", b"a,b\n1,2\n", "text/csv"),
        overwrite=False,
        auth=_auth(),
    )

    listing = await list_files(page=PageRequest(limit=25), auth=_auth())
    assert [f.id for f in listing.items] == [uploaded.id]

    found = await get_file(file_id=uploaded.id, auth=_auth())
    assert (found.name, found.path) == ("report.csv", "report.csv")


async def test_the_upload_returns_a_uri_agent_file_inputs_accept(
    workspace: dict[str, WorkspaceFile],
) -> None:
    """`workspace://<id>#<mime>` is what `store_media_file` resolves for a run."""
    uploaded = await upload_file(
        file=_upload("clip.mp4", b"\x00\x01", "video/mp4"),
        overwrite=False,
        auth=_auth(),
    )

    assert uploaded.file_uri == f"workspace://{uploaded.id}#video/mp4"


async def test_a_second_upload_of_the_same_name_conflicts_unless_overwritten(
    workspace: dict[str, WorkspaceFile],
) -> None:
    await upload_file(file=_upload("notes.txt", b"one"), overwrite=False, auth=_auth())

    with pytest.raises(HTTPException) as raised:
        await upload_file(
            file=_upload("notes.txt", b"two"), overwrite=False, auth=_auth()
        )
    assert raised.value.status_code == 409

    replaced = await upload_file(
        file=_upload("notes.txt", b"two"), overwrite=True, auth=_auth()
    )
    assert list(workspace) == [replaced.id]


async def test_an_upload_over_the_storage_quota_is_undone(
    workspace: dict[str, WorkspaceFile],
    quota_bytes: mock.AsyncMock,
) -> None:
    """A write that breaches the quota must not survive as an unlisted file."""
    quota_bytes.return_value = 1

    with pytest.raises(HTTPException) as raised:
        await upload_file(
            file=_upload("big.bin", b"0" * 64), overwrite=False, auth=_auth()
        )

    assert raised.value.status_code == 413
    assert workspace == {}


@pytest.fixture
def quota_bytes(mocker: pytest_mock.MockFixture) -> mock.AsyncMock:
    return mocker.patch(
        "backend.api.features.workspace.service.get_workspace_storage_limit_bytes",
        new=mock.AsyncMock(return_value=0),
    )


@pytest.fixture
def workspace(
    mocker: pytest_mock.MockFixture, quota_bytes: mock.AsyncMock
) -> dict[str, WorkspaceFile]:
    """One in-memory workspace behind both the write path and the read path.

    The point of the fixture: the upload endpoint and the listing endpoints
    reach the *same* store, so a test can only pass if they do in production.
    """
    files: dict[str, WorkspaceFile] = {}

    async def write_file(
        content: bytes, filename: str, *, overwrite: bool = False, metadata=None
    ) -> WorkspaceFile:
        existing = next((f for f in files.values() if f.path == filename), None)
        if existing and not overwrite:
            raise ValueError(f"File already exists at path: {filename}")
        if existing:
            del files[existing.id]
        file = _workspace_file(filename, len(content))
        files[file.id] = file
        return file

    async def delete_file(file_id: str) -> None:
        files.pop(file_id, None)

    manager = mocker.patch(
        "backend.api.features.workspace.service.WorkspaceManager"
    ).return_value
    manager.write_file = mock.AsyncMock(side_effect=write_file)
    manager.delete_file = mock.AsyncMock(side_effect=delete_file)

    mocker.patch(
        "backend.api.features.workspace.service.get_or_create_workspace",
        new=mock.AsyncMock(
            return_value=Workspace(
                id=WORKSPACE_ID,
                user_id=USER_ID,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ),
    )
    mocker.patch(
        "backend.api.features.workspace.service.get_workspace_total_size",
        new=mock.AsyncMock(
            side_effect=lambda _: sum(f.size_bytes for f in files.values())
        ),
    )
    mocker.patch(
        "backend.api.external.v2.files.file_upload_limiter.check",
        new=mock.AsyncMock(),
    )

    async def get_workspace(user_id: str) -> Workspace:
        return Workspace(
            id=WORKSPACE_ID,
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    mocker.patch(
        "backend.api.external.v2.files.get_workspace",
        new=mock.AsyncMock(side_effect=get_workspace),
    )
    mocker.patch(
        "backend.api.external.v2.files.count_workspace_files",
        new=mock.AsyncMock(side_effect=lambda _: len(files)),
    )
    mocker.patch(
        "backend.api.external.v2.files.list_workspace_files",
        new=mock.AsyncMock(side_effect=lambda **_: list(files.values())),
    )
    mocker.patch(
        "backend.api.external.v2.files.get_workspace_file",
        new=mock.AsyncMock(side_effect=lambda file_id, _: files.get(file_id)),
    )
    return files


def _workspace_file(name: str, size_bytes: int) -> WorkspaceFile:
    now = datetime.now(timezone.utc)
    return WorkspaceFile(
        id=str(uuid.uuid4()),
        workspace_id=WORKSPACE_ID,
        created_at=now,
        updated_at=now,
        name=name,
        path=name,
        storage_path=f"local://{name}",
        # Same derivation the real WorkspaceManager applies.
        mime_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
        size_bytes=size_bytes,
    )


def _upload(
    filename: str, content: bytes, content_type: Optional[str] = None
) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type} if content_type else {}),
    )


def _auth() -> TenantContext:
    return TenantContext(
        user_id=USER_ID,
        scopes=[APIKeyPermission.READ_FILES, APIKeyPermission.WRITE_FILES],
        type="api_key",
        organization_id=ORG_ID,
    )
