"""What an expert chat may attach to a message, and how it is announced.

An expert session is confined to its own conversations, so before this it
could not attach a file the user uploaded on the Files page at all. It can
now, and a folder can be attached whole: the message names it and the model
opens it with ``list_workspace_files``.
"""

import pytest

from backend.data import workspace
from backend.data.workspace import build_files_block, resolve_attachable_workspace_files
from backend.data.workspace_folder import WorkspaceFolder
from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope
from backend.util.workspace_test import _make_workspace_file

EXPERT_SCOPE = WorkspaceScope(
    expert_id="expert-a", session_ids=["expert-a"], reads_user_files=True
)


class _Row:
    """A prisma ``UserWorkspaceFile`` as the attachment resolver reads it."""

    def __init__(self, file_id: str, name: str, path: str):
        self.id = file_id
        self.name = name
        self.path = path
        self.mimeType = "application/pdf"
        self.sizeBytes = 2048


def _resolver(mocker, rows: list[_Row]):
    mocker.patch.object(
        workspace, "resolve_workspace_files", mocker.AsyncMock(return_value=rows)
    )
    mocker.patch.object(
        workspace,
        "resolve_expert_workspace_scope",
        mocker.AsyncMock(return_value=EXPERT_SCOPE),
    )


async def test_an_expert_may_attach_the_users_own_file(mocker):
    rows = [_Row("f1", "quarterly-report.pdf", "/quarterly-report.pdf")]
    _resolver(mocker, rows)

    resolved = await resolve_attachable_workspace_files(
        "user-1", ["f1"], session_id="expert-a", expert_id="expert-a"
    )

    assert [r.id for r in resolved] == ["f1"]


async def test_another_experts_conversation_file_is_still_refused(mocker):
    rows = [_Row("f2", "secret.pdf", "/sessions/expert-b/secret.pdf")]
    _resolver(mocker, rows)

    with pytest.raises(WorkspaceAccessDeniedError) as exc_info:
        await resolve_attachable_workspace_files(
            "user-1", ["f2"], session_id="expert-a", expert_id="expert-a"
        )

    assert "secret.pdf" in str(exc_info.value)


async def test_a_revoked_expert_cannot_attach_a_user_file(mocker):
    """``resolve_expert_workspace_scope`` fails closed for an archived or
    foreign expert, and the user-files grant is carried, never inferred."""
    rows = [_Row("f1", "quarterly-report.pdf", "/quarterly-report.pdf")]
    _resolver(mocker, rows)
    mocker.patch.object(
        workspace,
        "resolve_expert_workspace_scope",
        mocker.AsyncMock(return_value=WorkspaceScope(expert_id="expert-a")),
    )

    with pytest.raises(WorkspaceAccessDeniedError):
        await resolve_attachable_workspace_files(
            "user-1", ["f1"], session_id="other", expert_id="expert-a"
        )


def _folder(name: str = "Invoices", file_count: int = 4) -> WorkspaceFolder:
    return WorkspaceFolder(
        id="fld-1",
        workspace_id="ws-1",
        name=name,
        created_at=_make_workspace_file().created_at,
        updated_at=_make_workspace_file().updated_at,
        file_count=file_count,
    )


def test_an_attached_folder_is_named_rather_than_expanded():
    block = build_files_block([], [_folder()])

    assert "Invoices (folder, 4 file(s) directly inside), folder_id=fld-1" in block
    assert "list_workspace_files with the folder_id" in block
    # No per-file lines: attaching a folder costs one line whatever it holds.
    assert "read_workspace_file" not in block


def test_files_and_folders_share_one_block():
    row = _Row("f1", "report.pdf", "/report.pdf")
    block = build_files_block([row], [_folder()])  # type: ignore[list-item]

    assert block.count("[Attached files]") == 1
    assert "file_id=f1" in block
    assert "folder_id=fld-1" in block
    assert "read_workspace_file" in block


def test_nothing_attached_is_an_empty_block():
    assert build_files_block([], []) == ""
    assert build_files_block([], None) == ""
