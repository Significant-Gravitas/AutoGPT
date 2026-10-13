"""``write_workspace_file`` with no ``filename`` names the real cause.

``content`` / ``content_base64`` / ``source_path`` are named parameters, so
probing ``**kwargs`` for them always came up empty and every missing-filename
call was reported as a truncated tool call (SECRT-2441).
"""

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_files import WriteWorkspaceFileTool


def _session() -> ChatSession:
    return ChatSession.new("user-1", dry_run=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": "hello"},
        {"content": ""},
        {"content_base64": "aGVsbG8="},
        {"source_path": "/home/user/file.md"},
        {"path": "/reports/q1.md"},
    ],
)
async def test_any_argument_reports_the_missing_filename(kwargs: dict):
    resp = await WriteWorkspaceFileTool()._execute(
        user_id="user-1", session=_session(), **kwargs
    )

    assert isinstance(resp, ErrorResponse)
    assert "filename" in resp.message
    assert "truncated" not in resp.message


@pytest.mark.asyncio
async def test_no_arguments_at_all_still_reports_truncation():
    resp = await WriteWorkspaceFileTool()._execute(user_id="user-1", session=_session())

    assert isinstance(resp, ErrorResponse)
    assert "truncated" in resp.message
    assert "source_path" in resp.message
